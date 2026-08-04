from __future__ import annotations

import gc
import glob
import json
import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import types
    from typing import Self

import numpy as np
import torch
from fast_plaid import fast_plaid_rust
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from joblib import Parallel, delayed

from ..filtering import create, delete
from .kmeans import FastKMeans
from .load import _load_index_tensors_cpu, _reload_index, save_list_tensors_on_disk
from .update import process_update


class TorchWithCudaNotFoundError(Exception):
    """Exception raised when PyTorch with CUDA support is not found."""


def _load_torch_path(device: str) -> str:
    """Find the path to the shared library for PyTorch with CUDA.

    Args:
    ----
    device:
        The target device identifier (e.g., 'cuda', 'cpu').

    """
    search_paths = [
        os.path.join(os.path.dirname(torch.__file__), "lib", f"libtorch_{device}.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", f"libtorch_{device}.so"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch_cuda.dylib"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch.dylib"),
        os.path.join(os.path.dirname(torch.__file__), "lib", f"torch_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "torch.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", f"c10_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "c10.dll"),
        os.path.join(os.path.dirname(torch.__file__), "**", f"torch_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "**", "torch.dll"),
    ]

    for path_pattern in search_paths:
        found_libs = glob.glob(path_pattern, recursive=True)
        if found_libs:
            return found_libs[0]

    error = """
    Could not find torch binary.
    Please ensure PyTorch is installed.
    """
    raise TorchWithCudaNotFoundError(error)


def compute_kmeans(
    documents_embeddings: list[torch.Tensor] | torch.Tensor,
    dim: int,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
    num_partitions: int | None = None,
) -> torch.Tensor:
    """Compute K-means centroids for document embeddings.

    Args:
    ----
    documents_embeddings:
        The embeddings to cluster.
    dim:
        The dimensionality of the embeddings.
    device:
        The device to run the computation on.
    kmeans_niters:
        The number of iterations for the K-means algorithm.
    max_points_per_centroid:
        The maximum number of points to support per centroid.
    seed:
        The random seed for initialization.
    n_samples_kmeans:
        The number of samples to use for K-means training.
    use_triton_kmeans:
        Whether to use the Triton implementation of K-means.
    num_partitions:
        If provided, explicitly sets the number of centroids (K).
        If None, K is calculated using a heuristic based on dataset size.

    """
    num_documents = len(documents_embeddings)

    if n_samples_kmeans is None:
        n_samples_kmeans = min(
            1 + int(16 * math.sqrt(120 * num_documents)),
            num_documents,
        )

    n_samples_kmeans = min(num_documents, n_samples_kmeans)

    # Use a local Generator to avoid clobbering the global torch RNG state in
    # library code.
    generator = torch.Generator()
    generator.manual_seed(seed)
    # Memory optimization: Use torch.randperm for efficient sampling
    sampled_indices = torch.randperm(num_documents, generator=generator)[
        :n_samples_kmeans
    ]

    if isinstance(documents_embeddings, torch.Tensor):
        # Indexing a tensor directly is a view-operation or efficient gather
        samples_tensor = documents_embeddings[sampled_indices]
    else:
        # Optimization: Pre-calculate total tokens to allocate a single buffer
        sampled_indices_list = sampled_indices.tolist()
        total_sample_tokens = sum(
            documents_embeddings[i].shape[0] for i in sampled_indices_list
        )

        samples_tensor = torch.empty(
            (total_sample_tokens, dim),
            dtype=torch.float16,
            device="cpu",
        )

        current_offset = 0
        for i in sampled_indices_list:
            tensor_slice = documents_embeddings[i]
            length = tensor_slice.shape[0]
            # Direct copy into the pre-allocated buffer
            samples_tensor[current_offset : current_offset + length].copy_(tensor_slice)
            current_offset += length

    total_tokens = samples_tensor.shape[0]

    # Calculate num_partitions only if not provided by the caller
    if num_partitions is None:
        # Calculate num_partitions based on the density of the sample relative to
        # the whole dataset.
        avg_tokens_per_doc = total_tokens / n_samples_kmeans
        estimated_total_tokens = avg_tokens_per_doc * num_documents
        num_partitions = int(
            2 ** math.floor(math.log2(16 * math.sqrt(estimated_total_tokens)))
        )

    if samples_tensor.is_cuda:
        samples_tensor = samples_tensor.to(device="cpu", dtype=torch.float16)

    # The actual K that will be used by FastKMeans
    actual_k = min(num_partitions, total_tokens)

    kmeans = FastKMeans(
        d=dim,
        k=actual_k,
        niter=kmeans_niters,
        gpu=device.startswith("cuda"),
        verbose=False,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        use_triton=use_triton_kmeans,
    )

    centroids = kmeans.train(data=samples_tensor).to(
        device=device,
        dtype=torch.float32,
    )

    # Explicitly clear the large sample buffer before creating centroids
    del samples_tensor
    gc.collect()

    return torch.nn.functional.normalize(
        input=centroids,
        dim=-1,
    ).half()


def device_memory_budget(device: str, search_memory_fraction: float) -> int:
    """Scoring budget in bytes: `search_memory_fraction` of free VRAM; 0 on CPU."""
    if device.startswith("cuda") and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        return int(free * search_memory_fraction)
    return 0


def _resolve_batch_size(batch_size: int | Literal["auto"]) -> int:
    """Map 'auto' to 0 (memory-planned chunks); validate manual chunk sizes."""
    if batch_size == "auto":
        return 0
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be 'auto' or a positive int.")
    return batch_size


def _true_query_lengths(queries: list[torch.Tensor]) -> list[int]:
    """Last non-zero row + 1 per query (min 1), vectorized over same-shape chunks."""
    lengths: list[int] = []
    start = 0
    n = len(queries)
    chunk = 256
    while start < n:
        # Group contiguous equal-shaped queries (PyLate pads all to query_length).
        shape = queries[start].shape
        end = start + 1
        limit = min(start + chunk, n)
        while end < limit and queries[end].shape == shape:
            end += 1
        stacked = torch.stack(queries[start:end])
        row_is_nonzero = (stacked != 0).any(dim=-1)
        positions = torch.arange(
            1, row_is_nonzero.shape[1] + 1, device=row_is_nonzero.device
        )
        chunk_lengths = (row_is_nonzero * positions).amax(dim=1).clamp_min(1)
        lengths.extend(int(v) for v in chunk_lengths.tolist())
        start = end
    return lengths


def pack_queries(
    queries_embeddings: torch.Tensor | list[torch.Tensor],
) -> tuple[torch.Tensor, list[int]]:
    """Trim per-query zero-row padding and pack into a [total_tokens, dim] fp16 tensor.

    Returns the packed tensor and the true token count per query.
    """
    if isinstance(queries_embeddings, torch.Tensor):
        if queries_embeddings.dim() == 2:
            queries_embeddings = [queries_embeddings]
        else:
            queries_embeddings = list(queries_embeddings.unbind(0))

    queries = [q.squeeze(0) if q.dim() == 3 else q for q in queries_embeddings]
    if not queries:
        return torch.empty(0, 0, dtype=torch.float16), []

    lengths = _true_query_lengths(queries)
    pieces = [q[:length] for q, length in zip(queries, lengths)]
    packed = torch.cat(pieces, dim=0)
    return packed.to(torch.float16), lengths


def search_on_device(
    device: str,
    queries_embeddings: torch.Tensor,
    batch_size: int | Literal["auto"],
    n_full_scores: int,
    top_k: int,
    n_ivf_probe: int,
    search_memory_fraction: float,
    index_object: Any,
    show_progress: bool,
    query_lengths: list[int],
    subset: list[list[int]] | None = None,
) -> list[list[tuple[int, float]]]:
    """Perform a search on a single specified device using the passed object.

    Args:
    ----
    device:
        The device identifier to perform the search on.
    queries_embeddings:
        A packed 2D tensor (total_query_tokens, embedding_dim) holding all
        queries concatenated without padding.
    batch_size:
        Documents per scoring chunk; 'auto' plans chunks from the memory budget.
    n_full_scores:
        The number of full scores to compute per query.
    top_k:
        The number of top results to return.
    n_ivf_probe:
        The number of IVF clusters to probe.
    search_memory_fraction:
        Fraction of free VRAM the scoring stages may use.
    index_object:
        The loaded index object for the specific device.
    show_progress:
        Whether to show a progress bar.
    query_lengths:
        True token count per query in the packed tensor.
    subset:
        Optional subset of document IDs to search within.

    """
    # Guard clause to prevent the TypeError in Rust binding
    if index_object is None:
        error = f"""
        Index object is None for device '{device}'.
        This usually means the index was not found or failed to load.
        """
        raise ValueError(error)

    search_parameters = fast_plaid_rust.SearchParameters(
        batch_size=_resolve_batch_size(batch_size),
        n_full_scores=n_full_scores,
        top_k=top_k,
        n_ivf_probe=n_ivf_probe,
        memory_budget_bytes=device_memory_budget(device, search_memory_fraction),
    )

    scores = fast_plaid_rust.pysearch(
        index=index_object,
        device=device,
        queries_embeddings=queries_embeddings.to(dtype=torch.float16),
        search_parameters=search_parameters,
        show_progress=show_progress,
        subset=subset,
        query_lengths=query_lengths,
    )

    return [
        [
            (passage_id, score)
            for score, passage_id in zip(score.scores, score.passage_ids)
        ]
        for score in scores
    ]


def search_on_device_with_token_scores(
    device: str,
    queries_embeddings: torch.Tensor,
    batch_size: int | Literal["auto"],
    n_full_scores: int,
    top_k: int,
    n_ivf_probe: int,
    search_memory_fraction: float,
    index_object: Any,
    show_progress: bool,
    query_lengths: list[int],
    subset: list[list[int]] | None = None,
) -> list[list[tuple[int, float, torch.Tensor]]]:
    """Perform a search on a single device, returning token-level similarity matrices.

    Args:
    ----
    device:
        The device identifier to perform the search on.
    queries_embeddings:
        The query embeddings to search for.
    batch_size:
        Documents per scoring chunk; 'auto' plans chunks from the memory budget.
    n_full_scores:
        The number of full scores to compute per query.
    top_k:
        The number of top results to return.
    n_ivf_probe:
        The number of IVF clusters to probe.
    search_memory_fraction:
        Fraction of free VRAM the scoring stages may use.
    index_object:
        The loaded index object for the specific device.
    show_progress:
        Whether to show a progress bar.
    query_lengths:
        True token count per query in the packed tensor.
    subset:
        Optional subset of document IDs to search within.

    """
    if index_object is None:
        error = f"""
        Index object is None for device '{device}'.
        This usually means the index was not found or failed to load.
        """
        raise ValueError(error)

    search_parameters = fast_plaid_rust.SearchParameters(
        batch_size=_resolve_batch_size(batch_size),
        n_full_scores=n_full_scores,
        top_k=top_k,
        n_ivf_probe=n_ivf_probe,
        memory_budget_bytes=device_memory_budget(device, search_memory_fraction),
    )

    results = fast_plaid_rust.pysearch_with_token_scores(
        index=index_object,
        device=device,
        queries_embeddings=queries_embeddings.to(dtype=torch.float16),
        search_parameters=search_parameters,
        show_progress=show_progress,
        subset=subset,
        query_lengths=query_lengths,
    )

    return [
        [
            (passage_id, score, token_score)
            for score, passage_id, token_score in zip(
                result.scores, result.passage_ids, result.token_scores
            )
        ]
        for result in results
    ]


class FastPlaid:
    """A class for creating and searching a FastPlaid index with concurrent safety."""

    def __init__(
        self,
        index: str,
        device: str | list[str] | None = None,
        index_gpu_memory: Literal["auto", "low", "medium", "high"] = "auto",
        index_memory_fraction: float = 0.7,
        search_memory_fraction: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Initialize the FastPlaid instance.

        Args:
        ----
        index:
            Path to the directory where the index is stored.
        device:
            The device(s) to use for index operations (e.g., 'cuda:0', 'cpu').
        index_gpu_memory:
            GPU placement of the large document tensors: 'low' keeps codes and
            residuals on CPU, 'medium' moves codes to GPU, 'high' moves both;
            'auto' picks the highest tier that fits within index_memory_fraction
            of the device. Placement affects speed only.
        index_memory_fraction:
            Fraction of device memory 'auto' index placement may fill.
        search_memory_fraction:
            Fraction of free VRAM the scoring stages may use when batch_size
            is 'auto'.
        kwargs:
            Additional keyword arguments. Unknown keywords are ignored, so call
            sites written against older versions keep working.

        """
        # Before 1.5.0 the third parameter was `low_memory: bool`. Tolerate both the
        # keyword and the positional form: placement is now chosen automatically.
        legacy_low_memory = kwargs.pop("low_memory", None)
        if isinstance(index_gpu_memory, bool):
            legacy_low_memory = index_gpu_memory
            index_gpu_memory = "auto"
        if legacy_low_memory is not None:
            warnings.warn(
                "low_memory is deprecated and ignored since 1.5.0: index placement is"
                " now chosen automatically from the free VRAM. Pass"
                " index_gpu_memory='low', 'medium' or 'high' to force a tier.",
                DeprecationWarning,
                stacklevel=2,
            )

        if index_gpu_memory not in {"auto", "low", "medium", "high"}:
            error = "index_gpu_memory must be 'auto', 'low', 'medium' or 'high'."
            raise ValueError(error)
        if not 0.0 < index_memory_fraction <= 1.0:
            raise ValueError("index_memory_fraction must be in (0, 1].")
        if not 0.0 < search_memory_fraction <= 1.0:
            raise ValueError("search_memory_fraction must be in (0, 1].")
        if device is not None and isinstance(device, str):
            self.devices = [device]
        elif isinstance(device, list):
            self.devices = device
        elif torch.cuda.is_available():
            self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            self.devices = ["cpu"]

        # Normalize bare "cuda" to "cuda:0" for consistency
        self.devices = ["cuda:0" if d == "cuda" else d for d in self.devices]

        # Ensure devices are unique to avoid redundant loading
        self.devices = list(dict.fromkeys(self.devices))

        self.torch_path = _load_torch_path(device=self.devices[0])
        self.index = index
        self.index_gpu_memory = index_gpu_memory
        self.index_memory_fraction = index_memory_fraction
        self.search_memory_fraction = search_memory_fraction

        # Concurrency Control
        if not os.path.exists(self.index):
            os.makedirs(self.index, exist_ok=True)
        self.lock_path = os.path.join(self.index, "plaid.lock")
        self.lock = FileLock(self.lock_path)
        self._last_known_mtime = 0.0

        # In-memory lock for atomic index swaps (allows search during update)
        self._index_swap_lock = threading.Lock()

        # Initialize Torch environment once
        fast_plaid_rust.initialize_torch(torch_path=self.torch_path)

        # Load an index object for each device.
        self.indices: dict[str, Any] = {}

        # Initial Load
        self._check_and_reload_index()

    def close(self) -> None:
        """Release all resources held by the index.

        This method should be called before deleting the index directory,
        especially on Windows where memory-mapped files must be unmapped
        before they can be deleted.
        """
        with self._index_swap_lock:
            self.indices.clear()
        gc.collect()

    def __enter__(self) -> Self:
        """Enable context manager usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Clean up resources when exiting context."""
        self.close()

    def _format_embeddings(
        self, embeddings: list[torch.Tensor] | torch.Tensor
    ) -> list[torch.Tensor] | torch.Tensor:
        """Standardize embedding shapes without creating deep copies.

        Args:
        ----
        embeddings:
            The input embeddings to format.

        """
        if isinstance(embeddings, torch.Tensor):
            return embeddings.squeeze(0) if embeddings.dim() == 3 else embeddings

        return [e.squeeze(0) if e.dim() == 3 else e for e in embeddings]

    def _update_mtime(self) -> None:
        """Update internal state with current disk mtime to prevent dumb reloads."""
        meta_path = os.path.join(self.index, "metadata.json")
        if os.path.exists(meta_path):
            self._last_known_mtime = Path(meta_path).stat().st_mtime

    def _check_and_reload_index(self, blocking: bool = True) -> bool:
        """Check if index on disk is newer than memory and reload if necessary.

        This performs an optimistic check first, and if a change is detected,
        acquires the lock to perform a safe reload.

        Args:
        ----
        blocking:
            If True, waits for the file lock (default behavior for writes).
            If False, skips reload if lock is held (used by search to avoid blocking).

        Returns:
        -------
        True if reload was performed or not needed, False if skipped due to lock.

        """
        meta_path = os.path.join(self.index, "metadata.json")
        if not os.path.exists(meta_path):
            # No index to load
            with self._index_swap_lock:
                for device in self.devices:
                    self.indices[device] = None
            return True

        # 1. Optimistic Check (Fast, No Lock)
        current_mtime = Path(meta_path).stat().st_mtime
        if current_mtime <= self._last_known_mtime and any(
            idx is not None for idx in self.indices.values()
        ):
            return True

        # 2. Critical Section (Lock Required for disk operations)
        # For non-blocking mode (search), try to acquire lock without waiting
        if not blocking:
            try:
                _ = self.lock.acquire(timeout=0)
            except FileLockTimeout:
                # Lock is held by update - search can proceed with current index
                return False
            try:
                return self._reload_under_lock(current_mtime)
            finally:
                self.lock.release()
        else:
            with self.lock:
                return self._reload_under_lock(current_mtime)

    def _reload_under_lock(self, current_mtime: float) -> bool:
        """Perform the actual reload while holding the file lock.

        Args:
        ----
        current_mtime:
            The mtime that triggered the reload check.

        Returns:
        -------
        True after successful reload.

        """
        meta_path = os.path.join(self.index, "metadata.json")
        # Re-check mtime under lock (Double-Checked Locking pattern)
        current_mtime = Path(meta_path).stat().st_mtime
        if current_mtime <= self._last_known_mtime and any(
            idx is not None for idx in self.indices.values()
        ):
            return True

        new_indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
            index_gpu_memory=self.index_gpu_memory,
            index_memory_fraction=self.index_memory_fraction,
        )

        # Atomic swap of indices dictionary
        with self._index_swap_lock:
            self.indices = new_indices
            self._last_known_mtime = current_mtime

        return True

    @torch.inference_mode()
    def create(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        nbits: int = 4,
        n_samples_kmeans: int | None = None,
        batch_size: int = 25_000,
        seed: int = 42,
        use_triton_kmeans: bool | None = None,
        metadata: list[dict[str, Any]] | None = None,
        start_from_scratch: int = 1000,
        compress_only: bool = False,
    ) -> "FastPlaid":
        """Create and saves the FastPlaid index.

        Args:
        ----
        documents_embeddings:
            The embeddings used to build the index.
        kmeans_niters:
            The number of iterations for K-means clustering.
        max_points_per_centroid:
            The maximum number of points allowed per centroid.
        nbits:
            The number of bits used for compression.
        n_samples_kmeans:
            The number of samples used for K-means training.
        batch_size:
            The batch size for processing embeddings.
        seed:
            The random seed for initialization.
        use_triton_kmeans:
            Whether to use the Triton implementation of K-means.
        metadata:
            A list of metadata dictionaries corresponding to the documents.
        start_from_scratch:
            Threshold of documents below which the index is built from scratch.
        compress_only:
            If True, skip IVF construction. The index can be used with
            ``get_embeddings()`` but not ``search()``.

        """
        # Exclusive Lock for Modification
        with self.lock:
            documents_embeddings = self._format_embeddings(documents_embeddings)
            num_docs = len(documents_embeddings)
            self._prepare_index_directory(index_path=self.index)

            if metadata is not None:
                if len(metadata) != num_docs:
                    error = f"""
                    The length of metadata ({len(metadata)}) must match the number of
                    documents_embeddings ({num_docs}).
                    """
                    raise ValueError(error)
                create(index=self.index, metadata=metadata)

            if len(documents_embeddings) <= start_from_scratch:
                save_list_tensors_on_disk(
                    path=os.path.join(
                        self.index,
                        "embeddings.npy",
                    ),
                    tensors=documents_embeddings,
                )

            # Determine dimensionality from the first available element
            dim = (
                documents_embeddings[0].shape[-1]
                if isinstance(documents_embeddings, list)
                else documents_embeddings.shape[-1]
            )

            # Use the first device for creation logic
            primary_device = self.devices[0]

            centroids = compute_kmeans(
                documents_embeddings=documents_embeddings,
                dim=dim,
                kmeans_niters=kmeans_niters,
                device=primary_device,
                max_points_per_centroid=max_points_per_centroid,
                n_samples_kmeans=n_samples_kmeans,
                seed=seed,
                use_triton_kmeans=use_triton_kmeans,
            )

            fast_plaid_rust.create(
                index=self.index,
                torch_path=self.torch_path,
                device=primary_device,
                embedding_dim=dim,
                nbits=nbits,
                embeddings=documents_embeddings,
                centroids=centroids,
                batch_size=batch_size,
                seed=seed,
                compress_only=compress_only,
            )

            # Explicit cleanup of create objects
            del centroids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Materialize the merged on-disk files once (CPU-side); defer the
            # per-device GPU loads to the first search (all-None entries reload lazily).
            _load_index_tensors_cpu(index_path=self.index)
            with self._index_swap_lock:
                self.indices = dict.fromkeys(self.devices)
                self._update_mtime()

        return self

    @torch.inference_mode()
    def _assert_not_frozen(self, operation: str) -> None:
        """Refuse mutating operations on a frozen index.

        A frozen index has had its per-shard ``{i}.codes.npy`` /
        ``{i}.residuals.npy`` files deleted, so ``update`` and ``delete``
        (which operate per shard) can't run without re-sharding first.
        """
        meta_path = os.path.join(self.index, "metadata.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("frozen", False):
            error = (
                f"Cannot {operation} a frozen index. "
                f"Re-create the index from source embeddings if mutation is needed."
            )
            raise RuntimeError(error)

    def freeze(self) -> "FastPlaid":
        """Drop per-shard codes/residuals to halve on-disk storage.

        After indexing, FastPlaid keeps both the per-shard files
        (``{i}.codes.npy``, ``{i}.residuals.npy``) used by ``update``/``delete``
        and a merged file (``merged_codes.npy``, ``merged_residuals.npy``) used
        at search time. For a read-only index the shards are redundant.

        ``freeze()`` ensures the merged file is up to date, then deletes every
        per-shard ``codes``/``residuals`` file and writes ``"frozen": true``
        into ``metadata.json``. Subsequent ``update``/``delete`` calls raise.
        Search is unaffected and reloads become slightly faster (the chunk scan
        is skipped).

        This is irreversible without re-indexing from the source embeddings.
        """
        with self.lock:
            meta_path = os.path.join(self.index, "metadata.json")
            if not os.path.exists(meta_path):
                error = f"No index found at {self.index} to freeze."
                raise FileNotFoundError(error)

            # Force a reload so the merged file reflects the current shards.
            self._reload_under_lock(Path(meta_path).stat().st_mtime)

            with open(meta_path) as f:
                meta = json.load(f)

            if meta.get("frozen", False):
                return self

            num_chunks = meta.get("num_chunks", 0)

            for suffix in ("codes", "residuals"):
                merged_path = os.path.join(self.index, f"merged_{suffix}.npy")
                if not os.path.exists(merged_path):
                    error = (
                        f"Cannot freeze: {merged_path} does not exist. "
                        f"Load the index at least once before freezing."
                    )
                    raise FileNotFoundError(error)

            for i in range(num_chunks):
                for suffix in ("codes", "residuals"):
                    shard_path = os.path.join(self.index, f"{i}.{suffix}.npy")
                    if os.path.exists(shard_path):
                        try:
                            os.remove(shard_path)
                        except OSError:
                            pass

            meta["frozen"] = True
            # newline="\n" prevents Windows text-mode translation; Rust's
            # serde_json::to_writer_pretty always uses \n regardless of platform.
            with open(meta_path, "w", newline="\n") as f:
                json.dump(meta, f, indent=2)

            self._update_mtime()

        return self

    @staticmethod
    def _write_tch_compatible_npy(path: str, arr: np.ndarray) -> None:
        """Write a .npy file byte-identical to ``tch::Tensor::write_npy``.

        Differs from ``np.save`` in two places:

        - Multi-dim shapes are formatted ``(R,C,)`` (Rust style: trailing
          comma, no spaces), not ``(R, C)``.
        - Byte order is always ``<`` even for 1-byte types (numpy uses ``|``).
        - The header is padded so the prologue (10 + header) is a multiple
          of 16, matching tch's alignment.
        """
        arr = np.ascontiguousarray(arr)
        kind_size = arr.dtype.kind + str(arr.dtype.itemsize)
        descr = f"<{kind_size}"

        if arr.ndim == 1:
            shape_str = f"({arr.shape[0]},)"
        else:
            shape_str = "(" + ",".join(str(s) for s in arr.shape) + ",)"

        header_body = (
            f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
        )

        prologue_fixed = 10  # magic(6) + version(2) + header_len_field(2)
        body_with_newline = len(header_body) + 1
        target = ((prologue_fixed + body_with_newline + 15) // 16) * 16
        pad = target - prologue_fixed - body_with_newline

        header = header_body + (" " * pad) + "\n"
        header_bytes = header.encode("ascii")

        with open(path, "wb") as f:
            f.write(b"\x93NUMPY\x01\x00")
            f.write(len(header_bytes).to_bytes(2, "little"))
            f.write(header_bytes)
            f.write(arr.tobytes())

    def unfreeze(self) -> "FastPlaid":
        """Rebuild per-shard codes/residuals from the merged file.

        Inverse of ``freeze()``. Slices ``merged_codes.npy`` / ``merged_residuals.npy``
        back into ``{i}.codes.npy`` / ``{i}.residuals.npy`` according to each
        shard's ``doclens.{i}.json``, then clears the ``frozen`` flag.

        Restores the ability to call ``update()`` / ``delete()``.
        """
        with self.lock:
            meta_path = os.path.join(self.index, "metadata.json")
            if not os.path.exists(meta_path):
                error = f"No index found at {self.index} to unfreeze."
                raise FileNotFoundError(error)

            with open(meta_path) as f:
                meta = json.load(f)

            if not meta.get("frozen", False):
                return self

            num_chunks = meta.get("num_chunks", 0)

            merged_codes_path = os.path.join(self.index, "merged_codes.npy")
            merged_residuals_path = os.path.join(self.index, "merged_residuals.npy")
            for merged_path in (merged_codes_path, merged_residuals_path):
                if not os.path.exists(merged_path):
                    error = (
                        f"Cannot unfreeze: {merged_path} missing — index data is gone."
                    )
                    raise FileNotFoundError(error)

            # Release any open mmaps on the merged files before we read them.
            with self._index_swap_lock:
                self.indices.clear()
            gc.collect()

            codes_mmap = np.load(merged_codes_path, mmap_mode="r")
            residuals_mmap = np.load(merged_residuals_path, mmap_mode="r")

            offset = 0
            for i in range(num_chunks):
                doclens_path = os.path.join(self.index, f"doclens.{i}.json")
                if not os.path.exists(doclens_path):
                    error = f"Cannot unfreeze: {doclens_path} missing."
                    raise FileNotFoundError(error)
                with open(doclens_path) as f:
                    chunk_doclens = json.load(f)
                chunk_rows = sum(chunk_doclens)

                chunk_codes = np.ascontiguousarray(
                    codes_mmap[offset : offset + chunk_rows]
                )
                chunk_residuals = np.ascontiguousarray(
                    residuals_mmap[offset : offset + chunk_rows]
                )

                self._write_tch_compatible_npy(
                    os.path.join(self.index, f"{i}.codes.npy"),
                    chunk_codes,
                )
                self._write_tch_compatible_npy(
                    os.path.join(self.index, f"{i}.residuals.npy"),
                    chunk_residuals,
                )

                offset += chunk_rows

            del codes_mmap, residuals_mmap
            gc.collect()

            # Drop the frozen marker entirely to restore byte-identical metadata.
            meta.pop("frozen", None)
            with open(meta_path, "w", newline="\n") as f:
                json.dump(meta, f, indent=2)

            self._update_mtime()
            # Reload so the in-memory indices reflect the restored shard layout.
            self._reload_under_lock(Path(meta_path).stat().st_mtime)

        return self

    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 25_000,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_samples_kmeans: int | None = None,
        seed: int = 42,
        start_from_scratch: int = 999,
        buffer_size: int = 100,
        use_triton_kmeans: bool | None = False,
    ) -> "FastPlaid":
        """Update an existing FastPlaid index with new documents.

        Args:
        ----
        documents_embeddings:
            New embeddings to add to the index.
        metadata:
            Optional metadata for the new documents.
        batch_size:
            Batch size for processing the update.
        kmeans_niters:
            Number of iterations for K-Means (if new centroids are created).
        max_points_per_centroid:
            Constraint for centroid creation during updates.
        n_samples_kmeans:
            Number of samples to use for K-Means (if None, auto-calculated).
        seed:
            Random seed for K-Means.
        start_from_scratch:
            If the existing index has fewer documents than this,
            the index will be re-created from scratch.
        buffer_size:
            Number of embeddings needed to trigger centroid expansion.
        use_triton_kmeans:
            Whether to use the Triton implementation of K-means.

        """
        # Exclusive Lock for Modification
        with self.lock:
            self._assert_not_frozen("update")
            # Get current indices snapshot for the update operation
            with self._index_swap_lock:
                current_indices = dict(self.indices)

            new_indices = process_update(
                index_path=self.index,
                devices=self.devices,
                torch_path=self.torch_path,
                index_gpu_memory=self.index_gpu_memory,
                index_memory_fraction=self.index_memory_fraction,
                indices_dict=current_indices,
                documents_embeddings=documents_embeddings,
                metadata=metadata,
                batch_size=batch_size,
                kmeans_niters=kmeans_niters,
                max_points_per_centroid=max_points_per_centroid,
                n_samples_kmeans=n_samples_kmeans,
                seed=seed,
                start_from_scratch=start_from_scratch,
                buffer_size=buffer_size,
                use_triton_kmeans=use_triton_kmeans,
                create_fn=self.create,
                delete_fn=self.delete,
                compute_kmeans_fn=compute_kmeans,
                format_embeddings_fn=self._format_embeddings,
            )

            # Atomic swap of indices dictionary
            with self._index_swap_lock:
                self.indices = new_indices
                self._update_mtime()

        return self

    @staticmethod
    def _prepare_index_directory(index_path: str) -> None:
        """Prepare the index directory by cleaning or creating it.

        Args:
        ----
        index_path:
            The path to the index directory.

        """
        if os.path.exists(index_path) and os.path.isdir(index_path):
            for json_file in glob.glob(os.path.join(index_path, "*.json")):
                try:
                    os.remove(json_file)
                except OSError:
                    pass

            for npy_file in glob.glob(os.path.join(index_path, "*.npy")):
                try:
                    os.remove(npy_file)
                except OSError:
                    pass
        elif not os.path.exists(index_path):
            try:
                os.makedirs(index_path)
            except OSError as e:
                raise e

    def _prepare_search(
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        subset: list[list[int]] | list[int] | None,
    ) -> tuple[dict[str, Any], torch.Tensor, list[int], list[list[int]] | None]:
        """Shared setup for search methods: reload index, validate, pack queries."""
        self._check_and_reload_index(blocking=False)

        with self._index_swap_lock:
            search_indices = dict(self.indices)

        if any(idx is None for idx in search_indices.values()):
            self._check_and_reload_index(blocking=True)
            with self._index_swap_lock:
                search_indices = dict(self.indices)

        if not os.path.exists(os.path.join(self.index, "metadata.json")):
            error = f"""
            Index metadata not found in '{self.index}'.
            Please create the index before searching.
            """
            raise FileNotFoundError(error)

        for device in self.devices:
            if search_indices[device] is None:
                error = f"""Index could not be loaded on device '{device}'.
                Check CUDA memory or device availability."""
                raise RuntimeError(error)

        packed_queries, query_lengths = pack_queries(queries_embeddings)
        num_queries = len(query_lengths)

        if subset is not None:
            if isinstance(subset, int):
                subset = [subset] * num_queries
            if isinstance(subset, list) and len(subset) == 0:
                subset = None
            if isinstance(subset, list) and isinstance(subset[0], int):
                subset = [subset] * num_queries  # type: ignore

            if subset is not None and len(subset) != num_queries:
                raise ValueError("Subset length must match number of queries.")

        return search_indices, packed_queries, query_lengths, subset  # type: ignore

    def _dispatch_search(
        self,
        device_fn: Any,
        search_indices: dict[str, Any],
        packed_queries: torch.Tensor,
        query_lengths: list[int],
        subset: list[list[int]] | None,
        *,
        batch_size: int | Literal["auto"],
        n_full_scores: int,
        top_k: int,
        n_ivf_probe: int,
        show_progress: bool,
        n_processes: int | None = None,
    ) -> list:
        """Dispatch search across devices, including joblib CPU parallelism.

        Args:
        ----
        device_fn:
            The per-device search callable (search_on_device or
            search_on_device_with_token_scores).
        search_indices:
            Mapping of device name to loaded index object.
        packed_queries:
            A 2D packed tensor (total_query_tokens, embedding_dim), no padding.
        query_lengths:
            True token count per query.
        subset:
            Optional per-query document ID filters.
        batch_size:
            Documents per scoring chunk; 'auto' plans chunks from the memory budget.
        n_full_scores:
            The number of full scores to compute per query.
        top_k:
            The number of top results to return for each query.
        n_ivf_probe:
            The number of IVF clusters to probe.
        show_progress:
            Whether to display a progress bar during search.
        n_processes:
            Number of jobs for CPU parallelism via joblib.
            Ignored on GPU. Defaults to 1.

        """
        num_queries = len(query_lengths)
        if num_queries == 0:
            return []

        token_offsets = [0]
        for length in query_lengths:
            token_offsets.append(token_offsets[-1] + length)

        def query_slice(q_start: int, q_end: int) -> tuple[torch.Tensor, list[int]]:
            t_start, t_end = token_offsets[q_start], token_offsets[q_end]
            return (
                packed_queries.narrow(0, t_start, t_end - t_start),
                query_lengths[q_start:q_end],
            )

        # Joblib parallelism for CPU-only bulk search
        is_cpu_only = self.devices[0] == "cpu"
        use_joblib = (is_cpu_only and (num_queries > 10) and n_processes != 1) or (
            is_cpu_only
            and n_processes is not None
            and n_processes != 1
            and num_queries > 1
        )

        if n_processes is None:
            n_processes = min(num_queries // 10, os.cpu_count() or 1)

        if use_joblib:
            num_workers = n_processes
            chunk_size = math.ceil(num_queries / num_workers)
            bounds = [
                (start, min(start + chunk_size, num_queries))
                for start in range(0, num_queries, chunk_size)
            ]
            subset_chunks: list = []
            if subset is not None:
                subset_chunks = [subset[start:end] for start, end in bounds]
            else:
                subset_chunks = [None] * len(bounds)  # type: ignore

            results = Parallel(n_jobs=num_workers, prefer="threads")(
                delayed(device_fn)(
                    device="cpu",
                    queries_embeddings=query_slice(start, end)[0],
                    batch_size=batch_size,
                    n_full_scores=n_full_scores,
                    top_k=top_k,
                    n_ivf_probe=n_ivf_probe,
                    search_memory_fraction=self.search_memory_fraction,
                    index_object=search_indices["cpu"],
                    show_progress=(show_progress and i == 0),
                    subset=sub_chunk,
                    query_lengths=query_slice(start, end)[1],
                )
                for i, ((start, end), sub_chunk) in enumerate(
                    zip(bounds, subset_chunks)
                )
            )
            return [item for sublist in results for item in sublist]

        if len(self.devices) == 1:
            return device_fn(
                device=self.devices[0],
                queries_embeddings=packed_queries,
                batch_size=batch_size,
                n_full_scores=n_full_scores,
                top_k=top_k,
                n_ivf_probe=n_ivf_probe,
                search_memory_fraction=self.search_memory_fraction,
                index_object=search_indices[self.devices[0]],
                show_progress=show_progress,
                subset=subset,
                query_lengths=query_lengths,
            )

        # Multi-GPU split
        num_devices = len(self.devices)
        chunk_size = math.ceil(num_queries / num_devices)
        bounds = [
            (start, min(start + chunk_size, num_queries))
            for start in range(0, num_queries, chunk_size)
        ]
        subset_chunks_gpu: list = []
        if subset is not None:
            subset_chunks_gpu = [subset[start:end] for start, end in bounds]
        else:
            subset_chunks_gpu = [None] * len(bounds)  # type: ignore

        futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for i, device in enumerate(self.devices):
                if i >= len(bounds):
                    break
                chunk_queries, chunk_lengths = query_slice(*bounds[i])
                futures.append(
                    executor.submit(
                        device_fn,
                        device=device,
                        queries_embeddings=chunk_queries,
                        batch_size=batch_size,
                        n_full_scores=n_full_scores,
                        top_k=top_k,
                        n_ivf_probe=n_ivf_probe,
                        search_memory_fraction=self.search_memory_fraction,
                        index_object=search_indices[device],
                        show_progress=show_progress and (i == 0),
                        subset=subset_chunks_gpu[i],
                        query_lengths=chunk_lengths,
                    )
                )

        all_results = []
        for future in futures:
            all_results.extend(future.result())

        return all_results

    @torch.inference_mode()
    def search(
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int = 10,
        batch_size: int | Literal["auto"] = "auto",
        n_full_scores: int = 4096,
        n_ivf_probe: int = 8,
        show_progress: bool = True,
        subset: list[list[int]] | list[int] | None = None,
        n_processes: int | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Search the index for the given query embeddings.

        Args:
        ----
        queries_embeddings:
            A tensor of shape (num_queries, n_tokens, embedding_dim) or a list of
            tensors.
        top_k:
            The number of top results to return for each query.
        batch_size:
            Documents per scoring chunk; 'auto' plans chunks from the device
            memory budget.
        n_full_scores:
            The number of full scores to compute per query.
        n_ivf_probe:
            The number of IVF clusters to probe.
        show_progress:
            Whether to display a progress bar during search.
        subset:
            A list of lists specifying subsets of the index to search for each
            query, or a single list applied to all queries. If None, searches
            the entire index.
        n_processes:
            Number of jobs to use for CPU search via joblib.
            Ignored if running on GPU(s). Defaults to 1.

        """
        search_indices, packed_queries, query_lengths, subset = self._prepare_search(
            queries_embeddings, subset
        )

        return self._dispatch_search(
            search_on_device,
            search_indices,
            packed_queries,
            query_lengths,
            subset,  # type: ignore
            batch_size=batch_size,
            n_full_scores=n_full_scores,
            top_k=top_k,
            n_ivf_probe=n_ivf_probe,
            show_progress=show_progress,
            n_processes=n_processes,
        )

    @torch.inference_mode()
    def search_token_scores(
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int = 10,
        batch_size: int | Literal["auto"] = "auto",
        n_full_scores: int = 4096,
        n_ivf_probe: int = 8,
        show_progress: bool = True,
        subset: list[list[int]] | list[int] | None = None,
        n_processes: int | None = None,
    ) -> list[list[tuple[int, float, torch.Tensor]]]:
        """Search the index and return token-level similarity matrices.

        Same as search() but each result tuple includes a third element: a tensor
        of shape (query_tokens, doc_tokens) containing the dot-product similarity
        between each query token and each document token (equivalent to cosine
        similarity when embeddings are L2-normalized).

        Args:
        ----
        queries_embeddings:
            A tensor of shape (num_queries, n_tokens, embedding_dim) or a list of
            tensors.
        top_k:
            The number of top results to return for each query.
        batch_size:
            Documents per scoring chunk; 'auto' plans chunks from the device
            memory budget.
        n_full_scores:
            The number of full scores to compute per query.
        n_ivf_probe:
            The number of IVF clusters to probe.
        show_progress:
            Whether to display a progress bar during search.
        subset:
            A list of lists specifying subsets of the index to search for each
            query, or a single list applied to all queries. If None, searches
            the entire index.
        n_processes:
            Number of jobs to use for CPU search via joblib.
            Ignored if running on GPU(s). Defaults to 1.

        """
        search_indices, packed_queries, query_lengths, subset = self._prepare_search(
            queries_embeddings, subset
        )

        return self._dispatch_search(
            search_on_device_with_token_scores,
            search_indices,
            packed_queries,
            query_lengths,
            subset,  # type: ignore
            batch_size=batch_size,
            n_full_scores=n_full_scores,
            top_k=top_k,
            n_ivf_probe=n_ivf_probe,
            show_progress=show_progress,
            n_processes=n_processes,
        )

    @torch.inference_mode()
    def delete(
        self,
        subset: list[int],
        _delete_metadata: bool = True,
        _delete_buffer: bool = True,
    ) -> "FastPlaid":
        """Delete embeddings from an existing FastPlaid index.

        Args:
        ----
        subset:
            A list of document IDs (0-based) to delete.

        """
        # Exclusive Lock for Modification
        with self.lock:
            self._assert_not_frozen("delete from")
            primary_device = self.devices[0]

            fast_plaid_rust.delete(
                index=self.index,
                torch_path=self.torch_path,
                device=primary_device,
                subset=subset,
            )

            metadata_db_path = os.path.join(self.index, "metadata.db")
            if os.path.exists(metadata_db_path) and _delete_metadata:
                delete(index=self.index, subset=subset)

            # Get metadata to determine document counts
            meta_path = os.path.join(self.index, "metadata.json")
            num_documents = 0
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    num_documents = meta.get("num_documents", 0)

            # Determine buffer size if buffer exists
            buffer_path = os.path.join(self.index, "buffer.npy")
            num_buffer_docs = 0
            if os.path.exists(buffer_path):
                buffer_np = np.load(buffer_path, allow_pickle=True)
                num_buffer_docs = len(buffer_np)

            # Buffer documents are the most recent ones (at the end)
            buffer_start_idx = num_documents - num_buffer_docs

            # Update embeddings.npy if it exists
            embeddings_path = os.path.join(self.index, "embeddings.npy")
            if os.path.exists(embeddings_path):
                embeddings_np = np.load(embeddings_path, allow_pickle=True)
                num_embeddings = len(embeddings_np)

                # Filter subset to only include indices within embeddings range
                embeddings_to_delete = {idx for idx in subset if idx < num_embeddings}

                if embeddings_to_delete:
                    # Keep embeddings that are NOT in the delete set
                    remaining_embeddings = [
                        torch.from_numpy(embeddings_np[i])
                        for i in range(num_embeddings)
                        if i not in embeddings_to_delete
                    ]

                    if remaining_embeddings:
                        save_list_tensors_on_disk(
                            path=embeddings_path,
                            tensors=remaining_embeddings,
                        )
                    else:
                        os.remove(embeddings_path)

            # Update buffer.npy if it exists and contains documents to delete
            if os.path.exists(buffer_path) and num_buffer_docs > 0:
                # Calculate which subset indices fall within the buffer range
                buffer_indices_to_delete = {
                    idx - buffer_start_idx
                    for idx in subset
                    if buffer_start_idx <= idx < num_documents
                }

                if buffer_indices_to_delete:
                    buffer_np = np.load(buffer_path, allow_pickle=True)

                    # Keep buffer entries that are NOT in the delete set
                    remaining_buffer = [
                        torch.from_numpy(buffer_np[i])
                        for i in range(num_buffer_docs)
                        if i not in buffer_indices_to_delete
                    ]

                    if remaining_buffer:
                        save_list_tensors_on_disk(
                            path=buffer_path,
                            tensors=remaining_buffer,
                        )
                    else:
                        os.remove(buffer_path)

            new_indices = _reload_index(
                index_path=self.index,
                devices=self.devices,
                indices={},
                index_gpu_memory=self.index_gpu_memory,
                index_memory_fraction=self.index_memory_fraction,
            )

            # Atomic swap of indices dictionary
            with self._index_swap_lock:
                self.indices = new_indices
                self._update_mtime()

        return self

    @torch.inference_mode()
    def get_embeddings(self, subset: list[int]) -> list[torch.Tensor]:
        """Reconstruct original embeddings for the specified document IDs.

        This method leverages the Rust backend to efficiently decompress
        and reconstruct embeddings, utilizing parallelism where possible.

        Args:
        ----
        subset:
            A list of document IDs (0-based) to reconstruct.

        """
        # Non-blocking reload: if an update is in progress, continue with current index
        self._check_and_reload_index(blocking=False)

        if not subset:
            return []

        # Get a snapshot of current indices (atomic read)
        with self._index_swap_lock:
            current_index = self.indices[self.devices[0]]

        return fast_plaid_rust.reconstruct_embeddings(
            index=current_index,
            subset=subset,
            device=self.devices[0],
        )
