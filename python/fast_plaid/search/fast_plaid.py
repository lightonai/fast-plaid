from __future__ import annotations

import gc
import glob
import json
import math
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

import torch
from fast_plaid import fast_plaid_rust
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from joblib import Parallel, delayed

from ..filtering import create, delete
from .kmeans import FastKMeans
from .load import _reload_index, save_list_tensors_on_disk
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

    # Memory optimization: Use torch.randperm for efficient sampling
    sampled_indices = torch.randperm(num_documents)[:n_samples_kmeans]

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


def search_on_device(
    device: str,
    queries_embeddings: torch.Tensor,
    batch_size: int,
    n_full_scores: int,
    top_k: int,
    n_ivf_probe: int,
    index_object: Any,
    show_progress: bool,
    subset: list[list[int]] | None = None,
) -> list[list[tuple[int, float]]]:
    """Perform a search on a single specified device using the passed object.

    Args:
    ----
    device:
        The device identifier to perform the search on.
    queries_embeddings:
        The query embeddings to search for.
    batch_size:
        The batch size for processing queries.
    n_full_scores:
        The number of full scores to compute per query.
    top_k:
        The number of top results to return.
    n_ivf_probe:
        The number of IVF clusters to probe.
    index_object:
        The loaded index object for the specific device.
    show_progress:
        Whether to show a progress bar.
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
        batch_size=batch_size,
        n_full_scores=n_full_scores,
        top_k=top_k,
        n_ivf_probe=n_ivf_probe,
    )

    scores = fast_plaid_rust.pysearch(
        index=index_object,
        device=device,
        queries_embeddings=queries_embeddings.to(dtype=torch.float16),
        search_parameters=search_parameters,
        show_progress=show_progress,
        subset=subset,
    )

    return [
        [
            (passage_id, score)
            for score, passage_id in zip(score.scores, score.passage_ids)
        ]
        for score in scores
    ]


class FastPlaid:
    """A class for creating and searching a FastPlaid index with concurrent safety."""

    def __init__(
        self,
        index: str,
        device: str | list[str] | None = None,
        low_memory: bool = True,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the FastPlaid instance.

        Args:
        ----
        index:
            Path to the directory where the index is stored.
        device:
            The device(s) to use for index operations (e.g., 'cuda:0', 'cpu').
        low_memory:
            Whether to use low memory mode when loading the index.
        kwargs:
            Additional keyword arguments.

        """
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
        self.low_memory = low_memory

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

    def __enter__(self) -> "FastPlaid":
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
            low_memory=self.low_memory,
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
        batch_size: int = 50_000,
        seed: int = 42,
        use_triton_kmeans: bool | None = None,
        metadata: list[dict[str, Any]] | None = None,
        start_from_scratch: int = 1000,
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
            )

            # Explicit cleanup of create objects
            del centroids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Reload indices on all devices now that creation is complete
            new_indices = _reload_index(
                index_path=self.index,
                devices=self.devices,
                indices={},
                low_memory=self.low_memory,
            )

            # Atomic swap of indices dictionary
            with self._index_swap_lock:
                self.indices = new_indices
                self._update_mtime()

        return self

    @torch.inference_mode()
    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 50_000,
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
            # Get current indices snapshot for the update operation
            with self._index_swap_lock:
                current_indices = dict(self.indices)

            new_indices = process_update(
                index_path=self.index,
                devices=self.devices,
                torch_path=self.torch_path,
                low_memory=self.low_memory,
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

    @torch.inference_mode()
    def search(  # noqa: PLR0912
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int = 10,
        batch_size: int = 2000,
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
            The number of queries to process in each batch.
        n_full_scores:
            The number of full scores to compute per query.
        n_ivf_probe:
            The number of IVF clusters to probe during search.
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
        # Non-blocking reload: if an update is in progress, continue with current index
        self._check_and_reload_index(blocking=False)

        # Get a snapshot of current indices (atomic read)
        with self._index_swap_lock:
            search_indices = dict(self.indices)

        if any(idx is None for idx in search_indices.values()):
            # Try blocking reload if we have no valid index
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

        if isinstance(queries_embeddings, list):
            queries_embeddings = torch.nn.utils.rnn.pad_sequence(
                sequences=[
                    embedding[0] if embedding.dim() == 3 else embedding
                    for embedding in queries_embeddings
                ],
                batch_first=True,
                padding_value=0.0,
            )

        num_queries = queries_embeddings.shape[0]

        # Standardize subset
        if subset is not None:
            if isinstance(subset, int):
                subset = [subset] * num_queries
            if isinstance(subset, list) and len(subset) == 0:
                subset = None
            if isinstance(subset, list) and isinstance(subset[0], int):
                subset = [subset] * num_queries  # type: ignore

            if subset is not None and len(subset) != num_queries:
                raise ValueError("Subset length must match number of queries.")

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
            query_chunks = list(torch.split(queries_embeddings, chunk_size))
            subset_chunks = []
            if subset is not None:
                for i in range(0, num_queries, chunk_size):
                    subset_chunks.append(subset[i : i + chunk_size])
            else:
                subset_chunks = [None] * len(query_chunks)  # type: ignore

            results = Parallel(n_jobs=num_workers, prefer="threads")(
                delayed(search_on_device)(
                    device="cpu",
                    queries_embeddings=chunk,
                    batch_size=batch_size,
                    n_full_scores=n_full_scores,
                    top_k=top_k,
                    n_ivf_probe=n_ivf_probe,
                    index_object=search_indices["cpu"],
                    show_progress=(show_progress and i == 0),
                    subset=sub_chunk,
                )
                for i, (chunk, sub_chunk) in enumerate(zip(query_chunks, subset_chunks))
            )
            return [item for sublist in results for item in sublist]

        if len(self.devices) == 1:
            return search_on_device(
                device=self.devices[0],
                queries_embeddings=queries_embeddings,
                batch_size=batch_size,
                n_full_scores=n_full_scores,
                top_k=top_k,
                n_ivf_probe=n_ivf_probe,
                index_object=search_indices[self.devices[0]],
                show_progress=show_progress,
                subset=subset,  # type: ignore
            )

        # Multi-GPU Split
        num_devices = len(self.devices)
        chunk_size = math.ceil(num_queries / num_devices)
        futures = []
        query_chunks = list(torch.split(queries_embeddings, chunk_size))
        subset_chunks = []
        if subset is not None:
            for i in range(0, num_queries, chunk_size):
                subset_chunks.append(subset[i : i + chunk_size])
        else:
            subset_chunks = [None] * len(query_chunks)  # type: ignore

        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for i, device in enumerate(self.devices):
                if i >= len(query_chunks):
                    break
                futures.append(
                    executor.submit(
                        search_on_device,
                        device=device,
                        queries_embeddings=query_chunks[i],
                        batch_size=batch_size,
                        n_full_scores=n_full_scores,
                        top_k=top_k,
                        n_ivf_probe=n_ivf_probe,
                        index_object=search_indices[device],
                        show_progress=show_progress and (i == 0),
                        subset=subset_chunks[i],  # type: ignore
                    )
                )

        all_results = []
        for future in futures:
            all_results.extend(future.result())

        return all_results

    @torch.inference_mode()
    def delete(self, subset: list[int], _delete_metadata: bool = True) -> "FastPlaid":
        """Delete embeddings from an existing FastPlaid index.

        Args:
        ----
        subset:
            A list of document IDs (0-based) to delete.

        """
        # Exclusive Lock for Modification
        with self.lock:
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

            # Clear buffer to prevent stale embeddings from causing phantom documents
            # in search results. If deleted documents were buffered, keeping the buffer
            # would cause inconsistencies between the index and buffer state.
            buffer_path = os.path.join(self.index, "buffer.npy")
            if os.path.exists(buffer_path):
                os.remove(buffer_path)

            # Clear merged files to force regeneration after deletion
            # The Rust delete operation doesn't update these files, causing
            # loading failures on Windows and incorrect state on all platforms
            merged_files = [
                "merged_codes.npy",
                "merged_codes.manifest.json",
                "merged_residuals.npy",
                "merged_residuals.manifest.json",
            ]
            for merged_file in merged_files:
                merged_path = os.path.join(self.index, merged_file)
                if os.path.exists(merged_path):
                    try:
                        os.remove(merged_path)
                    except OSError:
                        pass

            # Update num_documents in metadata.json to reflect the deletion
            # This ensures subsequent updates use the correct document count
            metadata_json_path = os.path.join(self.index, "metadata.json")
            if os.path.exists(metadata_json_path):
                with open(metadata_json_path) as f:
                    meta = json.load(f)

                # Get the actual document count from the metadata DB if it exists
                if os.path.exists(metadata_db_path):
                    conn = sqlite3.connect(metadata_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM METADATA")
                    actual_count = cursor.fetchone()[0]
                    conn.close()
                    meta["num_documents"] = actual_count
                else:
                    # Fall back to decrementing by the number of deleted items
                    meta["num_documents"] = max(
                        0, meta.get("num_documents", 0) - len(subset)
                    )

                with open(metadata_json_path, "w") as f:
                    json.dump(meta, f, indent=4)

            new_indices = _reload_index(
                index_path=self.index,
                devices=self.devices,
                indices={},
                low_memory=self.low_memory,
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
