from __future__ import annotations

import gc
import glob
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from fast_plaid import fast_plaid_rust
from fastkmeans import FastKMeans
from joblib import Parallel, delayed

from ..filtering import create, delete, update
from .load import _reload_index


class TorchWithCudaNotFoundError(Exception):
    """Exception raised when PyTorch with CUDA support is not found."""


def _load_torch_path(device: str) -> str:
    """Find the path to the shared library for PyTorch with CUDA."""
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
    raise TorchWithCudaNotFoundError(error) from IndexError


def compute_kmeans(  # noqa: PLR0913
    documents_embeddings: list[torch.Tensor] | torch.Tensor,
    dim: int,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
) -> torch.Tensor:
    """Compute K-means centroids for document embeddings."""
    num_passages = len(documents_embeddings)

    if n_samples_kmeans is None:
        n_samples_kmeans = min(
            1 + int(16 * math.sqrt(120 * num_passages)),
            num_passages,
        )

    n_samples_kmeans = min(num_passages, n_samples_kmeans)

    # Memory optimization: Use torch.randperm for efficient sampling
    sampled_indices = torch.randperm(num_passages)[:n_samples_kmeans]

    if isinstance(documents_embeddings, torch.Tensor):
        # Indexing a tensor directly is a view-operation or efficient gather
        samples_tensor = documents_embeddings[sampled_indices]
    else:
        # Optimization: Pre-calculate total tokens to allocate a single buffer
        sampled_indices_list = sampled_indices.tolist()
        total_sample_tokens = sum(
            documents_embeddings[i].shape[0] for i in sampled_indices_list
        )

        # Optimization: Pre-allocate buffer in float16 to save 50% RAM immediately
        samples_tensor = torch.empty(
            (total_sample_tokens, dim), dtype=torch.float16, device="cpu"
        )

        # Optimization: Manual slice copy instead of torch.cat to avoid list-of-tensors copy
        current_offset = 0
        for i in sampled_indices_list:
            tensor_slice = documents_embeddings[i]
            length = tensor_slice.shape[0]
            # Direct copy into the pre-allocated buffer
            samples_tensor[current_offset : current_offset + length].copy_(tensor_slice)
            current_offset += length

    total_tokens = samples_tensor.shape[0]

    # Calculate num_partitions based on the density of the sample relative to the whole
    avg_tokens_per_doc = total_tokens / n_samples_kmeans
    estimated_total_tokens = avg_tokens_per_doc * num_passages
    num_partitions = int(
        2 ** math.floor(math.log2(16 * math.sqrt(estimated_total_tokens)))
    )

    if samples_tensor.is_cuda:
        samples_tensor = samples_tensor.to(device="cpu", dtype=torch.float16)

    kmeans = FastKMeans(
        d=dim,
        k=min(num_partitions, total_tokens),
        niter=kmeans_niters,
        gpu=device != "cpu",
        verbose=False,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        use_triton=use_triton_kmeans,
    )

    kmeans.train(data=samples_tensor)

    # Explicitly clear the large sample buffer before creating centroids
    del samples_tensor
    gc.collect()

    centroids = torch.from_numpy(
        kmeans.centroids,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    return torch.nn.functional.normalize(
        input=centroids,
        dim=-1,
    ).half()


def search_on_device(  # noqa: PLR0913
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
    """Perform a search on a single specified device using the passed object."""
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
    """A class for creating and searching a FastPlaid index.

    Args:
    ----
    index:
        Path to the directory where the index is stored or will be stored.
    device:
        The device(s) to use for computation (e.g., "cuda", ["cuda:0", "cuda:1"]).
        If None, defaults to ["cuda"].

    """

    def __init__(
        self,
        index: str,
        device: str | list[str] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the FastPlaid instance."""
        if device is not None and isinstance(device, str):
            self.devices = [device]
        elif isinstance(device, list):
            self.devices = device
        elif torch.cuda.is_available():
            self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            self.devices = ["cpu"]

        # Ensure devices are unique to avoid redundant loading
        self.devices = list(dict.fromkeys(self.devices))

        self.torch_path = _load_torch_path(device=self.devices[0])
        self.index = index

        # Initialize Torch environment once
        fast_plaid_rust.initialize_torch(torch_path=self.torch_path)

        # Load an index object for each device.
        # We duplicate the index object because it cannot be pickled or shared across
        # process boundaries easily, but it can be loaded multiple times.
        self.indices: dict[str, Any] = {}
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
        )

    def _format_embeddings(
        self, embeddings: list[torch.Tensor] | torch.Tensor
    ) -> list[torch.Tensor] | torch.Tensor:
        """Helper to standardize embedding shapes without creating deep copies."""
        if isinstance(embeddings, torch.Tensor):
            return embeddings.squeeze(0) if embeddings.dim() == 3 else embeddings

        # If it's a list, use a generator-like logic to squeeze items in-place (shallowly)
        return [e.squeeze(0) if e.dim() == 3 else e for e in embeddings]

    @torch.inference_mode()
    def create(  # noqa: PLR0913
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
    ) -> "FastPlaid":
        """Create and saves the FastPlaid index."""
        # Standardize format without forcing a full list-to-tensor or tensor-to-list conversion
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

        # Determine dimensionality from the first available element
        dim = (
            documents_embeddings[0].shape[-1]
            if isinstance(documents_embeddings, list)
            else documents_embeddings.shape[-1]
        )

        # Use the first device for creation logic
        primary_device = self.devices[0]

        print("Computing centroids of embeddings.")
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

        print("Creating FastPlaid index.")
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
        # Optimization: clear old indices first to prevent peaks during reload
        self.indices.clear()
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
        )

        return self

    @torch.inference_mode()
    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 50_000,
    ) -> "FastPlaid":
        """Update an existing FastPlaid index with new documents."""
        documents_embeddings = self._format_embeddings(documents_embeddings)
        num_docs = len(documents_embeddings)

        if not os.path.exists(self.index) or not os.path.exists(
            os.path.join(self.index, "metadata.json")
        ):
            error = f"""
            Index directory '{self.index}' does not exist or is invalid.
            Please create an index first using the .create() method.
            """
            raise FileNotFoundError(error)

        if os.path.exists(os.path.join(self.index, "metadata.db")):
            if metadata is None:
                metadata = [{} for _ in range(num_docs)]

            if len(metadata) != num_docs:
                error = f"""
                The length of metadata ({len(metadata)}) must match the number of
                documents_embeddings ({num_docs}).
                """
                raise ValueError(error)
            update(index=self.index, metadata=metadata)

        if self.indices[self.devices[0]] is None:
            raise RuntimeError("Index not loaded for update.")

        fast_plaid_rust.update(
            index_path=self.index,
            index=self.indices[self.devices[0]],
            torch_path=self.torch_path,
            device=self.devices[0],
            embeddings=documents_embeddings,
            batch_size=batch_size,
        )

        # Reload indices on all devices to reflect the updates
        self.indices.clear()
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
        )

        return self

    @staticmethod
    def _prepare_index_directory(index_path: str) -> None:
        """Prepare the index directory by cleaning or creating it."""
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
    def search(  # noqa: PLR0912, PLR0913, PLR0915, C901
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
        n_processes: Number of jobs to use for CPU search via joblib.
                        Ignored if running on GPU(s). Defaults to 1.

        """
        if any(idx is None for idx in self.indices.values()):
            self.indices = _reload_index(
                index_path=self.index,
                devices=self.devices,
                indices=self.indices,
            )

        if not os.path.exists(os.path.join(self.index, "metadata.json")):
            error = f"""
            Index metadata not found in '{self.index}'.
            Please create the index before searching.
            """
            raise FileNotFoundError(error)

        for device in self.devices:
            if self.indices[device] is None:
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
                error = """
                The length of the subset must match the number of queries.
                """
                raise ValueError(error)

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
                    index_object=self.indices["cpu"],
                    show_progress=(show_progress and i == 0),
                    subset=sub_chunk,
                )
                for i, (chunk, sub_chunk) in enumerate(zip(query_chunks, subset_chunks))
            )

            return [item for sublist in results for item in sublist]

        # Single device shortcut (GPU or CPU n=1)
        if len(self.devices) == 1:
            return search_on_device(
                device=self.devices[0],
                queries_embeddings=queries_embeddings,
                batch_size=batch_size,
                n_full_scores=n_full_scores,
                top_k=top_k,
                n_ivf_probe=n_ivf_probe,
                index_object=self.indices[self.devices[0]],
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
                        index_object=self.indices[device],
                        show_progress=show_progress and (i == 0),
                        subset=subset_chunks[i],  # type: ignore
                    )
                )

        all_results = []
        for future in futures:
            all_results.extend(future.result())

        return all_results

    def delete(self, subset: list[int]) -> "FastPlaid":
        """Delete embeddings from an existing FastPlaid index."""
        primary_device = self.devices[0]

        fast_plaid_rust.delete(
            index=self.index,
            torch_path=self.torch_path,
            device=primary_device,
            subset=subset,
        )

        metadata_db_path = os.path.join(self.index, "metadata.db")
        if os.path.exists(metadata_db_path):
            delete(index=self.index, subset=subset)

        self.indices.clear()
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
        )

        return self
