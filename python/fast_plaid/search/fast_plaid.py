from __future__ import annotations

import functools
import gc
import glob
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import psutil
import torch
from fast_plaid import fast_plaid_rust
from fastkmeans import FastKMeans
from joblib import Parallel, delayed

from ..filtering import create, delete, update
from .load import _reload_index, save_list_tensors_on_disk


def profile_resources(func):
    """Measure execution time, RAM usage (RSS), and GPU VRAM usage."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        process = psutil.Process(os.getpid())

        # 1. Snapshot Start State
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_vram = torch.cuda.memory_allocated() / (1024**2)
        else:
            start_vram = 0.0

        start_ram = process.memory_info().rss / (1024**2)  # Convert to MB
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_vram = torch.cuda.memory_allocated() / (1024**2)
                peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                end_vram = 0.0
                peak_vram = 0.0

            end_ram = process.memory_info().rss / (1024**2)
            end_time = time.time()

            delta_ram = end_ram - start_ram
            delta_vram = end_vram - start_vram

            print(f"\n[PROFILE] Function: {func.__name__}")
            print(f"  ├── Time:      {end_time - start_time:.4f}s")
            print(
                f"  ├── RAM (RSS): {start_ram:.2f}MB -> {end_ram:.2f}MB (Delta: {delta_ram:+.2f}MB)"  # noqa: E501
            )
            if torch.cuda.is_available():
                print(
                    f"  └── VRAM:      {start_vram:.2f}MB -> {end_vram:.2f}MB (Delta: {delta_vram:+.2f}MB, Peak: {peak_vram:.2f}MB)"  # noqa: E501
                )
            else:
                print("  └── VRAM:      N/A (CPU only)")
            print("-" * 40)

        return result

    return wrapper


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
    raise TorchWithCudaNotFoundError(error)


@profile_resources
def compute_kmeans(  # noqa: PLR0913
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
        num_partitions: If provided, explicitly sets the number of centroids (K).
                        If None, K is calculated using a heuristic based on dataset size.

    """
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
        # Calculate num_partitions based on the density of the sample relative to the whole
        avg_tokens_per_doc = total_tokens / n_samples_kmeans
        estimated_total_tokens = avg_tokens_per_doc * num_passages
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


@profile_resources
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
    """A class for creating and searching a FastPlaid index."""

    @profile_resources
    def __init__(
        self,
        index: str,
        device: str | list[str] | None = None,
        low_memory: bool = True,
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
        self.low_memory = low_memory

        # Initialize Torch environment once
        fast_plaid_rust.initialize_torch(torch_path=self.torch_path)

        # Load an index object for each device.
        self.indices: dict[str, Any] = {}
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
            low_memory=self.low_memory,
        )

    def _format_embeddings(
        self, embeddings: list[torch.Tensor] | torch.Tensor
    ) -> list[torch.Tensor] | torch.Tensor:
        """Standardize embedding shapes without creating deep copies."""
        if isinstance(embeddings, torch.Tensor):
            return embeddings.squeeze(0) if embeddings.dim() == 3 else embeddings

        return [e.squeeze(0) if e.dim() == 3 else e for e in embeddings]

    @torch.inference_mode()
    @profile_resources
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
        start_from_scratch: int = 1000,
    ) -> "FastPlaid":
        """Create and saves the FastPlaid index."""
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
        self.indices.clear()
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
            low_memory=self.low_memory,
        )

        return self

    @torch.inference_mode()
    @profile_resources
    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 50_000,
        cluster_threshold: float = 0.1,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_samples_kmeans: int | None = None,
        seed: int = 42,
        start_from_scratch: int = 1000,
        use_triton_kmeans: bool | None = False,
    ) -> "FastPlaid":
        """Update an existing FastPlaid index with new documents.

        Args:
            documents_embeddings: New embeddings to add.
            metadata: Optional metadata for the new documents.
            batch_size: Batch size for processing.
            cluster_threshold: L2 Distance threshold to trigger creation of new centroids.
                               Embeddings further than this from any existing centroid
                               will be clustered to form new centroids.
            kmeans_niters: Number of iterations for K-Means (if new centroids are created).
            max_points_per_centroid: Constraint for centroid creation.
            n_samples_kmeans: Number of samples to use for K-Means (if None, auto-calculated).
            seed: Random seed for K-Means.

        """
        # Brand new index creation
        if not os.path.exists(self.index) or not os.path.exists(
            os.path.join(self.index, "metadata.json")
        ):
            return self.create(
                documents_embeddings=documents_embeddings,
                kmeans_niters=kmeans_niters,
                max_points_per_centroid=max_points_per_centroid,
                n_samples_kmeans=n_samples_kmeans,
                batch_size=batch_size,
                seed=seed,
                use_triton_kmeans=use_triton_kmeans,
                metadata=metadata,
                start_from_scratch=start_from_scratch,
            )

        documents_embeddings = self._format_embeddings(documents_embeddings)

        with open(os.path.join(self.index, "metadata.json")) as f:
            meta = json.load(f)
            # If num_documents is missing, default to start_from_scratch + 1
            # Assert backward compatibility
            num_documents_in_index = meta.get("num_documents", start_from_scratch + 1)

        num_docs = len(documents_embeddings)

        if num_documents_in_index <= start_from_scratch:
            # Re-create index from scratch if there are too few documents used
            # to compute centroids.
            if os.path.exists(os.path.join(self.index, "embeddings.npy")):
                existing_embeddings = np.load(
                    os.path.join(self.index, "embeddings.npy"),
                    allow_pickle=True,
                )

                existing_embeddings = [
                    torch.from_numpy(tensor) for tensor in existing_embeddings
                ]

                documents_embeddings = existing_embeddings + documents_embeddings

            _ = self.create(
                documents_embeddings=documents_embeddings,
                kmeans_niters=kmeans_niters,
                max_points_per_centroid=max_points_per_centroid,
                n_samples_kmeans=n_samples_kmeans,
                batch_size=batch_size,
                seed=seed,
                use_triton_kmeans=use_triton_kmeans,
                metadata=metadata,
                start_from_scratch=start_from_scratch,
            )

            return self

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

        # 1. Expand Centroids
        # Perform outlier detection and centroid expansion BEFORE adding documents.
        _update_centroids(
            index_path=self.index,
            new_embeddings=documents_embeddings,
            cluster_threshold=cluster_threshold,
            device=self.devices[0],
            kmeans_niters=kmeans_niters,
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed,
            use_triton_kmeans=use_triton_kmeans,
        )

        # 2. Reload Index to pick up new centroids
        # The Python objects need to be refreshed so the Rust backend sees the updated
        # centroid files (centroids.npy, ivf_lengths.npy, etc.)
        self.indices.clear()
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
            low_memory=self.low_memory,
        )

        # 3. Append new documents (Rust logic)
        fast_plaid_rust.update(
            index_path=self.index,
            index=self.indices[self.devices[0]],
            torch_path=self.torch_path,
            device=self.devices[0],
            embeddings=documents_embeddings,
            batch_size=batch_size,
        )

        # 4. Final Reload for Searching
        self.indices.clear()
        self.indices = _reload_index(
            index_path=self.index,
            devices=self.devices,
            indices=self.indices,
            low_memory=self.low_memory,
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
    @profile_resources
    def search(  # noqa: PLR0912, PLR0913, C901
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
        """Search the index for the given query embeddings."""
        if any(idx is None for idx in self.indices.values()):
            self.indices = _reload_index(
                index_path=self.index,
                devices=self.devices,
                indices=self.indices,
                low_memory=self.low_memory,
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
                    index_object=self.indices["cpu"],
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

    @torch.inference_mode()
    @profile_resources
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
            low_memory=self.low_memory,
        )
        return self


def _update_centroids(
    index_path: str,
    new_embeddings: list[torch.Tensor] | torch.Tensor,
    cluster_threshold: float,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
) -> None:
    """Subsample new embeddings that are too far from existing centroids,
    run K-Means on them, and append new centroids to the index.
    """
    # 1. Load Existing Centroids
    centroids_path = os.path.join(index_path, "centroids.npy")
    if not os.path.exists(centroids_path):
        return

    existing_centroids_np = np.load(centroids_path)
    existing_centroids = torch.from_numpy(existing_centroids_np).to(device)

    # 2. Prepare New Embeddings
    # Flatten if list of tensors
    if isinstance(new_embeddings, list):
        flat_embeddings = torch.cat(new_embeddings).to(device)
    else:
        flat_embeddings = new_embeddings.to(device)

    if flat_embeddings.ndim == 3:
        flat_embeddings = flat_embeddings.squeeze(0)

    # Ensure Dtype Compatibility
    if existing_centroids.dtype != flat_embeddings.dtype:
        existing_centroids = existing_centroids.to(dtype=flat_embeddings.dtype)

    # 3. Compute Distances (L2) to find closest existing centroid for each new point
    # Optimization: Process in chunks if embeddings are very large to avoid OOM
    batch_size = 4096
    num_embeddings = flat_embeddings.shape[0]
    outlier_mask = torch.zeros(num_embeddings, dtype=torch.bool, device=device)

    # Square of threshold for comparison (avoids sqrt)
    threshold_sq = cluster_threshold**2

    for i in range(0, num_embeddings, batch_size):
        batch = flat_embeddings[i : i + batch_size]

        # L2 Distance squared: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        x2 = torch.sum(batch**2, dim=1, keepdim=True)
        y2 = torch.sum(existing_centroids**2, dim=1)

        dists_sq = x2 + y2 - 2 * torch.matmul(batch, existing_centroids.T)

        # Get min distance for each point
        min_dists_sq, _ = torch.min(dists_sq, dim=1)

        # Mark as outlier if closest centroid is too far
        outlier_mask[i : i + batch_size] = min_dists_sq > threshold_sq

    # 4. Filter "Subsample" Outliers
    outliers = flat_embeddings[outlier_mask]
    num_outliers = outliers.shape[0]

    # If no outliers found, nothing to do
    if num_outliers == 0:
        return

    # 5. Compute New Centroids using existing helper
    # We pass the outliers directly; compute_kmeans handles K calculation,
    # normalization, and memory optimization.
    target_k = math.ceil(num_outliers / max_points_per_centroid)
    k_update = max(1, target_k * 4)

    new_centroids_t = compute_kmeans(
        documents_embeddings=outliers,
        dim=outliers.shape[1],
        device=device,
        kmeans_niters=kmeans_niters,
        max_points_per_centroid=max_points_per_centroid,
        seed=seed,
        n_samples_kmeans=n_samples_kmeans,
        use_triton_kmeans=use_triton_kmeans,
        num_partitions=k_update,
    )

    # Convert back to numpy for concatenation and storage
    new_centroids_np = new_centroids_t.detach().cpu().numpy().astype(np.float32)
    k_new = new_centroids_np.shape[0]

    # 6. Update Set of Existing Centroids & Metadata
    final_centroids = np.concatenate([existing_centroids_np, new_centroids_np], axis=0)

    # Save Centroids
    np.save(centroids_path, final_centroids)

    # Update IVF Lengths (append 0s for new centroids)
    ivf_path = os.path.join(index_path, "ivf_lengths.npy")
    if os.path.exists(ivf_path):
        ivf_lengths = np.load(ivf_path)
        new_lengths = np.zeros(k_new, dtype=ivf_lengths.dtype)
        final_ivf = np.concatenate([ivf_lengths, new_lengths])
        np.save(ivf_path, final_ivf)

    # Update Average Residual Norms
    # We pad with the mean of existing norms as a safe default
    avg_res_path = os.path.join(index_path, "average_residual_norm.npy")
    if os.path.exists(avg_res_path):
        avg_res = np.load(avg_res_path)
        mean_res = np.mean(avg_res) if avg_res.size > 0 else 1.0
        new_res = np.full(k_new, mean_res, dtype=avg_res.dtype)
        final_avg_res = np.concatenate([avg_res, new_res])
        np.save(avg_res_path, final_avg_res)

    # Update Metadata JSON
    meta_path = os.path.join(index_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        meta["num_partitions"] = int(final_centroids.shape[0])

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    print(f"Added {k_new} new centroids. Total centroids: {final_centroids.shape[0]}")

    # Explicit cleanup
    del outliers, existing_centroids, flat_embeddings, new_centroids_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
