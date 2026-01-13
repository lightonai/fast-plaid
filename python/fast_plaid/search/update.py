from __future__ import annotations

import gc
import json
import math
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch
from fast_plaid import fast_plaid_rust
from usearch.index import Index

from ..filtering import update as update_metadata_db
from .load import (
    _construct_index_from_tensors,
    _load_index_tensors_cpu,
    _reload_index,
    save_list_tensors_on_disk,
)


def partial_reload(
    index_path: str,
    devices: list[str],
    indices_dict: dict[str, Any],
    low_memory: bool,
) -> dict[str, Any]:
    """Reload index data after update.

    Args:
    ----
    index_path:
        The path to the index directory.
    devices:
        List of devices to reload the index on.
    indices_dict:
        Dictionary mapping devices to index objects, updated in place.
    low_memory:
        Whether to use low memory mode when loading.

    """
    # Clear old indices first to release memory-mapped file handles
    # This is critical on Windows where files can't be modified while mapped
    indices_dict.clear()
    gc.collect()

    cpu_tensors = _load_index_tensors_cpu(index_path=index_path)
    if cpu_tensors is None:
        return indices_dict

    for device in devices:
        indices_dict[device] = _construct_index_from_tensors(
            data=cpu_tensors,
            device=device,
            low_memory=low_memory,
        )

    return indices_dict


def update_centroids(  # noqa: PLR0912
    index_path: str,
    new_embeddings: list[torch.Tensor] | torch.Tensor,
    cluster_threshold: float,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    compute_kmeans_fn: Callable,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
) -> None:
    """Update centroids by clustering embeddings far from existing centroids.

    Args:
    ----
    index_path:
        The path to the index directory.
    new_embeddings:
        New embeddings to potentially add as new centroids.
    cluster_threshold:
        Distance threshold for identifying outliers.
    device:
        The device to run the computation on.
    kmeans_niters:
        The number of iterations for the K-means algorithm.
    max_points_per_centroid:
        The maximum number of points to support per centroid.
    seed:
        The random seed for initialization.
    compute_kmeans_fn:
        Function to compute K-means centroids.
    n_samples_kmeans:
        The number of samples to use for K-means training.
    use_triton_kmeans:
        Whether to use the Triton implementation of K-means.

    """
    centroids_path = os.path.join(index_path, "centroids.npy")
    if not os.path.exists(centroids_path):
        return

    existing_centroids_np = np.load(centroids_path)
    existing_centroids = torch.from_numpy(existing_centroids_np).to(device)

    if isinstance(new_embeddings, list):
        flat_embeddings = torch.cat(new_embeddings).to(device)
    else:
        flat_embeddings = new_embeddings.to(device)

    if flat_embeddings.ndim == 3:
        flat_embeddings = flat_embeddings.squeeze(0)

    if existing_centroids.dtype != flat_embeddings.dtype:
        existing_centroids = existing_centroids.to(dtype=flat_embeddings.dtype)

    # Compute distances to find outliers (embeddings far from all centroids)
    batch_size = 4096
    num_embeddings = flat_embeddings.shape[0]
    outlier_mask = torch.zeros(num_embeddings, dtype=torch.bool, device=device)
    threshold_sq = cluster_threshold**2

    if device == "cpu":
        existing_centroids_np = existing_centroids_np.astype(np.float32)
        idx = Index(ndim=existing_centroids_np.shape[1], metric="l2sq")
        idx.add(
            np.arange(len(existing_centroids_np)),
            existing_centroids_np,
        )

        flat_embeddings_np = flat_embeddings.detach().cpu().numpy().astype(np.float32)

        for i in range(0, num_embeddings, batch_size):
            batch_np = flat_embeddings_np[i : i + batch_size]
            matches = idx.search(batch_np, 1)
            batch_dists = torch.from_numpy(matches.distances).flatten()
            outlier_mask[i : i + batch_size] = batch_dists > threshold_sq

    else:
        for i in range(0, num_embeddings, batch_size):
            batch = flat_embeddings[i : i + batch_size]
            x2 = torch.sum(batch**2, dim=1, keepdim=True)
            y2 = torch.sum(existing_centroids**2, dim=1)
            dists_sq = x2 + y2 - 2 * torch.matmul(batch, existing_centroids.T)
            min_dists_sq, _ = torch.min(dists_sq, dim=1)
            outlier_mask[i : i + batch_size] = min_dists_sq > threshold_sq

    outliers = flat_embeddings[outlier_mask]
    num_outliers = outliers.shape[0]

    if num_outliers == 0:
        del outliers, existing_centroids, flat_embeddings
        gc.collect()
        return

    # Compute new centroids from outliers
    target_k = math.ceil(num_outliers / max_points_per_centroid)
    k_update = max(1, target_k * 4)

    new_centroids_t = compute_kmeans_fn(
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

    # Update centroids and metadata on disk
    new_centroids_np = new_centroids_t.detach().cpu().numpy().astype(np.float32)
    k_new = new_centroids_np.shape[0]
    final_centroids = np.concatenate([existing_centroids_np, new_centroids_np], axis=0)

    np.save(centroids_path, final_centroids)

    ivf_path = os.path.join(index_path, "ivf_lengths.npy")
    if os.path.exists(ivf_path):
        ivf_lengths = np.load(ivf_path)
        new_lengths = np.zeros(k_new, dtype=ivf_lengths.dtype)
        final_ivf = np.concatenate([ivf_lengths, new_lengths])
        np.save(ivf_path, final_ivf)

    meta_path = os.path.join(index_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        meta["num_partitions"] = int(final_centroids.shape[0])

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    del outliers, existing_centroids, flat_embeddings, new_centroids_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_update(
    index_path: str,
    devices: list[str],
    torch_path: str,
    low_memory: bool,
    indices_dict: dict[str, Any],
    documents_embeddings: list[torch.Tensor] | torch.Tensor,
    metadata: list[dict[str, Any]] | None,
    batch_size: int,
    kmeans_niters: int,
    max_points_per_centroid: int,
    n_samples_kmeans: int | None,
    seed: int,
    start_from_scratch: int,
    buffer_size: int,
    use_triton_kmeans: bool | None,
    create_fn: Callable,
    delete_fn: Callable,
    compute_kmeans_fn: Callable,
    format_embeddings_fn: Callable,
) -> dict[str, Any]:
    """Execute the update logic for FastPlaid.

    Args:
    ----
    index_path:
        The path to the index directory.
    devices:
        List of devices to use for the update.
    torch_path:
        Path to the torch shared library.
    low_memory:
        Whether to use low memory mode when loading.
    indices_dict:
        Dictionary mapping devices to index objects.
    documents_embeddings:
        New embeddings to add to the index.
    metadata:
        Optional metadata for the new documents.
    batch_size:
        Batch size for processing the update.
    kmeans_niters:
        Number of iterations for K-means.
    max_points_per_centroid:
        Maximum points per centroid for K-means.
    n_samples_kmeans:
        Number of samples to use for K-means.
    seed:
        Random seed for initialization.
    start_from_scratch:
        Threshold below which to rebuild the index.
    buffer_size:
        Number of embeddings to trigger centroid expansion.
    use_triton_kmeans:
        Whether to use the Triton implementation of K-means.
    create_fn:
        Function to create a new index.
    delete_fn:
        Function to delete documents from the index.
    compute_kmeans_fn:
        Function to compute K-means centroids.
    format_embeddings_fn:
        Function to format embeddings.

    """
    if not os.path.exists(index_path) or not os.path.exists(
        os.path.join(index_path, "metadata.json")
    ):
        create_fn(
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
        return _reload_index(
            index_path=index_path,
            devices=devices,
            indices={},
            low_memory=low_memory,
        )

    documents_embeddings = format_embeddings_fn(documents_embeddings)

    with open(os.path.join(index_path, "metadata.json")) as f:
        meta = json.load(f)
        num_documents_in_index = meta.get("num_documents", start_from_scratch + 1)

    num_docs = len(documents_embeddings)

    if os.path.exists(os.path.join(index_path, "metadata.db")):
        if metadata is None:
            metadata = [{} for _ in range(num_docs)]

        if len(metadata) != num_docs:
            error = f"""
            The length of metadata ({len(metadata)}) must match the number of
            documents_embeddings ({num_docs}).
            """
            raise ValueError(error)
        update_metadata_db(index=index_path, metadata=metadata)

    # Rebuild index from scratch if below threshold
    if num_documents_in_index <= start_from_scratch and os.path.exists(
        os.path.join(index_path, "embeddings.npy")
    ):
        existing_embeddings_np = np.load(
            os.path.join(index_path, "embeddings.npy"),
            allow_pickle=True,
        )
        existing_embeddings = [
            torch.from_numpy(tensor) for tensor in existing_embeddings_np
        ]
        documents_embeddings = existing_embeddings + documents_embeddings

        create_fn(
            documents_embeddings=documents_embeddings,
            kmeans_niters=kmeans_niters,
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            batch_size=batch_size,
            seed=seed,
            use_triton_kmeans=use_triton_kmeans,
            metadata=None,
            start_from_scratch=start_from_scratch,
        )

        if len(documents_embeddings) > start_from_scratch and os.path.exists(
            os.path.join(index_path, "embeddings.npy")
        ):
            os.remove(os.path.join(index_path, "embeddings.npy"))

        return _reload_index(
            index_path=index_path,
            devices=devices,
            indices={},
            low_memory=low_memory,
        )

    # Ensure index is loaded before update
    if devices[0] not in indices_dict or indices_dict[devices[0]] is None:
        new_indices = _reload_index(
            index_path=index_path,
            devices=devices,
            indices=indices_dict,
            low_memory=low_memory,
        )
        indices_dict.clear()
        indices_dict.update(new_indices)

        if indices_dict[devices[0]] is None:
            raise RuntimeError("Index not loaded for update.")

    thresh_path = os.path.join(index_path, "cluster_threshold.npy")
    cluster_threshold = float(np.load(thresh_path))

    existing_buffer_embeddings: list[torch.Tensor] = []

    if os.path.exists(os.path.join(index_path, "buffer.npy")):
        existing_buffer_np = np.load(
            os.path.join(index_path, "buffer.npy"),
            allow_pickle=True,
        )
        existing_buffer_embeddings = [torch.from_numpy(t) for t in existing_buffer_np]

    total_new_docs = len(documents_embeddings) + len(existing_buffer_embeddings)

    # Buffer reached - expand centroids and process all buffered documents
    if total_new_docs >= buffer_size:
        if len(existing_buffer_embeddings) > 0:
            start_del_idx = num_documents_in_index - len(existing_buffer_embeddings)
            documents_to_delete = list(range(start_del_idx, num_documents_in_index))
            documents_embeddings = existing_buffer_embeddings + documents_embeddings
            delete_fn(subset=documents_to_delete, _delete_metadata=False)

        update_centroids(
            index_path=index_path,
            new_embeddings=documents_embeddings,
            cluster_threshold=cluster_threshold,
            device=devices[0],
            kmeans_niters=kmeans_niters,
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed,
            compute_kmeans_fn=compute_kmeans_fn,
            use_triton_kmeans=use_triton_kmeans,
        )

        indices_dict = _reload_index(
            index_path=index_path,
            devices=devices,
            indices=indices_dict,
            low_memory=low_memory,
        )

        if os.path.exists(os.path.join(index_path, "buffer.npy")):
            os.remove(os.path.join(index_path, "buffer.npy"))

        fast_plaid_rust.update(
            index_path=index_path,
            index=indices_dict[devices[0]],
            torch_path=torch_path,
            device=devices[0],
            embeddings=documents_embeddings,
            batch_size=batch_size,
            update_threshold_centroids=True,
        )

        return partial_reload(
            index_path=index_path,
            devices=devices,
            indices_dict=indices_dict,
            low_memory=low_memory,
        )

    # Buffer not reached - append to buffer and update without centroid expansion
    save_list_tensors_on_disk(
        path=os.path.join(index_path, "buffer.npy"),
        tensors=documents_embeddings,
    )

    fast_plaid_rust.update(
        index_path=index_path,
        index=indices_dict[devices[0]],
        torch_path=torch_path,
        device=devices[0],
        embeddings=documents_embeddings,
        batch_size=batch_size,
        update_threshold_centroids=False,
    )

    return partial_reload(
        index_path=index_path,
        devices=devices,
        indices_dict=indices_dict,
        low_memory=low_memory,
    )
