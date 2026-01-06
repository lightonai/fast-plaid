"""Wrapper around FastKmeans to avoid copy."""

from __future__ import annotations

import numpy as np
import torch
from fastkmeans import FastKMeans as BaseFastKMeans

try:
    from fastkmeans.triton_kernels import triton_kmeans

    HAS_TRITON = True
except ImportError:
    triton_kmeans = None
    HAS_TRITON = False


def _get_device(preset: str | int | torch.device | None = None) -> torch.device:
    """Resolve device from preset, defaulting to best available.

    Args:
    ----
    preset:
        Device specification (string, int for GPU index, or torch.device).

    """
    if isinstance(preset, torch.device):
        return preset
    if isinstance(preset, str):
        return torch.device(preset)
    if torch.cuda.is_available():
        return torch.device(
            f"cuda:{preset if isinstance(preset, int) and preset < torch.cuda.device_count() else 0}"  # noqa: E501
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(
            f"xpu:{preset if isinstance(preset, int) and preset < torch.xpu.device_count() else 0}"  # noqa: E501
        )
    return torch.device("cpu")


def _is_bfloat16_supported(device: torch.device) -> bool:
    """Check if bfloat16 is supported on the given device.

    Args:
    ----
    device:
        The torch device to check.

    """
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type == "xpu" and hasattr(torch.xpu, "is_bf16_supported"):
        return torch.xpu.is_bf16_supported()
    return False


@torch.inference_mode()
def _kmeans_torch_double_chunked(
    data: torch.Tensor,
    data_norms: torch.Tensor,
    k: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_iters: int = 25,
    tol: float = 1e-8,
    chunk_size_data: int = 51_200,
    chunk_size_centroids: int = 10_240,
    max_points_per_centroid: int | None = 256,
    verbose: bool = False,  # noqa: ARG001
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-means with double chunking for large datasets.

    Args:
    ----
    data:
        Input data tensor of shape (n_samples, n_features).
    data_norms:
        Precomputed squared norms of data points.
    k:
        Number of clusters.
    device:
        Device to run computation on.
    dtype:
        Data type for computation.
    max_iters:
        Maximum number of iterations.
    tol:
        Convergence tolerance.
    chunk_size_data:
        Chunk size for data batching.
    chunk_size_centroids:
        Chunk size for centroid batching.
    max_points_per_centroid:
        Maximum points per centroid for subsampling.
    verbose:
        Whether to print progress.
    use_triton:
        Whether to use Triton kernels.

    """
    if use_triton and not HAS_TRITON:
        raise ImportError(
            "Triton is not available. Please install Triton and try again."
        )

    if dtype is None:
        dtype = torch.float16 if device.type in ["cuda", "xpu"] else torch.float32

    n_samples_original, n_features = data.shape
    n_samples = n_samples_original

    if max_points_per_centroid is not None and n_samples > k * max_points_per_centroid:
        target_n_samples = k * max_points_per_centroid
        perm = torch.randperm(n_samples, device=data.device)
        indices = perm[:target_n_samples]
        data = data[indices]
        data_norms = data_norms[indices]
        n_samples = target_n_samples
        del perm, indices

    if n_samples < k:
        raise ValueError(
            f"Number of training points ({n_samples}) is less than k ({k})."
        )

    rand_indices = torch.randperm(n_samples)[:k]
    centroids = data[rand_indices].clone().to(device=device, dtype=dtype)
    prev_centroids = centroids.clone()
    labels = torch.empty(n_samples, dtype=torch.int64, device="cpu")

    for _iteration in range(max_iters):
        centroid_norms = (centroids**2).sum(dim=1)
        cluster_sums = torch.zeros((k, n_features), device=device, dtype=torch.float32)
        cluster_counts = torch.zeros((k,), device=device, dtype=torch.float32)

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + chunk_size_data, n_samples)

            data_chunk = data[start_idx:end_idx].to(
                device=device, dtype=dtype, non_blocking=True
            )
            data_chunk_norms = data_norms[start_idx:end_idx].to(
                device=device, dtype=dtype, non_blocking=True
            )
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=device, dtype=torch.int64)

            if use_triton:
                triton_kmeans(
                    data_chunk=data_chunk,
                    data_chunk_norms=data_chunk_norms,
                    centroids=centroids,
                    centroids_sqnorm=centroid_norms,
                    best_ids=best_ids,
                )
            else:
                best_dist = torch.full(
                    (batch_size,), float("inf"), device=device, dtype=dtype
                )
                c_start = 0
                while c_start < k:
                    c_end = min(c_start + chunk_size_centroids, k)
                    centroid_chunk = centroids[c_start:c_end]
                    centroid_chunk_norms = centroid_norms[c_start:c_end]

                    dist_chunk = data_chunk_norms.unsqueeze(
                        1
                    ) + centroid_chunk_norms.unsqueeze(0)
                    dist_chunk = dist_chunk.addmm_(
                        data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0
                    )

                    local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                    improved_mask = local_min_vals < best_dist
                    best_dist[improved_mask] = local_min_vals[improved_mask]
                    best_ids[improved_mask] = c_start + local_min_ids[improved_mask]

                    c_start = c_end

            cluster_sums.index_add_(0, best_ids, data_chunk.float())
            cluster_counts.index_add_(
                0,
                best_ids,
                torch.ones_like(best_ids, device=device, dtype=torch.float32),
            )

            labels[start_idx:end_idx] = best_ids.to("cpu", non_blocking=True)
            start_idx = end_idx

        new_centroids = torch.zeros_like(centroids, device=device, dtype=dtype)
        non_empty = cluster_counts > 0
        new_centroids[non_empty] = (
            cluster_sums[non_empty] / cluster_counts[non_empty].unsqueeze(1)
        ).to(dtype=dtype)

        empty_ids = (~non_empty).nonzero(as_tuple=True)[0]
        if len(empty_ids) > 0:
            reinit_indices = torch.randint(
                0, n_samples, (len(empty_ids),), device="cpu"
            )
            random_data = data[reinit_indices].to(
                device=device, dtype=dtype, non_blocking=True
            )
            new_centroids[empty_ids] = random_data

        shift = (
            torch.norm(new_centroids - prev_centroids.to(new_centroids.device), dim=1)
            .sum()
            .item()
        )
        centroids = new_centroids

        prev_centroids = centroids.clone()

        if shift < tol:
            break

    return centroids.to("cpu", dtype=torch.float32), labels


class FastKMeans(BaseFastKMeans):
    """FastKMeans with torch tensor support."""

    def train(self, data: torch.Tensor) -> torch.Tensor:
        """Train K-means and return centroids as torch tensor.

        Args:
        ----
        data:
            Input data as numpy array or torch tensor.

        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002

        data_torch = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
        data_norms_torch = (data_torch**2).sum(dim=1)

        device = _get_device(self.device)
        if device.type == "cuda" and self.pin_gpu_memory:
            data_torch = data_torch.pin_memory()
            data_norms_torch = data_norms_torch.pin_memory()

        centroids, _ = _kmeans_torch_double_chunked(
            data_torch,
            data_norms_torch,
            k=self.k,
            max_iters=self.niter,
            tol=self.tol,
            device=device,
            dtype=self.dtype,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
            max_points_per_centroid=self.max_points_per_centroid,
            verbose=self.verbose,
            use_triton=self.use_triton,
        )

        return centroids
