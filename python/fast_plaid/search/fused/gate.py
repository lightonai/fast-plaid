"""Activation gate for the fused CUDA search path.

Every check returns a human-readable reason, reported by ``fused_status`` and
by the one-time warning, so a decline never stays silent. The engine reads the
standard index in place and allocates only per-token norms plus bookkeeping, so
that is all the memory check budgets.
"""

from __future__ import annotations

from typing import Any

import torch

# Compute capabilities the kernels have been validated on.
TESTED_ARCHS = ((8, 0), (8, 6), (8, 9), (9, 0))

# Residual widths that divide a byte evenly; the unpack schedule needs that.
SUPPORTED_NBITS = (1, 2, 4)

# The exact kernel loads a 128-wide dimension tile and masks the tail.
MAX_DIM = 128

# Read in place from the standard index; the 'high' tier keeps all four on the device.
BORROWED = ("doc_codes", "doc_residuals", "centroids", "ivf")

# Share of free VRAM the engine's own allocation and its staging transient may use.
DEFAULT_MEMORY_FRACTION = 0.8

# The engine's only per-token allocation: one fp16 reconstruction norm.
_NORM_BYTES = 2

# Small on purpose: the allocator keeps the precompute's peak for the whole process.
NORM_CHUNK = 65_536

# Per token and per dim: int64 unpacked codes and partials, fp16 centroid, weight, sum.
_NORM_TRANSIENT_BYTES_PER_DIM = 22


def missing_on_device(data: dict[str, Any], device: str) -> list[str]:
    """Borrowed tensors that are not resident on ``device``."""
    target = torch.device(device)
    return [
        key
        for key in BORROWED
        if not (isinstance(data.get(key), torch.Tensor) and data[key].device == target)
    ]


def resident_bytes(*, n_tokens: int, n_docs: int, n_centroids: int) -> int:
    """Device bytes the engine allocates itself: norms, IVF offsets, per-doc arrays."""
    return (
        n_tokens * _NORM_BYTES + n_centroids * 8 + (n_centroids + 1) * 8 + n_docs * 12
    )


def staging_bytes(*, n_tokens: int, dim: int) -> int:
    """Peak transient of the norm precompute."""
    return min(n_tokens, NORM_CHUNK) * dim * _NORM_TRANSIENT_BYTES_PER_DIM


def check(  # noqa: PLR0911 - one branch per precondition, each with its reason
    data: dict[str, Any],
    device: str,
    *,
    n_tokens: int | None = None,
    free_bytes: int | None = None,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> str | None:
    """Return a reason the fused path cannot run, or ``None`` if it can.

    Args:
    ----
    data:
        The tensors the standard index was built from, as attached to the
        loaded index.
    device:
        Target device string, e.g. ``'cuda:0'``.
    n_tokens:
        Total indexed tokens; derived from ``data`` when not supplied.
    free_bytes:
        Free device memory to plan against; sampled when not supplied.
    memory_fraction:
        Share of free memory the engine may use, see :data:`DEFAULT_MEMORY_FRACTION`.

    """
    if not device.startswith("cuda"):
        return f"device '{device}' is not CUDA"

    if not torch.cuda.is_available():
        return "CUDA is not available"

    try:
        import triton  # noqa: F401
    except ImportError:
        return "triton is not installed"

    arch = torch.cuda.get_device_capability(device)
    if arch not in TESTED_ARCHS:
        return f"compute capability {arch[0]}.{arch[1]} is not validated"

    nbits = int(data["nbits"])
    if nbits not in SUPPORTED_NBITS:
        return f"nbits={nbits} is not supported by the fused kernels"

    if data.get("ivf") is None or data.get("ivf_lengths") is None:
        return "index has no IVF lists"

    dim = int(data["centroids"].shape[1])
    if dim > MAX_DIM:
        return f"dim={dim} exceeds the {MAX_DIM}-wide kernel tile"
    if (dim * nbits) % 8:
        return f"dim={dim} with nbits={nbits} is not byte-aligned"

    # Only the 'high' tier keeps everything the engine reads on the device.
    missing = missing_on_device(data, device)
    if missing:
        tier = data.get("index_gpu_memory", "unknown")
        return (
            f"index_gpu_memory='{tier}' keeps {' and '.join(missing)} off the "
            "device; the fused path serves only a 'high' placement"
        )

    if free_bytes is None:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device)

    doc_lengths = data["doc_lengths"].reshape(-1)
    if n_tokens is None:
        n_tokens = int(doc_lengths.sum())
    resident = resident_bytes(
        n_tokens=n_tokens,
        n_docs=int(doc_lengths.numel()),
        n_centroids=int(data["centroids"].shape[0]),
    )

    # Staging must survive its own peak, not just its steady state.
    transient = staging_bytes(n_tokens=n_tokens, dim=dim)
    required = resident + transient

    if required > memory_fraction * free_bytes:
        return (
            f"the fused engine needs {resident / 2**30:.2f} GiB for norms and "
            f"bookkeeping plus {transient / 2**30:.2f} GiB while staging, but only "
            f"{free_bytes / 2**30:.1f} GiB is free and the engine is capped at "
            f"{memory_fraction:g} of that, i.e. "
            f"{memory_fraction * free_bytes / 2**30:.1f} GiB"
        )

    return None
