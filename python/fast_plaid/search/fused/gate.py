"""Activation gate for the fused CUDA search path.

The fused path is an optimisation, never a behaviour change: whenever any
condition here is not met the caller runs the standard pipeline unchanged.
Every check returns a human-readable reason so that ``FAST_PLAID_FUSED_DEBUG``
can explain a fallback instead of leaving it silent.
"""

from __future__ import annotations

import os
from typing import Any

import torch

# (major, minor) compute capabilities the kernels have been validated on.
TESTED_ARCHS = ((8, 0), (8, 6), (8, 9), (9, 0))

# Residual layouts the kernels implement. Codes must divide a byte evenly so
# that the unpack schedule is a fixed shift sequence.
SUPPORTED_NBITS = (1, 2, 4)

# The exact kernel loads a 128-wide dimension tile and masks the tail.
MAX_DIM = 128

DISABLE_ENV = "FAST_PLAID_DISABLE_FUSED"
DEBUG_ENV = "FAST_PLAID_FUSED_DEBUG"

# Share of free memory the staged copy may occupy. Deliberately its own
# constant rather than the index's ``index_memory_fraction``: the fused copy is
# staged alongside the standard index rather than replacing it, so that budget
# has already been spent once by the time this check runs, and charging the
# second copy against what remains declines exactly the large indexes the fast
# path exists for. Revisit if residency ever becomes single.
DEFAULT_MEMORY_FRACTION = 0.8

# Per-token device bytes staged by the engine: int32 centroid id, packed
# residuals, fp16 reconstruction norm.
_CODE_BYTES = 4
_NORM_BYTES = 2


def bytes_per_token(dim: int, nbits: int) -> int:
    """Device bytes the fused engine stages per indexed token."""
    return _CODE_BYTES + (dim * nbits) // 8 + _NORM_BYTES


def is_debug() -> bool:
    """Whether fallback reasons should be reported to stderr."""
    return os.environ.get(DEBUG_ENV, "") not in ("", "0")


def check(  # noqa: PLR0911 - one branch per precondition, each with its reason
    data: dict[str, Any],
    device: str,
    *,
    n_tokens: int,
    free_bytes: int | None = None,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> str | None:
    """Return a reason the fused path cannot run, or ``None`` if it can.

    Args:
    ----
    data:
        Index tensors as returned by ``_load_index_tensors_cpu``.
    device:
        Target device string, e.g. ``'cuda:0'``.
    n_tokens:
        Total indexed tokens, i.e. the sum of document lengths.
    free_bytes:
        Free device memory to plan against. Sampled from the device when not
        supplied.
    memory_fraction:
        Share of free memory the staged copy may occupy. See
        :data:`DEFAULT_MEMORY_FRACTION` for why this is not the index's
        ``index_memory_fraction``.

    """
    if os.environ.get(DISABLE_ENV, "") not in ("", "0"):
        return f"disabled by {DISABLE_ENV}"

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

    if free_bytes is None:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device)

    centroids = data["centroids"]
    required = (
        n_tokens * bytes_per_token(dim=dim, nbits=nbits)
        + centroids.numel() * 2
        + data["ivf"].numel() * 4
    )
    # Staging needs headroom for the transient buffers of at least one query.
    if required > memory_fraction * free_bytes:
        return (
            f"index needs {required / 2**30:.1f} GiB resident but only "
            f"{free_bytes / 2**30:.1f} GiB is free and "
            f"the staged copy is capped at {memory_fraction:g} of that, "
            f"i.e. {memory_fraction * free_bytes / 2**30:.1f} GiB"
        )

    return None
