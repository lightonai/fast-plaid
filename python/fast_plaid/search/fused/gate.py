"""Activation gate for the fused CUDA search path.

The fused path is an optimisation, never a behaviour change: whenever any
condition here is not met the caller runs the standard pipeline unchanged.
Every check returns a human-readable reason so that ``fused_status`` and
``FAST_PLAID_FUSED_DEBUG`` can explain a decline instead of leaving it silent.

The engine reads the standard index's device tensors in place -- codes,
residuals, centroids and IVF lists -- and allocates only the per-token
reconstruction norms plus per-document and per-centroid bookkeeping. The memory
check budgets that allocation and the transient of computing it, nothing more.
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

DEBUG_ENV = "FAST_PLAID_FUSED_DEBUG"

# Index tensors the engine reads in place. All four must be resident on the
# device, which is what the 'high' placement tier guarantees.
BORROWED = ("doc_codes", "doc_residuals", "centroids", "ivf")

# Share of free memory the engine's own allocation, plus the transient of
# computing it, may occupy. Deliberately not the index's
# ``index_memory_fraction``: that budget governs where the standard index is
# placed, and has already been applied by the time this check runs.
DEFAULT_MEMORY_FRACTION = 0.8

# fp16 reconstruction norm per token: the engine's only per-token allocation.
_NORM_BYTES = 2

# Tokens per chunk of the reconstruction-norm precompute, owned here rather
# than by the engine because the transient it implies is part of what this
# module has to plan for. The engine imports it back.
#
# Deliberately small. The chunk is a pure scheduling knob on a bandwidth-bound
# loop that runs once per staging, and the caching allocator keeps whatever
# peak that loop reaches for the life of the process: at 500,000 tokens it left
# a gigabyte reserved behind a 74 MB index. At 65,536 the peak is ~140 MiB at
# dim 96, for a few thousand extra iterations on the largest corpora.
NORM_CHUNK = 65_536

# Device bytes the norm precompute holds per token of its chunk: the unpacked
# codes and the per-byte partials at int64 (8 each), then the gathered
# centroids, the looked-up bucket weights and their sum at fp16 (2 each).
_NORM_TRANSIENT_BYTES_PER_DIM = 22


def is_debug() -> bool:
    """Whether declines should be reported to stderr."""
    return os.environ.get(DEBUG_ENV, "") not in ("", "0")


def missing_on_device(data: dict[str, Any], device: str) -> list[str]:
    """Borrowed tensors that are not resident on ``device``."""
    target = torch.device(device)
    return [
        key
        for key in BORROWED
        if not (isinstance(data.get(key), torch.Tensor) and data[key].device == target)
    ]


def resident_bytes(*, n_tokens: int, n_docs: int, n_centroids: int) -> int:
    """Device bytes the engine allocates itself, matching ``FusedEngine``.

    Mirrors ``FusedEngine.resident_bytes`` term for term: the fp16 norms, the
    int64 IVF lengths and offsets (one longer), and per document an int64 token
    offset and an int32 length. The per-document and per-centroid arrays are
    small beside the norms but not nothing: on millions of documents they run
    to hundreds of megabytes.
    """
    return (
        n_tokens * _NORM_BYTES + n_centroids * 8 + (n_centroids + 1) * 8 + n_docs * 12
    )


def staging_bytes(*, n_tokens: int, dim: int) -> int:
    """Peak transient the norm precompute adds on top of the allocation."""
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
        The tensors the standard index was constructed from, as attached to
        the loaded index by the loader.
    device:
        Target device string, e.g. ``'cuda:0'``.
    n_tokens:
        Total indexed tokens. Derived from ``data`` when not supplied, which
        happens only after the checks that need no index at all -- a device
        without CUDA is declined without ever reading the tensors.
    free_bytes:
        Free device memory to plan against. Sampled from the device when not
        supplied.
    memory_fraction:
        Share of free memory the engine's allocation and staging transient may
        occupy. See :data:`DEFAULT_MEMORY_FRACTION`.

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

    # The engine never copies the index: whatever it reads must already be on
    # the device, which only the 'high' placement guarantees.
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

    # Staging has to survive its own peak, not just its steady state: the norm
    # precompute holds a chunk-sized working set on top of the norms already
    # allocated. Planning against the sum is what makes declining honest -- an
    # index whose norms fit but whose precompute does not would otherwise pass
    # this check and then fail inside the constructor.
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
