"""Fused CUDA search path.

An opt-in fast path that reads the loaded index's device tensors in place and
follows the standard scoring chain rounding step for rounding step. It runs
only on top of a ``'high'`` index placement -- codes and residuals resident on
the device -- and allocates nothing of its own beyond the per-token
reconstruction norms. When any precondition is unmet the caller runs the
standard pipeline instead.

It returns the same documents, and scores them to within a measured tolerance
rather than to the bit: the Half GEMM accumulates in a different order here
than in libtorch, which moves a per-token maximum by an ulp or two. How far
depends on the card and the shape together, since cuBLAS and Triton each pick
their kernel independently -- the parity suite sees 2.44e-4 on sm_86/sm_89,
1.22e-4 on sm_80 and zero on sm_90, while one real corpus deviates by zero on
an H100 and 4.9e-4 on an L4. Zero documents were substituted in any of it.

Importing this package must work everywhere the wheel installs, including the
CPU, macOS and Windows builds that ship no Triton at all. ``engine`` is
therefore reached lazily: it pulls in ``kernels``, which imports Triton
unconditionally, and an eager import would raise ``ModuleNotFoundError`` from
inside the very call that exists to decline gracefully.
"""

from __future__ import annotations

import sys
from typing import Any

from . import ceiling, gate
from .errors import (
    FusedCompilationError,
    FusedOutOfMemoryError,
    FusedUnavailableError,
)

__all__ = [
    "FusedCompilationError",
    "FusedEngine",
    "FusedOutOfMemoryError",
    "FusedUnavailableError",
    "build_engine",
    "ceiling",
    "gate",
]


def __getattr__(name: str) -> Any:
    """Resolve ``FusedEngine`` on first use rather than at import.

    Keeps the public name available to callers that can run the kernels while
    leaving the module importable on installs that cannot.
    """
    if name == "FusedEngine":
        from .engine import FusedEngine

        return FusedEngine
    error = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(error)


def _declined(reason: str) -> tuple[None, str]:
    if gate.is_debug():
        print(f"[fast-plaid] fused path unavailable: {reason}", file=sys.stderr)
    return None, reason


def build_engine(
    data: dict[str, Any],
    device: str,
    *,
    index_memory_fraction: float = gate.DEFAULT_MEMORY_FRACTION,
    search_memory_fraction: float = ceiling.BUDGET_FRACTION,
) -> tuple[Any | None, str | None]:
    """Build a fused engine over the loaded index, or explain why it cannot run.

    Returns ``(engine, None)`` when the fast path is available and
    ``(None, reason)`` otherwise. The reason is kept rather than discarded so
    that callers can report it without re-running the gate.

    Args:
    ----
    data:
        The tensors the standard index was constructed from, as attached to
        the loaded index by the loader.
    device:
        Target CUDA device.
    index_memory_fraction:
        Share of free memory the engine's allocation and staging transient may
        occupy.
    search_memory_fraction:
        Share of free memory this engine's per-search transients may occupy.

    """
    reason = gate.check(data=data, device=device, memory_fraction=index_memory_fraction)
    if reason is not None:
        return _declined(reason)

    # Only reached once the gate has confirmed a CUDA device with Triton
    # installed, so importing the kernels here cannot raise for want of it.
    from .engine import FusedEngine

    try:
        engine = FusedEngine(
            data=data, device=device, budget_fraction=search_memory_fraction
        )
    except RuntimeError as error:  # pragma: no cover - device dependent
        # torch.cuda.OutOfMemoryError derives from RuntimeError; staging that
        # cannot complete is a decline, not a failure.
        return _declined(f"staging failed: {error}")

    if gate.is_debug():
        print(
            f"[fast-plaid] fused path active on {device}: "
            f"{engine.n_tokens} tokens, {engine.n_docs} docs, "
            f"{engine.resident_bytes() / 2**20:.0f} MiB allocated, "
            f"{engine.shared_bytes() / 2**20:.0f} MiB read in place",
            file=sys.stderr,
        )
    return engine, None
