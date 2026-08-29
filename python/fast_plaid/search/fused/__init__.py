"""Fused CUDA search path.

An optional fast path that reads the standard index format unmodified and
follows the standard scoring chain rounding step for rounding step. When any
precondition is unmet the caller runs the standard pipeline instead.

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


def build_engine(
    data: dict[str, Any],
    device: str,
    *,
    index_memory_fraction: float = gate.DEFAULT_MEMORY_FRACTION,
    search_memory_fraction: float = ceiling.BUDGET_FRACTION,
) -> tuple[Any | None, str | None]:
    """Stage a fused engine, or explain why the fast path cannot run.

    Returns ``(engine, None)`` when the fast path is available and
    ``(None, reason)`` otherwise. The reason is kept rather than discarded so
    that callers can report it without re-running the gate.

    Args:
    ----
    data:
        Index tensors from ``_load_index_tensors_cpu``.
    device:
        Target CUDA device.
    index_memory_fraction:
        Share of free memory the resident copy may occupy.
    search_memory_fraction:
        Share of free memory this engine's per-search transients may occupy.

    """
    # The gate derives the token count itself, and only after the checks that
    # need no index at all. Reading ``data`` here instead would make a decline
    # depend on the tensors being well formed -- the same ordering mistake as
    # importing the kernels before deciding whether they can run.
    reason = gate.check(
        data=data,
        device=device,
        memory_fraction=index_memory_fraction,
    )
    if reason is not None:
        if gate.is_debug():
            print(f"[fast-plaid] fused path unavailable: {reason}", file=sys.stderr)
        return None, reason

    # Only reached once the gate has confirmed a CUDA device with Triton
    # installed, so importing the kernels here cannot raise for want of it.
    from .engine import FusedEngine

    try:
        engine = FusedEngine(
            data=data, device=device, budget_fraction=search_memory_fraction
        )
    except RuntimeError as error:  # pragma: no cover - device dependent
        # torch.cuda.OutOfMemoryError derives from RuntimeError; staging that
        # cannot complete is a fallback, not a failure.
        reason = f"staging failed: {error}"
        if gate.is_debug():
            print(f"[fast-plaid] fused {reason}", file=sys.stderr)
        return None, reason

    if gate.is_debug():
        print(
            f"[fast-plaid] fused path active on {device}: "
            f"{engine.n_tokens} tokens, {engine.n_docs} docs, "
            f"{engine.resident_bytes() / 2**30:.2f} GiB resident",
            file=sys.stderr,
        )
    return engine, None
