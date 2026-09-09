"""Fused CUDA search path.

An opt-in fast path that reads the loaded index's device tensors in place and
follows the standard scoring chain step for step. It returns the same documents;
scores can differ by an fp16 ulp or two because cuBLAS and Triton accumulate
the Half GEMM in different orders (zero on sm_90, up to 2.4e-4 elsewhere).

``engine`` is imported lazily because it pulls in Triton, which CPU, macOS and
Windows wheels do not ship.
"""

from __future__ import annotations

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
        return None, reason

    # The gate has confirmed CUDA and Triton, so this import cannot fail.
    from .engine import FusedEngine

    try:
        engine = FusedEngine(
            data=data, device=device, budget_fraction=search_memory_fraction
        )
    except RuntimeError as error:  # pragma: no cover - device dependent
        # Includes torch.cuda.OutOfMemoryError: a staging that cannot complete declines.
        return None, f"staging failed: {error}"

    return engine, None
