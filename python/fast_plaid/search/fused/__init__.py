"""Fused CUDA search path.

An optional fast path that reads the standard index format unmodified and
reproduces the standard scoring chain rounding point for rounding point. When
any precondition is unmet the caller runs the standard pipeline instead, so
enabling this module can change latency but never results.
"""

from __future__ import annotations

import sys
from typing import Any

from . import ceiling, gate
from .engine import FusedEngine

__all__ = ["FusedEngine", "build_engine", "ceiling", "gate"]


def build_engine(
    data: dict[str, Any], device: str
) -> tuple[FusedEngine | None, str | None]:
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

    """
    n_tokens = int(data["doc_lengths"].sum())
    reason = gate.check(data=data, device=device, n_tokens=n_tokens)
    if reason is not None:
        if gate.is_debug():
            print(f"[fast-plaid] fused path unavailable: {reason}", file=sys.stderr)
        return None, reason

    try:
        engine = FusedEngine(data=data, device=device)
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
