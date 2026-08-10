"""Failures the fused path is allowed to recover from.

Kept in their own module, importing nothing but the standard library, because
naming a fallback reason must not require the Triton kernels: an install
without Triton has to be able to ask the gate why the fast path is
unavailable, and the gate cannot answer if importing it has already raised.

The split between these classes and everything else is the whole fallback
policy. A failure that derives from :class:`FusedUnavailableError` names a
condition the standard pipeline is unaffected by, so ``search`` answers from
Rust instead. Anything else -- a bug in the kernels, or a device-side assert
that leaves the CUDA context poisoned for the rest of the process --
propagates, because falling back after those either hides a defect or fails a
second time on the same broken context.
"""

from __future__ import annotations


class FusedUnavailableError(RuntimeError):
    """The fused path cannot serve this call, but the standard pipeline can."""


class FusedOutOfMemoryError(FusedUnavailableError):
    """A single query's transients did not fit on the device.

    Transient by nature, and deliberately not a reason to retire the staged
    copy: the admission ceiling is recomputed from free memory on every call,
    so pressure from another process on the same device resolves itself
    without paying to stage the index again.
    """


class FusedCompilationError(FusedUnavailableError):
    """Triton could not compile or launch a kernel for this shape.

    Deterministic where :class:`FusedOutOfMemoryError` is transient: the same
    shape fails the same way every time, so the caller retires the staged copy
    rather than paying a doomed compilation on every subsequent search.
    """
