"""Closed-form batch admission for the fused search path.

Every transient is linear in the query batch, so the largest admissible batch
is arithmetic rather than trial and error. The coefficients are fitted to
measured peaks on MS MARCO (0.2 / 1.2 / 8.8 / 33.5 GiB at batch 1 / 8 / 64 / 250).
"""

from __future__ import annotations

import torch

# The query-by-centroid table is built once, in the layout the kernel reads.
QCT_COPIES = 1

# One byte of candidate bitmap per (query, document).
BITMAP_BYTES_PER_DOC = 1

# Per candidate: nonzero() pair, id, approximate score, top-k workspaces (~75 B).
BYTES_PER_CANDIDATE = 96

# Share of free VRAM transients may take; the rest absorbs fragmentation.
BUDGET_FRACTION = 0.6

# Batch-independent floor: allocator granularity and workspaces that do not scale.
FIXED_BYTES = 128 * 2**20

# Per-launch cap: scratch never exceeds the index footprint, floored for small indexes.
TRANSIENT_FLOOR_BYTES = 384 * 2**20
TRANSIENT_RESIDENT_RATIO = 1.0


def estimate_candidates(
    ivf_lengths: torch.Tensor,
    *,
    n_probe: int,
    max_query_tokens: int,
    n_docs: int,
) -> int:
    """Upper-bound the candidates a single query can produce.

    The sum of the largest ``n_probe * max_query_tokens`` posting lists. A mean
    would not do: query tokens preferentially pick dense centroids.

    Args:
    ----
    ivf_lengths:
        Posting list length per centroid.
    n_probe:
        Cells probed per query token.
    max_query_tokens:
        Padded query length.
    n_docs:
        Documents in the index, the hard cap on distinct candidates.

    """
    n_cells = min(int(ivf_lengths.numel()), int(n_probe * max_query_tokens))
    if n_cells <= 0:
        return 0
    largest = torch.topk(ivf_lengths.to(torch.float64), n_cells).values
    return int(min(float(largest.sum()), float(n_docs)))


def bytes_per_query(
    *,
    n_centroids: int,
    max_query_tokens: int,
    n_docs: int,
    candidates_per_query: int,
) -> int:
    """Device bytes of transient state a single query contributes."""
    qct = QCT_COPIES * n_centroids * max_query_tokens * 2
    bitmap = n_docs * BITMAP_BYTES_PER_DOC
    candidates = candidates_per_query * BYTES_PER_CANDIDATE
    return int(qct + bitmap + candidates)


def transient_bytes(
    *,
    batch: int,
    n_centroids: int,
    max_query_tokens: int,
    n_docs: int,
    candidates_per_query: int,
) -> int:
    """Total transient device bytes a batch of this size will allocate."""
    per_query = bytes_per_query(
        n_centroids=n_centroids,
        max_query_tokens=max_query_tokens,
        n_docs=n_docs,
        candidates_per_query=candidates_per_query,
    )
    return int(FIXED_BYTES + batch * per_query)


def usable_bytes(device: str) -> int:
    """Free device memory plus the allocator's cached-but-unused blocks."""
    free, _ = torch.cuda.mem_get_info(device)
    cached_unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
        device
    )
    return int(free + cached_unused)


def max_batch(
    *,
    n_centroids: int,
    max_query_tokens: int,
    n_docs: int,
    candidates_per_query: int,
    device: str,
    free_bytes: int | None = None,
    budget_fraction: float = BUDGET_FRACTION,
    resident_bytes: int | None = None,
) -> int:
    """Largest query batch whose transients fit the memory budget.

    Args:
    ----
    n_centroids:
        Centroid count of the index.
    max_query_tokens:
        Padded query length for this call.
    n_docs:
        Documents in the index.
    candidates_per_query:
        Bound from :func:`estimate_candidates`.
    device:
        Device to plan against.
    free_bytes:
        Free memory override. Sampled from the device when not supplied.
    budget_fraction:
        Share of free memory the transients may occupy. Callers pass the
        index's own ``search_memory_fraction``, so a deployment that lowered it
        to share the GPU is honoured here rather than overridden.
    resident_bytes:
        Device footprint of the index. When given, one launch may not use more
        scratch than max(:data:`TRANSIENT_FLOOR_BYTES`, ratio * footprint).

    """
    if free_bytes is None:
        free_bytes = usable_bytes(device)

    per_query = bytes_per_query(
        n_centroids=n_centroids,
        max_query_tokens=max_query_tokens,
        n_docs=n_docs,
        candidates_per_query=candidates_per_query,
    )
    if per_query <= 0:
        return 1

    budget = int(budget_fraction * free_bytes) - FIXED_BYTES
    if resident_bytes is not None:
        cap = max(TRANSIENT_FLOOR_BYTES, int(TRANSIENT_RESIDENT_RATIO * resident_bytes))
        budget = min(budget, cap - FIXED_BYTES)
    if budget <= 0:
        return 1
    return max(1, budget // per_query)
