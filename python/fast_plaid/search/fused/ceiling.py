"""Closed-form batch admission for the fused search path.

Fused search allocates no data-dependent scratch: every transient buffer is a
linear function of the query batch size, so the largest admissible batch can be
computed in arithmetic rather than discovered by catching out-of-memory errors.
This module owns that arithmetic.

The coefficients below are fitted to measured peaks, not derived from a single
tensor. An earlier version modelled only the query-by-centroid table and
underestimated the true transient by ~8x on a 262k-centroid index, because the
candidate-proportional buffers (the bitmap dump, the candidate matrix, the
approximate scores and the two top-k workspaces) dominate once posting lists are
long. Measured transients on MS MARCO v1 (8.84M documents, 262,144 centroids,
32 query tokens) were 0.2 / 1.2 / 8.8 / 33.5 GiB at batch 1 / 8 / 64 / 250 --
clean linearity in batch, which is what makes the closed form viable at all.
"""

from __future__ import annotations

import torch

# The query-by-centroid table is materialised by the einsum and again by the
# contiguous copy that feeds the approximate kernel.
QCT_COPIES = 2

# One byte of candidate bitmap per (query, document).
BITMAP_BYTES_PER_DOC = 1

# Per-candidate transient: the nonzero() index pair (16 B), the candidate id
# matrix (8 B), approximate scores (4 B), the selection gather and the two
# top-k workspaces. Measured ~75 B/candidate; rounded up for headroom.
BYTES_PER_CANDIDATE = 96

# Fraction of free device memory the fused path is willing to occupy with
# transients. The remainder absorbs allocator fragmentation.
BUDGET_FRACTION = 0.6

# Batch-independent floor: allocator block granularity and the workspaces that
# do not scale with batch. Measured transient at batch 1 exceeds the linear
# term alone, so a constant is carried rather than pretending the fit passes
# through the origin.
FIXED_BYTES = 128 * 2**20


def estimate_candidates(
    ivf_lengths: torch.Tensor,
    *,
    n_probe: int,
    max_query_tokens: int,
    n_docs: int,
) -> int:
    """Upper-bound the candidates a single query can produce.

    A query probes ``n_probe`` cells per query token. The largest number of
    postings those cells can hold is the sum of the largest
    ``n_probe * max_query_tokens`` posting lists, which bounds the candidate
    count before de-duplication.

    Averaging instead of bounding is not safe here: query tokens preferentially
    select dense centroids, so the cells actually probed are far longer than the
    mean cell. On MS MARCO the mean cell holds ~2.3k postings while a query's
    probed cells yielded ~1.33M candidates after de-duplication -- well above
    the ~584k a mean-based estimate predicts.

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
    """Device memory available to this process, including reusable cache.

    ``mem_get_info`` reports what the driver considers free, which excludes
    blocks the caching allocator is holding but not using. Those blocks are
    available without a round trip to the driver, so they count. Adding them
    back is exact and costs nothing, where releasing them with
    ``empty_cache()`` would synchronize the device and throw away warm blocks
    that the next allocation is about to want.
    """
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

    budget = int(BUDGET_FRACTION * free_bytes) - FIXED_BYTES
    if budget <= 0:
        return 1
    return max(1, budget // per_query)
