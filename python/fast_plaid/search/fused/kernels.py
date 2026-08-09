"""Triton kernels for fused PLAID scoring.

Two kernels replace the materialise-then-score stages of the standard pipeline:

``approx_maxsim``
    Scores every candidate against the query-by-centroid table, reading one
    contiguous row per document token.

``exact_maxsim``
    Decompresses residuals in registers and computes MaxSim in the same kernel,
    so the ``candidates x doc_tokens x dim`` embedding tensor is never
    materialised.

Both reproduce the arithmetic of the standard chain rounding point for rounding
point: Half reconstruction, Half normalise, Half GEMM output, fp32 sum. The
annotations in ``exact_maxsim`` mark each of those points.
"""

from __future__ import annotations

import triton
import triton.language as tl

# Shared-memory ceilings differ by architecture: sm_89 exposes ~100 KB per
# block against sm_90's ~227 KB, so the token tile is gated rather than assumed.
_TOKEN_TILE_BY_ARCH = {(8, 9): 64, (9, 0): 128}
_DEFAULT_TOKEN_TILE = 64


def token_tile(arch: tuple[int, int]) -> int:
    """Document-token tile width validated for this architecture."""
    return _TOKEN_TILE_BY_ARCH.get(arch, _DEFAULT_TOKEN_TILE)


def pad_pow2(value: int) -> int:
    """Round up to a power of two, as Triton block shapes require."""
    return 1 << max(0, int(value - 1)).bit_length() if value > 1 else 1


@triton.jit
def approx_maxsim(
    cand_ptr,
    ncand_ptr,
    qct_ptr,
    off_ptr,
    cid_ptr,
    dlen_ptr,
    out_ptr,
    n_cent,
    max_cand,
    MAXD: tl.constexpr,
    TB: tl.constexpr,
    MAXQ: tl.constexpr,
):
    """Approximate MaxSim from centroid identity alone.

    One program per (candidate slot, query). Reads the precomputed
    query-by-centroid scores instead of touching residuals.
    """
    slot = tl.program_id(0)
    b = tl.program_id(1)
    n_cand = tl.load(ncand_ptr + b)
    if slot < n_cand:
        doc = tl.load(cand_ptr + b.to(tl.int64) * max_cand + slot)
        offset = tl.load(off_ptr + doc)
        doc_len = tl.load(dlen_ptr + doc)
        qct_b = qct_ptr + b.to(tl.int64) * n_cent * MAXQ
        q_lanes = tl.arange(0, MAXQ)
        acc = tl.full((MAXQ,), float("-inf"), tl.float32)
        for tile in range(0, MAXD, TB):
            if tile < doc_len:
                tokens = tile + tl.arange(0, TB)
                mask = tokens < doc_len
                cid = tl.load(cid_ptr + offset + tokens, mask=mask, other=0)
                scores = tl.load(
                    qct_b + cid.to(tl.int64)[:, None] * MAXQ + q_lanes[None, :],
                    mask=mask[:, None],
                    other=float("-inf"),
                ).to(tl.float32)
                acc = tl.maximum(acc, tl.max(scores, axis=0))
        tl.store(out_ptr + b.to(tl.int64) * max_cand + slot, tl.sum(acc))


@triton.jit
def exact_maxsim(
    sel_ptr,
    ncand_ptr,
    off_ptr,
    res_ptr,
    cid_ptr,
    cent_ptr,
    bw_ptr,
    norm_ptr,
    q_ptr,
    dlen_ptr,
    out_ptr,
    n_sel,
    MAXD: tl.constexpr,
    TB: tl.constexpr,
    DIM: tl.constexpr,
    MAXQ: tl.constexpr,
    NBITS: tl.constexpr,
):
    """Exact MaxSim with residual decompression fused into the score.

    One program per (selected document, query). The reconstructed embeddings
    live in registers only.

    Queries in a batch rarely hold the same number of candidates, and the
    selection is a rectangle ``n_sel`` wide, so a query with fewer candidates
    than that carries padding in its tail. Those slots address document 0 and
    their scores are discarded, so the guard below skips them rather than
    decompressing a document whose result cannot be used. Without it a query
    holding 300 candidates against a selection depth of 1024 would spend 70%
    of this kernel on work that is thrown away.
    """
    slot = tl.program_id(0)
    b = tl.program_id(1)
    # Selection is sorted descending and padded slots score -inf, so the real
    # candidates occupy the first min(n_sel, n_cand) positions.
    n_valid = tl.load(ncand_ptr + b)
    if slot < n_valid:
        doc = tl.load(sel_ptr + b.to(tl.int64) * n_sel + slot)

        codes_per_byte: tl.constexpr = 8 // NBITS
        packed_bytes: tl.constexpr = (DIM * NBITS) // 8
        code_mask: tl.constexpr = (1 << NBITS) - 1

        offset = tl.load(off_ptr + doc)
        doc_len = tl.load(dlen_ptr + doc)

        lanes = tl.arange(0, 128)
        dim_mask = lanes < DIM
        q_lanes = tl.arange(0, MAXQ)
        query = tl.load(
            q_ptr
            + b.to(tl.int64) * MAXQ * DIM
            + q_lanes[:, None] * DIM
            + lanes[None, :],
            mask=dim_mask[None, :],
            other=0.0,
        )

        acc = tl.full((MAXQ,), float("-inf"), tl.float32)
        for tile in range(0, MAXD, TB):
            if tile < doc_len:
                tokens = tile + tl.arange(0, TB)
                mask = tokens < doc_len
                global_tok = offset + tokens

                byte_lane = (lanes // codes_per_byte)[None, :]
                packed = tl.load(
                    res_ptr + global_tok[:, None] * packed_bytes + byte_lane,
                    mask=mask[:, None] & dim_mask[None, :],
                    other=0,
                ).to(tl.int32)
                shift = (codes_per_byte - 1 - (lanes % codes_per_byte)) * NBITS
                code = (packed >> shift[None, :]) & code_mask
                residual = tl.where(
                    dim_mask[None, :], tl.load(bw_ptr + code).to(tl.float32), 0.0
                )

                cid = tl.load(cid_ptr + global_tok, mask=mask, other=0)
                centroid = tl.load(
                    cent_ptr + cid.to(tl.int64)[:, None] * DIM + lanes[None, :],
                    mask=mask[:, None] & dim_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                # Reconstruction rounds to Half, matching the standard chain.
                emb = (centroid + residual).to(tl.float16)
                # Dividing by a precomputed Half norm reproduces the ATen Half
                # divide, which rounds a single fp32 quotient (CUDA has no fp16
                # divide instruction).
                norm = tl.load(norm_ptr + global_tok, mask=mask, other=1.0).to(
                    tl.float32
                )
                emb = (emb.to(tl.float32) / norm[:, None]).to(tl.float16)

                scores = tl.dot(emb, tl.trans(query))
                # Rounding the fp32 accumulator to Half reproduces the Half
                # output of the standard HGEMM.
                scores = scores.to(tl.float16).to(tl.float32)
                # Masking with -inf mirrors the padded-token masking of the
                # standard reducer; the maximum is deliberately left unclamped.
                scores = tl.where(mask[:, None], scores, float("-inf"))
                acc = tl.maximum(acc, tl.max(scores, axis=0))

        tl.store(out_ptr + b.to(tl.int64) * n_sel + slot, tl.sum(acc))
