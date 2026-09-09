"""Fused-search engine over the loaded index's device tensors.

Reads the standard index's codes, residuals, centroids and IVF lists in place,
precomputes one fp16 reconstruction norm per token, and answers query batches
entirely in kernels. No index format change, no re-indexing, no second copy.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

import torch

from . import ceiling
from .errors import FusedCompilationError, FusedOutOfMemoryError
from .gate import NORM_CHUNK
from .kernels import approx_maxsim, exact_maxsim, pad_pow2, token_tile

# The standard pipeline exact-scores the top quarter of the approximate ranking.
_EXACT_FRACTION = 4

# Smallest block dimension tl.dot accepts.
_MIN_DOT_DIM = 16


def _compile_error_types() -> tuple[type[BaseException], ...]:
    """Triton's own failures, whose module paths move between releases.

    Resolved by name rather than imported directly so that a Triton version
    which has relocated one of them narrows what is caught instead of making
    this module unimportable.
    """
    found: list[type[BaseException]] = []
    for module_name, class_name in (
        ("triton.compiler.errors", "CompilationError"),
        ("triton.runtime.errors", "OutOfResources"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - depends on the Triton release
            continue
        found_type = getattr(module, class_name, None)
        if isinstance(found_type, type) and issubclass(found_type, BaseException):
            found.append(found_type)
    return tuple(found)


# Deterministic failures retire the engine; OSError covers a missing compiler or cache.
_COMPILE_ERRORS = (*_compile_error_types(), OSError)


class FusedEngine:
    """Fused Triton kernels over the standard index's device tensors."""

    # Engine attribute -> key of ``data`` it reads in place.
    _BORROWED: ClassVar[dict[str, str]] = {
        "codes": "doc_codes",
        "residuals": "doc_residuals",
        "centroids": "centroids",
        "ivf": "ivf",
    }

    def __init__(
        self,
        data: dict[str, Any],
        device: str,
        *,
        budget_fraction: float = ceiling.BUDGET_FRACTION,
    ) -> None:
        """Wrap the loaded index's tensors and precompute the norms.

        Args:
        ----
        data:
            The tensors the standard index was built from. Codes, residuals,
            centroids and IVF lists must already be on ``device``; they are
            read in place, codes at their stored int64 width.
        device:
            Target CUDA device.
        budget_fraction:
            Share of free memory per-search transients may occupy, from the
            index's ``search_memory_fraction``.

        """
        self.device = device
        self.budget_fraction = budget_fraction
        self.nbits = int(data["nbits"])
        self.dim = int(data["centroids"].shape[1])
        self.arch = torch.cuda.get_device_capability(device)

        target = torch.device(device)
        for key in self._BORROWED.values():
            if data[key].device != target:
                error = (
                    f"FusedEngine reads {key} in place from the standard index; "
                    f"it is on {data[key].device}, not {device}"
                )
                raise ValueError(error)

        doc_lengths = data["doc_lengths"].to(torch.int64).reshape(-1)
        self.n_docs = int(doc_lengths.numel())
        self.n_tokens = int(doc_lengths.sum())
        self.max_doc_len = pad_pow2(int(doc_lengths.max()))

        lengths = doc_lengths.to(device)
        self.offsets = torch.zeros(self.n_docs, dtype=torch.int64, device=device)
        self.offsets[1:] = lengths.cumsum(0)[:-1]
        self.doc_lengths = lengths.to(torch.int32).contiguous()

        # Slices of a contiguous tensor are views: nothing here allocates.
        packed_bytes = (self.dim * self.nbits) // 8
        self.codes = data["doc_codes"].reshape(-1)[: self.n_tokens]
        self.residuals = data["doc_residuals"].reshape(-1, packed_bytes)[
            : self.n_tokens
        ]
        self.centroids = data["centroids"]
        self.n_centroids = int(self.centroids.shape[0])
        self.ivf = data["ivf"].reshape(-1)

        # Weights are stored in code order but addressed bit-reversed by the nibbles.
        weights = data["bucket_weights"].to(torch.float32).reshape(-1)
        permutation = torch.tensor(
            [
                int(format(i, f"0{self.nbits}b")[::-1], 2)
                for i in range(1 << self.nbits)
            ],
            dtype=torch.int64,
        )
        self.bucket_weights = (
            weights[permutation].to(torch.float16).to(device).contiguous()
        )

        self.ivf_lengths = data["ivf_lengths"].to(torch.int64).reshape(-1).to(device)
        self.ivf_offsets = torch.zeros(
            self.ivf_lengths.numel() + 1, dtype=torch.int64, device=device
        )
        self.ivf_offsets[1:] = self.ivf_lengths.cumsum(0)

        self.norms = self._precompute_norms()
        self.token_tile = token_tile(self.arch)
        # The ceiling caps each launch's scratch at this footprint.
        self._footprint_bytes = self.resident_bytes() + self.shared_bytes()

        # Depends only on the index and the padded query length, so cached per shape.
        self._candidate_bound: dict[tuple[int, int], int] = {}

    def _precompute_norms(self) -> torch.Tensor:
        """Per-token Half reconstruction norms, the standard chain's divisor."""
        norms = torch.empty(self.n_tokens, dtype=torch.float16, device=self.device)

        # Codes are MSB-first in a byte; the permuted weights absorb the bit reversal.
        codes_per_byte = 8 // self.nbits
        code_mask = (1 << self.nbits) - 1
        shifts = [(codes_per_byte - 1 - i) * self.nbits for i in range(codes_per_byte)]

        for start in range(0, self.n_tokens, NORM_CHUNK):
            stop = min(self.n_tokens, start + NORM_CHUNK)
            packed = self.residuals[start:stop]
            parts = [
                ((packed >> shift) & code_mask).to(torch.int64) for shift in shifts
            ]
            codes = torch.stack(parts, dim=2).reshape(stop - start, self.dim)
            centroids = self.centroids[self.codes[start:stop].to(torch.int64)]
            embeddings = centroids + self.bucket_weights[codes]
            norms[start:stop] = embeddings.norm(p=2, dim=-1)
            del packed, parts, codes, centroids, embeddings
        return norms

    def _device_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "codes": self.codes,
            "residuals": self.residuals,
            "norms": self.norms,
            "centroids": self.centroids,
            "ivf": self.ivf,
            "ivf_lengths": self.ivf_lengths,
            "ivf_offsets": self.ivf_offsets,
            "offsets": self.offsets,
            "doc_lengths": self.doc_lengths,
        }

    def resident_bytes(self) -> int:
        """Device bytes this engine allocated itself."""
        return sum(
            t.numel() * t.element_size()
            for name, t in self._device_tensors().items()
            if name not in self._BORROWED
        )

    def shared_bytes(self) -> int:
        """Device bytes read in place from the standard index's tensors."""
        return sum(
            t.numel() * t.element_size()
            for name, t in self._device_tensors().items()
            if name in self._BORROWED
        )

    def _pad_queries(
        self,
        packed_queries: torch.Tensor,
        query_lengths: list[int],
        *,
        max_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack per-query tokens into a zero-padded ``[B, max_tokens, dim]`` tensor.

        Padded rows contribute nothing to MaxSim; the true lengths are returned
        so the IVF probe can ignore them. ``max_tokens`` is fixed for the whole
        call so every chunk compiles to the same kernel shape.
        """
        batch = len(query_lengths)

        # One transfer and one scatter for the whole chunk, not one copy per query.
        lengths = torch.tensor(query_lengths, dtype=torch.int64, device=self.device)
        starts = torch.zeros(batch, dtype=torch.int64, device=self.device)
        starts[1:] = lengths.cumsum(0)[:-1]

        flat = packed_queries.to(device=self.device, dtype=torch.float16)
        rows = torch.repeat_interleave(torch.arange(batch, device=self.device), lengths)
        cols = torch.arange(flat.shape[0], device=self.device) - starts[rows]

        out = torch.zeros(
            batch, max_tokens, self.dim, dtype=torch.float16, device=self.device
        )
        out[rows, cols] = flat
        return out, lengths

    def _candidates(
        self, queries: torch.Tensor, lengths: torch.Tensor, n_probe: int
    ) -> tuple[torch.Tensor, ...]:
        """Probe the IVF lists and build a dense candidate matrix."""
        batch, max_q, _ = queries.shape
        # Built directly in the layout the kernel reads, so only one copy is ever live.
        qct = torch.bmm(
            self.centroids.unsqueeze(0).expand(batch, -1, -1),
            queries.transpose(1, 2),
        )
        cells = qct.topk(n_probe, dim=1).indices.permute(0, 2, 1).reshape(batch, -1)

        flat_cells = cells.reshape(-1)
        segment_lengths = self.ivf_lengths[flat_cells]

        # Padded query rows tie every centroid; give them empty posting lists.
        row_of_cell = torch.arange(max_q, device=self.device).repeat_interleave(n_probe)
        keep = (row_of_cell[None, :] < lengths[:, None]).reshape(-1)
        segment_lengths = torch.where(
            keep, segment_lengths, torch.zeros_like(segment_lengths)
        )
        segment_starts = self.ivf_offsets[flat_cells]
        total = int(segment_lengths.sum())

        positions = torch.arange(total, device=self.device)
        segment_id = torch.repeat_interleave(
            torch.arange(flat_cells.numel(), device=self.device), segment_lengths
        )
        cumulative = torch.zeros(
            flat_cells.numel(), dtype=torch.int64, device=self.device
        )
        cumulative[1:] = segment_lengths.cumsum(0)[:-1]
        docs = self.ivf[
            segment_starts[segment_id] + (positions - cumulative[segment_id])
        ].to(torch.int64)

        bitmap = torch.zeros(batch, self.n_docs, dtype=torch.bool, device=self.device)
        bitmap[segment_id // cells.shape[1], docs] = True
        n_cand = bitmap.sum(1).to(torch.int32)
        # One transfer for both bounds keeps the batch-1 path to a single sync.
        bounds = torch.stack((n_cand.min(), n_cand.max())).cpu()
        min_cand, max_cand = int(bounds[0]), int(bounds[1])

        nonzero = bitmap.nonzero(as_tuple=False)
        row_offsets = torch.zeros(batch + 1, dtype=torch.int64, device=self.device)
        row_offsets[1:] = n_cand.to(torch.int64).cumsum(0)
        slots = (
            torch.arange(nonzero.shape[0], device=self.device)
            - row_offsets[nonzero[:, 0]]
        )
        cand = torch.zeros(batch, max_cand, dtype=torch.int64, device=self.device)
        cand[nonzero[:, 0], slots] = nonzero[:, 1]
        return cand, n_cand, min_cand, max_cand, qct

    def _search_batch(
        self,
        queries: torch.Tensor,
        lengths: torch.Tensor,
        *,
        top_k: int,
        n_full_scores: int,
        n_probe: int,
    ) -> list[list[tuple[int, float]]]:
        """Score one admissible batch of padded queries."""
        batch, max_q, _ = queries.shape
        cand, n_cand, min_cand, max_cand, qct = self._candidates(
            queries, lengths, n_probe
        )

        approx = torch.full((batch, max_cand), float("-inf"), device=self.device)
        approx_maxsim[(max_cand, batch)](
            cand,
            n_cand,
            qct,
            self.offsets,
            self.codes,
            self.doc_lengths,
            approx,
            self.n_centroids,
            max_cand,
            MAXD=self.max_doc_len,
            TB=self.token_tile,
            MAXQ=max_q,
        )

        n_sel = min(max(1, n_full_scores // _EXACT_FRACTION), max_cand)
        approx_top, approx_slots = approx.topk(n_sel, dim=1)
        selected = torch.gather(cand, 1, approx_slots).contiguous()

        # Padding slots read as document 0 and score -inf; mark them to drop them.
        filled = torch.isfinite(approx_top)

        # The kernel skips padded slots, so -inf is their final value.
        exact = torch.full((batch, n_sel), float("-inf"), device=self.device)
        exact_maxsim[(n_sel, batch)](
            selected,
            n_cand,
            self.offsets,
            self.residuals,
            self.codes,
            self.centroids,
            self.bucket_weights,
            self.norms,
            queries.contiguous(),
            self.doc_lengths,
            exact,
            n_sel,
            MAXD=self.max_doc_len,
            TB=self.token_tile,
            DIM=self.dim,
            MAXQ=max_q,
            NBITS=self.nbits,
        )

        k = min(top_k, n_sel)
        scores, order = exact.topk(k, dim=1)
        ids = torch.gather(selected, 1, order)
        ids_cpu = ids.cpu().tolist()
        scores_cpu = scores.to(torch.float32).cpu().tolist()
        if min_cand >= n_sel:
            # No query in this batch is short enough to have selected padding.
            return [
                list(zip(row_ids, row_scores))
                for row_ids, row_scores in zip(ids_cpu, scores_cpu)
            ]

        # Fewer candidates than top_k means fewer results, as in the standard pipeline.
        keep_cpu = torch.gather(filled, 1, order).cpu().tolist()
        return [
            [
                (doc, score)
                for doc, score, ok in zip(row_ids, row_scores, row_keep)
                if ok
            ]
            for row_ids, row_scores, row_keep in zip(ids_cpu, scores_cpu, keep_cpu)
        ]

    def search(
        self,
        packed_queries: torch.Tensor,
        query_lengths: list[int],
        *,
        top_k: int,
        n_full_scores: int,
        n_probe: int,
        max_batch: int | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Answer a batch of queries, splitting it to fit the memory budget.

        Args:
        ----
        packed_queries:
            Query tokens packed as ``(total_query_tokens, dim)``.
        query_lengths:
            True token count per query.
        top_k:
            Results to return per query.
        n_full_scores:
            Candidates kept after approximate ranking; a quarter of them are
            exact-scored, matching the standard pipeline.
        n_probe:
            IVF cells probed per query token.
        max_batch:
            Upper bound on the queries scored per launch, from the caller's
            ``batch_size``. The admission ceiling may still choose less.

        """
        if not query_lengths:
            # An empty request returns an empty list, as in the standard pipeline.
            return []

        # Decided from host-side lengths so admission happens before any allocation.
        max_q = max(_MIN_DOT_DIM, pad_pow2(max(query_lengths)))

        key = (n_probe, max_q)
        if key not in self._candidate_bound:
            self._candidate_bound[key] = ceiling.estimate_candidates(
                self.ivf_lengths,
                n_probe=n_probe,
                max_query_tokens=max_q,
                n_docs=self.n_docs,
            )
        candidates_per_query = self._candidate_bound[key]
        chunk = ceiling.max_batch(
            n_centroids=self.n_centroids,
            max_query_tokens=max_q,
            n_docs=self.n_docs,
            candidates_per_query=candidates_per_query,
            device=self.device,
            budget_fraction=self.budget_fraction,
            resident_bytes=self._footprint_bytes,
        )
        if max_batch is not None:
            chunk = max(1, min(chunk, max_batch))

        # Per-query token boundaries so each chunk transfers only its own slice.
        token_offsets = [0]
        for length in query_lengths:
            token_offsets.append(token_offsets[-1] + length)

        # The ceiling is a fitted model: on OOM, halve the chunk and retry the range.
        results: list[list[tuple[int, float]]] = []
        start = 0
        n_queries = len(query_lengths)
        while start < n_queries:
            stop = min(n_queries, start + chunk)
            try:
                queries, lengths = self._pad_queries(
                    packed_queries[token_offsets[start] : token_offsets[stop]],
                    query_lengths[start:stop],
                    max_tokens=max_q,
                )
                results.extend(
                    self._search_batch(
                        queries,
                        lengths,
                        top_k=top_k,
                        n_full_scores=n_full_scores,
                        n_probe=n_probe,
                    )
                )
            except torch.cuda.OutOfMemoryError as error:
                if chunk == 1:
                    raise FusedOutOfMemoryError(
                        "fused search ran out of memory at batch 1"
                    ) from error
                queries = lengths = None
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                continue
            except _COMPILE_ERRORS as error:
                # Recurs on every launch, so the caller retires the engine.
                raise FusedCompilationError(
                    f"fused kernels could not be compiled or launched: {error!r}"
                ) from error
            start = stop
        return results
