"""Tests for asymmetric int8-query scoring over stored residual codes."""

import gc
import math
import os
import shutil

import pytest
import torch
from fast_plaid import search


@pytest.fixture
def test_index_path(tmp_path):
    """Create a temporary index path for testing."""
    index_path = str(tmp_path / "asym_index")
    os.makedirs(index_path, exist_ok=True)
    yield index_path
    gc.collect()
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


def build_index(index_path, *, nbits=4, dim=128, n_docs=64, seed=0):
    """Build a small CPU index with variable document lengths."""
    torch.manual_seed(seed)
    documents = [
        torch.randn(120 + 7 * (i % 11), dim, device="cpu") for i in range(n_docs)
    ]
    index = search.FastPlaid(index=index_path, device="cpu")
    index.create(documents_embeddings=documents, kmeans_niters=4, nbits=nbits)
    return index


def queries(n=8, dim=128, seed=1):
    """Make queries of varying token counts, exercising the padded rows."""
    torch.manual_seed(seed)
    return [torch.randn(4 + 3 * (i % 5), dim, device="cpu") for i in range(n)]


class TestAsymEquivalence:
    """The asymmetric path answers the same questions as the float path."""

    def test_returns_the_same_shape_of_answer(self, test_index_path):
        """Every query gets a full result list, exactly as the float path does."""
        index = build_index(test_index_path)
        qs = queries()

        results = index.search(qs, top_k=5, residual_asym=True, n_processes=1)

        assert len(results) == len(qs)
        assert all(len(result) == 5 for result in results)

    def test_scores_track_the_float_path(self, test_index_path):
        """Scores are quantized, not equal; they must still track closely.

        MaxSim sums one maximum per query token, so the tolerance is stated
        per query token rather than per score -- a blanket bound would be
        vacuous for long queries and unreachable for short ones. It is on the
        score rather than the ranking, because an ordering assertion would
        turn on ties between documents the two paths score within an ulp of
        each other, which says nothing about the kernels.
        """
        index = build_index(test_index_path)
        qs = queries()

        floats = index.search(qs, top_k=10, n_processes=1)
        asyms = index.search(qs, top_k=10, residual_asym=True, n_processes=1)

        float_scores = {
            (q, doc): score for q, result in enumerate(floats) for doc, score in result
        }
        compared = 0
        for q, result in enumerate(asyms):
            tolerance = 0.05 * qs[q].shape[0]
            for doc, score in result:
                if (q, doc) in float_scores:
                    assert score == pytest.approx(float_scores[(q, doc)], abs=tolerance)
                    compared += 1
        assert compared > 0, "no shared documents to compare"

    @pytest.mark.parametrize("n_full_scores", [8, 64, 4096])
    def test_pruning_depth_does_not_change_agreement(
        self, test_index_path, n_full_scores
    ):
        """Candidate generation is exact, so pruning harder cannot diverge.

        Only the reranking scores are quantized. If the approximate stage had
        drifted, a shallow `n_full_scores` would prune a different candidate
        set and the two paths would disagree about documents, not just about
        the last decimal of a score -- so the shallow settings are the ones
        that would catch it.
        """
        index = build_index(test_index_path)
        qs = queries()

        floats = index.search(qs, top_k=5, n_full_scores=n_full_scores, n_processes=1)
        asyms = index.search(
            qs, top_k=5, n_full_scores=n_full_scores, residual_asym=True, n_processes=1
        )

        overlap = sum(
            len({doc for doc, _ in a} & {doc for doc, _ in f})
            for f, a in zip(floats, asyms)
        )
        assert overlap / sum(len(result) for result in floats) >= 0.8

    def test_top_results_mostly_agree(self, test_index_path):
        """The two paths retrieve substantially the same documents."""
        index = build_index(test_index_path)
        qs = queries()

        floats = index.search(qs, top_k=10, n_processes=1)
        asyms = index.search(qs, top_k=10, residual_asym=True, n_processes=1)

        overlap = sum(
            len({doc for doc, _ in a} & {doc for doc, _ in f})
            for f, a in zip(floats, asyms)
        )
        total = sum(len(result) for result in floats)
        assert overlap / total >= 0.8

    @pytest.mark.parametrize("nbits", [1, 2, 4])
    def test_every_supported_code_width(self, test_index_path, nbits):
        """Each width the packing supports produces usable results."""
        index = build_index(test_index_path, nbits=nbits)

        results = index.search(queries(n=4), top_k=5, residual_asym=True, n_processes=1)

        assert all(len(result) == 5 for result in results)
        assert all(math.isfinite(score) for result in results for _, score in result)


class TestAsymDefaults:
    """The flag is opt-in, reachable, and never changes the default path."""

    def test_default_is_the_float_path(self, test_index_path):
        """Omitting the flag reproduces the float path exactly."""
        index = build_index(test_index_path)
        qs = queries()

        default = index.search(qs, top_k=10, n_processes=1)
        explicit = index.search(qs, top_k=10, residual_asym=False, n_processes=1)

        assert default == explicit

    def test_flag_reaches_the_kernels(self, test_index_path):
        """A flag that cannot change the answer is a flag that never ran.

        Asymmetric scoring quantizes the query, so at least one score must
        differ from the float path's. Without this the feature could ship
        silently inert -- accepted, forwarded, and ignored.
        """
        index = build_index(test_index_path)
        qs = queries()

        floats = index.search(qs, top_k=10, n_processes=1)
        asyms = index.search(qs, top_k=10, residual_asym=True, n_processes=1)

        assert floats != asyms

    def test_threaded_search_agrees_with_sequential(self, test_index_path):
        """Candidate scoring is parallel; the answer must not depend on that."""
        index = build_index(test_index_path)
        qs = queries()

        sequential = index.search(qs, top_k=10, residual_asym=True, n_processes=1)
        threaded = index.search(qs, top_k=10, residual_asym=True, n_processes=4)

        assert sequential == threaded
