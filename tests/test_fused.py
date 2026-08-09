"""Tests for the fused CUDA search path.

The parity test is the important one: it asserts that fused search returns the
same documents and the same scores as the standard pipeline on the same index.
It skips itself when no supported GPU is present, so the CPU CI still exercises
the gate and the batch-ceiling arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fast_plaid.search import FastPlaid
from fast_plaid.search.fused import ceiling, gate

GIB = 2**30


def _cuda_ready() -> bool:
    """Whether this machine can run the fused kernels at all."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability("cuda:0") not in gate.TESTED_ARCHS:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


requires_fused = pytest.mark.skipif(
    not _cuda_ready(), reason="needs a CUDA device with a validated architecture"
)


@pytest.mark.parametrize(
    ("dim", "nbits", "expected"),
    [
        (96, 4, 4 + 48 + 2),
        (128, 4, 4 + 64 + 2),
        (96, 2, 4 + 24 + 2),
        (96, 1, 4 + 12 + 2),
    ],
)
def test_bytes_per_token_matches_layout(dim: int, nbits: int, expected: int) -> None:
    """Per-token residency is codes plus packed residuals plus the norm."""
    assert gate.bytes_per_token(dim=dim, nbits=nbits) == expected


def assert_same_ranking(actual, expected) -> None:
    """Assert the two engines agree, allowing only tie-order differences.

    Bit-equivalence is a claim about *scores*, not about how equal scores are
    ordered against each other. Asserting exact id order would test the
    tie-breaking of two different top-k implementations, which is not a
    property either engine promises.
    """
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        got_scores = [score for _, score in got]
        want_scores = [score for _, score in want]
        assert got_scores == pytest.approx(want_scores, abs=1e-3), (
            f"scores diverge, not a tie: {got_scores} vs {want_scores}"
        )
        for (got_id, got_score), (want_id, want_score) in zip(got, want):
            if got_id != want_id:
                assert got_score == pytest.approx(want_score, abs=1e-3), (
                    f"doc {got_id} replaced {want_id} at unequal scores "
                    f"({got_score} vs {want_score})"
                )


def test_gate_declines_cpu() -> None:
    """A CPU device is never eligible."""
    data = {
        "nbits": 4,
        "centroids": torch.zeros(8, 96, dtype=torch.float16),
        "ivf": torch.zeros(8, dtype=torch.int64),
        "ivf_lengths": torch.ones(8, dtype=torch.int32),
    }
    assert gate.check(data=data, device="cpu", n_tokens=64, free_bytes=GIB) is not None


@requires_fused
def test_gate_declines_unsupported_nbits() -> None:
    """Only 4-bit residuals are packed the way the kernels decode."""
    # 3 does not divide a byte evenly, so the unpack schedule does not exist.
    data = {
        "nbits": 3,
        "centroids": torch.zeros(8, 96, dtype=torch.float16),
        "ivf": torch.zeros(8, dtype=torch.int64),
        "ivf_lengths": torch.ones(8, dtype=torch.int32),
    }
    reason = gate.check(data=data, device="cuda:0", n_tokens=64, free_bytes=GIB)
    assert reason is not None
    assert "nbits" in reason


def test_gate_respects_the_kill_switch(monkeypatch) -> None:
    """The disable switch short-circuits every other check."""
    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    reason = gate.check(data={}, device="cuda:0", n_tokens=1, free_bytes=GIB)
    assert reason is not None
    assert gate.DISABLE_ENV in reason


def test_estimate_candidates_bounds_the_probed_cells() -> None:
    """The estimate bounds, rather than averages, the probed posting lists.

    Query tokens select dense centroids, so a mean-based estimate is not an
    upper bound. Here one cell holds almost every posting; probing two cells
    must account for it.
    """
    ivf_lengths = torch.tensor([1000, 10, 10, 10], dtype=torch.int64)
    estimate = ceiling.estimate_candidates(
        ivf_lengths, n_probe=1, max_query_tokens=2, n_docs=100_000
    )
    assert estimate == 1010

    mean_based = int(ivf_lengths.to(torch.float64).mean() * 2)
    assert estimate > mean_based


def test_estimate_candidates_caps_at_corpus_size() -> None:
    """Candidates can never exceed the number of documents."""
    ivf_lengths = torch.full((64,), 10_000, dtype=torch.int64)
    estimate = ceiling.estimate_candidates(
        ivf_lengths, n_probe=8, max_query_tokens=32, n_docs=5_000
    )
    assert estimate == 5_000


def test_ceiling_covers_measured_msmarco_transients() -> None:
    """The model must not underestimate transients measured on MS MARCO.

    Measured peaks on an 8.84M-document index with 262,144 centroids and 32
    query tokens: 0.2 / 1.2 / 8.8 / 33.5 GiB at batch 1 / 8 / 64 / 250. An
    earlier model counted only the query-by-centroid table and predicted 13.8
    GiB where 33.5 was consumed, which is the regression guarded here.
    """
    measured = {1: 0.2, 8: 1.2, 64: 8.8, 250: 33.5}
    for batch, observed_gib in measured.items():
        predicted_gib = (
            ceiling.transient_bytes(
                batch=batch,
                n_centroids=262_144,
                max_query_tokens=32,
                n_docs=8_841_823,
                candidates_per_query=1_331_302,
            )
            / GIB
        )
        assert predicted_gib >= observed_gib, (
            f"batch {batch}: predicted {predicted_gib:.1f} GiB "
            f"underestimates the measured {observed_gib} GiB"
        )


def test_ceiling_stays_within_an_order_of_magnitude() -> None:
    """Being conservative is required; being useless is not.

    Throughput is flat above batch 8, so over-refusing costs little, but the
    bound should still admit realistic batches.
    """
    predicted_at_250 = (
        ceiling.transient_bytes(
            batch=250,
            n_centroids=262_144,
            max_query_tokens=32,
            n_docs=8_841_823,
            candidates_per_query=1_331_302,
        )
        / GIB
    )
    assert 33.5 <= predicted_at_250 < 2 * 33.5


def test_max_batch_scales_with_budget() -> None:
    """A larger memory budget admits a proportionally larger batch."""
    kwargs = {
        "n_centroids": 4_096,
        "max_query_tokens": 32,
        "n_docs": 100_000,
        "candidates_per_query": 10_000,
        "device": "cuda:0",
    }
    small = ceiling.max_batch(free_bytes=GIB, **kwargs)
    large = ceiling.max_batch(free_bytes=8 * GIB, **kwargs)
    assert large > small
    assert small >= 1


def test_max_batch_never_returns_zero() -> None:
    """A batch of one is always attempted, even under a hopeless budget."""
    assert (
        ceiling.max_batch(
            n_centroids=262_144,
            max_query_tokens=32,
            n_docs=8_841_823,
            candidates_per_query=1_331_302,
            device="cuda:0",
            free_bytes=1024,
        )
        == 1
    )


@requires_fused
@pytest.mark.parametrize("nbits", [4, 2, 1])
def test_fused_matches_standard_pipeline(tmp_path, monkeypatch, nbits: int) -> None:
    """Fused search returns the standard pipeline's documents and scores.

    Both arms go through the public API on one index; the kill switch selects
    which path serves the call, so nothing private is reached into. Run at every
    supported residual width, since the packed-code unpack schedule and the
    bucket-weight permutation both depend on nbits.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_docs, dim = 2_000, 96

    documents = [
        torch.nn.functional.normalize(
            torch.randn(int(rng.integers(24, 96)), dim), p=2, dim=-1
        )
        for _ in range(n_docs)
    ]
    queries = torch.nn.functional.normalize(torch.randn(16, 32, dim), p=2, dim=-1)

    index_path = str(tmp_path / f"index-nbits{nbits}")
    engine = FastPlaid(index=index_path, device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=nbits)

    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    engine._invalidate_fused()
    expected = engine.search(
        queries_embeddings=queries, top_k=10, n_full_scores=4096, show_progress=False
    )

    monkeypatch.delenv(gate.DISABLE_ENV)
    engine._invalidate_fused()
    assert engine._maybe_fused() is not None, "fused path should be eligible here"
    actual = engine.search(
        queries_embeddings=queries, top_k=10, n_full_scores=4096, show_progress=False
    )

    assert_same_ranking(actual, expected)


@requires_fused
def test_fused_starves_without_inventing_documents(tmp_path, monkeypatch) -> None:
    """A query with fewer candidates than ``top_k`` returns fewer results.

    Candidates are held in a rectangle as wide as the batch's largest set, and
    the unused tail is zero-filled -- which reads as document 0. A short query
    probing one cell alongside a long query probing many is the case where that
    tail gets selected, and returning document 0 there would be inventing a
    result whose IVF cells were never probed.
    """
    torch.manual_seed(0)
    dim = 96
    documents = [
        torch.nn.functional.normalize(torch.randn(8, dim), p=2, dim=-1)
        for _ in range(256)
    ]

    # One long query probes many cells, one single-token query probes one.
    queries = [
        torch.nn.functional.normalize(torch.randn(32, dim), p=2, dim=-1),
        torch.nn.functional.normalize(torch.randn(1, dim), p=2, dim=-1),
    ]

    index_path = str(tmp_path / "index")
    engine = FastPlaid(index=index_path, device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=4)

    kwargs = {"top_k": 200, "n_ivf_probe": 1, "show_progress": False}

    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    engine._invalidate_fused()
    expected = engine.search(queries_embeddings=queries, **kwargs)

    monkeypatch.delenv(gate.DISABLE_ENV)
    engine._invalidate_fused()
    assert engine._maybe_fused() is not None, "fused path should be eligible here"
    actual = engine.search(queries_embeddings=queries, **kwargs)

    assert any(len(row) < kwargs["top_k"] for row in expected), (
        "setup did not starve any query, so this test would pass vacuously"
    )
    assert [len(row) for row in actual] == [len(row) for row in expected]
    assert_same_ranking(actual, expected)


@requires_fused
def test_fused_serves_documents_added_by_update(tmp_path) -> None:
    """A staged copy must not outlive the index it was built from.

    Asserting on the invalidation helper would only prove the helper works.
    The property that matters is that a search issued after ``update`` sees
    the documents that update added, which a stale copy cannot do: their ids
    did not exist when it was staged.
    """
    torch.manual_seed(0)
    dim = 96
    original = [
        torch.nn.functional.normalize(torch.randn(32, dim), p=2, dim=-1)
        for _ in range(512)
    ]
    added = [
        torch.nn.functional.normalize(torch.randn(32, dim), p=2, dim=-1)
        for _ in range(64)
    ]

    index_path = str(tmp_path / "index")
    engine = FastPlaid(index=index_path, device="cuda:0")
    engine.create(documents_embeddings=original, nbits=4)

    # A document that is about to be added, used as its own query.
    probe = [added[0]]

    engine.search(queries_embeddings=probe, top_k=5, show_progress=False)
    assert engine.fused_status()["active"] is True

    engine.update(documents_embeddings=added)

    results = engine.search(queries_embeddings=probe, top_k=5, show_progress=False)
    assert engine.fused_status()["active"] is True
    assert results[0], "fused search returned nothing after update"
    assert results[0][0][0] >= len(original), (
        "fused search did not return the added document, so it served the "
        "index generation that preceded update()"
    )


def test_max_batch_honours_the_search_memory_fraction() -> None:
    """A caller sharing the GPU is not overridden by the module default."""
    shared = {
        "n_centroids": 262_144,
        "max_query_tokens": 32,
        "n_docs": 8_840_000,
        "candidates_per_query": 1_330_000,
        "device": "cpu",
        "free_bytes": 60 * GIB,
    }
    default = ceiling.max_batch(**shared)
    restricted = ceiling.max_batch(**shared, budget_fraction=0.1)

    assert restricted < default
    # Linear in the fraction once the fixed floor is subtracted.
    assert restricted == pytest.approx(
        default * 0.1 / ceiling.BUDGET_FRACTION, rel=0.05
    )


@requires_fused
def test_gate_honours_its_memory_fraction() -> None:
    """Residency is judged against the fraction the caller passes."""
    data = {
        "nbits": 4,
        "centroids": torch.zeros(1024, 96, dtype=torch.float16),
        "ivf": torch.zeros(4096, dtype=torch.int32),
        "ivf_lengths": torch.zeros(1024, dtype=torch.int64),
        "doc_lengths": torch.zeros(1, dtype=torch.int64),
    }
    # Sized to sit between the two fractions: fits under 0.8, not under 0.2.
    n_tokens = int(0.5 * GIB / gate.bytes_per_token(dim=96, nbits=4))
    shared = {"data": data, "device": "cuda:0", "n_tokens": n_tokens}

    assert gate.check(**shared, free_bytes=GIB, memory_fraction=0.8) is None
    restricted = gate.check(**shared, free_bytes=GIB, memory_fraction=0.2)
    assert restricted is not None
    assert "capped at 0.2" in restricted


@requires_fused
def test_fused_falls_back_when_a_single_query_cannot_fit(tmp_path, monkeypatch) -> None:
    """An unrecoverable OOM answers the call instead of failing it.

    The admission model is fitted, so it can be optimistic on an index shaped
    unlike the ones it was fitted to. Raising there would fail a call the
    standard pipeline serves happily, so the engine signals and the caller
    falls through.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(16, 96), p=2, dim=-1)
        for _ in range(512)
    ]
    queries = torch.nn.functional.normalize(torch.randn(4, 32, 96), p=2, dim=-1)

    engine = FastPlaid(index=str(tmp_path / "index"), device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=4)

    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    engine._invalidate_fused()
    expected = engine.search(queries_embeddings=queries, top_k=5, show_progress=False)

    monkeypatch.delenv(gate.DISABLE_ENV)
    engine._invalidate_fused()
    fused = engine._maybe_fused()
    assert fused is not None

    calls = {"n": 0}

    def always_oom(*_args, **_kwargs):
        calls["n"] += 1
        raise torch.cuda.OutOfMemoryError("synthetic")

    monkeypatch.setattr(fused, "_search_batch", always_oom)

    actual = engine.search(queries_embeddings=queries, top_k=5, show_progress=False)

    assert calls["n"] > 1, "the engine should have halved the batch before giving up"
    assert_same_ranking(actual, expected)
    # Transient pressure is not a reason to restage.
    assert engine._fused_engine is fused


def test_stale_staging_is_not_published(tmp_path) -> None:
    """A copy finished after the index moved is dropped, not published.

    Staging runs outside the swap lock, so an update can land between reading
    the generation and publishing the result. Needs no GPU: the generation
    handshake is plain bookkeeping.
    """
    engine = FastPlaid(index=str(tmp_path / "index"), device="cpu")
    stale_generation = engine._fused_generation

    with engine._index_swap_lock:
        engine._invalidate_fused()

    sentinel = object()
    assert engine._publish_fused(stale_generation, sentinel, None) is None
    assert engine._fused_engine is None
    # The attempt goes unrecorded, so the next search stages again rather than
    # falling back to the standard path forever.
    assert engine._fused_attempted is False

    current = engine._fused_generation
    assert engine._publish_fused(current, sentinel, None) is sentinel
    assert engine._fused_engine is sentinel
    assert engine._fused_attempted is True


@pytest.mark.parametrize("mutation", ["create", "update", "delete", "close"])
def test_index_mutations_retire_the_staged_copy(tmp_path, mutation: str) -> None:
    """Every path that replaces ``indices`` also retires the fused copy.

    ``_reload_under_lock`` is not the only writer: ``create``, ``update`` and
    ``delete`` swap the index in place and then call ``_update_mtime``, which
    is exactly what stops the reload path from noticing the change. Each has
    to invalidate on its own. Needs no GPU -- a sentinel stands in for the
    staged copy, since what is under test is the bookkeeping around the swap.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(8, 96), p=2, dim=-1)
        for _ in range(64)
    ]

    engine = FastPlaid(index=str(tmp_path / "index"), device="cpu")
    engine.create(documents_embeddings=documents)

    sentinel = object()
    engine._fused_engine = sentinel
    engine._fused_attempted = True
    generation = engine._fused_generation

    if mutation == "create":
        engine.create(documents_embeddings=documents)
    elif mutation == "update":
        engine.update(documents_embeddings=documents[:8])
    elif mutation == "delete":
        engine.delete(subset=[0])
    else:
        engine.close()

    assert engine._fused_engine is None
    assert engine._fused_attempted is False
    assert engine._fused_generation > generation


@requires_fused
def test_fused_status_reports_activation(tmp_path) -> None:
    """A deployment can assert which path it got instead of guessing."""
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(32, 96), p=2, dim=-1)
        for _ in range(512)
    ]

    index_path = str(tmp_path / "index")
    engine = FastPlaid(index=index_path, device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=4)

    status = engine.fused_status()
    assert status["active"] is True
    assert status["reason"] is None
    assert status["n_docs"] == 512
    assert status["resident_bytes"] > 0


def test_fused_status_explains_unavailability(tmp_path, monkeypatch) -> None:
    """When the fast path declines, the reason is reported rather than hidden."""
    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(16, 96), p=2, dim=-1)
        for _ in range(64)
    ]

    index_path = str(tmp_path / "index")
    engine = FastPlaid(
        index=index_path, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    engine.create(documents_embeddings=documents, nbits=4)

    status = engine.fused_status()
    assert status["active"] is False
    assert status["reason"]


@requires_fused
@pytest.mark.parametrize(
    ("lengths", "label"),
    [
        ((5, 3, 1, 4), "all shorter than the minimum dot block"),
        ((32, 5, 17, 1), "full-length query batched with shorter ones"),
        ((32, 32, 32, 32), "uniform, the case every benchmark used"),
    ],
)
def test_fused_handles_variable_length_queries(
    tmp_path, monkeypatch, lengths: tuple[int, ...], label: str
) -> None:
    """Padded query rows must not influence results.

    A zero-padded row scores every centroid identically, so its IVF top-k is an
    arbitrary set of tied cells. Left unmasked those cells admit candidates the
    standard pipeline never visits, which shows up as fused scoring documents
    mainline never saw. Uniform-length batches cannot catch this because they
    have no padded rows at all -- which is why every benchmark missed it.
    """
    torch.manual_seed(0)
    dim = 96
    documents = [
        torch.nn.functional.normalize(torch.randn(48, dim), p=2, dim=-1)
        for _ in range(512)
    ]
    queries = [
        torch.nn.functional.normalize(torch.randn(length, dim), p=2, dim=-1)
        for length in lengths
    ]

    index_path = str(tmp_path / "index")
    engine = FastPlaid(index=index_path, device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=4)

    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    engine._invalidate_fused()
    expected = engine.search(queries_embeddings=queries, top_k=5, show_progress=False)

    monkeypatch.delenv(gate.DISABLE_ENV)
    engine._invalidate_fused()
    assert engine._maybe_fused() is not None, label
    actual = engine.search(queries_embeddings=queries, top_k=5, show_progress=False)

    assert_same_ranking(actual, expected)


@requires_fused
def test_fused_splits_batches_without_misaligning_queries(
    tmp_path, monkeypatch
) -> None:
    """Query lengths must follow their queries when a batch is split.

    The batch ceiling can admit fewer queries than were asked for, so the
    engine slices both the padded queries and their true lengths. A misaligned
    slice would only ever be visible once splitting actually happens.
    """
    torch.manual_seed(0)
    dim = 96
    documents = [
        torch.nn.functional.normalize(torch.randn(48, dim), p=2, dim=-1)
        for _ in range(512)
    ]
    queries = [
        torch.nn.functional.normalize(torch.randn(length, dim), p=2, dim=-1)
        for length in (32, 3, 19, 1, 27, 8, 12)
    ]

    index_path = str(tmp_path / "index")
    engine = FastPlaid(index=index_path, device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=4)

    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    engine._invalidate_fused()
    expected = engine.search(queries_embeddings=queries, top_k=5, show_progress=False)

    # Force the ceiling to admit two queries at a time so the split path runs.
    monkeypatch.setattr("fast_plaid.search.fused.ceiling.max_batch", lambda **_: 2)
    monkeypatch.delenv(gate.DISABLE_ENV)
    engine._invalidate_fused()
    assert engine._maybe_fused() is not None
    actual = engine.search(queries_embeddings=queries, top_k=5, show_progress=False)

    assert_same_ranking(actual, expected)
