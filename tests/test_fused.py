"""Tests for the fused CUDA search path.

The parity test is the important one: it asserts that fused search returns the
same documents and the same scores as the standard pipeline on the same index.
It skips itself when no supported GPU is present, so the CPU CI still exercises
the gate and the batch-ceiling arithmetic.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch
from filelock import FileLock

from fast_plaid.search import FastPlaid
from fast_plaid.search.fused import FusedCompilationError, ceiling, gate

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


# The two engines are not bit-identical, and how far apart they land depends on
# the architecture. Both round the same way through reconstruction,
# normalisation and the fp16 GEMM output, but `tl.dot` and libtorch's Half
# matmul accumulate in different orders, so a per-token maximum can land one or
# two fp16 ulps apart -- and a maximum that moves changes the sum.
#
# Measured by this suite against a real install:
#
#     sm_90 (H100)   0.0          exactly bit-identical
#     sm_80 (A100)   1.22e-4
#     sm_86 (A10G)   2.44e-4      2**-12, ~2 ulps at these magnitudes
#     sm_89 (L4)     2.44e-4
#
# The tolerance below is therefore an empirical bound with headroom, not a claim
# of exactness -- and measuring on H100 alone, as the first benchmarks did,
# would have suggested a guarantee the other three architectures do not honour.
#
# The *strict* part of this assertion is the document set: equivalence means the
# same documents come back, and that is asserted exactly at every distance.
SCORE_TOLERANCE = 1e-3

# Populated by every comparison so the suite can report what it actually saw
# rather than only that it stayed under the bound.
OBSERVED_DELTAS: list[float] = []


def assert_same_ranking(actual, expected) -> None:
    """Assert both engines return the same documents at near-identical scores.

    Ids are compared as sets rather than positionally, because equal scores may
    be ordered differently by two top-k implementations and neither engine
    promises a tie-break. Substitution is not tolerated at any score distance.

    An earlier version compared only rank-aligned scores within 1e-3 and made
    the id check *conditional* on the scores differing, which let fused
    substitute a document mainline never returned as long as the two scored
    closely -- precisely the failure "retrieval-equivalent" claims cannot
    happen, passing the test meant to prove it.
    """
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert len(got) == len(want), (
            f"returned {len(got)} results where the standard pipeline "
            f"returned {len(want)}"
        )
        got_ids = {doc for doc, _ in got}
        want_ids = {doc for doc, _ in want}
        assert got_ids == want_ids, (
            f"different documents returned: only fused {sorted(got_ids - want_ids)}, "
            f"only standard {sorted(want_ids - got_ids)}"
        )

        got_scores = [score for _, score in got]
        want_scores = [score for _, score in want]
        if got_scores:
            OBSERVED_DELTAS.append(
                max(abs(a - b) for a, b in zip(got_scores, want_scores))
            )
        assert got_scores == pytest.approx(want_scores, abs=SCORE_TOLERANCE), (
            f"scores diverge: {got_scores} vs {want_scores}"
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
    dim, nbits, n_tokens = 96, 4, 100_000
    data = {
        "nbits": nbits,
        "centroids": torch.zeros(1024, dim, dtype=torch.float16),
        "ivf": torch.zeros(4096, dtype=torch.int32),
        "ivf_lengths": torch.zeros(1024, dtype=torch.int64),
        "doc_lengths": torch.full((1_000,), n_tokens // 1_000, dtype=torch.int64),
    }
    # Sized from the gate's own accounting rather than a hand-tuned constant,
    # so that changing what staging costs cannot silently make this vacuous.
    required = gate.resident_bytes(
        n_tokens=n_tokens,
        n_docs=1_000,
        n_centroids=1024,
        n_ivf=4096,
        dim=dim,
        nbits=nbits,
    ) + gate.staging_bytes(n_tokens=n_tokens, dim=dim)
    free_bytes = int(required / 0.5)
    shared = {"data": data, "device": "cuda:0", "n_tokens": n_tokens}

    assert gate.check(**shared, free_bytes=free_bytes, memory_fraction=0.8) is None
    restricted = gate.check(**shared, free_bytes=free_bytes, memory_fraction=0.2)
    assert restricted is not None
    assert "capped at 0.2" in restricted


def test_resident_bytes_counts_more_than_the_per_token_arrays() -> None:
    """Residency includes the per-document and per-centroid arrays.

    They are small beside the per-token ones but not nothing: on millions of
    documents they run to hundreds of megabytes, and an estimate that omitted
    them understated exactly the indexes closest to declining.
    """
    dim, nbits, n_tokens, n_docs, n_centroids = 128, 4, 4_000_000, 100_000, 8_192
    resident = gate.resident_bytes(
        n_tokens=n_tokens,
        n_docs=n_docs,
        n_centroids=n_centroids,
        n_ivf=1_000_000,
        dim=dim,
        nbits=nbits,
    )
    per_token_only = n_tokens * gate.bytes_per_token(dim=dim, nbits=nbits)

    assert resident > per_token_only
    # The per-document arrays alone are worth counting at this corpus size.
    assert resident - per_token_only > n_docs * 12


def test_gate_admits_the_msmarco_index_that_was_measured() -> None:
    """The staging transient must not refuse an index already shown to serve.

    Measured on an 80GB H100 with the standard index resident: 43.7 GiB free,
    31.9 GiB staged, and the fused path answered the whole dev split. Counting
    the precompute transient is right -- an index that fits resident but not
    while being built should decline rather than fail inside the constructor --
    but at NORM_CHUNK = 2,000,000 that transient came to 3.9 GiB, which pushed
    resident + staging past the 0.8 cap and declined the flagship corpus by
    0.8 GiB. The chunk is a scheduling knob, so it was shrunk rather than the
    accounting loosened.
    """
    resident = int(31.9 * GIB)  # measured, not modelled
    transient = gate.staging_bytes(n_tokens=597_909_930, dim=96)
    free = int(43.7 * GIB)

    assert resident + transient <= gate.DEFAULT_MEMORY_FRACTION * free, (
        f"staging transient of {transient / GIB:.2f} GiB refuses an index "
        f"measured to stage and serve"
    )


@requires_fused
def test_gate_counts_the_staging_transient() -> None:
    """Residency alone is not what has to fit; the precompute peaks above it.

    An index that fits resident but not while being built would otherwise pass
    the gate and then fail inside the constructor, turning a clean decline into
    a staging exception.
    """
    dim, nbits, n_tokens = 128, 4, 4_000_000
    n_docs, n_centroids, n_ivf = 100_000, 8_192, 1_000_000
    resident = gate.resident_bytes(
        n_tokens=n_tokens,
        n_docs=n_docs,
        n_centroids=n_centroids,
        n_ivf=n_ivf,
        dim=dim,
        nbits=nbits,
    )
    transient = gate.staging_bytes(n_tokens=n_tokens, dim=dim)
    assert transient > 0, "no transient means this test proves nothing"

    data = {
        "nbits": nbits,
        "centroids": torch.zeros(n_centroids, dim, dtype=torch.float16),
        "ivf": torch.zeros(n_ivf, dtype=torch.int32),
        "ivf_lengths": torch.zeros(n_centroids, dtype=torch.int64),
        "doc_lengths": torch.full((n_docs,), n_tokens // n_docs, dtype=torch.int64),
    }
    shared = {"data": data, "device": "cuda:0", "n_tokens": n_tokens}

    # Free memory that covers residency but only half the staging peak.
    tight = int((resident + transient / 2) / 0.8)
    reason = gate.check(**shared, free_bytes=tight, memory_fraction=0.8)
    assert reason is not None
    assert "to stage" in reason

    # The same index passes once the peak is covered.
    ample = int((resident + transient) / 0.8) + 1
    assert gate.check(**shared, free_bytes=ample, memory_fraction=0.8) is None


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

    # Reporting never stages, so the question has to be asked explicitly.
    assert engine.fused_status()["active"] is False

    status = engine.prepare_fused()
    assert status["active"] is True
    assert status["reason"] is None
    assert status["n_docs"] == 512
    assert status["resident_bytes"] > 0

    # And now reporting agrees, without having caused it.
    assert engine.fused_status() == status


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

    status = engine.prepare_fused()
    assert status["active"] is False
    assert status["reason"]
    assert gate.DISABLE_ENV in status["reason"]


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


def test_fused_package_imports_without_triton() -> None:
    """Importing the package must not require Triton.

    The gate exists to decline in words on hardware that cannot run the
    kernels, but it used to be unreachable: the package imported ``engine``,
    which imports ``kernels``, which imports Triton unconditionally. Every
    CPU, macOS and Windows install therefore raised ``ModuleNotFoundError``
    from inside the call meant to report that Triton was missing -- 32 of the
    65 tests in the suite failed that way.

    Run in a subprocess with Triton masked, because this process may legitimately
    have it installed.
    """
    program = (
        "import sys; sys.modules['triton'] = None;"
        "import fast_plaid.search.fused as fused;"
        "reason = fused.gate.check(data={}, device='cpu', n_tokens=1);"
        "assert reason is not None, reason;"
        "engine, reason = fused.build_engine(data={}, device='cpu');"
        "assert engine is None and reason is not None;"
        "print('declined:', reason)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"the package is not importable without Triton:\n{result.stderr}"
    )
    assert "declined:" in result.stdout


def test_fused_can_be_disabled_per_instance(tmp_path) -> None:
    """The opt-out is per instance, not only per process.

    The environment variable disables the fast path everywhere, which is the
    wrong granularity for a library: one index sharing a GPU with a model may
    need to decline the extra residency while another does not.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(8, 96), p=2, dim=-1)
        for _ in range(64)
    ]

    engine = FastPlaid(index=str(tmp_path / "index"), device="cpu", fused=False)
    engine.create(documents_embeddings=documents)

    status = engine.prepare_fused()
    assert status["active"] is False
    assert "fused=False" in status["reason"]
    assert engine._maybe_fused() is None


def test_staging_declines_while_a_writer_holds_the_lock(tmp_path) -> None:
    """Staging reads a coherent snapshot or none at all.

    ``_load_index_tensors_cpu`` memory-maps the merged files and can pad them
    in place, so reading them while another process is mid-update yields
    tensors that never described any one state of the index. The generation
    counter cannot see that -- it is in-process bookkeeping -- so the file lock
    is what makes the read coherent.

    Declining must also leave the attempt *unmarked*, or one unlucky overlap
    with a writer would strand the index on the standard pipeline forever.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(8, 96), p=2, dim=-1)
        for _ in range(64)
    ]

    engine = FastPlaid(index=str(tmp_path / "index"), device="cpu")
    engine.create(documents_embeddings=documents)
    engine._invalidate_fused()

    writer = FileLock(engine.lock_path)
    with writer:
        assert engine._maybe_fused() is None
        assert engine._fused_attempted is False, (
            "a busy writer must not be remembered as a permanent decline"
        )


def test_invalid_batch_size_raises_rather_than_falling_back(tmp_path) -> None:
    """Argument validation happens before the fast path, not inside it.

    Validating within the fused attempt would let the fallback swallow a
    ``ValueError`` the caller needs to see, and answer the malformed call from
    the standard pipeline as though nothing were wrong.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(8, 96), p=2, dim=-1)
        for _ in range(64)
    ]
    queries = torch.nn.functional.normalize(torch.randn(2, 8, 96), p=2, dim=-1)

    engine = FastPlaid(index=str(tmp_path / "index"), device="cpu")
    engine.create(documents_embeddings=documents)

    with pytest.raises(ValueError, match="batch_size"):
        engine.search(queries_embeddings=queries, batch_size=-1, show_progress=False)


@requires_fused
def test_explicit_batch_size_is_served_by_the_standard_pipeline(
    tmp_path, monkeypatch
) -> None:
    """An explicit batch_size is declined by fused, not silently ignored.

    It budgets documents per scoring chunk; the fused path chunks by queries,
    so there is no honest translation. Whoever serves the call should be the
    one that can honour the setting.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(32, 96), p=2, dim=-1)
        for _ in range(512)
    ]
    queries = torch.nn.functional.normalize(torch.randn(4, 32, 96), p=2, dim=-1)

    engine = FastPlaid(index=str(tmp_path / "index"), device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=4)

    fused = engine._maybe_fused()
    assert fused is not None

    def refuse(*_args, **_kwargs):
        pytest.fail("the fused path served a call carrying an explicit batch_size")

    monkeypatch.setattr(fused, "search", refuse)
    results = engine.search(
        queries_embeddings=queries, top_k=5, batch_size=64, show_progress=False
    )
    assert len(results) == 4


@requires_fused
def test_compilation_failure_retires_the_staged_copy(tmp_path, monkeypatch) -> None:
    """A failure that will recur is not retried on every search.

    Out-of-memory is transient and deliberately keeps the staged copy, since
    the admission ceiling is recomputed from free memory each call. A kernel
    that cannot compile for this index's shapes is the opposite: it fails
    identically forever, so the copy is dropped and the standard pipeline
    takes over without paying the same failure again.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(32, 96), p=2, dim=-1)
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

    def cannot_compile(*_args, **_kwargs):
        calls["n"] += 1
        raise FusedCompilationError("synthetic compilation failure")

    monkeypatch.setattr(fused, "_search_batch", cannot_compile)

    actual = engine.search(queries_embeddings=queries, top_k=5, show_progress=False)
    assert_same_ranking(actual, expected)

    assert engine._fused_engine is None, "the doomed copy should have been retired"
    assert engine._fused_attempted is True, "retiring must not trigger a restage"

    # A second search must not pay the failure again.
    engine.search(queries_embeddings=queries, top_k=5, show_progress=False)
    assert calls["n"] == 1


@requires_fused
def test_fused_matches_standard_pipeline_on_empty_input(tmp_path, monkeypatch) -> None:
    """An empty request is answered, not raised on.

    Padding took the maximum of the query lengths, which is undefined for an
    empty batch, so fused turned an empty result into a ValueError.
    """
    torch.manual_seed(0)
    documents = [
        torch.nn.functional.normalize(torch.randn(8, 96), p=2, dim=-1)
        for _ in range(64)
    ]

    engine = FastPlaid(index=str(tmp_path / "index"), device="cuda:0")
    engine.create(documents_embeddings=documents, nbits=4)

    monkeypatch.setenv(gate.DISABLE_ENV, "1")
    engine._invalidate_fused()
    expected = engine.search(queries_embeddings=[], top_k=5, show_progress=False)

    monkeypatch.delenv(gate.DISABLE_ENV)
    engine._invalidate_fused()
    assert engine._maybe_fused() is not None
    actual = engine.search(queries_embeddings=[], top_k=5, show_progress=False)

    assert actual == expected


def test_zz_report_observed_score_deltas() -> None:
    """Report the worst score deviation this suite actually produced.

    Named to sort last. Not an assertion about a threshold -- the per-comparison
    bound already enforces that -- but a standing measurement, so the number in
    the PR description stays tied to something the suite prints rather than to
    a claim nobody re-checks.
    """
    if not OBSERVED_DELTAS:
        pytest.skip("no GPU comparisons ran")
    worst = max(OBSERVED_DELTAS)
    print(
        f"\nmax score delta across {len(OBSERVED_DELTAS)} query comparisons: "
        f"{worst:.10f}"
    )
    assert worst <= SCORE_TOLERANCE
