"""A/B the fused CUDA search path against the standard pipeline.

Both arms go through the public ``FastPlaid.search`` API on one index in one
process; ``FAST_PLAID_DISABLE_FUSED`` selects which path serves the call, so
query staging, host-to-device transfer, kernel time and top-k extraction are
counted identically on both sides.

Two details decide whether the resulting ratio means anything, and both are
enforced here rather than left to the operator:

*Arm order.* The standard pipeline sizes its scoring workspace as
``search_memory_fraction`` of *free* VRAM (``device_memory_budget``). Timing it
while the fused copy is already resident would shrink its chunks and inflate
the speedup, so the standard arm runs first, before anything has been staged.
The fused arm runs second, under the standard index's residency -- where
ordering biases the ratio at all, it biases it against the fused path.

*Timing boundaries.* CUDA launches are asynchronous, so a timer that does not
synchronise measures the launch and not the work. Every timed region is
bracketed by ``torch.cuda.synchronize()``, and a full warm-up pass runs outside
the timers to absorb Triton's first-call compilation.

``--batch`` is the number of queries per ``search`` call, and the timed sweep
answers the whole query file in calls of exactly that size -- the figure is
labelled by it, so it has to be what was run. Peak memory is reset per arm,
since the high-water mark of the process would otherwise report the first
arm's footprint alongside the second's.

Usage:
    python benchmarks/fused_ab.py --index path/to/index --queries queries.pt

``--queries`` is a torch file holding either a list of ``[tokens, dim]``
tensors or one ``[n_queries, tokens, dim]`` tensor.
"""

from __future__ import annotations

import argparse
import itertools
import os
import statistics
import time

import torch
from fast_plaid.search import FastPlaid

DISABLE_ENV = "FAST_PLAID_DISABLE_FUSED"


def load_queries(path: str) -> list[torch.Tensor]:
    """Read query embeddings as a list of ``[tokens, dim]`` tensors."""
    loaded = torch.load(path, map_location="cpu")
    return list(loaded)


def timed(fn) -> float:
    """Seconds spent in ``fn``, with the device quiesced on both sides."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_arm(
    engine: FastPlaid,
    queries: list[torch.Tensor],
    *,
    fused: bool,
    top_k: int,
    n_full_scores: int,
    n_ivf_probe: int,
    batch: int,
    repeats: int,
    batch1_queries: int,
) -> dict:
    """Measure one engine, asserting it is the one that actually served."""
    if fused:
        os.environ.pop(DISABLE_ENV, None)
    else:
        os.environ[DISABLE_ENV] = "1"

    # Drop any copy staged by a previous arm and re-evaluate the gate, so the
    # assertion below reflects this arm rather than a leftover decision.
    # Reporting is pure, so staging has to be asked for explicitly.
    engine._invalidate_fused()  # noqa: SLF001
    status = engine.prepare_fused()
    if status["active"] is not fused:
        raise RuntimeError(
            f"wanted fused={fused} but the gate reports {status}; "
            f"the arms would not be comparable"
        )

    # Each arm reports its own peak rather than the high-water mark of the
    # process, which after the first arm would carry the other arm's footprint.
    torch.cuda.reset_peak_memory_stats()

    search = lambda qs: engine.search(  # noqa: E731
        queries_embeddings=qs,
        top_k=top_k,
        n_full_scores=n_full_scores,
        n_ivf_probe=n_ivf_probe,
        show_progress=False,
    )

    def sweep(size: int) -> list:
        """Answer every query in ``size``-query calls, as a server would."""
        out = []
        for start in range(0, len(queries), size):
            out.extend(search(queries[start : start + size]))
        return out

    # Warm-up: Triton compiles on first call, and the caching allocator has to
    # reach steady state. Neither belongs in the reported number.
    search(queries[: min(len(queries), batch)])

    # ``batch`` is the number of queries per ``search`` call, which is what the
    # reported figure is labelled by. Passing the whole file in one call would
    # measure a different regime than the label claims.
    batched = [timed(lambda: sweep(batch)) for _ in range(repeats)]
    ms_per_query = statistics.median(batched) / len(queries) * 1e3

    # Batch-1 latency is a different regime: both engines are dominated by
    # fixed per-call work rather than by scoring.
    singles = [timed(lambda q=[q]: search(q)) for q in queries[:batch1_queries]]
    b1_ms = statistics.median(singles) * 1e3

    return {
        "ms_per_query": ms_per_query,
        "qps": 1e3 / ms_per_query,
        "b1_ms": b1_ms,
        "batch": batch,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "results": sweep(batch),
    }


def agreement(fused: list, standard: list, top_k: int) -> dict:
    """Compare the two arms, and measure whether a delta could reorder a ranking.

    Overlap of 1.0 is evidence that no ranking moved, not proof that none can.
    The quantity that decides it is the smallest *non-zero* gap between two
    adjacent ranks: a score deviation strictly below that gap cannot swap any
    pair, so the ranking is stable for arithmetic reasons rather than lucky
    ones. Exact ties are excluded because reordering equally scored documents
    is not a ranking change, and neither engine promises a tie-break.
    """
    max_delta = 0.0
    overlaps = []
    pairs = 0
    min_gap = float("inf")
    ties = 0
    positions_differing = 0
    positions = 0
    substituted = 0
    genuine_reorders = 0
    for got, want in zip(fused, standard):
        for (_, got_score), (_, want_score) in zip(got, want):
            max_delta = max(max_delta, abs(got_score - want_score))
            pairs += 1
        got_ids = {doc for doc, _ in got}
        want_ids = {doc for doc, _ in want}
        if want_ids:
            overlaps.append(len(got_ids & want_ids) / len(want_ids))
        substituted += len(want_ids - got_ids)

        # Set overlap cannot see reordering, which is the failure a score
        # deviation actually causes. Compare the sequences position by
        # position as well, and separate the two reasons a position can
        # differ: the standard ranking had an exact tie there, so the order
        # between those documents was arbitrary and neither engine promises
        # it -- or it did not, and the deviation genuinely moved a rank.
        got_seq = [doc for doc, _ in got]
        want_seq = [doc for doc, _ in want]
        want_scores_row = [score for _, score in want]
        for i, (got_doc, want_doc) in enumerate(zip(got_seq, want_seq)):
            positions += 1
            if got_doc == want_doc:
                continue
            positions_differing += 1
            here = want_scores_row[i]
            tied = (i > 0 and want_scores_row[i - 1] == here) or (
                i + 1 < len(want_scores_row) and want_scores_row[i + 1] == here
            )
            if not tied:
                genuine_reorders += 1

        scores = [score for _, score in want]
        for higher, lower in itertools.pairwise(scores):
            gap = abs(higher - lower)
            if gap == 0.0:
                ties += 1
            else:
                min_gap = min(min_gap, gap)

    return {
        "max_score_delta": max_delta,
        "mean_overlap": statistics.mean(overlaps) if overlaps else float("nan"),
        "pairs": pairs,
        "top_k": top_k,
        "min_rank_gap": min_gap,
        "ties": ties,
        # Whether the deviation is small enough that it cannot swap any
        # adjacent pair. Below 1.0 means reordering is arithmetically possible.
        "headroom": (min_gap / max_delta) if max_delta > 0 else float("inf"),
        # What actually happened, as opposed to what could.
        "positions": positions,
        "positions_differing": positions_differing,
        "substituted": substituted,
        "genuine_reorders": genuine_reorders,
    }


def main() -> None:
    """Run both arms and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="path to a fast-plaid index")
    parser.add_argument("--queries", required=True, help="torch file of query tokens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-full-scores", type=int, default=4096)
    parser.add_argument("--n-ivf-probe", type=int, default=8)
    parser.add_argument(
        "--batch", type=int, default=250, help="queries per search call"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch1-queries", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark compares CUDA search paths")

    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    print(
        f"device {torch.cuda.get_device_name(args.device)} | "
        f"capability {torch.cuda.get_device_capability(args.device)}"
    )
    try:
        import triton

        print(f"triton {triton.__version__}")
    except ImportError:
        print("triton not installed -- the fused arm cannot run")

    queries = load_queries(args.queries)
    print(f"{len(queries)} queries | index {args.index}")

    engine = FastPlaid(index=args.index, device=args.device)
    shared = {
        "top_k": args.top_k,
        "n_full_scores": args.n_full_scores,
        "n_ivf_probe": args.n_ivf_probe,
        "batch": args.batch,
        "repeats": args.repeats,
        "batch1_queries": args.batch1_queries,
    }

    # Order is deliberate: see the module docstring.
    standard = run_arm(engine, queries, fused=False, **shared)
    print(
        f"[standard] {standard['ms_per_query']:.2f} ms/q | "
        f"{standard['qps']:.0f} QPS | batch-1 {standard['b1_ms']:.2f} ms | "
        f"peak {standard['peak_gib']:.1f} GiB"
    )

    fused = run_arm(engine, queries, fused=True, **shared)
    print(
        f"[fused]    {fused['ms_per_query']:.2f} ms/q | "
        f"{fused['qps']:.0f} QPS | batch-1 {fused['b1_ms']:.2f} ms | "
        f"peak {fused['peak_gib']:.1f} GiB"
    )

    scores = agreement(fused["results"], standard["results"], args.top_k)
    print(
        f"speedup {standard['ms_per_query'] / fused['ms_per_query']:.1f}x "
        f"at batch {args.batch}, "
        f"{standard['b1_ms'] / fused['b1_ms']:.1f}x at batch 1"
    )
    print(
        f"agreement: max score delta {scores['max_score_delta']:.5f} over "
        f"{scores['pairs']} pairs | mean top-{args.top_k} overlap "
        f"{scores['mean_overlap']:.4f}"
    )
    print(
        f"ranking stability: smallest non-zero adjacent-rank gap "
        f"{scores['min_rank_gap']:.6f} ({scores['ties']} exact ties) | "
        f"gap/deviation = {scores['headroom']:.2f}"
        + (
            "  <-- above 1.0: no adjacent pair can swap"
            if scores["headroom"] >= 1.0
            else "  <-- below 1.0: reordering is arithmetically possible"
        )
    )
    print(
        f"observed: {scores['substituted']} documents substituted | "
        f"{scores['positions_differing']} of {scores['positions']} rank "
        f"positions differ, of which {scores['genuine_reorders']} are not "
        f"explained by an exact tie"
    )


if __name__ == "__main__":
    main()
