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

Usage:
    python benchmarks/fused_ab.py --index path/to/index --queries queries.pt

``--queries`` is a torch file holding either a list of ``[tokens, dim]``
tensors or one ``[n_queries, tokens, dim]`` tensor.
"""

from __future__ import annotations

import argparse
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
    engine._invalidate_fused()  # noqa: SLF001
    status = engine.fused_status()
    if status["active"] is not fused:
        raise RuntimeError(
            f"wanted fused={fused} but the gate reports {status}; "
            f"the arms would not be comparable"
        )

    search = lambda qs: engine.search(  # noqa: E731
        queries_embeddings=qs,
        top_k=top_k,
        n_full_scores=n_full_scores,
        n_ivf_probe=n_ivf_probe,
        show_progress=False,
    )

    # Warm-up: Triton compiles on first call, and the caching allocator has to
    # reach steady state. Neither belongs in the reported number.
    search(queries[: min(len(queries), batch)])

    batched = [timed(lambda: search(queries)) for _ in range(repeats)]
    ms_per_query = statistics.median(batched) / len(queries) * 1e3

    # Batch-1 latency is a different regime: both engines are dominated by
    # fixed per-call work rather than by scoring.
    singles = [timed(lambda q=[q]: search(q)) for q in queries[:batch1_queries]]
    b1_ms = statistics.median(singles) * 1e3

    return {
        "ms_per_query": ms_per_query,
        "qps": 1e3 / ms_per_query,
        "b1_ms": b1_ms,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "results": search(queries),
    }


def agreement(fused: list, standard: list, top_k: int) -> dict:
    """Max score deviation and mean top-k overlap between the two arms."""
    max_delta = 0.0
    overlaps = []
    pairs = 0
    for got, want in zip(fused, standard):
        for (_, got_score), (_, want_score) in zip(got, want):
            max_delta = max(max_delta, abs(got_score - want_score))
            pairs += 1
        got_ids = {doc for doc, _ in got}
        want_ids = {doc for doc, _ in want}
        if want_ids:
            overlaps.append(len(got_ids & want_ids) / len(want_ids))
    return {
        "max_score_delta": max_delta,
        "mean_overlap": statistics.mean(overlaps) if overlaps else float("nan"),
        "pairs": pairs,
        "top_k": top_k,
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
    parser.add_argument("--batch", type=int, default=250, help="warm-up batch size")
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
        f"speedup {standard['ms_per_query'] / fused['ms_per_query']:.1f}x batched, "
        f"{standard['b1_ms'] / fused['b1_ms']:.1f}x at batch 1"
    )
    print(
        f"agreement: max score delta {scores['max_score_delta']:.5f} over "
        f"{scores['pairs']} pairs | mean top-{args.top_k} overlap "
        f"{scores['mean_overlap']:.4f}"
    )


if __name__ == "__main__":
    main()
