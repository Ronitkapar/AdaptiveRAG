#!/usr/bin/env python
"""
scripts.measure_strategy_cost
-----------------------------
Phase 7.0b -- re-measure `RoutingConfig.strategy_cost_ms`.

The router prices each strategy by dividing its cost by the maximum cost across
available strategies (`routing/rule_based.py::_normalised_costs`), so what it
actually consumes is the ratio between strategies. The current table is seeded
from a Phase 5 run that recorded per-query means over 20 examples in a single
pass -- no warm-up, no repetitions -- and the 7.0 gate has since seen dense
retrieval swing between roughly 0.85 s and 4 s on this hardware, a spread wider
than the dense/hybrid gap those means encode. This script re-measures the table
under a defined protocol so the ratio is grounded in a distribution rather than
in one pass.

The protocol lives in `evaluation/measurement.py`:

* build every component before any clock starts;
* warm up each arm and discard those samples (ONNX session first-call cost, and
  the on-disk query-embedding cache at
  `storage/embeddings/embeddings_cache.sqlite3`);
* R repetitions over the query set, with the arm order *and* the query order
  rotated between repetitions;
* median as the estimator, with p95, mean, dispersion and raw samples recorded
  alongside.

The adaptive arm is measured and reported, but deliberately excluded from the
proposed table: its cost is a per-query mixture of the other strategies decided
at runtime, and the router chooses between strategies.

This script **measures and reports**. It writes one artifact and changes no
configuration. Freezing the proposed table into `RoutingConfig` is a separate,
reviewed edit, and E5 does not start until the values here have been read.

Usage:
    python scripts/measure_strategy_cost.py
    python scripts/measure_strategy_cost.py --repetitions 3 --arms bm25,dense
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    load_evaluation_dataset,
)
from adaptive_rag.evaluation.measurement import (  # noqa: E402
    COST_STRATEGIES,
    DEFAULT_REPETITIONS,
    DEFAULT_WARMUP_QUERIES,
    CostSample,
    arm_schedule,
    build_artifact,
    proposed_cost_table,
    sample_order,
    summarize_all,
    warmup_queries,
)
from adaptive_rag.errors import ConfigurationError  # noqa: E402
from adaptive_rag.experiments.arms import (  # noqa: E402
    build_arms,
    verify_arm_configs,
    verify_shared_conditions,
)
from adaptive_rag.experiments.config import (  # noqa: E402
    _shared_qdrant_client,
    build_experiment_config,
    compute_corpus_version,
    instantiate_components,
)
from adaptive_rag.schemas.config import RoutingConfig  # noqa: E402


def _load_gate_module() -> Any:
    """Import the sibling gate script so provenance is written one way.

    `scripts/` is a directory of standalone scripts with no `__init__.py`, so
    this cannot be a normal import. Loaded by path rather than duplicated
    because the gate report and this artifact must describe the machine with
    identical fields -- two copies of `_snapshot` would drift, and a sweep whose
    environment disagrees with the gate it followed is hard to trust.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "validate_environment.py"
    spec = importlib.util.spec_from_file_location("_phase7_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate_module()
_file_sha256 = _GATE._file_sha256
_git_commit = _GATE._git_commit
_snapshot = _GATE._snapshot

# The clock recorded for the table. This is the arm's end-to-end retrieval
# latency -- the same quantity Phase 5 recorded, so the new table is comparable
# to the seed rather than to a different measurement.
COST_CLOCK_FIELDS = (
    "query_embedding_latency_ms",
    "search_latency_ms",
    "dense_latency_ms",
    "bm25_latency_ms",
    "fusion_latency_ms",
    "candidate_generation_latency_ms",
    "rerank_latency_ms",
)

# Pacing between queries, in seconds. See `_pace` for why this exists and why it
# cannot bias a measurement.
DEFAULT_PACE_SECONDS = 0.75

# Arms that issue a live query-embedding call per query. BM25 is local and needs
# no pacing; `adaptive` is included because it may route into a dense-capable
# strategy at query time, so whether it calls the provider is not knowable here.
NETWORK_ARMS = frozenset({"dense", "hybrid", "hybrid_rerank", "adaptive"})


def _pace(arm: str, pace_seconds: float) -> None:
    """Sleep between queries for arms that call the embedding provider.

    Three of the four costed strategies embed every query remotely, so the
    protocol issues roughly `3 x queries x repetitions` calls in a burst. That
    burst reliably trips the provider's rate limit part-way through a sweep --
    a first attempt died at repetition 5/5 on HTTP 429 -- and the adapter's
    `2 ** attempt` backoff over `max_retries=4` is not enough to ride it out.

    The sleep sits *between* timed retrievals, never inside one, so it cannot
    enter any `latency_ms`. It changes request spacing only; the measured
    quantity, the sample count, the rotation, and the estimator are untouched.
    It is recorded in the artifact so a reader knows the sweep was paced.
    """
    if pace_seconds > 0 and arm in NETWORK_ARMS:
        time.sleep(pace_seconds)


def _sample_from(strategy: str, example: Any, repetition: int, response: Any) -> CostSample:
    """Build one sample from a retrieval response.

    A response with no usable `latency_ms` raises rather than contributing
    `0.0`: a missing clock would otherwise land in the median as an
    implausibly cheap retrieval and drag the arm's cost down, which is exactly
    the failure the gate's `_check_response_format` exists to prevent.
    """
    meta = response.retrieval_metadata
    latency = meta.latency_ms
    if latency is None or latency < 0:
        raise RuntimeError(
            f"{strategy} returned latency_ms={latency!r} for "
            f"{example.example_id}; refusing to record it as a measurement"
        )
    return CostSample(
        strategy=strategy,
        example_id=example.example_id,
        repetition=repetition,
        latency_ms=float(latency),
        stage_latency_ms={
            field: (
                float(value)
                if (value := getattr(meta, field, None)) is not None
                else None
            )
            for field in COST_CLOCK_FIELDS
        },
        rerank_fallback=meta.rerank_fallback,
    )


def _open_arms(
    common: Any, arms: Sequence[Any], client: Any
) -> dict[str, dict[str, Any]]:
    """Build every arm up front on the caller's shared Qdrant client.

    Built before any clock starts so that index loading, ONNX session creation
    and Qdrant opening are not billed to the first query. They share one client
    because local mode permits only one embedded client per process, and the
    rotated schedule needs every arm open at the same time.

    The client is passed in rather than opened here: opening a second one would
    fail outright against the first, and the caller has to hold the handle to
    close it.
    """
    opened: dict[str, dict[str, Any]] = {}
    for arm in arms:
        config = arm.build_config(common)
        components = instantiate_components(config, client=client)
        opened[arm.name] = {"retriever": components[2], "config": config}
    return opened


def _close_arms(opened: dict[str, dict[str, Any]], client: Any) -> None:
    for handle in opened.values():
        retriever = handle["retriever"]
        # A store may be wrapped in one or two layers (reranked over hybrid over
        # dense), so unwrap defensively rather than assuming a depth.
        for attribute in ("vector_store", "base_retriever"):
            inner = getattr(retriever, attribute, None)
            close = getattr(inner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001, S110 -- cleanup is best-effort
                    pass
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001, S110
            pass


def run_sweep(
    *,
    dataset_path: Path,
    top_k: int,
    repetitions: int,
    warmup: int,
    arm_names: Sequence[str],
    pace_seconds: float = DEFAULT_PACE_SECONDS,
) -> dict[str, Any]:
    """Execute the protocol and return the artifact."""
    examples = load_evaluation_dataset(dataset_path)
    if not examples:
        raise RuntimeError(f"no evaluation examples in {dataset_path}")

    common = build_experiment_config(name="phase7_cost_sweep_common")
    arms = build_arms(common)
    verify_shared_conditions(common, arms)
    verify_arm_configs(arms)

    selected = [arm for arm in arms if arm.name in set(arm_names)]
    if len(selected) != len(set(arm_names)):
        found = {arm.name for arm in selected}
        raise RuntimeError(f"unknown arms requested: {sorted(set(arm_names) - found)}")

    warm_set = warmup_queries(examples, warmup)
    # Every query is measured, including the ones used to warm up. Warm-up is a
    # separate discarded phase whose only job is to pay first-call costs before
    # the clock starts; excluding those queries would shrink the sample for no
    # benefit, since their first-call cost has already been absorbed.
    query_set = list(examples)
    schedule = arm_schedule([arm.name for arm in selected], repetitions)

    started = time.perf_counter()
    client = _shared_qdrant_client()
    try:
        opened = _open_arms(common, selected, client)

        for handle in opened.values():
            for example in warm_set:
                handle["retriever"].retrieve(example.query, top_k=top_k)
                _pace("warmup", pace_seconds)

        samples: list[CostSample] = []
        adaptive_samples: list[float] = []
        for repetition, order in enumerate(schedule):
            for name in order:
                retriever = opened[name]["retriever"]
                # Rotating the query order stops first-position effects inside a
                # pass from loading onto the same queries every repetition.
                for example in sample_order(query_set, repetition):
                    _pace(name, pace_seconds)
                    response = retriever.retrieve(example.query, top_k=top_k)
                    if name == "adaptive":
                        # Recorded as its own observation; the adaptive arm's cost
                        # is a per-query mixture and never becomes a table entry.
                        adaptive_samples.append(float(response.retrieval_metadata.latency_ms))
                        continue
                    samples.append(_sample_from(name, example, repetition, response))
            print(
                f"  repetition {repetition + 1}/{repetitions} done "
                f"(order: {', '.join(order)})",
                flush=True,
            )

        elapsed = time.perf_counter() - started
        summaries = summarize_all(samples)
        adaptive_observed: dict[str, Any] | None = None
        if adaptive_samples:
            ordered = sorted(adaptive_samples)
            adaptive_observed = {
                "n": len(ordered),
                "median_ms": ordered[len(ordered) // 2],
                "p95_ms": ordered[min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1)],
                "min_ms": ordered[0],
                "max_ms": ordered[-1],
                "note": (
                    "Reported for context only. The adaptive arm picks a strategy "
                    "and a stage count per query, so it has no single per-strategy "
                    "cost to enter in strategy_cost_ms."
                ),
            }

        current = dict(RoutingConfig().strategy_cost_ms)
        # The adaptive arm may be measured alongside the costed strategies as a
        # sanity check, but it must not reach the table: its cost is a per-query
        # mixture of the others. Dropped here rather than relied upon to be
        # absent, so `--arms ...,adaptive` cannot turn a complete sweep into an
        # "incomplete" artifact.
        table_summaries = [s for s in summaries if s.strategy != "adaptive"]
        # A partial sweep still produces a usable artifact: the arms that were
        # measured carry real distributions, and the arms that were not are a
        # finding rather than a reason to throw the whole sweep away. `proposed`
        # stays None whenever the table is incomplete, so a partial run can never
        # be mistaken for a freeze-ready one.
        table_error: str | None = None
        try:
            proposed = proposed_cost_table(table_summaries)
        except ConfigurationError as exc:
            proposed = None
            table_error = str(exc)

        artifact = build_artifact(
            summaries,
            proposed=proposed,
            current=current,
            environment=_snapshot(),
            provenance={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
                "corpus_version": common.corpus_version,
                "config_hash": common.config_hash,
                "dataset": {
                    "path": str(dataset_path.relative_to(REPO_ROOT)),
                    "n_examples": len(examples),
                    "sha256": _file_sha256(dataset_path),
                },
                "top_k": top_k,
                "n_measured_queries": len(query_set),
                "warmup_query_ids": [ex.example_id for ex in warm_set],
                "arm_schedule": schedule,
                "sweep_wall_seconds": round(elapsed, 1),
            },
            warmup=warmup,
            repetitions=repetitions,
            adaptive=adaptive_observed,
            incomplete_reason=table_error,
            pace_seconds=pace_seconds,
        )
        artifact["strategies"] = [
            s for s in artifact["strategies"] if s["n"] > 0
        ]
        return artifact
    finally:
        _close_arms(locals().get("opened", {}), client)


def _print_report(artifact: dict[str, Any]) -> None:
    print()
    print(f"strategy_cost_ms re-measurement -- {artifact['measurement_version']}")
    prov = artifact["provenance"]
    print(f"  corpus   {prov['corpus_version']}")
    print(f"  dataset  {prov['dataset']['path']} "
          f"({prov['dataset']['n_examples']} examples, "
          f"sha256={prov['dataset']['sha256']})")
    print(f"  commit   {prov['git_commit']}")
    protocol = artifact["protocol"]
    print(f"  protocol {protocol['repetitions']} reps x "
          f"{prov['n_measured_queries']} queries, {protocol['warmup_queries']} "
          f"warm-up discarded, wall={prov['sweep_wall_seconds']}s")
    print()
    header = f"  {'strategy':<15}{'n':>5}{'p50':>10}{'p95':>10}{'mean':>10}{'stdev':>10}{'max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in artifact["strategies"]:
        print(
            f"  {row['strategy']:<15}{row['n']:>5}"
            f"{row['median_ms']:>10.2f}{row['p95_ms']:>10.2f}"
            f"{row['mean_ms']:>10.2f}{row['stdev_ms']:>10.2f}{row['max_ms']:>10.2f}"
        )

    adaptive = artifact.get("adaptive_observed")
    if adaptive:
        print(f"  {'adaptive':<15}{adaptive['n']:>5}{adaptive['median_ms']:>10.2f}"
              f"{adaptive['p95_ms']:>10.2f}{'':>10}{'':>10}{adaptive['max_ms']:>10.2f}"
              "   (reported, not a table entry)")

    if not artifact["complete"]:
        print()
        print(f"  INCOMPLETE SWEEP: {artifact['incomplete_reason']}")
        print("  No proposed table was produced. Every costed strategy must be "
              "measured before this artifact can inform a freeze.")
        return None

    print()
    print("  proposed vs current (Phase 5 seed):")
    print(f"  {'strategy':<15}{'current':>12}{'proposed':>12}{'delta':>12}")
    for row in artifact["comparison"]:
        flag = "  *" if row["changed"] else ""
        print(
            f"  {row['strategy']:<15}{row['current_ms']:>12.2f}"
            f"{row['proposed_ms']:>12.2f}{row['delta_ms']:>12.2f}{flag}"
        )
    print()
    print("  p95 is nearest-rank: the ceil(0.95*n)-th smallest observed sample, "
          "not interpolated.")
    print("  NOTHING HAS BEEN FROZEN. Review these values before E5.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--repetitions", type=int, default=DEFAULT_REPETITIONS,
        help="Full passes over the query set per arm.",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_QUERIES,
        help="Discarded warm-up queries per arm.",
    )
    parser.add_argument(
        "--arms", default=",".join(COST_STRATEGIES),
        help="Comma-separated subset of arms to measure.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "experiments" / "phase7" / "strategy_cost_ms.json",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE_SECONDS,
        help=(
            "Seconds to sleep between queries on provider-calling arms, to "
            "avoid the embedding API rate limit. 0 disables pacing."
        ),
    )
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    artifact = run_sweep(
        dataset_path=args.dataset,
        top_k=args.top_k,
        repetitions=args.repetitions,
        warmup=args.warmup,
        arm_names=[name.strip() for name in args.arms.split(",") if name.strip()],
        pace_seconds=args.pace,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    if not args.json_only:
        _print_report(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())