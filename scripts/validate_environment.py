#!/usr/bin/env python
"""
scripts.validate_environment
----------------------------
Phase 7.0 -- the environment validation gate.

The Phase 7 benchmark is only meaningful if every arm actually runs against the
real corpus and real indexes. This script establishes that, or records precisely
why it cannot. Nothing downstream should be built on an unvalidated assumption
that, say, the ONNX reranker loads or that Qdrant opens in local mode.

For each of the five benchmark arms this checks:

1. **Build** -- components construct without raising.
2. **Query** -- one real retrieval call returns results.
3. **Output format** -- the response satisfies the Phase 7 metadata contract:
   latency fields present and non-negative, `top_k` honoured, ranking
   contiguous, scores finite, provenance populated.
4. **Routing trace** (adaptive arm only) -- decision, sufficiency, escalation,
   and the split latency clocks are all present and mutually consistent.
5. **Determinism** -- a second identical call returns the same ranked chunk ids.

Exit status is 0 only when every arm that was *required* passed. An arm that
cannot run is reported as BLOCKED with its reason; it is not silently skipped,
because "the reranker would not load" is a finding, not a footnote.

This script is deliberately read-mostly and side-effect-light: it writes one
JSON report and nothing else. It does not run the benchmark, does not tune
anything, and does not modify Phase 2-6 state.

Usage:
    python scripts/validate_environment.py
    python scripts/validate_environment.py --json-only
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    load_evaluation_dataset,
)
from adaptive_rag.experiments.arms import (  # noqa: E402
    ARM_NAMES,
    build_arms,
    verify_arm_configs,
    verify_shared_conditions,
)
from adaptive_rag.experiments.config import (  # noqa: E402
    build_experiment_config,
    instantiate_components,
)
from adaptive_rag.errors import AdaptiveRAGError  # noqa: E402
from adaptive_rag.schemas.experiment import RetrievalResponse  # noqa: E402

GATE_VERSION = "phase7_gate_v1"

# One real query is enough to prove an arm executes; the benchmark's own
# repetitions are the suite's job, not the gate's.
PROBE_TOP_K = 5


# --- result records -----------------------------------------------------------


def _record(name: str, **fields: Any) -> dict[str, Any]:
    return {"name": name, **fields}


def _blocked(name: str, reason: str, detail: str = "") -> dict[str, Any]:
    return _record(
        name,
        status="blocked",
        reason=reason,
        detail=detail.strip().splitlines()[-1] if detail else "",
    )


# --- format checks ------------------------------------------------------------


def _check_response_format(arm: str, response: Any, top_k: int) -> list[str]:
    """Return a list of contract violations; empty means the response is well-formed.

    These are the fields Phase 7 aggregates over. A missing or NaN latency would
    silently become 0.0 in a p95 calculation and quietly flatter whichever arm
    produced it, so each is checked explicitly rather than assumed.
    """
    problems: list[str] = []

    if not isinstance(response, RetrievalResponse):
        return [f"returned {type(response).__name__}, not a RetrievalResponse"]

    meta = response.retrieval_metadata
    if not response.results:
        problems.append("no results for a query that should match the corpus")

    # Latency clocks must exist and be real numbers.
    for field in ("latency_ms", "search_latency_ms"):
        value = getattr(meta, field, None)
        if value is None:
            problems.append(f"metadata.{field} is None")
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"metadata.{field} is not finite: {value!r}")
        elif value < 0:
            problems.append(f"metadata.{field} is negative: {value!r}")

    if meta.top_k != top_k:
        problems.append(f"metadata.top_k={meta.top_k}, requested {top_k}")
    if not meta.corpus_version:
        problems.append("metadata.corpus_version is empty")
    if not meta.retriever_version:
        problems.append("metadata.retriever_version is empty")

    if len(response.results) > top_k:
        problems.append(f"returned {len(response.results)} results for top_k={top_k}")

    for index, result in enumerate(response.results):
        if result.rank != index + 1:
            problems.append(f"result {index} has rank {result.rank}, expected {index + 1}")
        if not math.isfinite(result.score):
            problems.append(f"result {index} has non-finite score {result.score!r}")
        if not result.chunk_id:
            problems.append(f"result {index} has an empty chunk_id")
        if not result.metadata.document_id:
            problems.append(f"result {index} has no source document_id")
        if not result.provenance.document_id:
            problems.append(f"result {index} has no provenance document_id")
        if not result.provenance.source_sha256:
            problems.append(f"result {index} has no provenance source_sha256")

    return problems


def _check_routing_trace(arm: str, response: Any) -> list[str]:
    """Adaptive-only: the routing story must be complete enough to analyse.

    Phase 7 treats per-query routing behaviour as a research result, so a
    partially-populated trace is a real failure -- it would leave the ablation
    tables unable to say *why* a strategy was chosen.
    """
    problems: list[str] = []
    meta = response.retrieval_metadata

    if meta.routing is None:
        return ["adaptive arm returned no RoutingTrace"]

    trace = meta.routing
    if trace.decision is None:
        problems.append("routing trace has no decision")
    if trace.features is None:
        problems.append("routing trace has no analyzed features")
    if trace.routing_latency_ms is None:
        problems.append("routing trace has no routing_latency_ms")

    if not meta.adaptive_initial_strategy:
        problems.append("adaptive_initial_strategy is empty")
    if not meta.adaptive_final_strategy:
        problems.append("adaptive_final_strategy is empty")
    if meta.adaptive_routing_latency_ms is None:
        problems.append("adaptive_routing_latency_ms is None")

    # Total latency must account for routing, or the adaptive arm would be
    # charged only for retrieval and appear artificially cheap next to the
    # fixed arms. This is the single most important consistency check here.
    if meta.adaptive_routing_latency_ms is not None and meta.latency_ms is not None:
        if meta.latency_ms < meta.adaptive_routing_latency_ms:
            problems.append(
                f"total latency {meta.latency_ms:.2f}ms is below routing latency "
                f"{meta.adaptive_routing_latency_ms:.2f}ms -- routing is not being priced"
            )

    if trace.decision is not None:
        strategy = getattr(trace.decision, "strategy", None)
        if strategy != meta.adaptive_initial_strategy:
            problems.append(
                f"decision.strategy={strategy!r} does not match "
                f"adaptive_initial_strategy={meta.adaptive_initial_strategy!r}"
            )

    return problems


# --- per-arm validation -------------------------------------------------------


def validate_arm(
    arm_name: str,
    config: Any,
    query: str,
    top_k: int,
) -> dict[str, Any]:
    """Build, query, and verify one arm. Never raises; failures become records."""
    started = time.perf_counter()
    try:
        components = instantiate_components(config)
    except Exception as exc:  # noqa: BLE001 -- a failing build is a finding
        return _blocked(arm_name, type(exc).__name__, traceback.format_exc())

    retriever = components[2]
    resource = components[1]

    try:
        first = retriever.retrieve(query, top_k=top_k)
        first_wall = (time.perf_counter() - started) * 1000.0

        problems = _check_response_format(arm_name, first, top_k)
        if arm_name == "adaptive":
            problems += _check_routing_trace(arm_name, first)

        # Determinism: identical call, identical ranking. BM25 and dense are both
        # deterministic given fixed inputs; a mismatch would mean the benchmark is
        # measuring noise, and the suite's repetition design assumes it is not.
        second = retriever.retrieve(query, top_k=top_k)
        first_ids = [r.chunk_id for r in first.results]
        second_ids = [r.chunk_id for r in second.results]
        deterministic = first_ids == second_ids

        meta = first.retrieval_metadata
        record = _record(
            arm_name,
            status="ok" if not problems else "failed",
            retrieval_method=first.retrieval_method,
            n_results=len(first.results),
            top_ids=first_ids,
            latency_ms=meta.latency_ms,
            search_latency_ms=meta.search_latency_ms,
            wall_ms=first_wall,
            deterministic=deterministic,
            rerank_latency_ms=meta.rerank_latency_ms,
            rerank_candidate_count=meta.candidate_count,
            reranker_model_id=meta.reranker_model_id,
            routing_latency_ms=meta.adaptive_routing_latency_ms,
            initial_strategy=meta.adaptive_initial_strategy,
            final_strategy=meta.adaptive_final_strategy,
            escalated=meta.adaptive_escalated,
            sufficient=meta.adaptive_sufficient,
            stage_count=meta.adaptive_stage_count,
            problems=problems,
        )
        if not deterministic:
            record["status"] = "failed"
            record["problems"].append(
                "two identical calls returned different rankings"
            )
        return record

    except Exception as exc:  # noqa: BLE001 -- a crashing arm is a finding
        return _blocked(arm_name, type(exc).__name__, traceback.format_exc())
    finally:
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001, S110 -- cleanup is best-effort
                pass


# --- environment snapshot -----------------------------------------------------


def _snapshot() -> dict[str, Any]:
    """Record what the numbers were measured on.

    Latency comparisons across machines are meaningless, and the Phase 7 report
    has to state the hardware it ran on. Captured without adding a dependency:
    `platform` for the host, and the installed distributions via importlib.
    """
    from importlib.metadata import distributions

    packages: dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name in {
            "numpy", "pydantic", "qdrant-client", "onnxruntime",
            "optimum", "transformers", "tokenizers",
        }:
            packages[name] = dist.version or "unknown"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": __import__("os").cpu_count(),
        "packages": packages,
    }


# --- main ---------------------------------------------------------------------


def run_gate(dataset_path: Path, top_k: int) -> dict[str, Any]:
    """Validate every arm and return the gate report."""
    examples = load_evaluation_dataset(dataset_path)
    if not examples:
        raise RuntimeError(f"no evaluation examples in {dataset_path}")

    # Use the first example's query verbatim: a real corpus query, not a
    # synthetic probe string that might match nothing.
    query = examples[0].query

    common = build_experiment_config(name="phase7_gate_common")
    arms = build_arms(common)
    verify_shared_conditions(common, arms)
    verify_arm_configs(arms)

    results = [
        validate_arm(arm.name, arm.build_config(common), query, top_k)
        for arm in arms
    ]

    by_arm = {r["name"]: r for r in results}
    # All five are required: the benchmark compares them to each other, and a
    # missing arm is a hole in the comparison rather than a partial result.
    passed = [n for n in ARM_NAMES if by_arm.get(n, {}).get("status") == "ok"]

    return {
        "gate_version": GATE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "corpus_version": common.corpus_version,
        "config_hash": common.config_hash,
        "dataset": {
            "path": str(dataset_path.relative_to(REPO_ROOT)),
            "n_examples": len(examples),
            "dataset_version": examples[0].dataset_version,
            "sha256": _file_sha256(dataset_path),
        },
        "probe": {"query": query, "top_k": top_k},
        "environment": _snapshot(),
        "arms": results,
        "summary": {
            "required": list(ARM_NAMES),
            "passed": passed,
            "blocked": [
                r["name"] for r in results if r["status"] == "blocked"
            ],
            "failed": [r["name"] for r in results if r["status"] == "failed"],
            "gate_open": set(passed) == set(ARM_NAMES),
        },
    }


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return "unknown"


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _print_report(report: dict[str, Any]) -> None:
    print(f"Phase 7.0 environment gate -- {report['gate_version']}")
    print(f"  corpus   {report['corpus_version']}")
    print(f"  dataset  {report['dataset']['path']} "
          f"({report['dataset']['n_examples']} examples, "
          f"sha256={report['dataset']['sha256']})")
    print(f"  commit   {report['git_commit']}")
    env = report["environment"]
    print(f"  host     {env['platform']} / {env['cpu_count']} cpus / py{env['python']}")
    print()

    for arm in report["arms"]:
        mark = {"ok": "PASS", "failed": "FAIL", "blocked": "BLOCKED"}.get(
            arm["status"], "?"
        )
        print(f"  [{mark:>7}] {arm['name']}")
        if arm["status"] == "ok":
            print(f"             results={arm['n_results']} "
                  f"latency={arm['latency_ms']:.1f}ms "
                  f"deterministic={arm['deterministic']}")
            if arm.get("rerank_latency_ms") is not None:
                print(f"             rerank={arm['rerank_latency_ms']:.1f}ms "
                      f"on {arm['rerank_candidate_count']} candidates")
            if arm.get("routing_latency_ms") is not None:
                print(f"             routing={arm['routing_latency_ms']:.2f}ms "
                      f"{arm['initial_strategy']} -> {arm['final_strategy']} "
                      f"(escalated={arm['escalated']}, stages={arm['stage_count']})")
        elif arm["status"] == "failed":
            for problem in arm["problems"]:
                print(f"             - {problem}")
        else:
            print(f"             {arm['reason']}: {arm['detail']}")

    summary = report["summary"]
    print()
    print(f"  passed={len(summary['passed'])}/{len(summary['required'])}"
          f" blocked={summary['blocked']} failed={summary['failed']}")
    print(f"  GATE {'OPEN' if summary['gate_open'] else 'CLOSED'}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Evaluation dataset used as the probe source.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=PROBE_TOP_K,
        help="Number of results to request from each arm.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "phase7" / "gate.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Write the report without the human-readable summary.",
    )
    args = parser.parse_args()

    report = run_gate(args.dataset, args.top_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")

    if not args.json_only:
        _print_report(report)

    return 0 if report["summary"]["gate_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main())