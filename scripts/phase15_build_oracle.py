#!/usr/bin/env python3
"""Phase 15 oracle build -- per-query BM25-vs-Dense need truth, powered set.

Same frozen oracle as Phase 14 (`phase14_cheapfirst_v1`; ties/harm stop),
applied to fresh retrieval runs. Differences from `phase14_build_oracle.py`
are all pre-registered in `docs/phases/phase-15.md` §§15.4-15.9:

* inputs are parameterized (powered suite runs + powered dataset), not the
  frozen Phase 8/9 artifacts;
* BM25-state signals are computed directly from the BM25 traces with the
  Phase 9 functions (no frozen signal table exists for new queries);
* T0 query features come from the powered adaptive-arm traces, exactly as
  Phase 14 read them from the frozen adaptive traces;
* all powered queries carry split `test` (nothing is fitted).

Reads only persisted artifacts, runs no retrieval, calls no provider.
Integrity gates (any failure aborts rather than writing a partial oracle):
both arms fully `ok` (strict `load_arm`), identical query sets across
bm25/dense/adaptive-trace sources, frozen retrieval config (top_k=10 both
arms), expected split coverage, T0 features present on every query.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_rag.config.hashing import compute_file_sha256  # noqa: E402
from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.agreement import (  # noqa: E402
    distinct_doc_ratio,
    score_decay_slope,
    score_gap_top1_top2,
)
from adaptive_rag.evaluation.cheapfirst import (  # noqa: E402
    NEED_ORACLE_VERSION,
    SIGNAL_K,
    oracle_label,
)
from adaptive_rag.evaluation.dispersion import (  # noqa: E402
    EPSILON,
    scores_from_trace,
)

ORACLE_VERSION = NEED_ORACLE_VERSION
ORACLE_EPSILON = EPSILON

OUT_DIR = REPO_ROOT / "experiments" / "phase15"
FROZEN_POLICY_PATH = REPO_ROOT / "experiments" / "phase14" / "frozen_policy.json"

ARMS = ("after", "before")

EXPECTED_TOP_K = 10
FLOAT_TOL = 1e-9

#: T0 query-feature representatives (docs/phases/phase-15.md §15.5): numeric
#: fields of the powered adaptive-trace `routing.features`, nothing else.
QUERY_FEATURE_NAMES = ("complexity_score", "content_term_count")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _single(path_pattern: str, suite_root: Path, what: str) -> Path:
    matches = sorted(suite_root.glob(path_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {what} under {suite_root}, found "
            f"{len(matches)}"
        )
    return matches[0]


def _bm25_traces(suite_root: Path) -> dict[str, dict[str, Any]]:
    """Powered BM25 traces keyed by query id (example_id == query_id)."""
    path = _single(
        "*E1_baseline_comparison__bm25/traces.jsonl", suite_root, "bm25 traces"
    )
    traces = {}
    for record in _read_jsonl(path):
        if record.get("status") != "ok":
            raise ValueError(
                f"{suite_root}: bm25 trace {record.get('example_id')} "
                f"status={record.get('status')!r}; refusing a partial join"
            )
        traces[str(record["example_id"])] = record
    return traces


def _query_features(suite_root: Path) -> dict[str, dict[str, float]]:
    """T0 query features from the powered adaptive arm's persisted routing."""
    path = _single(
        "*E1_baseline_comparison__adaptive/traces.jsonl",
        suite_root,
        "adaptive traces",
    )
    features: dict[str, dict[str, float]] = {}
    for record in _read_jsonl(path):
        if record.get("status") != "ok":
            continue
        feats = (record.get("routing") or {}).get("features") or {}
        query_id = str(record["example_id"])
        try:
            features[query_id] = {
                name: float(feats[name]) for name in QUERY_FEATURE_NAMES
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{suite_root}: adaptive trace {query_id} lacks numeric "
                f"T0 features {QUERY_FEATURE_NAMES}: {exc}"
            ) from exc
    return features


def _retrieval_config(suite_root: Path, system: str) -> dict[str, Any]:
    path = _single(
        f"*E1_baseline_comparison__{system}/config.json",
        suite_root,
        f"{system} config",
    )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["retrieval"]


def _manifest(suite_root: Path, system: str) -> dict[str, Any]:
    path = _single(
        f"*E1_baseline_comparison__{system}/manifest.json",
        suite_root,
        f"{system} manifest",
    )
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "config_hash": manifest.get("config_hash"),
        "corpus_version": manifest.get("corpus_version"),
        "git_commit": manifest.get("git_commit"),
        "trace_count": manifest.get("trace_count"),
    }


def _load_rows_strict(suite_root: Path, system: str) -> list[dict[str, Any]]:
    """Read one E1 arm's rows, refusing anything short of full coverage.

    Mirrors `compare_phase8_arms.load_arm`'s guarantees for the Phase 15
    suite layout (unprefixed `E1_baseline_comparison__{system}` run dirs):
    exactly one run dir, every row `status == "ok"`, every row labelled
    with the expected system, and full agreement with that arm's traces
    (checked by the caller through the query-set join).
    """
    rows_path = _single(
        f"*E1_baseline_comparison__{system}/rows.jsonl", suite_root, f"{system} rows"
    )
    rows = _read_jsonl(rows_path)
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(
                f"{suite_root}: {system} row {row.get('query_id')} "
                f"status={row.get('status')!r}; refusing a partial arm"
            )
        if str(row.get("system")) != system:
            raise ValueError(
                f"{suite_root}: {system} rows carry system "
                f"{row.get('system')!r}; refusing a mixed arm"
            )
    traces_path = _single(
        f"*E1_baseline_comparison__{system}/traces.jsonl",
        suite_root,
        f"{system} traces",
    )
    trace_ids = {
        str(record["example_id"])
        for record in _read_jsonl(traces_path)
        if record.get("status") == "ok"
    }
    row_ids = {str(row["query_id"]) for row in rows}
    if trace_ids != row_ids:
        raise ValueError(
            f"{suite_root}: {system} rows/traces disagree: "
            f"{len(row_ids)} rows vs {len(trace_ids)} ok traces"
        )
    return rows


def build_arm(
    arm: str,
    suite_root: Path,
    dataset_path: Path,
    expected_splits: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join one corpus arm's rows, traces, computed signals, and splits."""
    # Strict: any non-ok trace aborts (the Phase 8 measurement-integrity
    # lesson -- a silently short arm inverts conclusions).
    arm_rows = {
        system: _load_rows_strict(suite_root, system) for system in ("bm25", "dense")
    }
    bm25 = {r["query_id"]: r for r in arm_rows["bm25"]}
    dense = {r["query_id"]: r for r in arm_rows["dense"]}
    bm25_traces = _bm25_traces(suite_root)
    qfeat = _query_features(suite_root)

    sets = {
        "bm25": set(bm25),
        "dense": set(dense),
        "bm25_traces": set(bm25_traces),
        "qfeat": set(qfeat),
    }
    if len({frozenset(s) for s in sets.values()}) != 1:
        only = sorted(
            {q for s in sets.values() for q in s},
            key=lambda q: sum(q not in s for s in sets.values()),
            reverse=True,
        )[:8]
        raise ValueError(
            f"{arm}: query sets disagree: "
            + ", ".join(f"{k}={len(v)}" for k, v in sets.items())
            + f"; most-lopsided={only}"
        )

    bm25_cfg = _retrieval_config(suite_root, "bm25")
    dense_cfg = _retrieval_config(suite_root, "dense")
    for name, got in (
        ("bm25.top_k", bm25_cfg.get("top_k")),
        ("dense.top_k", dense_cfg.get("top_k")),
    ):
        if got != EXPECTED_TOP_K:
            raise ValueError(
                f"{arm}: frozen retrieval config moved: {name}={got}, "
                f"expected {EXPECTED_TOP_K}; refusing to join the wrong runs"
            )

    split = {
        json.loads(line)["example_id"]: json.loads(line)["split"]
        for line in open(dataset_path, encoding="utf-8")
        if line.strip()
    }

    records: list[dict[str, Any]] = []
    missing_counts = {"bm25_gap": 0, "bm25_slope": 0}
    for query_id in sorted(sets["bm25"]):
        if query_id not in split:
            raise ValueError(f"{arm}: {query_id} has no split in the dataset")
        if split[query_id] not in expected_splits:
            raise ValueError(
                f"{arm}: {query_id} split={split[query_id]!r} outside "
                f"expected {sorted(expected_splits)}"
            )
        b_row, d_row = bm25[query_id], dense[query_id]
        if b_row["status"] != "ok" or d_row["status"] != "ok":
            # Unreachable when load_arm is strict, kept as a second lock.
            raise ValueError(f"{arm}: {query_id} carries a non-ok row")
        trace = bm25_traces[query_id]
        scores = scores_from_trace(trace)

        # BM25-state signals computed directly with the Phase 9 functions
        # (no frozen signal table exists for fresh queries).
        fresh_ddr = round(
            distinct_doc_ratio(b_row["retrieved_document_ids"], SIGNAL_K), 6
        )
        fresh_gap = score_gap_top1_top2(scores, SIGNAL_K)
        fresh_slope = score_decay_slope(scores, SIGNAL_K)
        results = ((trace.get("retrieval") or {}).get("results") or [])
        if not results or results[0].get("score") is None:
            raise ValueError(
                f"{arm}: {query_id} bm25 trace has no top-1 native score"
            )
        top1 = float(results[0]["score"])

        missing_counts["bm25_gap"] += fresh_gap is None
        missing_counts["bm25_slope"] += fresh_slope is None
        rule_signals = {
            "ddr_bm25": float(fresh_ddr),
            "bm25_top1": top1,
            "bm25_gap": None if fresh_gap is None else float(fresh_gap),
            "bm25_slope": None if fresh_slope is None else float(fresh_slope),
            "complexity_score": float(qfeat[query_id]["complexity_score"]),
            "content_term_count": float(qfeat[query_id]["content_term_count"]),
        }
        delta_r5 = float(d_row["recall_at_5"]) - float(b_row["recall_at_5"])
        delta_mrr = float(d_row["mrr"]) - float(b_row["mrr"])
        records.append(
            {
                "query_id": query_id,
                "corpus_arm": arm,
                "split": split[query_id],
                "category": b_row["category"],
                "signals": rule_signals,
                "bm25_recall_at_5": float(b_row["recall_at_5"]),
                "dense_recall_at_5": float(d_row["recall_at_5"]),
                "delta_recall_at_5": delta_r5,
                "bm25_mrr": float(b_row["mrr"]),
                "dense_mrr": float(d_row["mrr"]),
                "delta_mrr": delta_mrr,
                "bm25_hit_at_5": float(b_row["hit_at_5"]),
                "dense_hit_at_5": float(d_row["hit_at_5"]),
                "bm25_latency_ms": float(b_row["retrieval_latency_ms"]),
                "dense_latency_ms": float(d_row["retrieval_latency_ms"]),
                "oracle_escalate": bool(oracle_label(delta_r5, ORACLE_EPSILON)),
                "bm25_retrieved_document_ids": list(
                    b_row["retrieved_document_ids"]
                ),
                "dense_retrieved_document_ids": list(
                    d_row["retrieved_document_ids"]
                ),
            }
        )

    provenance = {
        "suite": str(suite_root),
        "bm25_manifest": _manifest(suite_root, "bm25"),
        "dense_manifest": _manifest(suite_root, "dense"),
        "adaptive_trace_source": "E1 adaptive traces (T0 routing.features)",
        "bm25_top_k": bm25_cfg.get("top_k"),
        "dense_top_k": dense_cfg.get("top_k"),
        "n_queries": len(records),
        "n_oracle_yes": sum(1 for r in records if r["oracle_escalate"]),
        "missing_signal_counts": missing_counts,
        "signal_integrity": (
            "ddr/gap/slope computed directly from powered bm25 "
            "rows/traces with the Phase 9 functions (no frozen table "
            "exists for fresh queries)"
        ),
        "query_feature_integrity": (
            f"T0 features present on {len(records)}/{len(records)}"
        ),
    }
    return records, provenance


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Build the Phase 15 cheap-first need oracle (powered set)."
    )
    parser.add_argument("--suite-after", required=True, help="after-arm suite root")
    parser.add_argument("--suite-before", required=True, help="before-arm suite root")
    parser.add_argument("--dataset", required=True, help="evaluation dataset JSONL")
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory")
    parser.add_argument(
        "--expect-splits",
        default="test",
        help="comma-separated allowed split values (default: test)",
    )
    parser.add_argument("--phase", type=int, default=15, help="phase stamp")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset)
    expected_splits = {s.strip() for s in str(args.expect_splits).split(",")}
    suites = {"after": Path(args.suite_after), "before": Path(args.suite_before)}

    per_arm: dict[str, list[dict[str, Any]]] = {}
    arm_provenance: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        records, provenance = build_arm(
            arm, suites[arm], dataset_path, expected_splits
        )
        per_arm[arm] = records
        arm_provenance[arm] = provenance
        path = out / f"oracle_powered_{arm}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    split_counts: dict[str, dict[str, int]] = {}
    for arm, records in per_arm.items():
        counts: dict[str, int] = {}
        for record in records:
            counts[record["split"]] = counts.get(record["split"], 0) + 1
        split_counts[arm] = counts

    with open(FROZEN_POLICY_PATH, encoding="utf-8") as handle:
        frozen_policy = json.load(handle)

    provenance_doc = {
        "phase": args.phase,
        "artifact": "cheap-first need oracle (powered set)",
        "oracle_version": ORACLE_VERSION,
        "oracle_rule": (
            "oracle_escalate = (recall@5(dense) - recall@5(bm25)) >= EPSILON"
        ),
        "oracle_epsilon": ORACLE_EPSILON,
        "primary_quality_metric": "recall_at_5",
        "secondary_quality_metric": "mrr",
        "latency_model": (
            "adaptive = bm25.retrieval_latency_ms + dense.retrieval_latency_ms "
            "iff escalated; decision cost recorded as 0 (threshold comparison)"
        ),
        "cost_model": (
            "primary currency is the Dense-call rate (embedding calls avoided); "
            "latency is the secondary proxy; no USD figure exists for "
            "retrieval-only runs"
        ),
        "dataset_path": str(dataset_path),
        "dataset_sha256": compute_file_sha256(dataset_path),
        "frozen_rule_source": str(FROZEN_POLICY_PATH),
        "frozen_rule_sha256": compute_file_sha256(FROZEN_POLICY_PATH),
        "frozen_rule": frozen_policy["frozen_rule"],
        "signal_source": (
            "BM25-state signals computed directly from powered bm25 "
            "rows/traces with the Phase 9 functions; T0 features from "
            "powered adaptive-arm traces"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": arm_provenance,
        "split_counts": split_counts,
    }
    (out / "oracle_provenance.json").write_text(
        json.dumps(provenance_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance_doc, indent=2, sort_keys=True))
    return provenance_doc


if __name__ == "__main__":
    main()
