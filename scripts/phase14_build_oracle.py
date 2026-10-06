#!/usr/bin/env python3
"""Phase 14 oracle build -- per-query BM25-vs-Dense need truth.

Reads only persisted artifacts, runs no retrieval, calls no provider:

* bm25 + dense E1 rows from `experiments/phase8/combined/p8a_e1`
  (shipping, `phase8_after`) and `.../p8b_e1` (replication, `phase8_before`);
* bm25 traces for the top-1 native score and the score-geometry integrity
  recompute (gap/slope via the same functions Phase 9 used);
* the frozen Phase 9 signal tables for the BM25-state scalars (DDR, gap,
  slope), each re-verified before use;
* the frozen adaptive-arm traces for the T0 query features
  (`complexity_score`, `content_term_count`, numeric fields only);
* the frozen ADR-027 calibration/test split from `phase7_eval_v1.jsonl`.

For each query it records both arms' outcomes, the six pre-Dense signals,
the adaptive latency model (BM25 clock always; dense clock only when the
policy escalates -- the embedding call is the currency), and the frozen
oracle label from `docs/phases/phase-14.md` §4:

    oracle_escalate(q) = YES  iff  recall@5(dense,q) - recall@5(bm25,q) >= 0.01

Integrity gates (any failure aborts rather than writing a partial oracle):
both arms 107/107 `ok` (the `load_arm` contamination check), identical
query sets across bm25/dense/signal/query-feature sources, frozen
retrieval config (top_k=10 both arms), DDR/gap/slope recomputed from the
frozen rows/traces matching the signal-table values on every query, query
features present on every query, and the 47/60 split coverage.

Outputs (under `experiments/phase14/`):
`oracle_phase8_after.jsonl`, `oracle_phase8_before.jsonl`, and
`oracle_provenance.json` carrying every reproducibility field.
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
from adaptive_rag.evaluation.signals import read_signal_table  # noqa: E402
import compare_phase8_arms  # noqa: E402

ORACLE_VERSION = NEED_ORACLE_VERSION
ORACLE_EPSILON = EPSILON

PHASE8_DIR = REPO_ROOT / "experiments" / "phase8" / "combined"
PHASE9_DIR = REPO_ROOT / "experiments" / "phase9"
DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "phase7_eval_v1.jsonl"
OUT_DIR = REPO_ROOT / "experiments" / "phase14"

ARMS = ("after", "before")
SUITE_BY_ARM = {"after": "p8a_e1", "before": "p8b_e1"}

EXPECTED_TOP_K = 10
EXPECTED_SPLIT_COUNTS = {"calibration": 47, "test": 60}
FLOAT_TOL = 1e-9

#: T0 query-feature representatives (docs/phases/phase-14.md §6): numeric
#: fields of the frozen adaptive-trace `routing.features`, nothing else.
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
    """Frozen BM25 traces keyed by query id (example_id == query_id)."""
    path = _single(
        "*__E1_baseline_comparison__bm25/traces.jsonl", suite_root, "bm25 traces"
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
    """Frozen T0 query features from the adaptive arm's persisted routing."""
    path = _single(
        "*__E1_baseline_comparison__adaptive/traces.jsonl",
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
        f"*__E1_baseline_comparison__{system}/config.json",
        suite_root,
        f"{system} config",
    )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["retrieval"]


def _manifest(suite_root: Path, system: str) -> dict[str, Any]:
    path = _single(
        f"*__E1_baseline_comparison__{system}/manifest.json",
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


def _check_close(label: str, frozen: Any, fresh: Any) -> None:
    if frozen is None and fresh is None:
        return
    if frozen is None or fresh is None:
        raise ValueError(f"{label}: null mismatch: frozen={frozen} fresh={fresh}")
    if abs(float(frozen) - float(fresh)) > FLOAT_TOL:
        raise ValueError(
            f"{label}: mismatch: frozen={frozen} recomputed={fresh}"
        )


def build_arm(arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join one corpus arm's rows, traces, signal values, and splits."""
    suite_root = PHASE8_DIR / SUITE_BY_ARM[arm]
    # Strict: any non-ok trace aborts (the Phase 8 measurement-integrity
    # lesson -- a silently short arm inverts conclusions).
    arm_rows = compare_phase8_arms.load_arm(suite_root, ["bm25", "dense"])
    bm25 = {r["query_id"]: r for r in arm_rows["bm25"]}
    dense = {r["query_id"]: r for r in arm_rows["dense"]}
    bm25_traces = _bm25_traces(suite_root)
    qfeat = _query_features(suite_root)
    signals = {
        r["query_id"]: r
        for r in read_signal_table(
            PHASE9_DIR / f"signal_table_phase8_{arm}.jsonl"
        )
    }

    sets = {
        "bm25": set(bm25),
        "dense": set(dense),
        "bm25_traces": set(bm25_traces),
        "signal": set(signals),
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
        for line in open(DATASET_PATH, encoding="utf-8")
        if line.strip()
    }

    records: list[dict[str, Any]] = []
    missing_counts = {"bm25_gap": 0, "bm25_slope": 0}
    for query_id in sorted(sets["bm25"]):
        if query_id not in split:
            raise ValueError(f"{arm}: {query_id} has no split in the dataset")
        b_row, d_row = bm25[query_id], dense[query_id]
        if b_row["status"] != "ok" or d_row["status"] != "ok":
            # Unreachable when load_arm is strict, kept as a second lock.
            raise ValueError(f"{arm}: {query_id} carries a non-ok row")
        frozen = signals[query_id]
        trace = bm25_traces[query_id]
        scores = scores_from_trace(trace)

        # Integrity: every frozen BM25-state scalar is recomputed from the
        # frozen rows/traces it was built over; any drift aborts the build.
        fresh_ddr = round(
            distinct_doc_ratio(b_row["retrieved_document_ids"], SIGNAL_K), 6
        )
        _check_close(
            f"{arm}:{query_id} ddr_bm25",
            frozen["distinct_doc_ratio"]["bm25"],
            fresh_ddr,
        )
        fresh_gap = score_gap_top1_top2(scores, SIGNAL_K)
        _check_close(
            f"{arm}:{query_id} bm25_gap",
            frozen["score_gap_top1_top2"]["bm25"],
            round(fresh_gap, 6) if fresh_gap is not None else None,
        )
        fresh_slope = score_decay_slope(scores, SIGNAL_K)
        _check_close(
            f"{arm}:{query_id} bm25_slope",
            frozen["score_decay_slope"]["bm25"],
            round(fresh_slope, 6) if fresh_slope is not None else None,
        )
        results = ((trace.get("retrieval") or {}).get("results") or [])
        if not results or results[0].get("score") is None:
            raise ValueError(
                f"{arm}: {query_id} bm25 trace has no top-1 native score"
            )
        top1 = float(results[0]["score"])

        gap = frozen["score_gap_top1_top2"]["bm25"]
        slope = frozen["score_decay_slope"]["bm25"]
        missing_counts["bm25_gap"] += gap is None
        missing_counts["bm25_slope"] += slope is None
        rule_signals = {
            "ddr_bm25": float(frozen["distinct_doc_ratio"]["bm25"]),
            "bm25_top1": top1,
            "bm25_gap": None if gap is None else float(gap),
            "bm25_slope": None if slope is None else float(slope),
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
        "suite": str(suite_root.relative_to(REPO_ROOT)),
        "bm25_manifest": _manifest(suite_root, "bm25"),
        "dense_manifest": _manifest(suite_root, "dense"),
        "bm25_top_k": bm25_cfg.get("top_k"),
        "dense_top_k": dense_cfg.get("top_k"),
        "n_queries": len(records),
        "n_oracle_yes": sum(1 for r in records if r["oracle_escalate"]),
        "missing_signal_counts": missing_counts,
        "signal_integrity": (
            f"ddr/gap/slope recomputed==frozen on "
            f"{len(records)}/{len(records)}"
        ),
        "query_feature_integrity": (
            f"T0 features present on {len(records)}/{len(records)}"
        ),
    }
    return records, provenance


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Build the frozen Phase 14 cheap-first need oracle."
    )
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_arm: dict[str, list[dict[str, Any]]] = {}
    arm_provenance: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        records, provenance = build_arm(arm)
        per_arm[arm] = records
        arm_provenance[arm] = provenance
        path = out / f"oracle_phase8_{arm}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    split_counts: dict[str, dict[str, int]] = {}
    for arm, records in per_arm.items():
        counts = {"calibration": 0, "test": 0}
        for record in records:
            counts[record["split"]] += 1
        split_counts[arm] = counts
        if counts != EXPECTED_SPLIT_COUNTS:
            raise ValueError(
                f"{arm}: split coverage {counts} != frozen {EXPECTED_SPLIT_COUNTS}"
            )

    provenance_doc = {
        "phase": 14,
        "artifact": "cheap-first need oracle",
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
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "dataset_sha256": compute_file_sha256(DATASET_PATH),
        "signal_source": (
            "experiments/phase9/signal_table_phase8_{after,before}.jsonl "
            "(frozen BM25-state scalars; recomputed from frozen rows/traces "
            "and matched exactly) + frozen bm25/adaptive traces"
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
