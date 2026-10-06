#!/usr/bin/env python3
"""Phase 11 mechanism analysis -- why escalation helps vs harms.

Reads only persisted artifacts (no retrieval, no provider): the Phase 10
oracle tables, the frozen Phase 9 signal tables, the phase8 E1 traces, and
the benchmark's relevance labels. Every query becomes one mechanism record
(group, three-state label, dense-confidence and disagreement features, and
the exact dense->hybrid top-5 change with source attribution); relevance
labels characterize *what happened* and never enter a candidate signal.

Discovery discipline (frozen in `docs/phases/phase-11.md`): hypotheses come
from calibration rows only; test rows appear solely as side-by-side
consistency descriptions. The bounded `separation_check` over six
pre-declared numeric features is a falsification demo -- perfect separation
on ~5 points is expected by chance and selects nothing. No threshold is
fitted anywhere in Phase 11.

Outputs (under `experiments/phase11/`): `mechanism_table_{after,before}.jsonl`
and `mechanism_summary.json` with full provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.hashing import compute_file_sha256  # noqa: E402
from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.mechanism import (  # noqa: E402
    MECHANISM_VERSION,
    build_mechanism_record,
    separation_check,
)
from adaptive_rag.evaluation.signals import read_signal_table  # noqa: E402

PHASE8_DIR = REPO_ROOT / "experiments" / "phase8" / "combined"
PHASE9_DIR = REPO_ROOT / "experiments" / "phase9"
PHASE10_DIR = REPO_ROOT / "experiments" / "phase10"
DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "phase7_eval_v1.jsonl"
OUT_DIR = REPO_ROOT / "experiments" / "phase11"

ARMS = ("after", "before")
SUITE_BY_ARM = {"after": "p8a_e1", "before": "p8b_e1"}
SYSTEMS = ("dense", "bm25", "hybrid")

#: The six pre-declared numeric features for the bounded separation check.
#: Fixed here so the check cannot shop for a separating column.
CHECK_FEATURES = (
    "ddr_dense",
    "dense_top1_score",
    "dense_score_gap",
    "dense_score_slope",
    "jaccard_bm25_dense",
    "union_concentration",
)

DESCRIPTIVE_FEATURES = CHECK_FEATURES + ("rank_correlation_bm25_dense",)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_traces(suite_root: Path, system: str) -> dict[str, dict[str, Any]]:
    matches = sorted(
        suite_root.glob(f"*__E1_baseline_comparison__{system}/traces.jsonl")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {system} trace set under {suite_root}"
        )
    traces = {}
    for row in _read_jsonl(matches[0]):
        traces[str(row["example_id"])] = row
    return traces


def _ordering(
    records: Sequence[dict[str, Any]], feature: str
) -> dict[str, Any]:
    """Descriptive group ordering for one feature: values and means by
    group, with None counted rather than dropped silently."""
    out: dict[str, Any] = {}
    for group in ("helps", "harms", "ties"):
        vals = [r[feature] for r in records if r["group"] == group]
        present = sorted(v for v in vals if v is not None)
        out[group] = {
            "n": len(vals),
            "n_missing": len(vals) - len(present),
            "values": present,
            "mean": (sum(present) / len(present)) if present else None,
        }
    return out


def build_arm(arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join one corpus arm's oracle, signal, trace, and label evidence."""
    suite_root = PHASE8_DIR / SUITE_BY_ARM[arm]
    oracle_rows = {
        r["query_id"]: r
        for r in _read_jsonl(PHASE10_DIR / f"oracle_phase8_{arm}.jsonl")
    }
    signal_rows = {
        r["query_id"]: r
        for r in read_signal_table(
            PHASE9_DIR / f"signal_table_phase8_{arm}.jsonl"
        )
    }
    traces = {
        system: _load_traces(suite_root, system) for system in SYSTEMS
    }
    relevant = {
        json.loads(line)["example_id"]: json.loads(line)["relevant_documents"]
        for line in open(DATASET_PATH, encoding="utf-8")
        if line.strip()
    }
    if not (
        set(oracle_rows) == set(signal_rows) == set(traces["dense"])
        and set(oracle_rows) == set(traces["bm25"]) == set(traces["hybrid"])
    ):
        raise ValueError(
            f"{arm}: query sets disagree across oracle/signal/trace inputs"
        )

    records = [
        build_mechanism_record(
            oracle_row=oracle_rows[qid],
            signal_row=signal_rows[qid],
            dense_trace=traces["dense"][qid],
            bm25_trace=traces["bm25"][qid],
            hybrid_trace=traces["hybrid"][qid],
            relevant_docs=relevant[qid],
        )
        for qid in sorted(oracle_rows)
    ]
    return records, {"n_queries": len(records)}


def summarize_arm(
    arm: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Counts, three-state labels, calibration orderings, test consistency,
    the bounded separation check, and full non-tie case blocks."""
    calibration = [r for r in records if r["split"] == "calibration"]
    test = [r for r in records if r["split"] == "test"]
    summary: dict[str, Any] = {
        "corpus_arm": arm,
        "n": len(records),
        "group_counts": dict(Counter(r["group"] for r in records)),
        "group_counts_calibration": dict(
            Counter(r["group"] for r in calibration)
        ),
        "group_counts_test": dict(Counter(r["group"] for r in test)),
        "state_counts": dict(Counter(r["state"] for r in records)),
        "calibration_orderings": {
            feature: _ordering(calibration, feature)
            for feature in DESCRIPTIVE_FEATURES
        },
        "test_consistency": {
            feature: _ordering(test, feature)
            for feature in DESCRIPTIVE_FEATURES
        },
        "separation_check_calibration": separation_check(
            calibration, CHECK_FEATURES
        ),
        "non_tie_cases": [
            {
                key: r[key]
                for key in (
                    "query_id", "split", "category", "group", "state",
                    "dense_recall_at_5", "hybrid_recall_at_5",
                    "delta_recall_at_5", "ddr_dense", "dense_top1_score",
                    "dense_score_gap", "dense_score_slope",
                    "jaccard_bm25_dense", "union_concentration",
                    "top1_agreement_bm25_dense", "dense_top_k",
                    "hybrid_top_k", "entered_docs", "left_docs",
                    "relevant_entered", "relevant_left", "entered_source",
                )
            }
            for r in records
            if r["group"] != "ties"
        ],
    }
    return summary


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Phase 11 per-query escalation mechanism analysis."
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms: dict[str, Any] = {}
    for arm in ARMS:
        records, info = build_arm(arm)
        path = out / f"mechanism_table_{arm}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        arms[arm] = summarize_arm(arm, records)
        arms[arm]["table_rows"] = info["n_queries"]

    summary = {
        "phase": 11,
        "artifact": "escalation mechanism analysis",
        "mechanism_version": MECHANISM_VERSION,
        "oracle_version": "phase10_escalation_v1 (reused unchanged)",
        "signal_source": "experiments/phase9/signal_table_phase8_{after,before}.jsonl (frozen)",
        "trace_source": "experiments/phase8/combined/p8{a,b}_e1 E1 traces",
        "dataset_sha256": compute_file_sha256(DATASET_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
    }
    (out / "mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = {
        arm: {
            "groups": a["group_counts"],
            "states": a["state_counts"],
            "separation_check": [
                {k: c[k] for k in ("feature", "separates", "direction")}
                for c in a["separation_check_calibration"]
            ],
        }
        for arm, a in arms.items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
