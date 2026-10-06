#!/usr/bin/env python3
"""Phase 10 oracle build -- per-query dense-vs-hybrid escalation truth.

Reads only persisted artifacts, runs no retrieval, calls no provider:

* dense + hybrid E1 rows from `experiments/phase8/combined/p8a_e1`
  (shipping, `phase8_after`) and `.../p8b_e1` (replication, `phase8_before`) --
  the exact row set Phase 9's signal tables were built over;
* hybrid traces for the per-query non-embedding stage clocks;
* the frozen Phase 9 signal tables for `distinct_doc_ratio@dense`;
* the frozen ADR-027 calibration/test split from `phase7_eval_v1.jsonl`.

For each query it records both arms' outcomes, the incremental escalation
latency (hybrid's non-embedding stages; the embedding is already paid at the
dense stage), and the frozen oracle label from `docs/phases/phase-10.md` §4:

    oracle_escalate(q) = YES  iff  recall@5(hybrid,q) - recall@5(dense,q) >= 0.01

Integrity gates (any failure aborts the build rather than writing a partial
oracle): both arms 107/107 `ok` (the `load_arm` contamination check),
identical query sets across dense/hybrid/signal rows, frozen retrieval config
(top_k=10, candidate_k=20, rrf_k=60), DDR recomputed from the dense rows
matching the frozen signal-table value on every query, and the 47/60 split
coverage.

Outputs (under `experiments/phase10/`):
`oracle_phase8_after.jsonl`, `oracle_phase8_before.jsonl`, and
`oracle_provenance.json` carrying every reproducibility field §17 requires.
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
from adaptive_rag.evaluation.dispersion import EPSILON  # noqa: E402
from adaptive_rag.evaluation.escalation import (  # noqa: E402
    DDR_K,
    ESCALATION_ORACLE_VERSION,
    incremental_latency_ms,
    oracle_label,
    recompute_ddr,
)
from adaptive_rag.evaluation.signals import read_signal_table  # noqa: E402
import compare_phase8_arms  # noqa: E402

ORACLE_VERSION = ESCALATION_ORACLE_VERSION
ORACLE_EPSILON = EPSILON

PHASE8_DIR = REPO_ROOT / "experiments" / "phase8" / "combined"
PHASE9_DIR = REPO_ROOT / "experiments" / "phase9"
DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "phase7_eval_v1.jsonl"
OUT_DIR = REPO_ROOT / "experiments" / "phase10"

ARMS = ("after", "before")
SUITE_BY_ARM = {"after": "p8a_e1", "before": "p8b_e1"}

EXPECTED_TOP_K = 10
EXPECTED_CANDIDATE_K = 20
EXPECTED_RRF_K = 60
EXPECTED_SPLIT_COUNTS = {"calibration": 47, "test": 60}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _hybrid_stage_latencies(suite_root: Path) -> dict[str, dict[str, Any]]:
    """Per-query hybrid trace metadata clocks needed for the latency model."""
    matches = sorted(
        suite_root.glob("*__E1_baseline_comparison__hybrid/traces.jsonl")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one hybrid trace set under {suite_root}, "
            f"found {[m.parent.name for m in matches]}"
        )
    stages: dict[str, dict[str, Any]] = {}
    with open(matches[0], encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            meta = (trace.get("retrieval") or {}).get("retrieval_metadata") or {}
            stages[str(trace["example_id"])] = {
                "bm25_latency_ms": meta.get("bm25_latency_ms"),
                "fusion_latency_ms": meta.get("fusion_latency_ms"),
                "search_latency_ms": meta.get("search_latency_ms"),
                "hybrid_retrieval_latency_ms": trace.get("retrieval_latency_ms"),
            }
    return stages


def _retrieval_config(suite_root: Path, system: str) -> dict[str, Any]:
    matches = sorted(
        suite_root.glob(f"*__E1_baseline_comparison__{system}/config.json")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {system} config under {suite_root}"
        )
    return _read_json(matches[0])["retrieval"]


def _manifest(suite_root: Path, system: str) -> dict[str, Any]:
    matches = sorted(
        suite_root.glob(f"*__E1_baseline_comparison__{system}/manifest.json")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {system} manifest under {suite_root}"
        )
    manifest = _read_json(matches[0])
    return {
        "config_hash": manifest.get("config_hash"),
        "corpus_version": manifest.get("corpus_version"),
        "git_commit": manifest.get("git_commit"),
        "trace_count": manifest.get("trace_count"),
    }


def build_arm(arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join one corpus arm's rows, traces, signal values, and splits."""
    suite_root = PHASE8_DIR / SUITE_BY_ARM[arm]
    # Strict: any non-ok trace aborts (the Phase 8 measurement-integrity
    # lesson -- a silently short arm inverts conclusions).
    arm_rows = compare_phase8_arms.load_arm(suite_root, ["dense", "hybrid"])
    dense = {r["query_id"]: r for r in arm_rows["dense"]}
    hybrid = {r["query_id"]: r for r in arm_rows["hybrid"]}
    stages = _hybrid_stage_latencies(suite_root)
    signals = {
        r["query_id"]: r
        for r in read_signal_table(
            PHASE9_DIR / f"signal_table_phase8_{arm}.jsonl"
        )
    }

    dense_ids, hybrid_ids, signal_ids = (
        set(dense), set(hybrid), set(signals),
    )
    if not (dense_ids == hybrid_ids == signal_ids):
        raise ValueError(
            f"{arm}: query sets disagree: dense={len(dense_ids)} "
            f"hybrid={len(hybrid_ids)} signal={len(signal_ids)}; "
            f"only-in-one={sorted((dense_ids ^ hybrid_ids) | (dense_ids ^ signal_ids))[:8]}"
        )

    hybrid_cfg = _retrieval_config(suite_root, "hybrid")
    dense_cfg = _retrieval_config(suite_root, "dense")
    for name, got, want in (
        ("hybrid.top_k", hybrid_cfg.get("top_k"), EXPECTED_TOP_K),
        ("hybrid.candidate_k", hybrid_cfg.get("candidate_k"), EXPECTED_CANDIDATE_K),
        ("hybrid.rrf_k", hybrid_cfg.get("rrf_k"), EXPECTED_RRF_K),
        ("dense.top_k", dense_cfg.get("top_k"), EXPECTED_TOP_K),
    ):
        if got != want:
            raise ValueError(
                f"{arm}: frozen retrieval config moved: {name}={got}, "
                f"expected {want}; refusing to join the wrong runs"
            )

    split = {
        json.loads(line)["example_id"]: json.loads(line)["split"]
        for line in open(DATASET_PATH, encoding="utf-8")
        if line.strip()
    }

    records: list[dict[str, Any]] = []
    fallback_count = 0
    for query_id in sorted(dense_ids):
        if query_id not in split:
            raise ValueError(f"{arm}: {query_id} has no split in the dataset")
        d_row, h_row = dense[query_id], hybrid[query_id]
        if d_row["status"] != "ok" or h_row["status"] != "ok":
            # Unreachable when load_arm is strict, kept as a second lock:
            # a failed retrieval has no metric values, so it must never
            # become an oracle row.
            raise ValueError(f"{arm}: {query_id} carries a non-ok row")
        frozen_ddr = signals[query_id]["distinct_doc_ratio"]["dense"]
        fresh_ddr = round(
            recompute_ddr(d_row["retrieved_document_ids"], DDR_K), 6
        )
        if abs(float(frozen_ddr) - fresh_ddr) > 1e-9:
            raise ValueError(
                f"{arm}: {query_id} DDR mismatch: frozen={frozen_ddr} "
                f"recomputed={fresh_ddr}; the oracle is not joining the rows "
                "Phase 9 measured"
            )
        delta_r5 = float(h_row["recall_at_5"]) - float(d_row["recall_at_5"])
        delta_mrr = float(h_row["mrr"]) - float(d_row["mrr"])
        stage = stages[query_id]
        incr, method = incremental_latency_ms(
            bm25_latency_ms=stage["bm25_latency_ms"],
            fusion_latency_ms=stage["fusion_latency_ms"],
            search_latency_ms=stage["search_latency_ms"],
            fallback_latency_ms=stage["hybrid_retrieval_latency_ms"],
        )
        fallback_count += method == "full_fallback"
        records.append(
            {
                "query_id": query_id,
                "corpus_arm": arm,
                "split": split[query_id],
                "category": d_row["category"],
                "ddr_dense": float(frozen_ddr),
                "dense_recall_at_5": float(d_row["recall_at_5"]),
                "hybrid_recall_at_5": float(h_row["recall_at_5"]),
                "delta_recall_at_5": delta_r5,
                "dense_mrr": float(d_row["mrr"]),
                "hybrid_mrr": float(h_row["mrr"]),
                "delta_mrr": delta_mrr,
                "dense_hit_at_5": float(d_row["hit_at_5"]),
                "hybrid_hit_at_5": float(h_row["hit_at_5"]),
                "dense_latency_ms": float(d_row["retrieval_latency_ms"]),
                "hybrid_latency_ms": float(h_row["retrieval_latency_ms"]),
                "incremental_latency_ms": incr,
                "latency_method": method,
                "oracle_escalate": bool(
                    oracle_label(delta_r5, ORACLE_EPSILON)
                ),
                "dense_retrieved_document_ids": list(
                    d_row["retrieved_document_ids"]
                ),
            }
        )

    provenance = {
        "suite": str(suite_root.relative_to(REPO_ROOT)),
        "dense_manifest": _manifest(suite_root, "dense"),
        "hybrid_manifest": _manifest(suite_root, "hybrid"),
        "dense_top_k": dense_cfg.get("top_k"),
        "hybrid_top_k": hybrid_cfg.get("top_k"),
        "hybrid_candidate_k": hybrid_cfg.get("candidate_k"),
        "hybrid_rrf_k": hybrid_cfg.get("rrf_k"),
        "n_queries": len(records),
        "n_oracle_yes": sum(1 for r in records if r["oracle_escalate"]),
        "latency_fallback_count": fallback_count,
        "ddr_integrity": f"recomputed==frozen on {len(records)}/{len(records)}",
    }
    return records, provenance


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Build the frozen Phase 10 per-query escalation oracle."
    )
    parser.add_argument(
        "--out", default=str(OUT_DIR), help="output directory"
    )
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
                f"{arm}: split coverage {counts} != frozen "
                f"{EXPECTED_SPLIT_COUNTS}"
            )

    provenance_doc = {
        "phase": 10,
        "artifact": "escalation oracle",
        "oracle_version": ORACLE_VERSION,
        "oracle_rule": (
            "oracle_escalate = (recall@5(hybrid) - recall@5(dense)) >= EPSILON"
        ),
        "oracle_epsilon": ORACLE_EPSILON,
        "primary_quality_metric": "recall_at_5",
        "secondary_quality_metric": "mrr",
        "latency_model": (
            "incremental = bm25_latency_ms + fusion_latency_ms + "
            "search_latency_ms from the hybrid trace; fallback = full "
            "hybrid retrieval_latency_ms when a stage clock is missing"
        ),
        "cost_model": (
            "latency is the project's established computational proxy; no "
            "USD figure exists for retrieval-only runs; API call count is "
            "flat (1 embedding call per query under every policy)"
        ),
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "dataset_sha256": compute_file_sha256(DATASET_PATH),
        "signal_source": (
            "experiments/phase9/signal_table_phase8_{after,before}.jsonl "
            "(frozen; recomputed from dense rows and matched exactly)"
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
