#!/usr/bin/env python3
"""Phase 10 policy evaluation -- the single test-split pass.

Reads `frozen_policy.json` (written by `phase10_signal_analysis.py` from
calibration rows only) and the frozen oracle tables, then evaluates exactly
once on the ADR-027 `test` split (n=60) per corpus arm:

  A  dense-only (stop everywhere);
  B  always-escalate (the hybrid rows);
  C  adaptive (frozen threshold policy);
  D  random escalation at C's test escalation rate (1000 seeded draws).

Per query it records what dense did, what escalation would have done, what the
policy chose, and whether that choice was correct (true/false/missed
escalation, correct stop). Paired statistics follow the project convention
(`statistics.select_test` chooses the test per metric kind; Holm step-down
within each pre-registered family; seeded bootstrap CIs). The replication arm
reuses the frozen policy with no refitting.

Outputs (under `experiments/phase10/`): `policy_eval_{after,before}.json`
(aggregate + statistics + verdict inputs) and `policy_rows_{arm}.jsonl`
(per-query decisions). The test split is opened here for the first time in
Phase 10.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation import stats  # noqa: E402
from adaptive_rag.evaluation.escalation import (  # noqa: E402
    ESCALATION_ORACLE_VERSION,
    RANDOM_ABLATION_DRAWS,
    RANDOM_ABLATION_SEED,
    apply_threshold,
    classify_decision,
    confusion_counts,
    random_escalation,
)
from adaptive_rag.evaluation.statistics import (  # noqa: E402
    DEFAULT_SEED,
    select_test,
)

OUT_DIR = REPO_ROOT / "experiments" / "phase10"
ARMS = ("after", "before")
PRIMARY_QUALITY = "recall_at_5"
SECONDARY_QUALITY = "mrr"
ALPHA = 0.05


def _load_test(arm: str) -> list[dict[str, Any]]:
    path = OUT_DIR / f"oracle_phase8_{arm}.jsonl"
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    test = [r for r in rows if r["split"] == "test"]
    if not test:
        raise ValueError(f"{arm}: no test rows in {path}")
    return test


def _paired_stats(
    treatment: list[float], baseline: list[float], metric: str
) -> dict[str, Any]:
    """Paired comparison under the project test-selection convention."""
    differences = [float(t) - float(b) for t, b in zip(treatment, baseline)]
    selection = select_test(metric, differences)
    full = stats.compare_paired(treatment, baseline, seed=DEFAULT_SEED)
    if selection["p_value"] is None:
        # Latency/cost kinds select the sign test as primary; the paired
        # result carries its exact p-value.
        selection = dict(selection)
        selection["p_value"] = full["wilcoxon"]["sign_test_p"]
    mean_ci = stats.bootstrap_ci(
        differences, statistic=np.mean, seed=DEFAULT_SEED
    )
    return {
        "metric": metric,
        "n_pairs": len(differences),
        "selected_test": selection["selected_test"],
        "test_reason": selection["reason"],
        "p_value": selection["p_value"],
        "p_value_method": selection["p_value_method"],
        "sign_test_p": selection["sign_test_p"],
        "wilcoxon_p_value": selection["wilcoxon_p_value"],
        "mean_difference": round(float(np.mean(differences)), 6),
        "mean_difference_ci95": [mean_ci[0], mean_ci[1]],
        "median_difference": full["effect"]["median_difference"],
        "median_difference_ci95": [
            full["effect"]["ci_low"],
            full["effect"]["ci_high"],
        ],
        "wins": full["effect"]["wins"],
        "ties": full["effect"]["ties"],
        "losses": full["effect"]["losses"],
    }


def _holmize(comparisons: list[dict[str, Any]]) -> None:
    """Holm-adjust the selected p-values of one pre-registered family."""
    pvalues = [c["p_value"] if c["p_value"] is not None else 1.0
               for c in comparisons]
    for comparison, adjusted in zip(
        comparisons, stats.holm_bonferroni(pvalues)
    ):
        comparison["p_adjusted_holm"] = adjusted
        comparison["significant_at_0_05"] = bool(adjusted < ALPHA)


def evaluate_arm(
    arm: str, policy: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One test-split evaluation of arms A/B/C/D under the frozen policy."""
    records = _load_test(arm)
    threshold = float(policy["threshold"])
    direction = str(policy["direction"])

    decisions = [
        apply_threshold(r["ddr_dense"], threshold, direction) for r in records
    ]
    labels = [bool(r["oracle_escalate"]) for r in records]

    per_query: list[dict[str, Any]] = []
    dense_q = {PRIMARY_QUALITY: [], SECONDARY_QUALITY: []}
    hybrid_q = {PRIMARY_QUALITY: [], SECONDARY_QUALITY: []}
    adaptive_q = {PRIMARY_QUALITY: [], SECONDARY_QUALITY: []}
    dense_lat, adaptive_lat = [], []
    for record, decision in zip(records, decisions):
        chosen = "hybrid" if decision else "dense"
        per_query.append(
            {
                "query_id": record["query_id"],
                "corpus_arm": arm,
                "split": "test",
                "ddr_dense": record["ddr_dense"],
                "oracle_escalate": bool(record["oracle_escalate"]),
                "policy_escalated": decision,
                "classification": classify_decision(
                    bool(record["oracle_escalate"]), decision
                ),
                "dense_recall_at_5": record["dense_recall_at_5"],
                "hybrid_recall_at_5": record["hybrid_recall_at_5"],
                "adaptive_recall_at_5": (
                    record["hybrid_recall_at_5"]
                    if decision
                    else record["dense_recall_at_5"]
                ),
                "dense_mrr": record["dense_mrr"],
                "hybrid_mrr": record["hybrid_mrr"],
                "adaptive_mrr": (
                    record["hybrid_mrr"] if decision else record["dense_mrr"]
                ),
                "dense_latency_ms": record["dense_latency_ms"],
                "adaptive_latency_ms": (
                    record["dense_latency_ms"]
                    + (record["incremental_latency_ms"] if decision else 0.0)
                ),
                "incremental_latency_ms": (
                    record["incremental_latency_ms"] if decision else 0.0
                ),
                "chosen_arm": chosen,
            }
        )
        for quality in (PRIMARY_QUALITY, SECONDARY_QUALITY):
            dense_q[quality].append(float(record[f"dense_{quality}"]))
            hybrid_q[quality].append(float(record[f"hybrid_{quality}"]))
            adaptive_q[quality].append(
                float(record[f"{'hybrid' if decision else 'dense'}_{quality}"])
            )
        dense_lat.append(float(record["dense_latency_ms"]))
        adaptive_lat.append(
            float(record["dense_latency_ms"])
            + (float(record["incremental_latency_ms"]) if decision else 0.0)
        )

    n = len(records)
    n_esc = sum(1 for d in decisions if d)
    esc_rate = n_esc / n
    rate_ci = stats.bootstrap_ci(
        [1.0 if d else 0.0 for d in decisions],
        statistic=np.mean,
        seed=DEFAULT_SEED,
    )

    # Pre-registered primary family: adaptive vs each fixed arm on recall@5.
    primary = [
        {"comparison": "adaptive_vs_dense_only", **_paired_stats(
            adaptive_q[PRIMARY_QUALITY], dense_q[PRIMARY_QUALITY],
            PRIMARY_QUALITY)},
        {"comparison": "adaptive_vs_always_escalate", **_paired_stats(
            adaptive_q[PRIMARY_QUALITY], hybrid_q[PRIMARY_QUALITY],
            PRIMARY_QUALITY)},
    ]
    _holmize(primary)
    # Pre-registered secondary family: same pair on MRR.
    secondary = [
        {"comparison": "adaptive_vs_dense_only", **_paired_stats(
            adaptive_q[SECONDARY_QUALITY], dense_q[SECONDARY_QUALITY],
            SECONDARY_QUALITY)},
        {"comparison": "adaptive_vs_always_escalate", **_paired_stats(
            adaptive_q[SECONDARY_QUALITY], hybrid_q[SECONDARY_QUALITY],
            SECONDARY_QUALITY)},
    ]
    _holmize(secondary)
    # Latency is descriptive (sign test primary per the project convention,
    # reported through the paired effect's win/tie/loss counts and CIs).
    latency = _paired_stats(adaptive_lat, dense_lat, "retrieval_latency_ms")

    ablation = random_escalation(
        records,
        rate=esc_rate,
        n_draws=RANDOM_ABLATION_DRAWS,
        seed=RANDOM_ABLATION_SEED,
        quality=PRIMARY_QUALITY,
    )
    adaptive_mean_q = float(np.mean(adaptive_q[PRIMARY_QUALITY]))
    draws = ablation["mean_quality_values"]
    ablation["adaptive_mean_quality"] = adaptive_mean_q
    ablation["p_random_ge_adaptive"] = float(
        sum(1 for v in draws if v >= adaptive_mean_q) / len(draws)
    )

    def _mean(values: list[float]) -> float:
        return float(np.mean(values))

    result = {
        "corpus_arm": arm,
        "split": "test",
        "n": n,
        "policy": {
            "threshold": threshold,
            "direction": direction,
            "quality_metric": policy["quality_metric"],
        },
        "n_escalated": n_esc,
        "escalation_rate": esc_rate,
        "escalation_rate_ci95": [rate_ci[0], rate_ci[1]],
        "n_oracle_yes": sum(1 for lab in labels if lab),
        "confusion": confusion_counts(decisions, labels),
        "means": {
            "dense_only_recall_at_5": _mean(dense_q[PRIMARY_QUALITY]),
            "always_escalate_recall_at_5": _mean(hybrid_q[PRIMARY_QUALITY]),
            "adaptive_recall_at_5": _mean(adaptive_q[PRIMARY_QUALITY]),
            "dense_only_mrr": _mean(dense_q[SECONDARY_QUALITY]),
            "always_escalate_mrr": _mean(hybrid_q[SECONDARY_QUALITY]),
            "adaptive_mrr": _mean(adaptive_q[SECONDARY_QUALITY]),
            "dense_only_latency_ms": _mean(dense_lat),
            "adaptive_latency_ms": _mean(adaptive_lat),
            # Descriptive efficiency (recall per second of retrieval
            # latency): the project's quality-vs-cost framing as a rate, not
            # a dollars figure that does not exist for retrieval-only runs.
            "recall_per_second": {
                "dense_only": _mean(dense_q[PRIMARY_QUALITY])
                / (_mean(dense_lat) / 1000.0),
                "always_escalate": _mean(hybrid_q[PRIMARY_QUALITY])
                / (
                    sum(
                        float(r["hybrid_latency_ms"]) for r in records
                    )
                    / n
                    / 1000.0
                ),
                "adaptive": _mean(adaptive_q[PRIMARY_QUALITY])
                / (_mean(adaptive_lat) / 1000.0),
            },
        },
        "primary_family_recall_at_5": primary,
        "secondary_family_mrr": secondary,
        "latency_adaptive_vs_dense": latency,
        "random_ablation": {
            k: v
            for k, v in ablation.items()
            if k not in ("mean_quality_values", "mean_latency_ms_values")
        },
        "random_ablation_mean_quality_distribution": ablation[
            "mean_quality_values"
        ],
    }
    return result, per_query


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Phase 10 single test-split policy evaluation."
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "frozen_policy.json", encoding="utf-8") as handle:
        policy = json.load(handle)

    evaluations: dict[str, Any] = {}
    for arm in ARMS:
        result, per_query = evaluate_arm(arm, policy)
        evaluations[arm] = result
        (out / f"policy_eval_{arm}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with open(out / f"policy_rows_{arm}.jsonl", "w",
                   encoding="utf-8") as handle:
            for row in per_query:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    # Verdict inputs, computed mechanically from the frozen mapping in
    # docs/phases/phase-10.md §10. The verdict itself is written in the
    # results report; these booleans are the auditable inputs to it.
    shipping = evaluations["after"]
    c_vs_a = shipping["primary_family_recall_at_5"][0]
    ci_low, ci_high = c_vs_a["mean_difference_ci95"]
    replication = evaluations["before"]["primary_family_recall_at_5"][0]
    verdict_inputs = {
        "adaptive_beats_dense_ci_excludes_zero": bool(ci_low > 0),
        "adaptive_minus_dense_mean": c_vs_a["mean_difference"],
        "adaptive_minus_dense_ci95": [ci_low, ci_high],
        "escalation_rate": shipping["escalation_rate"],
        "escalation_well_under_100pct": bool(
            shipping["escalation_rate"] < 0.80
        ),
        "replication_same_direction": bool(
            replication["mean_difference"] > 0
        ),
        "replication_mean_difference": replication["mean_difference"],
    }
    doc = {
        "phase": 10,
        "artifact": "test-split policy evaluation",
        "oracle_version": ESCALATION_ORACLE_VERSION,
        "policy": policy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": evaluations,
        "verdict_inputs": verdict_inputs,
    }
    (out / "policy_evaluation.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {arm: {
            "means": evaluations[arm]["means"],
            "escalation_rate": evaluations[arm]["escalation_rate"],
            "confusion": evaluations[arm]["confusion"],
            "primary": [
                {k: c[k] for k in (
                    "comparison", "selected_test", "p_value",
                    "p_adjusted_holm", "significant_at_0_05",
                    "mean_difference", "mean_difference_ci95",
                )}
                for c in evaluations[arm]["primary_family_recall_at_5"]
            ],
            "random_ablation": evaluations[arm]["random_ablation"],
        } for arm in ARMS} | {"verdict_inputs": verdict_inputs},
        indent=2, sort_keys=True,
    ))
    return doc


if __name__ == "__main__":
    main()
