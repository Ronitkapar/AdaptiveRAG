#!/usr/bin/env python3
"""Phase 15 one-pass powered evaluation of the frozen Phase 14 rule.

Reads the powered oracle rows (`oracle_powered_{arm}.jsonl`, all split
`test`), applies the exact frozen rule from
`experiments/phase14/frozen_policy.json` (Option A -- no re-estimation),
and evaluates per the frozen protocol (`docs/phases/phase-15.md` §§15.9
-15.15):

* per arm: C-vs-A and C-vs-B paired tests on recall@5 (Holm within the
  pair) + bootstrap CIs; same for MRR; descriptive latency; random
  ablation at C's arm rate (1000 draws, seed 20250101);
* combined: randomization test over both arms (P_comb) + cluster
  bootstrap CI on the combined adaptive-minus-BM25 mean;
* mechanical evaluation of the frozen SUCCESS/FAILURE mapping (§§15.13
  -15.15) from the numbers above (the results document reports it; the
  script only computes it deterministically).

Runs no retrieval, calls no provider, fits nothing.
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
from adaptive_rag.evaluation.cheapfirst import (  # noqa: E402
    NEED_ORACLE_VERSION,
    RANDOM_ABLATION_DRAWS,
    RANDOM_ABLATION_SEED,
    apply_rule,
    classify_decision,
    confusion_counts,
    random_escalation,
    signal_value,
)
from adaptive_rag.evaluation.powered import (  # noqa: E402
    POWERED_BOOTSTRAP_SEED,
    POWERED_N_DRAWS,
    cluster_bootstrap_ci,
    combined_randomization_test,
)
from adaptive_rag.evaluation.statistics import (  # noqa: E402
    DEFAULT_SEED,
    select_test,
)

OUT_DIR = REPO_ROOT / "experiments" / "phase15"
FROZEN_POLICY_PATH = REPO_ROOT / "experiments" / "phase14" / "frozen_policy.json"
ARMS = ("after", "before")
PRIMARY_QUALITY = "recall_at_5"
SECONDARY_QUALITY = "mrr"
ALPHA = 0.05
DENSE_MARGIN = 0.02
MAX_RATE = 0.50


def _load_test(arm: str, oracle_dir: Path) -> list[dict[str, Any]]:
    path = oracle_dir / f"oracle_powered_{arm}.jsonl"
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
        differences, statistic=np.mean, seed=POWERED_BOOTSTRAP_SEED
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
    pvalues = [
        c["p_value"] if c["p_value"] is not None else 1.0 for c in comparisons
    ]
    adjusted = stats.holm_bonferroni(pvalues)
    for comparison, adj in zip(comparisons, adjusted):
        comparison["p_adjusted_holm"] = adjusted
        comparison["significant_at_0_05"] = bool(adj < ALPHA)


def evaluate_arm(
    arm: str,
    signal: str,
    threshold: float,
    direction: str,
    oracle_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One-pass powered evaluation of the frozen rule on one corpus arm."""
    records = _load_test(arm, oracle_dir)
    decisions = [
        apply_rule(signal_value(r["signals"], signal), threshold, direction)
        for r in records
    ]
    labels = [bool(r["oracle_escalate"]) for r in records]

    adaptive_q: dict[str, list[float]] = {
        PRIMARY_QUALITY: [], SECONDARY_QUALITY: [],
    }
    bm25_q: dict[str, list[float]] = {
        PRIMARY_QUALITY: [], SECONDARY_QUALITY: [],
    }
    dense_q: dict[str, list[float]] = {
        PRIMARY_QUALITY: [], SECONDARY_QUALITY: [],
    }
    bm25_lat: list[float] = []
    adaptive_lat: list[float] = []
    per_query: list[dict[str, Any]] = []
    for record, decision, label in zip(records, decisions, labels):
        for metric in (PRIMARY_QUALITY, SECONDARY_QUALITY):
            b_val = float(record[f"bm25_{metric}"])
            d_val = float(record[f"dense_{metric}"])
            bm25_q[metric].append(b_val)
            dense_q[metric].append(d_val)
            adaptive_q[metric].append(d_val if decision else b_val)
        b_lat = float(record["bm25_latency_ms"])
        bm25_lat.append(b_lat)
        adaptive_lat.append(
            b_lat + (float(record["dense_latency_ms"]) if decision else 0.0)
        )
        per_query.append(
            {
                "query_id": record["query_id"],
                "corpus_arm": arm,
                "oracle_escalate": label,
                "policy_escalate": decision,
                "outcome": classify_decision(label, decision),
                "bm25_recall_at_5": float(record["bm25_recall_at_5"]),
                "dense_recall_at_5": float(record["dense_recall_at_5"]),
                "adaptive_recall_at_5": adaptive_q[PRIMARY_QUALITY][-1],
                "bm25_mrr": float(record["bm25_mrr"]),
                "dense_mrr": float(record["dense_mrr"]),
                "adaptive_mrr": adaptive_q[SECONDARY_QUALITY][-1],
                "adaptive_latency_ms": adaptive_lat[-1],
                "dense_call": decision,
            }
        )

    n = len(records)
    n_esc = sum(1 for d in decisions if d)
    esc_rate = n_esc / n
    rate_ci = stats.bootstrap_ci(
        [1.0 if d else 0.0 for d in decisions],
        statistic=np.mean,
        seed=POWERED_BOOTSTRAP_SEED,
    )

    # Pre-registered primary family: adaptive vs each fixed arm on recall@5.
    primary = [
        {"comparison": "adaptive_vs_bm25_only", **_paired_stats(
            adaptive_q[PRIMARY_QUALITY], bm25_q[PRIMARY_QUALITY],
            PRIMARY_QUALITY)},
        {"comparison": "adaptive_vs_always_escalate", **_paired_stats(
            adaptive_q[PRIMARY_QUALITY], dense_q[PRIMARY_QUALITY],
            PRIMARY_QUALITY)},
    ]
    _holmize(primary)
    # Pre-registered secondary family: same pair on MRR.
    secondary = [
        {"comparison": "adaptive_vs_bm25_only", **_paired_stats(
            adaptive_q[SECONDARY_QUALITY], bm25_q[SECONDARY_QUALITY],
            SECONDARY_QUALITY)},
        {"comparison": "adaptive_vs_always_escalate", **_paired_stats(
            adaptive_q[SECONDARY_QUALITY], dense_q[SECONDARY_QUALITY],
            SECONDARY_QUALITY)},
    ]
    _holmize(secondary)
    # Latency is descriptive (sign test primary per the project convention,
    # reported through the paired effect's win/tie/loss counts and CIs).
    latency = _paired_stats(adaptive_lat, bm25_lat, "retrieval_latency_ms")

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
            "signal": signal,
            "threshold": threshold,
            "direction": direction,
            "quality_metric": PRIMARY_QUALITY,
        },
        "n_escalated": n_esc,
        "escalation_rate": esc_rate,
        "escalation_rate_ci95": [rate_ci[0], rate_ci[1]],
        "dense_calls_avoided_fraction": 1.0 - esc_rate,
        "n_oracle_yes": sum(1 for lab in labels if lab),
        "confusion": confusion_counts(decisions, labels),
        "means": {
            "bm25_only_recall_at_5": _mean(bm25_q[PRIMARY_QUALITY]),
            "always_escalate_recall_at_5": _mean(dense_q[PRIMARY_QUALITY]),
            "adaptive_recall_at_5": _mean(adaptive_q[PRIMARY_QUALITY]),
            "bm25_only_mrr": _mean(bm25_q[SECONDARY_QUALITY]),
            "always_escalate_mrr": _mean(dense_q[SECONDARY_QUALITY]),
            "adaptive_mrr": _mean(adaptive_q[SECONDARY_QUALITY]),
            "bm25_only_latency_ms": _mean(bm25_lat),
            "adaptive_latency_ms": _mean(adaptive_lat),
        },
        "primary_family_recall_at_5": primary,
        "secondary_family_mrr": secondary,
        "latency_adaptive_vs_bm25": latency,
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


def _apply_mapping(evaluations: dict[str, Any], combined: dict[str, Any]) -> dict[str, Any]:
    """Mechanical evaluation of the frozen verdict mapping (§§15.13-15.15)."""
    after, before = evaluations["after"], evaluations["before"]
    comb_adapt = (
        after["means"]["adaptive_recall_at_5"]
        + before["means"]["adaptive_recall_at_5"]
    ) / 2.0
    comb_dense = (
        after["means"]["always_escalate_recall_at_5"]
        + before["means"]["always_escalate_recall_at_5"]
    ) / 2.0
    comb_bm25 = (
        after["means"]["bm25_only_recall_at_5"]
        + before["means"]["bm25_only_recall_at_5"]
    ) / 2.0
    delta_bm25_after = (
        after["means"]["adaptive_recall_at_5"]
        - after["means"]["bm25_only_recall_at_5"]
    )
    delta_bm25_before = (
        before["means"]["adaptive_recall_at_5"]
        - before["means"]["bm25_only_recall_at_5"]
    )
    p_after = after["random_ablation"]["p_random_ge_adaptive"]
    p_before = before["random_ablation"]["p_random_ge_adaptive"]
    p_comb = combined["randomization_test"]["p_comb"]
    ci = combined["cluster_bootstrap_ci"]

    success = {
        "(a) combined adaptive within 0.02 of combined dense": bool(
            comb_adapt >= comb_dense - DENSE_MARGIN
        ),
        "(b) escalation rate <= 50% on each arm": bool(
            after["escalation_rate"] <= MAX_RATE
            and before["escalation_rate"] <= MAX_RATE
        ),
        "(c) TP >= 1 on each arm": bool(
            after["confusion"]["tp"] >= 1 and before["confusion"]["tp"] >= 1
        ),
        "(d) P_comb < 0.05 and neither per-arm P >= 0.5": bool(
            p_comb < ALPHA and p_after < 0.5 and p_before < 0.5
        ),
        "(e) combined CI excludes 0 and per-arm benefit >= 0": bool(
            (ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0)
            and delta_bm25_after >= 0.0
            and delta_bm25_before >= 0.0
        ),
    }
    failure = {
        "(i) TP = 0 on both arms with rate > 5%": bool(
            after["confusion"]["tp"] == 0
            and before["confusion"]["tp"] == 0
            and after["escalation_rate"] > 0.05
            and before["escalation_rate"] > 0.05
        ),
        "(ii) combined adaptive < combined BM25-only": bool(
            comb_adapt < comb_bm25
        ),
        "(iii) P_comb >= 0.5": bool(p_comb >= 0.5),
        "(iv) per-arm P >= 0.5 on BOTH arms": bool(
            p_after >= 0.5 and p_before >= 0.5
        ),
    }
    if all(success.values()):
        verdict = "SUCCESS"
    elif any(failure.values()):
        verdict = "FAILURE"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
    return {
        "combined_means": {
            "adaptive_recall_at_5": comb_adapt,
            "always_escalate_recall_at_5": comb_dense,
            "bm25_only_recall_at_5": comb_bm25,
        },
        "per_arm_benefit_over_bm25": {
            "after": delta_bm25_after,
            "before": delta_bm25_before,
        },
        "success_conditions": success,
        "failure_conditions": failure,
        "mechanical_verdict": verdict,
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Phase 15 one-pass powered policy evaluation."
    )
    parser.add_argument("--oracle-dir", default=str(OUT_DIR))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    oracle_dir = Path(args.oracle_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(FROZEN_POLICY_PATH, encoding="utf-8") as handle:
        policy = json.load(handle)["frozen_rule"]
    signal = str(policy["signal"])
    threshold = float(policy["threshold"])
    direction = str(policy["direction"])
    if policy.get("degenerate"):
        raise ValueError(
            "frozen policy is degenerate; Phase 15 tests the Phase 14 rule "
            "as frozen and cannot proceed on a degenerate rule"
        )

    evaluations: dict[str, Any] = {}
    per_arm_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        result, per_query = evaluate_arm(
            arm, signal, threshold, direction, oracle_dir
        )
        evaluations[arm] = result
        per_arm_rows[arm] = per_query
        with open(out / f"policy_eval_{arm}.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
        with open(out / f"policy_rows_{arm}.jsonl", "w", encoding="utf-8") as fh:
            for row in per_query:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def _arm_quality(arm: str) -> tuple[list[float], list[float], list[float]]:
        rows = per_arm_rows[arm]
        return (
            [float(r["adaptive_recall_at_5"]) for r in rows],
            [float(r["bm25_recall_at_5"]) for r in rows],
            [float(r["dense_recall_at_5"]) for r in rows],
        )

    adapt_a, b_a, d_a = _arm_quality("after")
    adapt_b, b_b, d_b = _arm_quality("before")
    k_a = evaluations["after"]["n_escalated"]
    k_b = evaluations["before"]["n_escalated"]
    combined_test = combined_randomization_test(
        adaptive_after=adapt_a,
        adaptive_before=adapt_b,
        bm25_after=b_a,
        bm25_before=b_b,
        dense_after=d_a,
        dense_before=d_b,
        k_after=k_a,
        k_before=k_b,
        n_draws=POWERED_N_DRAWS,
        seed=RANDOM_ABLATION_SEED,
    )
    combined_ci = cluster_bootstrap_ci(
        adaptive_after=adapt_a,
        adaptive_before=adapt_b,
        bm25_after=b_a,
        bm25_before=b_b,
        seed=POWERED_BOOTSTRAP_SEED,
    )
    combined = {
        "randomization_test": combined_test,
        "cluster_bootstrap_ci": combined_ci,
    }
    (out / "policy_eval_combined.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    mapping = _apply_mapping(evaluations, combined)
    summary = {
        "phase": 15,
        "oracle_version": NEED_ORACLE_VERSION,
        "frozen_policy": policy,
        "frozen_policy_source": str(FROZEN_POLICY_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": {
            arm: {
                "n": evaluations[arm]["n"],
                "n_oracle_yes": evaluations[arm]["n_oracle_yes"],
                "escalation_rate": evaluations[arm]["escalation_rate"],
                "confusion": evaluations[arm]["confusion"],
                "adaptive_recall_at_5": evaluations[arm]["means"][
                    "adaptive_recall_at_5"
                ],
                "bm25_only_recall_at_5": evaluations[arm]["means"][
                    "bm25_only_recall_at_5"
                ],
                "always_escalate_recall_at_5": evaluations[arm]["means"][
                    "always_escalate_recall_at_5"
                ],
                "p_random_ge_adaptive": evaluations[arm]["random_ablation"][
                    "p_random_ge_adaptive"
                ],
            }
            for arm in ARMS
        },
        "combined": combined,
        "verdict_mapping": mapping,
    }
    (out / "policy_evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
