#!/usr/bin/env python3
"""Phase 14 policy evaluation -- the single test-split pass.

Reads `frozen_policy.json` (written by `phase14_signal_analysis.py` from
calibration rows only) and the frozen oracle tables, then evaluates
exactly once on the ADR-027 `test` split (n=60) per corpus arm:

  A  BM25-only (stop everywhere; zero embedding calls);
  B  always-escalate (the dense rows; an embedding call per query);
  C  adaptive (frozen cheap-first rule);
  D  random escalation at C's test escalation rate (1000 seeded draws).

Per query it records what BM25 did, what Dense would have done, what the
policy chose, and whether that choice was correct (true/false/missed
escalation, correct stop), plus the dense-call indicator and the latency
model (BM25 clock always; dense clock only when escalated). Paired
statistics follow the project convention (`statistics.select_test`;
Holm step-down within each pre-registered family; seeded bootstrap CIs).
The replication arm reuses the frozen policy with no refitting.

Outputs (under `experiments/phase14/`): `policy_eval_{after,before}.json`
(aggregate + statistics + verdict inputs) and `policy_rows_{arm}.jsonl`
(per-query decisions). The test split is opened here for the first time
in Phase 14.
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
from adaptive_rag.evaluation.statistics import (  # noqa: E402
    DEFAULT_SEED,
    select_test,
)

OUT_DIR = REPO_ROOT / "experiments" / "phase14"
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
    pvalues = [
        c["p_value"] if c["p_value"] is not None else 1.0 for c in comparisons
    ]
    adjusted = stats.holm_bonferroni(pvalues)
    for comparison, adj in zip(comparisons, adjusted):
        comparison["p_adjusted_holm"] = adjusted
        comparison["significant_at_0_05"] = bool(adj < ALPHA)


def evaluate_arm(
    arm: str, signal: str, threshold: float, direction: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One-pass test evaluation of the frozen rule on one corpus arm."""
    records = _load_test(arm)
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
        seed=DEFAULT_SEED,
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
            # Descriptive efficiency (recall per second of retrieval
            # latency): the project's quality-vs-cost framing as a rate, not
            # a dollars figure that does not exist for retrieval-only runs.
            "recall_per_second": {
                "bm25_only": _mean(bm25_q[PRIMARY_QUALITY])
                / (_mean(bm25_lat) / 1000.0),
                "always_escalate": _mean(dense_q[PRIMARY_QUALITY])
                / (
                    sum(float(r["dense_latency_ms"]) for r in records)
                    / n
                    / 1000.0
                ),
                "adaptive": _mean(adaptive_q[PRIMARY_QUALITY])
                / (_mean(adaptive_lat) / 1000.0),
            },
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


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Phase 14 single test-split policy evaluation."
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "frozen_policy.json", encoding="utf-8") as handle:
        policy = json.load(handle)["frozen_rule"]
    signal = str(policy["signal"])
    threshold = float(policy["threshold"])
    direction = str(policy["direction"])
    if policy.get("degenerate"):
        print(
            "WARNING: frozen policy is degenerate; evaluating anyway and "
            "reporting the degeneracy (no re-sweep per prereg §8.4)"
        )

    evaluations: dict[str, Any] = {}
    for arm in ARMS:
        result, per_query = evaluate_arm(arm, signal, threshold, direction)
        evaluations[arm] = result
        with open(out / f"policy_eval_{arm}.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
        with open(out / f"policy_rows_{arm}.jsonl", "w", encoding="utf-8") as fh:
            for row in per_query:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "phase": 14,
        "oracle_version": NEED_ORACLE_VERSION,
        "frozen_policy": policy,
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
    }
    (out / "policy_evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
