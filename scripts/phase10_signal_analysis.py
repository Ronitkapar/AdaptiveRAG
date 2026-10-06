#!/usr/bin/env python3
"""Phase 10 signal analysis -- calibration split only, then freeze the policy.

Reads the frozen oracle tables built by `phase10_build_oracle.py` and answers,
on the ADR-027 `calibration` split (n=47) and nothing else:

  (A) DDR distribution for oracle_escalate YES vs NO;
  (B) ranking ability: ROC-AUC, PR-AUC (average precision), and
      Spearman(DDR, delta_recall@5);
  (C) the pre-registered threshold sweep in both direction families.

The frozen selection rule (`escalation.select_threshold`) picks the policy on
the shipping arm's calibration rows; the control family wins only if its best
calibration mean recall@5 exceeds the primary family's best by more than the
primary best's own standard error (declared in `docs/phases/phase-10.md` §8 so
the direction choice cannot become a post-hoc flip). The winner is written to
`frozen_policy.json`.

Leakage control is structural: this script filters to `split ==
"calibration"` immediately after loading and has no code path that opens test
rows. The `test` split is first touched by `phase10_evaluate_policy.py`, after
the policy file exists.
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

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.agreement import spearman  # noqa: E402
from adaptive_rag.evaluation.escalation import (  # noqa: E402
    DIRECTIONS,
    ESCALATION_ORACLE_VERSION,
    THRESHOLDS,
    average_precision,
    roc_auc,
    select_threshold,
    sweep_thresholds,
)

OUT_DIR = REPO_ROOT / "experiments" / "phase10"
ARMS = ("after", "before")
PRIMARY_QUALITY = "recall_at_5"
SECONDARY_QUALITY = "mrr"


def _load_calibration(arm: str) -> list[dict[str, Any]]:
    path = OUT_DIR / f"oracle_phase8_{arm}.jsonl"
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    calibration = [r for r in rows if r["split"] == "calibration"]
    if not calibration:
        raise ValueError(f"{arm}: no calibration rows in {path}")
    if any(r["split"] != "calibration" for r in calibration):
        raise ValueError(f"{arm}: split filter leaked a non-calibration row")
    return calibration


def analyze_arm(arm: str) -> dict[str, Any]:
    """Descriptive signal/oracle analysis on one arm's calibration rows."""
    records = _load_calibration(arm)
    ddrs = [float(r["ddr_dense"]) for r in records]
    labels = [bool(r["oracle_escalate"]) for r in records]
    deltas = [float(r["delta_recall_at_5"]) for r in records]
    n_pos = sum(1 for lab in labels if lab)

    by_class: dict[str, dict[str, Any]] = {}
    for name, wanted in (("yes", True), ("no", False)):
        vals = [d for d, lab in zip(ddrs, labels) if lab is wanted]
        by_class[name] = {
            "n": len(vals),
            "ddr_value_counts": dict(sorted(Counter(vals).items())),
            "ddr_mean": (
                sum(vals) / len(vals) if vals else None
            ),
        }

    sweeps = {
        direction: {
            PRIMARY_QUALITY: sweep_thresholds(
                records, thresholds=THRESHOLDS, direction=direction,
                quality=PRIMARY_QUALITY,
            ),
            SECONDARY_QUALITY: sweep_thresholds(
                records, thresholds=THRESHOLDS, direction=direction,
                quality=SECONDARY_QUALITY,
            ),
        }
        for direction in DIRECTIONS
    }
    return {
        "corpus_arm": arm,
        "split": "calibration",
        "n": len(records),
        "n_oracle_yes": n_pos,
        "ddr_by_oracle_class": by_class,
        "roc_auc_ddr_vs_oracle": roc_auc(ddrs, labels),
        "pr_auc_ddr_vs_oracle": average_precision(ddrs, labels),
        "spearman_ddr_vs_delta_recall_at_5": spearman(ddrs, deltas),
        "sweeps": sweeps,
    }


def choose_direction(
    primary_sweep: list[dict[str, Any]],
    control_sweep: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Apply the frozen direction rule and return (direction, selection)."""
    primary_pick = select_threshold(primary_sweep)
    if primary_pick["degenerate"]:
        # A degenerate primary family cannot be rescued by the control
        # family: report it, do not shop for a better-looking direction.
        return "high", primary_pick
    primary_best = max(primary_sweep, key=lambda r: float(r["mean_quality"]))
    control_best = max(control_sweep, key=lambda r: float(r["mean_quality"]))
    bar = float(primary_best["mean_quality"])
    se = primary_best["quality_stderror"]
    if se is not None and float(control_best["mean_quality"]) > bar + se:
        return "low", select_threshold(control_sweep)
    return "high", primary_pick


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Phase 10 calibration-only signal analysis + policy freeze."
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    analyses = {arm: analyze_arm(arm) for arm in ARMS}
    for arm, analysis in analyses.items():
        path = out / f"signal_analysis_{arm}.json"
        path.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Selection happens on the shipping arm's calibration rows only; the
    # replication arm is analysed descriptively above and never selected on.
    shipping = analyses["after"]
    direction, pick = choose_direction(
        shipping["sweeps"]["high"][PRIMARY_QUALITY],
        shipping["sweeps"]["low"][PRIMARY_QUALITY],
    )
    policy = {
        "phase": 10,
        "threshold": pick["threshold"],
        "direction": direction,
        "quality_metric": PRIMARY_QUALITY,
        "degenerate": pick["degenerate"],
        "selection_reason": pick["reason"],
        "selected_on": "phase8_after calibration (n=47)",
        "oracle_version": ESCALATION_ORACLE_VERSION,
        "threshold_family": list(THRESHOLDS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "frozen_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        arm: {
            "n": a["n"],
            "n_oracle_yes": a["n_oracle_yes"],
            "roc_auc": a["roc_auc_ddr_vs_oracle"],
            "pr_auc": a["pr_auc_ddr_vs_oracle"],
            "spearman_ddr_delta": a[
                "spearman_ddr_vs_delta_recall_at_5"
            ],
        }
        for arm, a in analyses.items()
    }
    summary["frozen_policy"] = policy
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
