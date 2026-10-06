#!/usr/bin/env python3
"""Phase 14 signal analysis -- calibration-only rule freeze.

Reads ONLY calibration rows from the frozen Phase 14 oracle tables (the
`after` arm for selection, the `before` arm for the consistency check).
This script has no code path that loads the test split: the loader
asserts `split == "calibration"` on every row it touches, so test data
cannot leak into selection even by accident.

For each of the six pre-registered candidate signals
(`docs/phases/phase-14.md` §7) it builds the label-free decile grid,
sweeps the primary and control directions, and applies the frozen 1-SE
selection rule (`cheapfirst.select_rule`). The winner is checked for
cross-arm consistency on `before`-calibration (frozen rule, no refit):
its adaptive mean recall@5 must reach the BM25-only mean, else the
verdict is capped at INSUFFICIENT EVIDENCE per the preregistration.

Outputs (under `experiments/phase14/`): `signal_analysis_after.json`
(full sweeps + selection + descriptive ranking metrics),
`signal_analysis_before_calib.json` (consistency check), and
`frozen_policy.json` (signal, threshold, direction, provenance). The test
split is first opened by `phase14_evaluate_policy.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.cheapfirst import (  # noqa: E402
    CANDIDATE_SIGNALS,
    NEED_ORACLE_VERSION,
    decile_thresholds,
    roc_auc,
    select_rule,
    signal_value,
    sweep_rule,
)

OUT_DIR = REPO_ROOT / "experiments" / "phase14"
PRIMARY_QUALITY = "recall_at_5"


def _load_calibration(arm: str) -> list[dict[str, Any]]:
    """Calibration rows only; the test split is never opened here."""
    path = OUT_DIR / f"oracle_phase8_{arm}.jsonl"
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 107:
        raise ValueError(f"{arm}: expected 107 oracle rows, found {len(rows)}")
    calib = [r for r in rows if r.get("split") == "calibration"]
    if len(calib) != 47:
        raise ValueError(
            f"{arm}: expected 47 calibration rows, found {len(calib)}"
        )
    return calib


def _describe_signal(
    records: list[dict[str, Any]], name: str
) -> dict[str, Any]:
    """Descriptive operating numbers for one signal (no selection)."""
    values = [signal_value(r["signals"], name) for r in records]
    labels = [bool(r["oracle_escalate"]) for r in records]
    present = [(v, lab) for v, lab in zip(values, labels) if v is not None]
    if not present:
        return {
            "signal": name, "n": len(records), "n_present": 0,
            "n_missing": len(records),
        }
    p_vals = [v for v, lab in present if lab]
    n_vals = [v for v, lab in present if not lab]
    ordered = sorted(v for v, _ in present)
    # Direction-aware AUC: the primary direction defines "higher score =
    # more likely to need Dense", so LOW-primary signals are negated first.
    # Descriptive only (≤6 calibration positives on the shipping arm).
    primary = next(s["direction"] for s in CANDIDATE_SIGNALS if s["name"] == name)
    oriented = [v if primary == "high" else -v for v in ordered]
    oriented_labels = [lab for _, lab in sorted(present, key=lambda p: p[0])]
    return {
        "signal": name,
        "n": len(records),
        "n_present": len(present),
        "n_missing": len(records) - len(present),
        "n_oracle_yes": sum(labels),
        "mean_yes": sum(p_vals) / len(p_vals) if p_vals else None,
        "mean_no": sum(n_vals) / len(n_vals) if n_vals else None,
        "min": ordered[0],
        "max": ordered[-1],
        "roc_auc_oriented_descriptive": roc_auc(oriented, oriented_labels),
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Phase 14 calibration-only signal analysis."
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calib_after = _load_calibration("after")
    calib_before = _load_calibration("before")

    descriptions = {
        spec["name"]: _describe_signal(calib_after, spec["name"])
        for spec in CANDIDATE_SIGNALS
    }

    # Sweep every candidate in both directions on after-calibration.
    sweeps: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for spec in CANDIDATE_SIGNALS:
        name = spec["name"]
        grid = decile_thresholds(
            [signal_value(r["signals"], name) for r in calib_after]
        )
        if not grid:
            raise ValueError(f"{name}: empty threshold grid on calibration")
        sweeps[name] = {
            "primary": sweep_rule(
                calib_after, signal=name, thresholds=grid,
                direction=spec["direction"], quality=PRIMARY_QUALITY,
            ),
            "control": sweep_rule(
                calib_after, signal=name, thresholds=grid,
                direction=(
                    "low" if spec["direction"] == "high" else "high"
                ),
                quality=PRIMARY_QUALITY,
            ),
        }

    # Family selection, pooled across signals exactly as the preregistration
    # defines "family": all primary-direction rules form one family (the
    # Phase 10 analogue pooled one signal's thresholds; here the direction
    # defines the family). Control precedence mirrors Phase 10's
    # choose_direction: the control family wins only if its best exceeds the
    # primary best by more than the primary best's own SE.
    pooled_primary: list[dict[str, Any]] = []
    pooled_control: list[dict[str, Any]] = []
    for pair in sweeps.values():
        pooled_primary.extend(pair["primary"])
        pooled_control.extend(pair["control"])
    primary_choice = select_rule(pooled_primary)
    if primary_choice["degenerate"]:
        # A degenerate primary family cannot be rescued by the control
        # family: report it, do not shop for a better-looking direction.
        frozen = {
            "signal": primary_choice["signal"],
            "threshold": primary_choice["threshold"],
            "direction": primary_choice["direction"],
            "degenerate": True,
            "selection_note": primary_choice["reason"],
        }
        frozen_row = next(
            r for r in pooled_primary
            if r["signal"] == frozen["signal"]
            and r["threshold"] == frozen["threshold"]
            and r["direction"] == frozen["direction"]
        )
        control_wins = False
    else:
        primary_best = max(
            pooled_primary, key=lambda r: float(r["mean_quality"])
        )
        control_best = max(
            pooled_control, key=lambda r: float(r["mean_quality"])
        )
        bar = float(primary_best["mean_quality"])
        se = primary_best["quality_stderror"]
        control_wins = (
            se is not None
            and float(control_best["mean_quality"]) > bar + float(se)
        )
        if control_wins:
            control_choice = select_rule(pooled_control)
            frozen = {
                "signal": control_choice["signal"],
                "threshold": control_choice["threshold"],
                "direction": control_choice["direction"],
                "degenerate": bool(control_choice["degenerate"]),
                "selection_note": (
                    "control family best exceeds primary best by >1 SE "
                    f"(control {control_best['mean_quality']:.4f} vs "
                    f"primary {bar:.4f} + SE {se})"
                ),
            }
        else:
            frozen = {
                "signal": primary_choice["signal"],
                "threshold": primary_choice["threshold"],
                "direction": primary_choice["direction"],
                "degenerate": False,
                "selection_note": primary_choice["reason"],
            }
        frozen_row = next(
            r for r in pooled_primary + pooled_control
            if r["signal"] == frozen["signal"]
            and r["threshold"] == frozen["threshold"]
            and r["direction"] == frozen["direction"]
        )

    # Cross-arm consistency on before-calibration (frozen rule, no refit).
    before_rows = sweep_rule(
        calib_before,
        signal=str(frozen["signal"]),
        thresholds=[float(frozen["threshold"])],
        direction=str(frozen["direction"]),
        quality=PRIMARY_QUALITY,
    )
    assert len(before_rows) == 1
    before_row = before_rows[0]
    bm25_before = sum(
        float(r["bm25_recall_at_5"]) for r in calib_before
    ) / len(calib_before)
    consistent = (
        float(before_row["mean_quality"]) >= bm25_before
        and not bool(frozen["degenerate"])
    )
    frozen["consistency_before_calib"] = {
        "adaptive_mean_quality": before_row["mean_quality"],
        "bm25_only_mean_quality": bm25_before,
        "escalation_rate": before_row["escalation_rate"],
        "confusion": {
            k: before_row[k] for k in ("tp", "fp", "fn", "tn")
        },
        "consistent": consistent,
    }
    if not consistent:
        frozen["verdict_cap"] = (
            "consistency check failed or rule degenerate: verdict capped at "
            "INSUFFICIENT EVIDENCE per docs/phases/phase-14.md §8"
        )

    policy_doc = {
        "phase": 14,
        "oracle_version": NEED_ORACLE_VERSION,
        "quality_metric": PRIMARY_QUALITY,
        "frozen_rule": frozen,
        "frozen_row_after_calib": frozen_row,
        "provenance": {
            "selected_on": "after/calibration (n=47)",
            "consistency_on": "before/calibration (n=47, no refit)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    (out / "frozen_policy.json").write_text(
        json.dumps(policy_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    analysis = {
        "phase": 14,
        "oracle_version": NEED_ORACLE_VERSION,
        "arm": "after",
        "split": "calibration",
        "n": len(calib_after),
        "n_oracle_yes": sum(1 for r in calib_after if r["oracle_escalate"]),
        "signal_descriptions": descriptions,
        "family_selection": {
            "n_primary_rules_swept": len(pooled_primary),
            "n_control_rules_swept": len(pooled_control),
            "control_wins": control_wins,
        },
        "frozen_rule": frozen,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "signal_analysis_after.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before_doc = {
        "phase": 14,
        "oracle_version": NEED_ORACLE_VERSION,
        "arm": "before",
        "split": "calibration",
        "n": len(calib_before),
        "n_oracle_yes": sum(1 for r in calib_before if r["oracle_escalate"]),
        "frozen_rule_applied": frozen,
        "operating_row": before_row,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "signal_analysis_before_calib.json").write_text(
        json.dumps(before_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(policy_doc, indent=2, sort_keys=True))
    return policy_doc


if __name__ == "__main__":
    main()
