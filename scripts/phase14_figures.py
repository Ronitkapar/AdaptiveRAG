#!/usr/bin/env python3
"""Phase 14 figures -- four pre-registered panels from written artifacts.

1. `signal_vs_value`: frozen signal vs oracle delta on after-calibration,
   by oracle group (helps / harms / ties), frozen threshold marked.
2. `operating_curve`: test quality vs escalation rate over the frozen
   signal's full threshold sweep (post-freeze descriptive; no selection),
   frozen operating point marked.
3. `quality_vs_callrate`: BM25-only / dense-only / adaptive / random-mean
   on both test arms (quality vs dense-call rate).
4. `waterfall`: per-query adaptive-minus-BM25 deltas on after-test,
   oracle positives marked.

All data comes from written artifacts under `experiments/phase14/`, never
from in-memory intermediates. Manifest with sha256 is written alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.cheapfirst import (  # noqa: E402
    apply_rule,
    decile_thresholds,
    signal_value,
    sweep_rule,
)

OUT_DIR = REPO_ROOT / "experiments" / "phase14"
EPSILON = 0.01


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _group(record: dict[str, Any]) -> str:
    delta = float(record["delta_recall_at_5"])
    if delta >= EPSILON:
        return "helps"
    if delta <= -EPSILON:
        return "harms"
    return "ties"


def fig_signal_vs_value(fig_dir: Path) -> Path:
    oracle = [r for r in _read_jsonl(OUT_DIR / "oracle_phase8_after.jsonl")
              if r["split"] == "calibration"]
    policy = json.loads((OUT_DIR / "frozen_policy.json").read_text())[
        "frozen_rule"
    ]
    signal = str(policy["signal"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"helps": ("o", "tab:green"), "harms": ("x", "tab:red"),
               "ties": (".", "0.6")}
    for group, (marker, color) in markers.items():
        xs = [signal_value(r["signals"], signal) for r in oracle
              if _group(r) == group]
        ys = [float(r["delta_recall_at_5"]) for r in oracle
              if _group(r) == group]
        present = [(x, y) for x, y in zip(xs, ys) if x is not None]
        if present:
            ax.scatter([p[0] for p in present], [p[1] for p in present],
                       marker=marker, color=color, label=group, s=36)
    ax.axhline(EPSILON, color="k", linestyle="--", linewidth=1)
    ax.axhline(-EPSILON, color="k", linestyle=":", linewidth=1)
    ax.axvline(float(policy["threshold"]), color="tab:blue", linestyle="-",
               linewidth=1,
               label=f"frozen t={policy['threshold']:.3g} ({policy['direction']})")
    ax.set_xlabel(f"{signal} (pre-Dense)")
    ax.set_ylabel("dense - bm25 recall@5 (oracle delta)")
    ax.set_title("Frozen signal vs escalation value (after/calibration)")
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "signal_vs_value.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_operating_curve(fig_dir: Path) -> Path:
    policy = json.loads((OUT_DIR / "frozen_policy.json").read_text())[
        "frozen_rule"
    ]
    signal = str(policy["signal"])
    test = [r for r in _read_jsonl(OUT_DIR / "oracle_phase8_after.jsonl")
            if r["split"] == "test"]
    calib = [r for r in _read_jsonl(OUT_DIR / "oracle_phase8_after.jsonl")
             if r["split"] == "calibration"]
    grid = decile_thresholds(
        [signal_value(r["signals"], signal) for r in calib]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for direction, style in (("high", "o-"), ("low", "s--")):
        rows = sweep_rule(test, signal=signal, thresholds=grid,
                          direction=direction)
        ax.plot([r["escalation_rate"] for r in rows],
                [r["mean_quality"] for r in rows], style,
                label=f"test/{direction}", markersize=4)
    eval_after = json.load(open(OUT_DIR / "policy_eval_after.json"))
    ax.scatter([eval_after["escalation_rate"]],
               [eval_after["means"]["adaptive_recall_at_5"]],
               color="tab:blue", s=80, zorder=5, label="frozen operating point")
    ax.axhline(eval_after["means"]["bm25_only_recall_at_5"], color="0.5",
               linestyle=":", label="BM25-only")
    ax.axhline(eval_after["means"]["always_escalate_recall_at_5"],
               color="k", linestyle=":", label="dense-only")
    ax.set_xlabel("escalation (dense-call) rate")
    ax.set_ylabel("mean recall@5")
    ax.set_title("Operating curve on after/test (post-freeze, descriptive)")
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "operating_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_quality_vs_callrate(fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, marker in (("after", "o"), ("before", "s")):
        ev = json.load(open(OUT_DIR / f"policy_eval_{arm}.json"))
        m = ev["means"]
        ax.scatter([0.0], [m["bm25_only_recall_at_5"]], color="tab:blue",
                   marker=marker, s=60, label=f"{arm} BM25-only" if arm == "after" else None)
        ax.scatter([1.0], [m["always_escalate_recall_at_5"]], color="k",
                   marker=marker, s=60)
        ax.scatter([ev["escalation_rate"]], [m["adaptive_recall_at_5"]],
                   color="tab:green", marker=marker, s=80)
        ax.scatter([ev["escalation_rate"]],
                   [ev["random_ablation"]["mean_quality"]],
                   color="tab:orange", marker=marker, s=60)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="BM25-only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k", label="dense-only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green", label="adaptive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", label="random-at-rate mean"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="k", label="circle=after square=before"),
    ])
    ax.set_xlabel("dense-call rate")
    ax.set_ylabel("mean recall@5 (test)")
    ax.set_title("Quality vs dense-call rate")
    fig.tight_layout()
    path = fig_dir / "quality_vs_callrate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_waterfall(fig_dir: Path) -> Path:
    rows = [r for r in _read_jsonl(OUT_DIR / "policy_rows_after.jsonl")]
    oracle = {r["query_id"]: r
              for r in _read_jsonl(OUT_DIR / "oracle_phase8_after.jsonl")}
    deltas = sorted(
        (float(r["adaptive_recall_at_5"]) - float(r["bm25_recall_at_5"]),
         r["query_id"]) for r in rows
    )
    colors = ["tab:green" if oracle[q]["oracle_escalate"] else "0.55"
              for _, q in deltas]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(deltas)), [d for d, _ in deltas], color=colors)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel("test queries, ordered by adaptive-minus-BM25 delta")
    ax.set_ylabel("adaptive - BM25 recall@5")
    ax.set_title("Per-query adaptive effect (after/test; green = oracle YES)")
    fig.tight_layout()
    path = fig_dir / "waterfall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Phase 14 figures.")
    parser.add_argument("--out", default=str(OUT_DIR / "figures"))
    args = parser.parse_args()
    fig_dir = Path(args.out)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_signal_vs_value(fig_dir), fig_operating_curve(fig_dir),
             fig_quality_vs_callrate(fig_dir), fig_waterfall(fig_dir)]
    manifest = {
        "phase": 14,
        "figures": [
            {"file": p.name,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths
        ],
    }
    (fig_dir / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
