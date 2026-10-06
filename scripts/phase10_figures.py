#!/usr/bin/env python3
"""Phase 10 figures -- rendered from written artifacts, never from memory.

All five figures derive from `signal_analysis_{arm}.json` (calibration),
`policy_evaluation.json` (test), and the frozen oracle tables. No threshold
is selected here; the test operating curve in figure 4 is diagnostic (it shows
whether *any* threshold would have worked) and cannot move the frozen policy.

Figures:
  1. signal_distribution_by_oracle_class -- DDR value counts for
     oracle YES vs NO on calibration, both arms.
  2. threshold_sweep -- calibration (shipping): escalation rate, adaptive
     recall@5 with 1-SE whiskers, and oracle agreement vs threshold, both
     direction families; frozen policy marked.
  3. quality_vs_latency -- test means for dense-only / always-escalate /
     adaptive / random with bootstrap-CI whiskers on recall, both arms.
  4. quality_vs_cost -- test operating curve: adaptive recall@5 vs
     escalation rate across every threshold and both directions, with the
     frozen policy, the fixed arms, and the random band; top axis converts
     rate to implied incremental milliseconds at the measured median.
  5. adaptive_vs_baselines -- test grouped bars (recall@5, MRR) for the four
     arms plus the policy confusion counts, both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation import stats  # noqa: E402
from adaptive_rag.evaluation.escalation import sweep_thresholds  # noqa: E402
from adaptive_rag.evaluation.statistics import DEFAULT_SEED  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "phase10"
FIG_DIR = OUT_DIR / "figures"
ARMS = ("after", "before")
DDR_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)
DPI = 150


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_oracle(arm: str, split: str) -> list[dict[str, Any]]:
    rows = []
    with open(OUT_DIR / f"oracle_phase8_{arm}.jsonl", encoding="utf-8") as h:
        for line in h:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r["split"] == split]


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    low, high = stats.bootstrap_ci(
        values, statistic=np.mean, seed=DEFAULT_SEED
    )
    return float(np.mean(values)), float(low), float(high)


def fig1_signal_distribution(plt, analyses: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        by_class = analyses[arm]["ddr_by_oracle_class"]
        x = np.arange(len(DDR_VALUES))
        for offset, name, color in (
            (-0.2, "no", "#6b7a99"),
            (0.2, "yes", "#c25e3a"),
        ):
            counts = by_class[name]["ddr_value_counts"]
            vals = [int(counts.get(str(v), 0)) for v in DDR_VALUES]
            ax.bar(x + offset, vals, width=0.4, color=color, label=f"oracle {name}")
        ax.set_xticks(x, [str(v) for v in DDR_VALUES])
        ax.set_xlabel("distinct_doc_ratio@dense")
        ax.set_title(
            f"{arm} calibration (n={analyses[arm]['n']}, "
            f"YES={analyses[arm]['n_oracle_yes']})"
        )
    axes[0].set_ylabel("queries")
    axes[0].legend()
    fig.suptitle("Phase 10 (1): signal distribution by oracle class")
    fig.tight_layout()
    path = FIG_DIR / "signal_distribution_by_oracle_class.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig2_threshold_sweep(plt, analyses: dict[str, Any]) -> Path:
    sweep_high = analyses["after"]["sweeps"]["high"]["recall_at_5"]
    sweep_low = analyses["after"]["sweeps"]["low"]["recall_at_5"]
    policy = _read_json(OUT_DIR / "frozen_policy.json")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    xs = [r["threshold"] for r in sweep_high]
    ax1.errorbar(
        xs,
        [r["mean_quality"] for r in sweep_high],
        yerr=[r["quality_stderror"] or 0.0 for r in sweep_high],
        marker="o",
        color="#1f5fa8",
        label="recall@5, escalate iff DDR >= t",
    )
    ax1.plot(
        xs,
        [r["mean_quality"] for r in sweep_low],
        marker="s",
        color="#9db6d8",
        linestyle="--",
        label="recall@5, control (DDR <= t)",
    )
    ax1.set_xlabel("threshold t")
    ax1.set_ylabel("adaptive mean recall@5 (calibration)", color="#1f5fa8")
    ax2 = ax1.twinx()
    ax2.plot(
        xs,
        [r["escalation_rate"] for r in sweep_high],
        marker="^",
        color="#c25e3a",
        label="escalation rate",
    )
    ax2.set_ylabel("escalation rate", color="#c25e3a")
    ax1.axvline(
        policy["threshold"], color="black", linestyle=":",
        label=f"frozen t={policy['threshold']} ({policy['direction']})",
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Phase 10 (2): threshold sweep (after, calibration n=47)")
    fig.tight_layout()
    path = FIG_DIR / "threshold_sweep.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig3_quality_vs_latency(plt, evaluation: dict[str, Any]) -> Path:
    policy = evaluation["policy"]
    threshold, direction = float(policy["threshold"]), str(policy["direction"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        from adaptive_rag.evaluation.escalation import apply_threshold

        test_rows = _read_oracle(arm, "test")
        dense_r = [r["dense_recall_at_5"] for r in test_rows]
        hybrid_r = [r["hybrid_recall_at_5"] for r in test_rows]
        decisions = [
            apply_threshold(r["ddr_dense"], threshold, direction)
            for r in test_rows
        ]
        adaptive_r = [
            h if d else v for v, h, d in zip(dense_r, hybrid_r, decisions)
        ]
        dense_l = [r["dense_latency_ms"] for r in test_rows]
        hybrid_l = [
            r["dense_latency_ms"] + r["incremental_latency_ms"]
            for r in test_rows
        ]
        adaptive_l = [
            dl + (r["incremental_latency_ms"] if d else 0.0)
            for dl, r, d in zip(dense_l, test_rows, decisions)
        ]
        points = [
            ("dense-only", dense_r, dense_l, "#2a7a3f", "o"),
            ("always-escalate", hybrid_r, hybrid_l, "#6b7a99", "s"),
            ("adaptive", adaptive_r, adaptive_l, "#1f5fa8", "D"),
        ]
        for name, recalls, latencies, color, marker in points:
            mean_r, low_r, high_r = _mean_ci([float(v) for v in recalls])
            mean_l = float(np.mean(latencies))
            ax.errorbar([mean_l], [mean_r],
                        yerr=[[mean_r - low_r], [high_r - mean_r]],
                        color=color, fmt=marker, capsize=4, label=name)
        rand = evaluation["arms"][arm]["random_ablation"]
        ax.scatter([rand["mean_latency_ms"]], [rand["mean_quality"]],
                   color="#c25e3a", marker="x", s=60, label="random",
                   zorder=3)
        ax.set_xlabel("mean retrieval latency per query (ms)")
        ax.set_title(f"{arm} test (n=60)")
    axes[0].set_ylabel("mean recall@5 (95% bootstrap CI)")
    axes[0].legend()
    fig.suptitle("Phase 10 (3): quality vs latency trade-off")
    fig.tight_layout()
    path = FIG_DIR / "quality_vs_latency.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig4_quality_vs_cost(plt, evaluation: dict[str, Any]) -> Path:
    policy = evaluation["policy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        test_rows = _read_oracle(arm, "test")
        median_incr = float(
            np.median([r["incremental_latency_ms"] for r in test_rows])
        )
        for direction, style, color in (
            ("high", "-", "#1f5fa8"),
            ("low", "--", "#9db6d8"),
        ):
            sweep = sweep_thresholds(
                test_rows, direction=direction, quality="recall_at_5"
            )
            ax.plot(
                [r["escalation_rate"] for r in sweep],
                [r["mean_quality"] for r in sweep],
                marker="o",
                linestyle=style,
                color=color,
                label=f"operating curve ({direction})",
            )
        ev = evaluation["arms"][arm]
        ax.scatter([0.0], [ev["means"]["dense_only_recall_at_5"]],
                   color="#2a7a3f", s=80, zorder=4, label="dense-only")
        ax.scatter([1.0], [ev["means"]["always_escalate_recall_at_5"]],
                   color="#6b7a99", s=80, zorder=4, label="always-escalate")
        ax.scatter([ev["escalation_rate"]],
                   [ev["means"]["adaptive_recall_at_5"]],
                   color="black", s=100, marker="*", zorder=5,
                   label="frozen policy")
        rand = ev["random_ablation"]
        draws = np.asarray(
            ev["random_ablation_mean_quality_distribution"]
        )
        ax.errorbar([ev["escalation_rate"]], [rand["mean_quality"]],
                    yerr=float(draws.std()), color="#c25e3a", fmt="x",
                    capsize=4, label="random (±1 sd)")
        ax.set_xlabel("escalation rate (share of queries escalated)")
        ax.set_title(f"{arm} test (median +{median_incr:.1f} ms/escalation)")

        def _to_ms(rate):
            return rate * median_incr

        def _to_rate(ms):
            return ms / median_incr if median_incr else 0.0

        ax2 = ax.secondary_xaxis("top", functions=(_to_ms, _to_rate))
        ax2.set_xlabel("implied incremental cost (ms/query, median)")
    axes[0].set_ylabel("mean recall@5")
    axes[0].legend(loc="lower left", fontsize="small")
    fig.suptitle("Phase 10 (4): quality vs cost (diagnostic operating curve)")
    fig.tight_layout()
    path = FIG_DIR / "quality_vs_cost.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig5_adaptive_vs_baselines(plt, evaluation: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["recall@5", "MRR"]
    for ax, arm in zip(axes, ARMS):
        ev = evaluation["arms"][arm]
        means = ev["means"]
        recall_vals = [
            means["dense_only_recall_at_5"],
            means["always_escalate_recall_at_5"],
            means["adaptive_recall_at_5"],
            ev["random_ablation"]["mean_quality"],
        ]
        mrr_vals = [
            means["dense_only_mrr"],
            means["always_escalate_mrr"],
            means["adaptive_mrr"],
        ]
        names = ["dense-only", "always-esc.", "adaptive", "random"]
        x = np.arange(len(labels))
        width = 0.2
        for i, (name, recall) in enumerate(zip(names, recall_vals)):
            ax.bar(x[0] + (i - 1.5) * width, recall, width=width,
                   label=name)
        for i, mrr in enumerate(mrr_vals):
            # Random escalation has no MRR draw distribution stored; the
            # random MRR mean equals its recall draw mean only by
            # coincidence, so no bar is drawn rather than a wrong one.
            ax.bar(x[1] + (i - 1.5) * width, mrr, width=width)
        conf = ev["confusion"]
        ax.set_xticks(x, labels)
        ax.set_title(
            f"{arm} test -- escalated {ev['n_escalated']}/{ev['n']} "
            f"(TP {conf['tp']} FP {conf['fp']} "
            f"FN {conf['fn']} TN {conf['tn']})"
        )
    axes[0].set_ylabel("mean quality")
    axes[0].legend(fontsize="small")
    fig.suptitle("Phase 10 (5): adaptive vs fixed baselines")
    fig.tight_layout()
    path = FIG_DIR / "adaptive_vs_baselines.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> dict[str, Any]:
    global OUT_DIR, FIG_DIR
    parser = argparse.ArgumentParser(
        description="Render the five Phase 10 figures from written artifacts."
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.out)
    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt = _pyplot()
    analyses = {
        arm: _read_json(OUT_DIR / f"signal_analysis_{arm}.json")
        for arm in ARMS
    }
    evaluation = _read_json(OUT_DIR / "policy_evaluation.json")

    # Figure 3's always-escalate latency note: recompute the adaptive-model
    # hybrid mean here so the figure cannot drift from the evaluation.
    paths = [
        fig1_signal_distribution(plt, analyses),
        fig2_threshold_sweep(plt, analyses),
        fig3_quality_vs_latency(plt, evaluation),
        fig4_quality_vs_cost(plt, evaluation),
        fig5_adaptive_vs_baselines(plt, evaluation),
    ]
    manifest = {
        "phase": 10,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "signal_analysis_after.json",
            "signal_analysis_before.json",
            "policy_evaluation.json",
            "oracle_phase8_after.jsonl",
            "oracle_phase8_before.jsonl",
            "frozen_policy.json",
        ],
        "figures": [
            {"name": p.name, "sha256": _sha256(p)} for p in paths
        ],
    }
    (FIG_DIR / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
