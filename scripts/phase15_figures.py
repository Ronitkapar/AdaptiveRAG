#!/usr/bin/env python3
"""Phase 15 figures -- five pre-registered panels from written artifacts.

1. oracle-positive distribution (per arm: YES/NO counts with split note);
2. policy decision / escalation behavior (confusion + rate per arm);
3. Adaptive vs BM25 vs Dense quality (grouped bars per arm);
4. Adaptive vs Random-at-rate (null distributions with adaptive marker);
5. quality versus Dense-call rate (frontier over A/B/C per arm).

Reads only `experiments/phase15/*.json(l)`; writes PNGs plus a sha256
manifest. No statistics are computed here beyond bar heights the artifacts
already carry; the combined randomization panel re-uses the stored null
draws verbatim.
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

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "phase15"
ARMS = ("after", "before")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load(oracle_dir: Path) -> tuple[dict[str, Any], dict[str, list], dict[str, Any]]:
    evaluation = json.loads(
        (oracle_dir / "policy_evaluation.json").read_text(encoding="utf-8")
    )
    rows = {
        arm: _read_jsonl(oracle_dir / f"policy_rows_{arm}.jsonl") for arm in ARMS
    }
    arm_full = {
        arm: json.loads(
            (oracle_dir / f"policy_eval_{arm}.json").read_text(encoding="utf-8")
        )
        for arm in ARMS
    }
    return evaluation, rows, arm_full


def fig_oracle_distribution(
    evaluation: dict[str, Any], fig_dir: Path
) -> Path:
    arms = ARMS
    yes = [evaluation["arms"][a]["n_oracle_yes"] for a in arms]
    n = [evaluation["arms"][a]["n"] for a in arms]
    no = [t - y for t, y in zip(n, yes)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(len(arms))
    ax.bar(x, yes, label="oracle YES")
    ax.bar(x, no, bottom=yes, label="oracle NO")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{a} (n={t})" for a, t in zip(arms, n)])
    ax.set_ylabel("queries")
    ax.set_title("Powered oracle-positive distribution (all test)")
    ax.legend()
    path = fig_dir / "oracle_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_escalation_behavior(
    evaluation: dict[str, Any], fig_dir: Path
) -> Path:
    labels = ["TP", "FP", "FN", "TN"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, arm in zip(axes, ARMS):
        arm_eval = evaluation["arms"][arm]
        conf = arm_eval["confusion"]
        vals = [conf["tp"], conf["fp"], conf["fn"], conf["tn"]]
        ax.bar(labels, vals)
        ax.set_title(
            f"{arm}: rate={arm_eval['escalation_rate']:.3f} "
            f"TP={conf['tp']} FN={conf['fn']}"
        )
    axes[0].set_ylabel("queries")
    fig.suptitle("Frozen-rule decisions vs oracle (powered test)")
    path = fig_dir / "escalation_behavior.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_quality_bars(
    evaluation: dict[str, Any], arm_full: dict[str, Any], fig_dir: Path
) -> Path:
    systems = ["bm25_only_recall_at_5", "adaptive_recall_at_5",
               "always_escalate_recall_at_5"]
    names = ["BM25-only", "Adaptive", "Dense-only"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(ARMS))
    width = 0.22
    for i, (key, name) in enumerate(zip(systems, names)):
        vals = [arm_full[a]["means"][key] for a in ARMS]
        ax.bar([p + (i - 1) * width for p in x], vals, width=width, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(ARMS))
    ax.set_ylabel("recall@5")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Powered retrieval quality by system and arm")
    ax.legend()
    path = fig_dir / "quality_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_adaptive_vs_random(
    evaluation: dict[str, Any],
    rows: dict[str, list],
    arm_full: dict[str, Any],
    fig_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, arm in zip(axes, ARMS):
        arm_eval = evaluation["arms"][arm]
        null = arm_full[arm]["random_ablation_mean_quality_distribution"]
        ax.hist(null, bins=30, alpha=0.7, label="random-at-rate null")
        ax.axvline(
            arm_eval["adaptive_recall_at_5"], color="red",
            label=f"adaptive ({arm_eval['adaptive_recall_at_5']:.3f})",
        )
        ax.set_title(
            f"{arm}: P={arm_eval['p_random_ge_adaptive']:.3f} "
            f"(TP={arm_eval['confusion']['tp']})"
        )
        ax.set_xlabel("mean recall@5")
    axes[0].set_ylabel("draws")
    axes[0].legend(fontsize=8)
    p_comb = evaluation["combined"]["randomization_test"]["p_comb"]
    fig.suptitle(f"Adaptive vs random-at-rate (combined P={p_comb:.3f})")
    path = fig_dir / "adaptive_vs_random.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_quality_vs_callrate(
    evaluation: dict[str, Any], arm_full: dict[str, Any], fig_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for arm in ARMS:
        arm_eval = evaluation["arms"][arm]
        means = arm_full[arm]["means"]
        ax.scatter([0.0], [means["bm25_only_recall_at_5"]], label=f"{arm} BM25")
        ax.scatter(
            [arm_eval["escalation_rate"]],
            [means["adaptive_recall_at_5"]],
            label=f"{arm} adaptive",
        )
        ax.scatter([1.0], [means["always_escalate_recall_at_5"]],
                   label=f"{arm} dense")
    ax.set_xlabel("dense-call rate")
    ax.set_ylabel("recall@5")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Quality vs dense-call rate (powered test)")
    ax.legend(fontsize=8)
    path = fig_dir / "quality_vs_callrate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Phase 15 figures.")
    parser.add_argument("--oracle-dir", default=str(OUT_DIR))
    parser.add_argument("--out", default=str(OUT_DIR / "figures"))
    args = parser.parse_args()
    oracle_dir = Path(args.oracle_dir)
    fig_dir = Path(args.out)
    fig_dir.mkdir(parents=True, exist_ok=True)

    evaluation, rows, arm_full = _load(oracle_dir)
    paths = [
        fig_oracle_distribution(evaluation, fig_dir),
        fig_escalation_behavior(evaluation, fig_dir),
        fig_quality_bars(evaluation, arm_full, fig_dir),
        fig_adaptive_vs_random(evaluation, rows, arm_full, fig_dir),
        fig_quality_vs_callrate(evaluation, arm_full, fig_dir),
    ]
    manifest = {
        "phase": 15,
        "figures": [
            {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
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
