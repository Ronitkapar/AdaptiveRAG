#!/usr/bin/env python3
"""Phase 11 figures -- rendered from written artifacts, never from memory.

Exactly the four pre-registered mechanism figures (`docs/phases/phase-11.md`
§6): DDR vs escalation value, dense confidence vs escalation value,
disagreement vs dense quality, and the per-query dense->hybrid change chart.
No candidate-signal panel exists because no candidate was established -- per
the brief, no filler figures are generated to inflate the count.
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

OUT_DIR = REPO_ROOT / "experiments" / "phase11"
FIG_DIR = OUT_DIR / "figures"
ARMS = ("after", "before")
DPI = 150

GROUP_COLORS = {"helps": "#2a7a3f", "ties": "#9db6d8", "harms": "#c25e3a"}


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


def _read_table(arm: str) -> list[dict[str, Any]]:
    with open(OUT_DIR / f"mechanism_table_{arm}.jsonl", encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def _jitter(rng: np.random.Generator, n: int, span: float) -> np.ndarray:
    return (rng.random(n) - 0.5) * 2 * span


def fig1_ddr_vs_delta(plt, tables: dict[str, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    rng = np.random.default_rng(20250101)
    for ax, arm in zip(axes, ARMS):
        rows = tables[arm]
        for group in ("ties", "harms", "helps"):
            xs = [r["ddr_dense"] for r in rows if r["group"] == group]
            ys = [r["delta_recall_at_5"] for r in rows if r["group"] == group]
            ax.scatter(
                np.asarray(xs) + _jitter(rng, len(xs), 0.02), ys,
                color=GROUP_COLORS[group], label=f"{group} (n={len(xs)})",
                alpha=0.7, zorder=3,
            )
        for r in rows:
            if r["group"] == "helps":
                ax.annotate(r["query_id"], (r["ddr_dense"], r["delta_recall_at_5"]),
                            fontsize=7, xytext=(4, 4),
                            textcoords="offset points")
        ax.axhline(0.01, color="black", linestyle=":", linewidth=1)
        ax.axhline(-0.01, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("distinct_doc_ratio@dense")
        ax.set_title(f"{arm} (n={len(rows)})")
    axes[0].set_ylabel("delta recall@5 (hybrid - dense)")
    axes[0].legend(fontsize="small")
    fig.suptitle("Phase 11 (1): DDR vs escalation value")
    fig.tight_layout()
    path = FIG_DIR / "ddr_vs_delta.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig2_gap_vs_delta(plt, tables: dict[str, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    rng = np.random.default_rng(20250101)
    for ax, arm in zip(axes, ARMS):
        rows = [r for r in tables[arm] if r["dense_score_gap"] is not None]
        for group in ("ties", "harms", "helps"):
            xs = [r["dense_score_gap"] for r in rows if r["group"] == group]
            ys = [r["delta_recall_at_5"] for r in rows if r["group"] == group]
            ax.scatter(
                np.asarray(xs) + _jitter(rng, len(xs), 0.002), ys,
                color=GROUP_COLORS[group], label=f"{group} (n={len(xs)})",
                alpha=0.7, zorder=3,
            )
        ax.axhline(0.01, color="black", linestyle=":", linewidth=1)
        ax.axhline(-0.01, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("dense top1-top2 score gap")
        ax.set_title(f"{arm}")
    axes[0].set_ylabel("delta recall@5 (hybrid - dense)")
    axes[0].legend(fontsize="small")
    fig.suptitle("Phase 11 (2): dense confidence vs escalation value")
    fig.tight_layout()
    path = FIG_DIR / "gap_vs_delta.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig3_disagreement_vs_quality(
    plt, tables: dict[str, list[dict[str, Any]]]
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    rng = np.random.default_rng(20250101)
    for ax, arm in zip(axes, ARMS):
        rows = tables[arm]
        for group in ("ties", "harms", "helps"):
            xs = [r["jaccard_bm25_dense"] for r in rows if r["group"] == group]
            ys = [r["dense_recall_at_5"] for r in rows if r["group"] == group]
            ax.scatter(
                np.asarray(xs) + _jitter(rng, len(xs), 0.01),
                np.asarray(ys) + _jitter(rng, len(ys), 0.015),
                color=GROUP_COLORS[group], label=f"{group} (n={len(xs)})",
                alpha=0.7, zorder=3,
            )
        ax.set_xlabel("jaccard@5 (bm25, dense)")
        ax.set_title(f"{arm}")
    axes[0].set_ylabel("dense recall@5")
    axes[0].legend(fontsize="small")
    fig.suptitle("Phase 11 (3): retrieval disagreement vs dense quality")
    fig.tight_layout()
    path = FIG_DIR / "disagreement_vs_quality.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig4_nontie_waterfall(plt, tables: dict[str, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, arm in zip(axes, ARMS):
        rows = sorted(
            (r for r in tables[arm] if r["group"] != "ties"),
            key=lambda r: r["delta_recall_at_5"],
        )
        ys = np.arange(len(rows))
        deltas = [r["delta_recall_at_5"] for r in rows]
        colors = [GROUP_COLORS[r["group"]] for r in rows]
        ax.barh(ys, deltas, color=colors, height=0.6)
        ax.set_yticks(ys, [r["query_id"] for r in rows], fontsize=7)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("delta recall@5 (hybrid - dense)")
        ax.set_title(f"{arm}: all non-tie queries (n={len(rows)})")
    fig.suptitle("Phase 11 (4): per-query dense to hybrid change")
    fig.tight_layout()
    path = FIG_DIR / "nontie_waterfall.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> dict[str, Any]:
    global OUT_DIR, FIG_DIR
    parser = argparse.ArgumentParser(
        description="Render the four Phase 11 mechanism figures."
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.out)
    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt = _pyplot()
    tables = {arm: _read_table(arm) for arm in ARMS}
    paths = [
        fig1_ddr_vs_delta(plt, tables),
        fig2_gap_vs_delta(plt, tables),
        fig3_disagreement_vs_quality(plt, tables),
        fig4_nontie_waterfall(plt, tables),
    ]
    manifest = {
        "phase": 11,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "mechanism_table_after.jsonl",
            "mechanism_table_before.jsonl",
            "mechanism_summary.json",
        ],
        "figures": [{"name": p.name, "sha256": _sha256(p)} for p in paths],
        "omitted": (
            "candidate-signal panel: no candidate was established, and the "
            "brief forbids filler figures"
        ),
    }
    (FIG_DIR / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
