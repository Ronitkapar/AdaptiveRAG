#!/usr/bin/env python3
"""
compare_reranking.py
--------------------
Compare each first-stage baseline against its second-stage reranked variant.
Reads run directories (config/manifest/traces/metrics) and prints two tables:

  1. Base vs reranked — one row per metric per strategy pair, with deltas.
  2. Depth ablation  — quality and rerank latency at each candidate depth, with
     the final top_k held constant, so the depth/cost curve is visible.

Every run directory must be supplied explicitly: no globbing, no auto-discovery,
because a rerank run is only comparable against the exact baseline it was built
from. Exits 2 when the corpus, benchmark size, or retrieval methods disagree.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from adaptive_rag.config.logging import setup_logging

logger = logging.getLogger("compare_reranking")

# The strategy pairs this tool knows how to line up.
PAIRS = [
    ("dense", "dense", "dense_rerank", "--dense-rerank-run"),
    ("bm25", "bm25", "bm25_rerank", "--bm25-rerank-run"),
    ("hybrid", "hybrid", "hybrid_rerank", "--hybrid-rerank-run"),
]

_RETRIEVAL_ROWS = [
    ("recall_at_k", 1),
    ("recall_at_k", 3),
    ("recall_at_k", 5),
    ("recall_at_k", 10),
    ("precision_at_k", 5),
    ("hit_at_k", 5),
    ("mrr", None),
    ("ndcg_at_k", 5),
    ("ndcg_at_k", 10),
]

_EFFICIENCY_ROWS = [
    "retrieval_latency_ms_mean",
    "candidate_generation_latency_ms_mean",
    "rerank_latency_ms_mean",
    "rerank_latency_ms_p50",
    "rerank_latency_ms_p95",
]

_ABLATION_ROWS = [
    "recall_at_k@5",
    "mrr",
    "retrieval_latency_ms_mean",
    "candidate_generation_latency_ms_mean",
    "rerank_latency_ms_mean",
    "rerank_candidate_count_mean",
]


def _load_metrics(run_dir: Path) -> dict:
    data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    lookup: dict[tuple[str, int | None], float | None] = {}
    for payload in data.values():
        for metric in payload.get("metrics", []):
            lookup[(metric.get("name"), metric.get("k"))] = metric.get("value")
    return lookup  # type: ignore[return-value]


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def _candidate_depth(run_dir: Path) -> int | None:
    retrieval = _load_config(run_dir).get("retrieval", {})
    return retrieval.get("rerank_candidate_k")


def _is_run_dir(path: Path) -> bool:
    return path is not None and (path / "metrics.json").is_file()


def _fmt(value, width: int = 12, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:>{width}.{digits}f}"
    return f"{str(value):>{width}}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare first-stage baselines against their reranked variants."
    )
    for label, _base, _rerank, flag in PAIRS:
        parser.add_argument(
            f"--{label}-run",
            type=Path,
            default=None,
            help=f"{label} first-stage baseline run directory",
        )
        parser.add_argument(
            flag,
            type=Path,
            default=None,
            help=f"{label} reranked run directory (never auto-discovered)",
        )
    parser.add_argument(
        "--ablation-run",
        type=Path,
        action="append",
        default=[],
        help="Reranked run for the depth ablation table; repeatable",
    )
    args = parser.parse_args()

    setup_logging()

    columns: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    for label, _base, _rerank, flag in PAIRS:
        base_dir = getattr(args, f"{label}_run")
        rerank_dir = getattr(args, flag.lstrip("-").replace("-", "_"))
        if rerank_dir is None:
            continue
        if not _is_run_dir(base_dir):
            missing.append(f"--{label}-run (metrics.json not found at {base_dir})")
            continue
        if not _is_run_dir(rerank_dir):
            missing.append(f"{flag} (metrics.json not found at {rerank_dir})")
            continue
        columns.append((label, base_dir, rerank_dir))

    if missing:
        for problem in missing:
            logger.error("Missing run: %s", problem)
        return 1
    if not columns:
        logger.error(
            "No reranked run supplied. Pass at least one --<strategy>-rerank-run "
            "together with its matching --<strategy>-run."
        )
        return 1

    # --- Run consistency: same corpus, same benchmark size, correct methods ---
    mismatches: list[str] = []
    reference_corpus = None
    reference_count = None
    print("Run consistency:")
    for label, base_dir, rerank_dir in columns:
        base_manifest = _load_manifest(base_dir)
        rerank_manifest = _load_manifest(rerank_dir)

        corpus_base = base_manifest.get("corpus_version")
        corpus_rerank = rerank_manifest.get("corpus_version")
        corpus_ok = corpus_base is not None and corpus_base == corpus_rerank
        print(
            f"  {label:<7} corpus: base={corpus_base} rerank={corpus_rerank} "
            f"{'MATCH' if corpus_ok else 'MISMATCH'}"
        )
        if not corpus_ok:
            mismatches.append(f"{label}:corpus_version")

        n_base = base_manifest.get("trace_count")
        n_rerank = rerank_manifest.get("trace_count")
        n_ok = n_base is not None and n_base == n_rerank
        print(
            f"  {label:<7} n:     base={n_base} rerank={n_rerank} "
            f"{'MATCH' if n_ok else 'MISMATCH'}"
        )
        if not n_ok:
            mismatches.append(f"{label}:trace_count")

        expected_base, expected_rerank, _flag = next(
            (b, r, f) for (label_, b, r, f) in PAIRS if label_ == label
        )
        method_base = base_manifest.get("retrieval_method")
        method_rerank = rerank_manifest.get("retrieval_method")
        method_ok = method_base == expected_base and method_rerank == expected_rerank
        print(
            f"  {label:<7} method: base={method_base} rerank={method_rerank} "
            f"{'OK' if method_ok else 'UNEXPECTED'}"
        )
        if not method_ok:
            mismatches.append(f"{label}:retrieval_method")

        if reference_corpus is None:
            reference_corpus = corpus_base
            reference_count = n_base
        elif corpus_base != reference_corpus:
            mismatches.append(f"{label}:corpus_version_not_shared_across_columns")
        elif n_base != reference_count:
            mismatches.append(f"{label}:trace_count_not_shared_across_columns")

    for problem in mismatches:
        print(f"  UNEXPECTED: {problem}")
    if mismatches:
        logger.error("Configuration/corpus mismatches detected: %s", mismatches)
        return 2
    print()

    # --- Table 1: base vs reranked, one block per strategy pair ---
    for label, base_dir, rerank_dir in columns:
        base_metrics = _load_metrics(base_dir)
        rerank_metrics = _load_metrics(rerank_dir)
        print(f"{label}: base vs reranked")
        header = (
            f"{'metric':<22}{'base':>12}{'reranked':>12}{'delta':>12}"
        )
        print(header)
        print("-" * len(header))
        for name, k in _RETRIEVAL_ROWS:
            b = base_metrics.get((name, k))
            r = rerank_metrics.get((name, k))
            label_text = f"{name}@k={k}" if k is not None else name
            delta = (r - b) if isinstance(b, (int, float)) and isinstance(r, (int, float)) else None
            print(
                f"{label_text:<22}{_fmt(b)}{_fmt(r)}"
                f"{_fmt(delta) if delta is not None else 'n/a':>12}"
            )
        print()
        print("Efficiency (lower is better):")
        for name in _EFFICIENCY_ROWS:
            b = base_metrics.get((name, None))
            r = rerank_metrics.get((name, None))
            delta = (r - b) if isinstance(b, (int, float)) and isinstance(r, (int, float)) else None
            print(
                f"{name:<34}{_fmt(b, 12, 2)}{_fmt(r, 12, 2)}"
                f"{_fmt(delta, 12, 2) if delta is not None else 'n/a':>12}"
            )
        print()

    # --- Table 2: depth ablation ---
    ablation_dirs = [d for d in args.ablation_run if _is_run_dir(d)]
    if ablation_dirs:
        print("Depth ablation (final top_k held constant):")
        depths = [_candidate_depth(d) for d in ablation_dirs]
        header = f"{'metric':<34}" + "".join(f"{f'k={k}':>12}" for k in depths)
        print(header)
        print("-" * len(header))
        loaded = [_load_metrics(d) for d in ablation_dirs]
        for name in _ABLATION_ROWS:
            if "@" in name:
                metric_name, k_text = name.split("@")
                key = (metric_name, int(k_text))
                row_label = name
            else:
                key = (name, None)
                row_label = name
            values = [m.get(key) for m in loaded]
            print(f"{row_label:<34}" + "".join(_fmt(v, 12, 2) for v in values))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())