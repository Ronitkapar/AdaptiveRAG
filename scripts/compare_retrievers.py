#!/usr/bin/env python3
"""
compare_retrievers.py
---------------------
Compare dense vs BM25 (and optionally hybrid) baseline experiment metrics
side-by-side. Reads run directories (config/manifest/traces/metrics) and prints
retrieval quality (Recall/Precision/Hit/MRR/nDCG) plus latency tables.

The hybrid column is opt-in via --hybrid-run. When the flag is absent the output
is byte-identical to the dense-vs-BM25-only table.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.paths import EXPERIMENTS_DIR

logger = logging.getLogger("compare_retrievers")

_RETRIEVAL_ROWS = [
    ("recall_at_k", 1),
    ("recall_at_k", 3),
    ("recall_at_k", 5),
    ("recall_at_k", 10),
    ("precision_at_k", 1),
    ("precision_at_k", 5),
    ("hit_at_k", 1),
    ("hit_at_k", 5),
    ("mrr", None),
    ("ndcg_at_k", 5),
    ("ndcg_at_k", 10),
]

_EFFICIENCY_ROWS = [
    "retrieval_latency_ms_mean",
    "retrieval_latency_ms_p50",
    "retrieval_latency_ms_p95",
]


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dense vs BM25 retrieval experiments.")
    parser.add_argument("--dense-run", type=Path, default=None)
    parser.add_argument("--bm25-run", type=Path, default=None)
    parser.add_argument(
        "--hybrid-run",
        type=Path,
        default=None,
        help="Optional hybrid run directory; adds a third column when provided",
    )
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    args = parser.parse_args()

    setup_logging()

    def _resolve(prefix: str | None, fallback: Path | None) -> Path | None:
        if fallback is not None:
            return fallback
        candidates = sorted(args.experiments_dir.glob(f"*{prefix}*"), reverse=True)
        return candidates[0] if candidates else None

    dense_dir = _resolve("dense_baseline", args.dense_run)
    bm25_dir = _resolve("bm25_baseline", args.bm25_run)
    # Hybrid is never auto-discovered: the third column is opt-in, so omitting
    # --hybrid-run must keep the dense-vs-BM25 table byte-identical.
    hybrid_dir = args.hybrid_run

    if dense_dir is None or not (dense_dir / "metrics.json").is_file():
        logger.error("Dense run metrics not found (looked for *dense_baseline* runs).")
        return 1
    if bm25_dir is None or not (bm25_dir / "metrics.json").is_file():
        logger.error("BM25 run metrics not found (looked for *bm25_baseline* runs).")
        return 1

    use_hybrid = hybrid_dir is not None and (hybrid_dir / "metrics.json").is_file()
    if args.hybrid_run is not None and not use_hybrid:
        logger.error("Hybrid run metrics not found at %s.", hybrid_dir)
        return 1

    dense = _load_metrics(dense_dir)
    bm25 = _load_metrics(bm25_dir)
    dense_manifest = _load_manifest(dense_dir)
    bm25_manifest = _load_manifest(bm25_dir)
    hybrid = _load_metrics(hybrid_dir) if use_hybrid else None
    hybrid_manifest = _load_manifest(hybrid_dir) if use_hybrid else {}
    logger.info("Dense run: %s (n=%s)", dense_dir, dense_manifest.get("trace_count"))
    logger.info("BM25 run:  %s (n=%s)", bm25_dir, bm25_manifest.get("trace_count"))
    if use_hybrid:
        logger.info("Hybrid run: %s (n=%s)", hybrid_dir, hybrid_manifest.get("trace_count"))

    # --- Run consistency: same corpus, same benchmark size, correct methods ---
    print("Run consistency:")
    mismatches: list[str] = []
    corpus_dense = dense_manifest.get("corpus_version")
    corpus_bm25 = bm25_manifest.get("corpus_version")
    corpus_ok = corpus_dense is not None and corpus_dense == corpus_bm25
    print(f"  corpus_version: dense={corpus_dense} bm25={corpus_bm25} "
          f"{'MATCH' if corpus_ok else 'MISMATCH'}")
    if not corpus_ok:
        mismatches.append("corpus_version")

    n_dense = dense_manifest.get("trace_count")
    n_bm25 = bm25_manifest.get("trace_count")
    n_ok = n_dense is not None and n_dense == n_bm25
    print(f"  trace_count:    dense={n_dense} bm25={n_bm25} {'MATCH' if n_ok else 'MISMATCH'}")
    if not n_ok:
        mismatches.append("trace_count")

    method_dense = dense_manifest.get("retrieval_method")
    method_bm25 = bm25_manifest.get("retrieval_method")
    methods_ok = method_dense == "dense" and method_bm25 == "bm25"
    print(f"  method:         dense={method_dense} bm25={method_bm25} "
          f"{'OK' if methods_ok else 'UNEXPECTED'}")
    if not methods_ok:
        mismatches.append("retrieval_method")

    if use_hybrid:
        corpus_hybrid = hybrid_manifest.get("corpus_version")
        corpus_hybrid_ok = corpus_hybrid is not None and corpus_hybrid == corpus_dense
        print(f"  corpus_version: hybrid={corpus_hybrid} "
              f"{'MATCH' if corpus_hybrid_ok else 'MISMATCH'}")
        if not corpus_hybrid_ok:
            mismatches.append("corpus_version")

        n_hybrid = hybrid_manifest.get("trace_count")
        n_hybrid_ok = n_hybrid is not None and n_hybrid == n_dense
        print(f"  trace_count:    hybrid={n_hybrid} {'MATCH' if n_hybrid_ok else 'MISMATCH'}")
        if not n_hybrid_ok:
            mismatches.append("trace_count")

        method_hybrid = hybrid_manifest.get("retrieval_method")
        method_hybrid_ok = method_hybrid == "hybrid"
        print(f"  method:         hybrid={method_hybrid} "
              f"{'OK' if method_hybrid_ok else 'UNEXPECTED'}")
        if not method_hybrid_ok:
            mismatches.append("retrieval_method")

    if mismatches:
        logger.error("Configuration/corpus mismatches detected: %s", mismatches)
        return 2
    print()

    if use_hybrid:
        header = (f"{'metric':<22}{'dense':>12}{'bm25':>12}{'hybrid':>12}"
                  f"{'Δhyb-dense':>12}{'Δhyb-bm25':>12}")
        print(header)
        print("-" * len(header))
        for name, k in _RETRIEVAL_ROWS:
            d = dense.get((name, k))
            b = bm25.get((name, k))
            h = hybrid.get((name, k))
            label = f"{name}@k={k}" if k is not None else name
            if all(isinstance(v, (int, float)) for v in (d, b, h)):
                print(f"{label:<22}{d:>12.4f}{b:>12.4f}{h:>12.4f}"
                      f"{(h - d):>+12.4f}{(h - b):>+12.4f}")
            else:
                print(f"{label:<22}{str(d):>12}{str(b):>12}{str(h):>12}{'n/a':>12}{'n/a':>12}")
        print()
        print("Efficiency (lower is better):")
        for name in _EFFICIENCY_ROWS:
            d = dense.get((name, None))
            b = bm25.get((name, None))
            h = hybrid.get((name, None))
            if all(isinstance(v, (int, float)) for v in (d, b, h)):
                print(f"{name:<30}{d:>12.2f}{b:>12.2f}{h:>12.2f}"
                      f"{(h - d):>+12.2f}{(h - b):>+12.2f}")
        return 0

    header = f"{'metric':<22}{'dense':>12}{'bm25':>12}{'delta':>12}"
    print(header)
    print("-" * len(header))
    for name, k in _RETRIEVAL_ROWS:
        d = dense.get((name, k))
        b = bm25.get((name, k))
        label = f"{name}@k={k}" if k is not None else name
        if isinstance(d, (int, float)) and isinstance(b, (int, float)):
            print(f"{label:<22}{d:>12.4f}{b:>12.4f}{(b - d):>+12.4f}")
        else:
            print(f"{label:<22}{str(d):>12}{str(b):>12}{'n/a':>12}")
    print()
    print("Efficiency (lower is better):")
    for name in _EFFICIENCY_ROWS:
        d = dense.get((name, None))
        b = bm25.get((name, None))
        if isinstance(d, (int, float)) and isinstance(b, (int, float)):
            print(f"{name:<30}{d:>12.2f}{b:>12.2f}{(b - d):>+12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
