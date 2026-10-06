"""Corpus manifest and benchmark-query loading.

Loads the two committed data sources that survive a fresh clone:

* ``data/metadata/papers.json``  -- the 14-paper corpus manifest.
* ``data/evaluation/*.jsonl``     -- the frozen benchmark query sets.

Both are read-only inputs. The module also derives the Playground's
suggested questions from the *real* benchmark queries, so the demo
never invents a question the project did not actually ask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_rag.config.paths import (
    EVALUATION_DIR,
    PAPERS_MANIFEST_PATH,
)

# Benchmark files that are committed to git (not gitignored) and are
# therefore readable on a fresh clone.
COMMITTED_BENCHMARKS: tuple[str, ...] = (
    "phase7_eval_v1.jsonl",
    "phase15_eval_v1.jsonl",
    "dense_eval_v1.jsonl",
)


def load_papers() -> list[dict[str, Any]]:
    """Load the corpus manifest (14 papers)."""
    if not PAPERS_MANIFEST_PATH.is_file():
        return []
    with PAPERS_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def papers_by_id() -> dict[str, dict[str, Any]]:
    """Index the manifest by ``document_id``."""
    return {paper["document_id"]: paper for paper in load_papers()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_benchmark(name: str) -> list[dict[str, Any]]:
    """Load a committed benchmark query set by filename."""
    if name not in COMMITTED_BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}")
    return _read_jsonl(EVALUATION_DIR / name)


def load_phase7_queries() -> list[dict[str, Any]]:
    """The frozen 107-query benchmark (47 calibration / 60 test)."""
    return load_benchmark("phase7_eval_v1.jsonl")


def suggested_questions(limit: int = 8) -> list[dict[str, Any]]:
    """Suggested Playground questions, drawn from the real benchmark.

    Returns a small, varied selection of the actual ``phase7_eval_v1``
    queries so the demo's suggestions are real research questions
    rather than invented ones. Selection is deterministic (first
    ``limit`` by example_id) so the demo is reproducible.
    """
    rows = load_phase7_queries()
    # Prefer a spread across the set: take evenly spaced examples so
    # the suggestions are not all from the calibration head.
    if not rows:
        return []
    if len(rows) <= limit:
        picked = rows
    else:
        step = len(rows) / limit
        picked = [rows[int(i * step)] for i in range(limit)]
    return [
        {
            "example_id": row.get("example_id", ""),
            "query": row.get("query", ""),
            "category": row.get("category", ""),
            "relevant_documents": row.get("relevant_documents", []),
        }
        for row in picked
    ]


def category_counts(papers: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Count papers per corpus category."""
    papers = papers if papers is not None else load_papers()
    counts: dict[str, int] = {}
    for paper in papers:
        category = paper.get("category", "uncategorized")
        counts[category] = counts.get(category, 0) + 1
    return counts
