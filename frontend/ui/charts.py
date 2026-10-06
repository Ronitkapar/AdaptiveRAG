"""Chart-spec builders for the Benchmarks page.

These functions are deliberately Streamlit-free: they turn the
frozen result tables into plain-Python data structures (lists of
dicts) that the page renders with Streamlit's native Vega-Lite
charts (``st.bar_chart`` / ``st.line_chart``). Keeping them pure
means the ``tests/test_frontend`` suite can assert the graphs are
built from real data without a Streamlit runtime.

No new dependency is introduced: no Plotly, no matplotlib at
runtime. The project's own matplotlib PNGs are displayed with
``st.image`` where they exist on the working tree.
"""

from __future__ import annotations

from typing import Any

from frontend.ui.benchmark import resolve_row_system


def e1_quality_bars(phase7_e1: dict[str, Any]) -> list[dict[str, Any]]:
    """Recall@5 / MRR / Hit@5 per strategy (Graph 1).

    Rows may carry ``None`` for metrics the source table lacks (the
    working-tree E1 artifact has no Hit column); such rows are skipped
    for quality but still shown on latency graphs when possible.
    """
    specs: list[dict[str, Any]] = []
    for row in phase7_e1["rows"]:
        system = resolve_row_system(row)
        if not system:
            continue
        recall = row.get("recall_at_5")
        mrr = row.get("mrr")
        if recall is None or mrr is None:
            continue
        specs.append(
            {
                "system": system,
                "Recall@5": recall,
                "MRR": mrr,
                "Hit@5": row.get("hit_at_5"),
            }
        )
    return specs


def quality_vs_latency(phase7_e1: dict[str, Any]) -> list[dict[str, Any]]:
    """Quality-vs-efficiency frontier points (Graph 2).

    ``log_latency`` is provided because the BM25-to-reranker latency
    spans three orders of magnitude; the page can plot either axis.
    Rows missing quality or usable latency are skipped.
    """
    import math

    specs: list[dict[str, Any]] = []
    for row in phase7_e1["rows"]:
        system = resolve_row_system(row)
        if not system:
            continue
        recall = row.get("recall_at_5")
        latency = row.get("median_latency_ms")
        if recall is None or latency is None or latency <= 0:
            continue
        specs.append(
            {
                "system": system,
                "Recall@5": recall,
                "median_latency_ms": latency,
                "log10_latency_ms": round(math.log10(latency), 3),
            }
        )
    return specs


def phase8_before_after(phase8_e1: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-arm recall before/after the corpus fix, with the paired delta."""
    return [
        {
            "arm": row["arm"],
            "recall_before": row["recall_before"],
            "recall_after": row["recall_after"],
            "delta": round(row["recall_after"] - row["recall_before"], 4),
        }
        for row in phase8_e1["rows"]
    ]


def oracle_comparison(oracle: dict[str, Any]) -> list[dict[str, Any]]:
    """Oracle-ceiling policies by Recall@5 (graph) — latency rides along."""
    return [
        {
            "policy": row["policy"],
            "Recall@5": row["recall_at_5"],
            "MRR": row["mrr"],
            "median_latency_ms": row["median_latency_ms"],
        }
        for row in oracle["rows"]
    ]


def arm_query_series(
    rows: list[dict[str, Any]], metric: str = "recall_at_5"
) -> list[dict[str, Any]]:
    """Per-query values of one metric across an arm's stored rows.

    Used for the query-level explorer over the on-disk ``rows.jsonl``
    (working tree only). ``metric`` must be a numeric per-query field.
    """
    series: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            series.append(
                {
                    "query_id": row.get("query_id", ""),
                    "system": row.get("system", ""),
                    metric: value,
                }
            )
    return series


def per_query_strategy_frame(
    arm_rows: dict[str, list[dict[str, Any]]],
    query_id: str,
    metric: str = "recall_at_5",
) -> list[dict[str, Any]]:
    """One query's metric across all loaded arms (Graph 3 data)."""
    frame: list[dict[str, Any]] = []
    for arm, rows in arm_rows.items():
        for row in rows:
            if row.get("query_id") == query_id and isinstance(
                row.get(metric), (int, float)
            ):
                frame.append(
                    {
                        "system": arm,
                        metric: row[metric],
                        "latency_ms": row.get("retrieval_latency_ms"),
                    }
                )
                break
    return frame
