"""Experiment Explorer: the stored per-query runs behind the tables."""

from __future__ import annotations

import streamlit as st

from frontend.ui import benchmark, charts, corpus

ARMS = ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")


def render() -> None:
    st.title("Experiment Explorer")
    st.markdown(
        "The per-query records behind the frozen tables: per-arm result "
        "rows (`rows.jsonl`), run manifests, and the query-level explorer. "
        "These artifacts are gitignored — present on the research machine, "
        "absent on a fresh clone."
    )
    tree = benchmark.experiment_tree()
    if not tree:
        st.warning(
            "No experiment artifacts in this deployment (`experiments/` holds "
            "only the committed README and .gitkeep). The Benchmarks page "
            "covers the same frozen results from the committed snapshot, and "
            "the full record is in `docs/phase*_results.md`."
        )
        return

    st.subheader("Stored runs on this machine")
    dirs = [entry for entry in tree if entry["kind"] == "directory"]
    st.markdown(f"`{len(dirs)}` run/analysis directories under `experiments/`.")
    with st.expander("Directory listing"):
        for entry in dirs:
            st.markdown(f"* `{entry['name']}`")

    st.subheader("Query-level explorer (Phase 7 E1 arms)")
    arm_rows: dict[str, list[dict]] = {}
    sources: dict[str, str] = {}
    for arm in ARMS:
        rows, source = benchmark.load_arm_rows(arm)
        arm_rows[arm] = rows
        sources[arm] = source
    loaded = {arm: rows for arm, rows in arm_rows.items() if rows}
    if not loaded:
        st.info(
            "No E1 arm rows on this machine. The frozen arm table is on the Benchmarks page."
        )
        return
    st.caption(
        "Loaded arms: " + ", ".join(f"`{arm}` ({len(rows)} rows)" for arm, rows in loaded.items())
    )
    queries = corpus.load_phase7_queries()
    query_index = {q["example_id"]: q for q in queries}
    query_id = st.selectbox(
        "Query",
        sorted({row["query_id"] for rows in loaded.values() for row in rows}),
    )
    question = query_index.get(query_id, {})
    if question:
        st.markdown(f"**{question.get('query', '')}**")
        st.caption(
            f"`{query_id}` · {question.get('category', '')} · split {question.get('split', '')} · "
            f"relevant: {', '.join(question.get('relevant_documents', []))}"
        )
    metric = st.selectbox(
        "Metric",
        ("recall_at_5", "mrr", "hit_at_5", "precision_at_5", "retrieval_latency_ms"),
    )
    frame = charts.per_query_strategy_frame(loaded, query_id, metric=metric)
    if frame:
        st.bar_chart(frame, x="system", y=metric)
        st.dataframe(frame, width="stretch")
    else:
        st.info("No stored values for this query/metric combination.")

    with st.expander("Raw stored row (selected arm)"):
        arm = st.selectbox("Arm", sorted(loaded), key="raw_arm")
        row = next((r for r in loaded[arm] if r.get("query_id") == query_id), None)
        if row is not None:
            st.json(row)
