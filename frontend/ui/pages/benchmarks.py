"""Benchmarks: frozen results and the quality–cost frontier."""

from __future__ import annotations

import streamlit as st

from frontend.ui import benchmark, charts
from frontend.ui.components import source_caption


def _e1_tab() -> None:
    st.header("E1 — five-arm comparison (107 queries)")
    table, source = benchmark.load_e1_table()
    rows = table["rows"]
    st.markdown(
        "Phase 7 E1 on the pre-fix 713-chunk corpus — the historical record "
        "the Phase 7 closure gates were computed on. Three facts carry the "
        "rest of the project: **Dense is the quality leader** (0.9283); "
        "**the reranker is the worst arm on every quality metric and the "
        "most expensive**; **adaptive ≡ hybrid on quality** at ~59 ms overhead."
    )
    quality = charts.e1_quality_bars({"rows": rows})
    st.subheader("Graph 1 — Retrieval quality (Recall@5)")
    y_cols = ["Recall@5", "MRR"]
    if any(spec.get("Hit@5") is not None for spec in quality):
        y_cols.append("Hit@5")
    st.bar_chart(quality, x="system", y=y_cols)
    if "Hit@5" not in y_cols:
        st.caption(
            "Hit@5 is absent from the working-tree E1 artifact; "
            "the committed snapshot carries it."
        )
    source_caption(source, table.get("source", table.get("title", "")))

    frontier = charts.quality_vs_latency({"rows": rows})
    st.subheader("Graph 2 — Quality vs efficiency")
    st.scatter_chart(
        frontier,
        x="median_latency_ms",
        y="Recall@5",
    )
    st.caption(
        "The central trade-off. BM25 retrieves in ~2 ms at 0.7788; Dense "
        "spends ~473 ms (mostly one embedding call) for 0.9283; the reranker "
        "spends ~3.5 s and lowers quality to 0.7165."
    )
    source_caption(source, table.get("source", table.get("title", "")))
    st.subheader("Full table")
    st.dataframe(rows, width="stretch")


def _phase8_tab() -> None:
    st.header("Phase 8 — corpus fix before/after (107 queries)")
    snapshot = benchmark.load_headline_snapshot()
    table = snapshot["phase8_e1"]
    st.markdown(
        "The same five arms on the before (618 chunks) and after (613 "
        "chunks, shipping) corpora. Headline: **the defect was real, was "
        "fixed to a measured standard, and did not change retrieval quality "
        "— and did not rescue the reranker** (−0.0016 recall@5, 99/107 ties)."
    )
    deltas = charts.phase8_before_after(table)
    st.subheader("Recall@5 before → after (delta)")
    st.bar_chart(deltas, x="arm", y="delta")
    source_caption("snapshot", table["source"])
    st.dataframe(table["rows"], width="stretch")
    st.markdown(
        "**Methodology note:** 13 of 29 arms in the first sweep were silently "
        "invalid (dropped embedding calls) and were re-run; the comparison "
        "driver now refuses contaminated arms. Trace status — not trace count "
        "— is what establishes that a measurement covers its query set."
    )


def _oracle_tab() -> None:
    st.header("Oracle routing ceiling — is there headroom at all?")
    snapshot = benchmark.load_headline_snapshot()
    oracle = snapshot["oracle_ceiling"]
    st.markdown(
        "A per-query oracle over the four selectable strategies beats the best "
        "fixed strategy (dense) by **+0.0218** recall@5 — but the gain is "
        "carried by only **4 of 107 queries** (98 tie at max), and the latency "
        "prize is not detectable by any existing signal. Gate verdict: "
        "`routing_has_headroom`, which should not be read at face value."
    )
    frame = charts.oracle_comparison(oracle)
    st.subheader("Recall@5 by policy (oracle vs fixed)")
    st.bar_chart(frame, x="policy", y="Recall@5")
    source_caption("snapshot", oracle["source"])
    st.dataframe(oracle["rows"], width="stretch")
    gate = oracle["gate"]
    st.markdown(
        f"*Quality headroom:* +{gate['quality_headroom_after']} (after) / "
        f"+{gate['quality_headroom_before']} (before), threshold +{gate['quality_threshold']}. "
        f"*Frontier(0.01) latency saving:* {gate['frontier_saving_after_ms']} ms — "
        "under a detectability test that failed. *Oracle MRR* (0.8224) is worse "
        "than dense's (0.8645): the oracle maximises recall per query and ignores rank."
    )


def _cheapfirst_tab() -> None:
    st.header("Cheap-first track — Phase 15 powered confirmation")
    snapshot = benchmark.load_headline_snapshot()
    table = snapshot["cheap_first_phase15"]
    st.markdown(
        "The frozen Phase 14 rule (`bm25_slope ≥ −0.915`) applied unchanged to "
        "110 fresh queries per arm. It caught **18 of 25** oracle positives at "
        "~30% spend, avoided ~70% of Dense calls, and beat random spending "
        "(P = 0.001 combined) — yet trailed full Dense by 0.018–0.030 and "
        "missed the pre-registered margin bar by **0.004**. Verdict: "
        "`INSUFFICIENT EVIDENCE` (4 of 5 bars)."
    )
    st.dataframe(table["rows"], width="stretch")
    source_caption("snapshot", table["source"])


def _figures_tab() -> None:
    st.header("Project figures (working tree)")
    figures = benchmark.load_figures()
    if not figures:
        st.info(
            "The project's matplotlib figures live under experiments/phase7/figures/, "
            "which is gitignored and therefore absent on a fresh clone. "
            "The interactive graphs on the other tabs cover the same frozen results."
        )
        return
    for name, path in figures.items():
        st.subheader(name)
        st.image(str(path))


def render() -> None:
    st.title("Benchmarks")
    st.markdown(
        "Frozen, pre-registered results. Every number here comes from a "
        "recorded experiment artifact — never from a live ad-hoc query, "
        "which has no ground-truth labels to score against."
    )
    tabs = st.tabs(
        ["E1 five-arm", "Phase 8 before/after", "Oracle ceiling", "Cheap-first", "Figures"]
    )
    with tabs[0]:
        _e1_tab()
    with tabs[1]:
        _phase8_tab()
    with tabs[2]:
        _oracle_tab()
    with tabs[3]:
        _cheapfirst_tab()
    with tabs[4]:
        _figures_tab()
