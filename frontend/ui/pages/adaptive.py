"""Adaptive Decision: the live routing trace for one query."""

from __future__ import annotations

import streamlit as st

from adaptive_rag.errors import AdaptiveRAGError
from frontend.ui import availability, corpus, service
from frontend.ui.components import (
    deployment_banner,
    live_caption,
    result_card,
    routing_trace_viewer,
)


def render() -> None:
    st.title("Adaptive Decision")
    st.markdown(
        "Watch the Phase 6 router work on **your** query: deterministic "
        "query features, the rule-based routing decision with per-strategy "
        "evidence, the label-free sufficiency check, and the bounded "
        "escalation outcome."
    )
    probe = availability.probe()
    deployment_banner(availability.summarize())

    runnable = service.available_adaptive_strategies()
    st.info(
        "In this deployment the router can select: **"
        + "**, **".join(runnable)
        + "**. "
        + (
            "Without provider credentials this is BM25-only and fully offline — "
            "the trace still shows the real analyze → decide → check → (no-op) mechanics."
            if runnable == ["bm25"]
            else "With credentials set, all four rungs are selectable."
        )
    )
    st.warning(
        "This viewer shows a research prototype, not a proven router. At the "
        "shipped defaults the sufficiency gate never fires on the benchmark, "
        "so adaptive collapses onto its initial pick (Phase 7: adaptive ≡ hybrid)."
    )

    if "query_adaptive" not in st.session_state:
        st.session_state["query_adaptive"] = ""
    suggestions = corpus.suggested_questions(limit=4)
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(
            suggestion["query"][:80] + "…", key=f"adapt_{suggestion['example_id']}"
        ):
            st.session_state["query_adaptive"] = suggestion["query"]
    query = st.text_area(
        "Query", value=st.session_state["query_adaptive"], height=80
    )
    top_k = st.slider("Results (top_k)", 1, 10, 5)

    if st.button("Route and retrieve", type="primary"):
        if not query.strip():
            st.warning("Type a query first.")
            return
        routing = service.adaptive_routing(runnable)
        try:
            with st.spinner("Analyzing, routing, retrieving…"):
                response = service.retrieve(
                    "adaptive", query, top_k=top_k, routing=routing
                )
        except AdaptiveRAGError as exc:
            st.error(service.credential_hint(exc))
            return
        live_caption()
        meta = response.retrieval_metadata
        st.markdown(
            f"**Initial strategy:** `{meta.adaptive_initial_strategy}` → "
            f"**Final strategy:** `{meta.adaptive_final_strategy}` · "
            f"**Escalated:** `{meta.adaptive_escalated}` · "
            f"**Sufficient:** `{meta.adaptive_sufficient}` · "
            f"**Total latency:** `{meta.latency_ms:.2f} ms`"
        )
        trace = meta.routing
        if trace is None:
            st.warning("No routing trace was carried on this response.")
        else:
            routing_trace_viewer(trace)
        st.subheader("Final retrieved evidence")
        paper_index = corpus.papers_by_id()
        for i, result in enumerate(response.results):
            result_card(result, paper_index, index=i)
