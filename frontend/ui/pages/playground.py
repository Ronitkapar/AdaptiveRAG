"""Query Playground: single-query live retrieval over the corpus."""

from __future__ import annotations

import streamlit as st

from adaptive_rag.errors import AdaptiveRAGError
from frontend.ui import availability, benchmark, corpus, service
from frontend.ui.components import (
    deployment_banner,
    latency_panel,
    live_caption,
    result_card,
    source_caption,
)

STRATEGY_ORDER = ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")


def _frozen_fallback(strategy: str) -> None:
    """Show a strategy's frozen benchmark number when it cannot run live."""
    snapshot = benchmark.load_headline_snapshot()
    rows = snapshot["phase7_e1"]["rows"]
    row = next((r for r in rows if benchmark.resolve_row_system(r) == strategy), None)
    if row is None:
        st.info("No frozen benchmark entry for this strategy.")
        return
    st.markdown(
        f"Frozen benchmark (107 queries): **Recall@5 {row['recall_at_5']}**, "
        f"MRR {row['mrr']}, median latency {row['median_latency_ms']} ms."
    )
    source_caption("snapshot", snapshot["phase7_e1"]["source"])


def render() -> None:
    st.title("Query Playground")
    st.markdown(
        "Ask a question about the 14-paper corpus and see what a strategy "
        "retrieves — live, on this machine, right now."
    )
    probe = availability.probe()
    deployment_banner(availability.summarize())

    with st.expander("What can I ask about? The corpus", expanded=False):
        papers = corpus.load_papers()
        paper_index = corpus.papers_by_id()
        for paper in papers:
            st.markdown(
                f"* **{paper['title']}** ({paper['year']}) — "
                f"`{paper['document_id']}` · {paper['category']}"
            )
        st.caption(f"{len(papers)} papers · manifest: data/metadata/papers.json")

    st.subheader("Try a real benchmark question")
    suggestions = corpus.suggested_questions()
    if "playground_query" not in st.session_state:
        st.session_state["playground_query"] = ""
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(
            suggestion["query"][:90] + "…",
            key=f"suggest_{suggestion['example_id']}",
            help=f"Benchmark id {suggestion['example_id']} ({suggestion['category']})",
        ):
            st.session_state["playground_query"] = suggestion["query"]
    st.caption(
        "Suggestions are real questions from the frozen phase7_eval_v1 benchmark, not invented examples."
    )

    query = st.text_area(
        "Your question",
        value=st.session_state["playground_query"],
        height=90,
        placeholder="e.g. What is the difference between RAG-Sequence and RAG-Token?",
    )

    live_strategies = [s for s in STRATEGY_ORDER if probe[s].is_live]
    options = [
        f"{s} — LIVE" if probe[s].is_live else f"{s} — {probe[s].status}"
        for s in STRATEGY_ORDER
    ]
    choice = st.selectbox("Strategy", options, index=0)
    strategy = STRATEGY_ORDER[options.index(choice)]
    top_k = st.slider("Results (top_k)", min_value=1, max_value=10, value=5)
    corpus_arm = st.selectbox(
        "Corpus arm",
        ("phase8_after", "phase8_before", "phase7"),
        index=0,
        help="phase8_after is the shipping corpus. phase7 is the historical pre-fix namespace.",
    )
    for name in STRATEGY_ORDER:
        if not probe[name].is_live:
            st.caption(f"`{name}`: {probe[name].reason}")

    if st.button("Retrieve", type="primary"):
        if not query.strip():
            st.warning("Type a question first — or pick one of the benchmark suggestions above.")
            return
        if strategy == "adaptive":
            st.info(
                "Adaptive routing composes the strategies this deployment can run. "
                "Its full decision trace is shown on the Adaptive Decision page; "
                "here are its final retrieved results."
            )
            routing = service.adaptive_routing(service.available_adaptive_strategies())
        else:
            routing = None
        if not probe[strategy].is_live:
            st.warning(
                f"`{strategy}` cannot run live in this deployment: {probe[strategy].reason}"
            )
            _frozen_fallback(strategy)
            return
        try:
            with st.spinner(f"Retrieving with {strategy}…"):
                if strategy == "adaptive":
                    response = service.retrieve(
                        strategy, query, top_k=top_k, corpus_arm=corpus_arm, routing=routing
                    )
                else:
                    response = service.retrieve(
                        strategy, query, top_k=top_k, corpus_arm=corpus_arm
                    )
        except AdaptiveRAGError as exc:
            st.error(service.credential_hint(exc))
            _frozen_fallback(strategy)
            return
        live_caption()
        latency_panel(response)
        if response.status == "no_results" or not response.results:
            st.info("No chunks retrieved for this query.")
            return
        paper_index = corpus.papers_by_id()
        for i, result in enumerate(response.results):
            result_card(result, paper_index, index=i)
