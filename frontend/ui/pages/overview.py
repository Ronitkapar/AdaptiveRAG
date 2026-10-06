"""Overview page: hero, problem, pipeline, strategies."""

from __future__ import annotations

import streamlit as st

from frontend.ui import availability, benchmark, corpus
from frontend.ui.components import deployment_banner, strategy_badge

GITHUB_URL = "https://github.com/Ronitkapar/AdaptiveRAG"


def render() -> None:
    st.title("AdaptiveRAG")
    st.subheader("An investigation into query-aware adaptive retrieval")
    st.markdown(
        "AdaptiveRAG asks one question: **when does a query actually need "
        "expensive retrieval, and can retrieval effort be adapted per query "
        "without sacrificing too much quality?** It built four fixed "
        "retrieval strategies plus a rule-based adaptive router over a "
        "frozen 107-query academic-RAG benchmark — and investigated, "
        "measured, failed honestly, and stopped deliberately."
    )
    st.markdown(
        "**Role:** end-to-end research project (retrieval, routing, "
        f"evaluation) · **Status:** research complete, no optimal routing "
        f"policy claimed · **Code:** [{GITHUB_URL}]({GITHUB_URL})"
    )
    deployment_banner(availability.summarize())

    st.header("The problem: a quality–cost frontier")
    st.markdown(
        "Every query in a standard RAG system pays for the most expensive "
        "retrieval configuration, including the many queries a cheap "
        "retriever already answers. Measured on 107 paired queries "
        "(Phase 7 E1):"
    )
    snapshot = benchmark.load_headline_snapshot()
    rows = snapshot["phase7_e1"]["rows"]
    st.dataframe(
        [
            {
                "Strategy": benchmark.resolve_row_system(row),
                "Recall@5": row["recall_at_5"],
                "Median latency": f"{row['median_latency_ms']} ms",
            }
            for row in rows
            if benchmark.resolve_row_system(row)
        ],
        width="stretch",
    )
    st.caption(
        "FROZEN BENCHMARK RESULT (committed snapshot). Dense costs ~250x "
        "BM25's latency, mostly one remote embedding call per query."
    )

    st.header("How it works")
    st.code(
        "Query\n"
        "  → QueryFeatureAnalyzer   (deterministic query features, no model)\n"
        "  → RuleBasedRouter         (strategy + evidence + confidence)\n"
        "  → Retriever               (BM25 / Dense / Hybrid RRF / Hybrid+Rerank)\n"
        "  → SufficiencyChecker      (is the evidence enough? label-free)\n"
        "  → EscalationPolicy        (bounded: at most 1 step up the ladder)\n"
        "  → Retrieved evidence\n"
        "  → strategy-agnostic evaluators (Recall@5, MRR, latency)",
        language="text",
    )
    st.caption(
        "The real pipeline. The demo reuses it through "
        "`instantiate_components` — it reimplements nothing."
    )

    st.header("The strategies")
    cards = [
        (
            "BM25",
            "Okapi BM25 (k1=1.2, b=0.75) over a deterministic tokenizer. "
            "No embedding, no API call, ~2 ms.",
            "Observed: the cheapest arm (Recall@5 0.7788) and the no-API latency control.",
        ),
        (
            "Dense",
            "Bi-encoder `text-embedding-3-large` + Qdrant cosine. One remote "
            "embedding call per query (~374 ms of its ~472 ms).",
            "Observed: the best fixed strategy in every condition measured (Recall@5 0.9283) — the bar every adaptive policy is judged against.",
        ),
        (
            "Hybrid (RRF)",
            "Reciprocal-rank fusion of dense@20 and BM25@20 into a top-10 (rrf_k=60).",
            "Observed: better than BM25 but below Dense (0.8723). Fusion flattens ranked scores.",
        ),
        (
            "Hybrid + Rerank",
            "Hybrid candidates rescored by a cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2` on ONNX Runtime, CPU).",
            "Observed: net-negative on this corpus — worst arm on every quality metric at ~7.5x hybrid's latency. The rung was removed from escalation.",
        ),
        (
            "Adaptive",
            "Analyze → route → check sufficiency → escalate boundedly over the unchanged Phase 2–5 retrievers.",
            "Observed: at shipped defaults the sufficiency gate never fires, so adaptive collapses onto hybrid (identical quality, ~59 ms overhead). A research prototype, not a proven router.",
        ),
    ]
    for title, what, observed in cards:
        with st.expander(f"**{title}**"):
            st.markdown(what)
            st.markdown(f"*{observed}*")

    st.header("The corpus")
    papers = corpus.load_papers()
    counts = corpus.category_counts(papers)
    st.markdown(
        f"**{len(papers)} foundational IR/NLP papers**, each a canonical "
        "open-access PDF with a recorded SHA-256 digest. Categories: "
        + ", ".join(f"{name} ({count})" for name, count in sorted(counts.items()))
        + "."
    )
    st.markdown(
        "Known defects, contained rather than hidden: `pyserini_lin_2021` "
        "holds a biomedical survey, not the Pyserini paper; a two-column "
        "extraction defect was fixed in Phase 8; 47 of 140 gold section "
        "labels are corrupt, so nDCG@5 is excluded from every quality claim."
    )

    st.header("Strategy availability in this deployment")
    summary = availability.summarize()
    for name, info in summary["strategies"].items():
        strategy_badge(name, info)

    st.header("Start exploring")
    st.markdown(
        "* **Query Playground** — ask a live question over the 14-paper corpus.\n"
        "* **Strategy Comparison** — run one query across strategies, side by side.\n"
        "* **Adaptive Decision** — watch the router analyze, decide, check and (sometimes) escalate.\n"
        "* **Benchmarks** — the frozen results and the quality–cost frontier.\n"
        "* **Experiments** — the stored per-query runs behind the tables.\n"
        "* **Research Journey** — fifteen phases, including the failures.\n"
        "* **Findings** — what the evidence supports, and what it does not."
    )
