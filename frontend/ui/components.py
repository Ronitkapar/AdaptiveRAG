"""Shared Streamlit render helpers.

Small, reusable presentational pieces used by several pages:
deployment/strategy badges, honesty labels, the retrieval result
card (the Retrieval Explorer), the latency breakdown, and the
routing-trace viewer. All rendering lives here so the pages stay
thin dispatchers and the wording of every disclaimer is stated
once.

Nothing here computes retrieval results; everything renders data
structures produced by ``frontend.ui.service`` or the loaders.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.ui.availability import LIVE, NEEDS_CREDENTIALS, UNAVAILABLE


def deployment_banner(summary: dict[str, Any]) -> None:
    """One banner stating what this deployment can actually run."""
    live = [
        name
        for name, info in summary["strategies"].items()
        if info["status"] == LIVE
    ]
    if "bm25" in live and len(live) == 1:
        st.info(
            "Running in **offline mode**: BM25 retrieval is live. Provider "
            "strategies show their frozen benchmark results instead of live "
            "retrieval. Nothing on this page is simulated.",
        )
    elif len(live) > 1:
        st.success(
            "Live retrieval available for: **"
            + "**, **".join(live)
            + "**. Latencies below are measured on this machine, right now.",
        )
    else:
        st.warning(
            "No retrieval strategy can run in this deployment (indexes or "
            "credentials missing). The demo shows frozen benchmark results, "
            "clearly labelled as such.",
        )


def strategy_badge(name: str, info: dict[str, Any]) -> None:
    """A one-line availability badge for a strategy."""
    status = info["status"]
    if status == LIVE:
        st.success(f"**{info['label']}** — LIVE · {info['reason']}")
    elif status == NEEDS_CREDENTIALS:
        st.warning(f"**{info['label']}** — OFFLINE-ONLY · {info['reason']}")
    elif status == UNAVAILABLE:
        st.error(f"**{info['label']}** — UNAVAILABLE · {info['reason']}")
    else:  # pragma: no cover -- defensive
        st.caption(f"**{info['label']}** — {status}: {info['reason']}")


def source_caption(source: str, detail: str = "") -> None:
    """Label a number/graph with where it came from."""
    if source == "snapshot":
        st.caption(f"FROZEN BENCHMARK RESULT (committed snapshot). {detail}")
    else:
        st.caption(f"WORKING-TREE ARTIFACT (research machine). {detail}")


def live_caption() -> None:
    """Label a live retrieval result."""
    st.caption("LIVE QUERY RESULT — retrieved just now on this machine. No quality metric such as Recall@5 applies to an ad-hoc query (no ground-truth labels).")


def result_card(result: Any, papers_by_id: dict[str, Any], index: int = 0) -> None:
    """Render one retrieved chunk as an expandable evidence card.

    This is the Retrieval Explorer: rank, score, document, section,
    pages and provenance up front; full chunk text behind an
    expander.
    """
    metadata = result.metadata
    provenance = result.provenance
    paper = papers_by_id.get(metadata.document_id, {})
    title = metadata.doc_title or paper.get("title", metadata.document_id)
    section = " / ".join(metadata.section_path) if metadata.section_path else "—"
    pages = (
        f"{metadata.page_start}-{metadata.page_end}"
        if metadata.page_start is not None and metadata.page_end is not None
        else "—"
    )

    header = f"#{result.rank} · {title} · score {result.score:.4f}"
    with st.expander(header, expanded=(index == 0)):
        st.markdown(f"**Document:** {title} (`{metadata.document_id}`)")
        st.markdown(f"**Section:** {section}")
        cols = st.columns(3)
        cols[0].markdown(f"**Pages:** {pages}")
        cols[1].markdown(f"**Rank:** {result.rank}")
        cols[2].markdown(f"**Chunk:** `{result.chunk_id}`")
        if result.retrieval_score is not None or result.retrieval_rank is not None:
            st.markdown(
                f"*First-stage signal preserved:* score "
                f"`{result.retrieval_score}`, rank `{result.retrieval_rank}`"
            )
        st.markdown("**Chunk text:**")
        st.write(result.text)
        with st.expander("Provenance"):
            st.json(
                {
                    "document_id": provenance.document_id,
                    "pages": provenance.pages,
                    "source_sha256": provenance.source_sha256,
                    "element_types": metadata.element_types,
                    "token_count": metadata.token_count,
                    "char_count": metadata.char_count,
                }
            )


def latency_panel(response: Any) -> None:
    """Render what a retrieval run cost, from the response metadata."""
    meta = response.retrieval_metadata
    st.markdown(
        f"**Retrieval method:** `{response.retrieval_method}` · "
        f"**Status:** `{response.status}` · "
        f"**Total latency:** `{meta.latency_ms:.2f} ms` · "
        f"**Results:** `{len(response.results)}`"
    )
    stages: dict[str, Any] = {}
    for field in (
        "search_latency_ms",
        "query_embedding_latency_ms",
        "dense_latency_ms",
        "bm25_latency_ms",
        "fusion_latency_ms",
        "candidate_generation_latency_ms",
        "rerank_latency_ms",
        "adaptive_routing_latency_ms",
        "adaptive_initial_latency_ms",
    ):
        value = getattr(meta, field, None)
        if value is not None:
            stages[field] = round(value, 3)
    detail = {
        "top_k": meta.top_k,
        "retriever_version": meta.retriever_version,
        "index_id": meta.index_id,
        "corpus_version": meta.corpus_version,
    }
    if meta.fusion_method:
        detail["fusion_method"] = meta.fusion_method
        detail["rrf_k"] = meta.rrf_k
        detail["candidate_k"] = meta.candidate_k
        detail["dense_candidate_count"] = meta.dense_candidate_count
        detail["bm25_candidate_count"] = meta.bm25_candidate_count
    if meta.rerank_enabled:
        detail["reranker_model_id"] = meta.reranker_model_id
        detail["rerank_candidate_k"] = meta.rerank_candidate_k
    with st.expander("Latency breakdown & provenance"):
        if stages:
            st.json(stages)
        st.json(detail)


def routing_trace_viewer(trace: Any) -> None:
    """Render a full Phase 6 routing trace for one query.

    Four stages mirror the pipeline: query features, the routing
    decision with per-strategy evidence, the sufficiency judgement,
    and the escalation outcome. The confidence note is rendered every
    time, because a normalized score margin is not a probability.
    """
    st.subheader("1 · Query features")
    features = trace.features
    st.dataframe(
        [
            {"feature": "query_length_words", "value": features.query_length_words},
            {"feature": "query_length_chars", "value": features.query_length_chars},
            {"feature": "content_term_count", "value": features.content_term_count},
            {"feature": "lexical_density", "value": round(features.lexical_density, 4)},
            {"feature": "entity_indicator_count", "value": features.entity_indicator_count},
            {"feature": "technical_term_count", "value": features.technical_term_count},
            {"feature": "semantic_indicator_count", "value": features.semantic_indicator_count},
            {"feature": "question_type", "value": features.question_type},
            {"feature": "concept_count", "value": features.concept_count},
            {"feature": "multi_concept", "value": features.multi_concept},
            {"feature": "comparison_indicator_count", "value": features.comparison_indicator_count},
            {"feature": "complexity_score", "value": round(features.complexity_score, 4)},
        ],
        width="stretch",
    )

    st.subheader("2 · Routing decision")
    decision = trace.decision
    st.markdown(
        f"**Selected strategy:** `{decision.strategy}` · "
        f"**Confidence:** `{decision.confidence:.4f}` · "
        f"**Router:** `{decision.router_kind} {decision.router_version}`"
    )
    st.caption(
        "Confidence is a normalized score margin relative to the runner-up — "
        "a routing-strength signal, NOT a calibrated probability, and it never "
        "gates sufficiency on its own."
    )
    evidence_rows = [
        {
            "strategy": ev.strategy,
            "score": round(ev.score, 4),
            "cost_penalty": round(ev.cost_penalty, 4),
            **{group: round(value, 4) for group, value in ev.contributions.items()},
        }
        for ev in decision.evidence
    ]
    if evidence_rows:
        st.markdown("**Per-strategy evidence (score = weighted contributions − cost penalty):**")
        st.dataframe(evidence_rows, width="stretch")
    st.markdown(
        f"`candidate_k={decision.candidate_k}` · `final_top_k={decision.final_top_k}` · "
        f"`reranking_required={decision.reranking_required}` · `cost_weight={decision.cost_weight}`"
    )
    if decision.feature_groups_disabled:
        st.markdown(f"Disabled feature groups: `{decision.feature_groups_disabled}`")

    st.subheader("3 · Sufficiency judgement")
    sufficiency = trace.sufficiency
    if sufficiency is None:
        st.markdown("No sufficiency check ran for this query.")
    else:
        st.markdown(
            f"**Sufficient:** `{sufficiency.sufficient}` · "
            f"**Score:** `{sufficiency.score:.4f}` vs threshold `{sufficiency.threshold}`"
        )
        st.markdown(f"*Reason:* {sufficiency.reason}")
        if sufficiency.signals:
            st.dataframe(
                [
                    {
                        "signal": signal.name,
                        "value": round(signal.value, 4),
                        "passed": signal.passed,
                        "weight": signal.weight,
                    }
                    for signal in sufficiency.signals
                ],
                width="stretch",
            )

    st.subheader("4 · Escalation outcome")
    escalation = trace.escalation
    if escalation is None or not escalation.escalated:
        st.markdown("No escalation — the initial strategy's results were kept.")
    else:
        st.markdown(
            f"**Escalated:** `{escalation.from_strategy}` → `{escalation.to_strategy}` · "
            f"step {escalation.step_index + 1}/{escalation.max_steps + 1}"
        )
        st.markdown(f"*Reason:* {escalation.reason}")

    st.subheader("Trace summary")
    st.markdown(
        f"`{trace.initial_strategy}` → `{trace.final_strategy}` · "
        f"{trace.stage_count} stage(s) · "
        f"stage latencies `{[round(v, 2) for v in trace.stage_latencies_ms]} ms` · "
        f"routing latency `{trace.routing_latency_ms:.2f} ms`"
    )
    if trace.initial_chunk_ids is not None:
        st.caption(
            "Pre-escalation evidence was discarded by escalation; "
            f"{len(trace.initial_chunk_ids)} initial chunk ids were recorded."
        )
