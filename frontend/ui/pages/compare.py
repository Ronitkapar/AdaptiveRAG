"""Strategy Comparison: one query, several strategies, side by side."""

from __future__ import annotations

import streamlit as st

from frontend.ui import availability, corpus, service
from frontend.ui.components import deployment_banner, live_caption, result_card


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


def _overlap_table(records: list[dict]) -> list[dict]:
    """Pairwise Jaccard@k over retrieved chunk-id sets (computed, not stored)."""
    table: list[dict] = []
    live = [r for r in records if r["response"] is not None]
    for i, left in enumerate(live):
        left_ids = {r.chunk_id for r in left["response"].results}
        for right in live[i + 1 :]:
            right_ids = {r.chunk_id for r in right["response"].results}
            table.append(
                {
                    "pair": f"{left['strategy']} × {right['strategy']}",
                    "Jaccard@k (chunks)": round(_jaccard(left_ids, right_ids), 3),
                    f"{left['strategy']} docs": len(
                        {r.metadata.document_id for r in left["response"].results}
                    ),
                    f"{right['strategy']} docs": len(
                        {r.metadata.document_id for r in right["response"].results}
                    ),
                }
            )
    return table


def render() -> None:
    st.title("Strategy Comparison")
    st.markdown(
        "Run the **same query** through several strategies and compare what "
        "each one retrieves. Arms run **sequentially** (local Qdrant holds an "
        "exclusive lock, and provider arms embed remotely), with a short "
        "pace between provider calls."
    )
    probe = availability.probe()
    deployment_banner(availability.summarize())

    live = [s for s in ("bm25", "dense", "hybrid", "hybrid_rerank") if probe[s].is_live]
    if len(live) < 2:
        st.info(
            "Comparison needs at least two live strategies. Right now only "
            f"`{live}` can run here — the Playground still works for single-strategy retrieval, "
            "and Benchmarks holds the frozen five-arm table."
        )
        if not live:
            return
    default = [s for s in ("bm25", "dense") if s in live] or live[:1]
    chosen = st.multiselect(
        "Strategies to compare (run sequentially)",
        options=live,
        default=default,
    )
    if "hybrid_rerank" in chosen:
        st.warning(
            "The reranked arm runs a local ONNX cross-encoder (~3 s CPU for 20 "
            "candidates). Expect this comparison to take a while."
        )
    if "query_compare" not in st.session_state:
        st.session_state["query_compare"] = ""
    query = st.text_area(
        "Query",
        value=st.session_state["query_compare"],
        height=80,
        placeholder="Ask about the corpus — or paste a benchmark question from the Playground.",
    )
    top_k = st.slider("Results per strategy (top_k)", 1, 10, 5)

    if st.button("Run comparison", type="primary"):
        if not query.strip():
            st.warning("Type a query first.")
            return
        if len(chosen) < 1:
            st.warning("Select at least one live strategy.")
            return
        records: list[dict] = []
        progress = st.progress(0.0, text="Starting…")
        for i, strategy in enumerate(chosen):
            progress.progress(
                i / len(chosen), text=f"Retrieving with {strategy} ({i + 1}/{len(chosen)})…"
            )
            try:
                response = service.retrieve(strategy, query, top_k=top_k)
                records.append(
                    {"strategy": strategy, "response": response, "error": None}
                )
            except Exception as exc:  # noqa: BLE001 -- reported, not hidden
                records.append(
                    {
                        "strategy": strategy,
                        "response": None,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
            if strategy != "bm25" and i < len(chosen) - 1:
                import time

                time.sleep(service.PROVIDER_PACE_SECONDS)
        progress.progress(1.0, text="Done.")
        live_caption()

        st.subheader("Headline comparison")
        st.dataframe(
            [
                (
                    {
                        "strategy": rec["strategy"],
                        "results": len(rec["response"].results),
                        "top-1 document": rec["response"].results[0].metadata.doc_title
                        if rec["response"].results
                        else "—",
                        "top-1 score": round(rec["response"].results[0].score, 4)
                        if rec["response"].results
                        else None,
                        "latency_ms": round(
                            rec["response"].retrieval_metadata.latency_ms, 2
                        ),
                        "method": rec["response"].retrieval_method,
                    }
                    if rec["response"] is not None
                    else {
                        "strategy": rec["strategy"],
                        "error": f"{rec['error']['type']}: {rec['error']['message']}",
                    }
                )
                for rec in records
            ],
            width="stretch",
        )

        overlap = _overlap_table(records)
        if overlap:
            st.subheader("Result-set overlap (Jaccard over chunk ids)")
            st.dataframe(overlap, width="stretch")
            st.caption(
                "Overlap is computed from the chunk ids retrieved just now — "
                "it measures agreement between strategies on this query, not quality."
            )

        st.subheader("Side-by-side results")
        paper_index = corpus.papers_by_id()
        columns = st.columns(len(records))
        for column, rec in zip(columns, records):
            with column:
                st.markdown(f"### `{rec['strategy']}`")
                if rec["response"] is None:
                    st.error(f"{rec['error']['type']}: {rec['error']['message']}")
                    continue
                st.markdown(
                    f"`{rec['response'].retrieval_metadata.latency_ms:.1f} ms` · "
                    f"{len(rec['response'].results)} results"
                )
                for i, result in enumerate(rec["response"].results):
                    result_card(result, paper_index, index=i)
