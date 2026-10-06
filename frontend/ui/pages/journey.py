"""Research Journey: fifteen phases, including the failures."""

from __future__ import annotations

import streamlit as st

from frontend.ui import timeline


def _phase_card(phase: dict) -> None:
    header = f"Phase {phase['n']} — {phase['name']} · `{phase['verdict']}`"
    with st.expander(header):
        st.markdown(f"**Objective.** {phase['objective']}")
        st.markdown(f"**Hypothesis.** {phase['hypothesis']}")
        st.markdown(f"**Experiment.** {phase['experiment']}")
        st.markdown(f"**Result.** {phase['result']}")
        st.markdown(f"**Decision.** {phase['decision']}")


def _journey_tab() -> None:
    st.header("Fifteen phases, one stopping point")
    track = st.selectbox(
        "Research track",
        ("all", "foundation", "adaptive", "post-dense", "cheap-first"),
        format_func=lambda t: "All phases" if t == "all" else timeline.TRACK_LABELS[t],
    )
    phases = timeline.phases() if track == "all" else timeline.phases_by_track(track)
    for phase in phases:
        _phase_card(phase)
    st.caption("Source: docs/history/experiment_timeline.md (committed).")


def _failure_tab() -> None:
    st.header("Failure analysis — why escalation helps or hurts")
    st.markdown(
        "Phase 10's post-dense escalation policy failed cleanly (AUC 0.550, "
        "test Δ −0.050 vs dense-only, dominated by dense-only *and* by random "
        "spending). Phase 11 then decomposed every helped and harmed query to "
        "find the mechanism — and found two different ones."
    )
    st.subheader("The three states")
    st.markdown(
        "* **State 2 — Dense missed it.** Dense's Recall@5 is 0–0.33; BM25 "
        "holds the relevant document and RRF keeps it in the top-5. Every "
        "helped query (5 unique across both arms) is State 2: BM25-side "
        "rescue of Dense-missed evidence.\n"
        "* **State 3 — Dense had it.** `dense_recall_at_5 > 0` and fusion "
        "corrupted the ranking. Every harmed query (9–12) is State 3.\n\n"
        "The mapping is exact, not approximate — and it cannot be told apart "
        "before paying (bounded separation 1/6 and 0/6 over six pre-declared "
        "features; only 5 positives)."
    )
    st.subheader("H1 — intruder promotion")
    st.markdown(
        "`bm25_only` documents that BM25 ranks confidently but irrelevantly "
        "(e.g. crag, retro, realm, contriever) displace the relevant document "
        "from the top-5. The cheap retriever's confidence is not evidence of "
        "relevance."
    )
    st.subheader("H2 — re-ranking among already-shared documents")
    st.markdown(
        "For several harmed queries **no new document enters at all**: RRF "
        "reorders the shared pool and the relevant document's chunks fall "
        "below the cutoff (one query's Hybrid top-5 is five chunks of a "
        "single wrong document). This is the Phase 4 RRF-flattening effect at "
        "per-query resolution — even an agreeing BM25 can preside over harm. "
        "Agreement is not safety either."
    )
    st.subheader("The lesson")
    st.markdown(
        "> **Retrieval disagreement ≠ retrieval insufficiency ≠ escalation value.**\n\n"
        "Phase 9's correlation stands — Dense distinct-document ratio tracks "
        "cross-strategy dispersion (ρ +0.343/+0.406, out-of-sample +0.410). "
        "But dispersion here is dominated by *Dense succeeding where fusion "
        "degrades the ranking*, so the signal marks disagreement without "
        "marking escalation value. Correlation with dispersion did not predict "
        "gain — and reporting that mismatch honestly is the project's most "
        "valuable result."
    )
    st.caption(
        "Sources: docs/phase10_results.md, docs/phase11_results.md, docs/ADAPTIVERAG_REPORT.md §5.4–5.5 (all committed)."
    )


def render() -> None:
    st.title("Research Journey")
    st.markdown(
        "How the investigation moved from building retrieval strategies to a "
        "deliberate stopping point — with the negative results preserved as "
        "the main scientific output."
    )
    tabs = st.tabs(["Phase timeline", "Failure analysis"])
    with tabs[0]:
        _journey_tab()
    with tabs[1]:
        _failure_tab()
