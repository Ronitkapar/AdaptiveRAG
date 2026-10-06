"""Findings: what the evidence supports, and what it does not."""

from __future__ import annotations

import streamlit as st

GITHUB_URL = "https://github.com/Ronitkapar/AdaptiveRAG"


def render() -> None:
    st.title("Findings")
    st.markdown(
        "Separated by evidential strength — because conflating these is how "
        "negative results get lost. Source: `docs/ADAPTIVERAG_REPORT.md` §6 (committed)."
    )

    st.header("Confirmed findings")
    st.markdown(
        "1. **Per-query retrieval dispersion is real.** T-disp3 > 0 on 21/107 "
        "and 29/107 queries (after/before); not a tie-break artefact.\n"
        "2. **Dense distinct-document ratio tracks that dispersion and "
        "generalizes out of sample** (ρ +0.343/+0.406, held-out +0.410). "
        "Disagreement is measurable, real, and mechanically tied to evidence diffuseness.\n"
        "3. **The reranker is a net-negative rung on this corpus** — worst arm "
        "on every quality metric at 6.2–7.5× hybrid's latency, unrescued by "
        "the corpus fix (−0.0016).\n"
        "4. **Dense is the best fixed strategy in every condition measured**, "
        "and adaptive routing collapsed onto the second-best one at the "
        "shipped configuration.\n"
        "5. **Disagreement does not imply escalation value** — the DDR policy "
        "failed at chance AUC and was dominated by dense-only and random spending.\n"
        "6. **Helps and harms have different mechanisms**: helps are BM25-side "
        "rescue of Dense-missed evidence (State 2); harms are fusion "
        "corrupting an already-correct Dense result (State 3).\n"
        "7. **Cheap first-stage information can select Dense calls better than "
        "random**: 18 of 25 oracle positives at ~30% spend on 220 fresh "
        "query-instances (P = 0.001 combined).\n"
        "8. **Trace status — not trace count — establishes measurement coverage.** "
        "13 of 29 Phase 8 arms were silently invalid while reporting ok."
    )

    st.header("Promising but insufficient")
    st.markdown(
        "* Flat BM25 score decay predicts Dense's marginal value (AUC 0.757 on "
        "calibration) — the strongest pre-cost candidate found, but selected "
        "on 5 positives.\n"
        "* The cheap-first decision point is where the headroom is (16/21 "
        "positives, EV-positive rung) — feasibility established, policy not.\n"
        "* Escalation is affordable when it happens (~31 ms of local fusion "
        "vs ~450 ms of provider call avoided)."
    )

    st.header("Tested and rejected")
    st.markdown(
        "* Rule-based query-aware routing does not beat a fixed strategy here "
        "(gate never opens; 0/24 feature hits).\n"
        "* The reranker rung does not pay for itself.\n"
        "* The corpus defect did not cause the reranker's collapse.\n"
        "* Post-dense escalation on DDR fails (FAILURE).\n"
        "* The observable-signal search is exhausted for this setting.\n"
        "* The RRF-hybrid rung harms more queries than it helps (4 vs 9–12)."
    )

    st.header("What this project does NOT claim")
    st.markdown(
        "No optimal adaptive retrieval algorithm. No production-ready routing "
        "policy. No universal generalization (one corpus, one embedding model, "
        "one fusion method). No causal explanations for the correlations. No "
        "answer-quality conclusion (retrieval-only evaluation). Adaptive "
        "retrieval does not outperform Dense retrieval here: every cheap-first "
        "configuration trailed full Dense by 0.018–0.030 Recall@5."
    )

    st.header("Tech stack")
    st.markdown(
        "Python · **Streamlit** (this demo) · Qdrant (local embedded vector "
        "store) · Okapi BM25 · `text-embedding-3-large` (AICredits) · ONNX "
        "Runtime cross-encoder (`ms-marco-MiniLM-L-6-v2`) · RRF fusion · "
        "pydantic / pydantic-settings · Groq (generation, optional) · pytest "
        "(1056 offline tests)."
    )

    st.header("Links")
    st.markdown(
        f"* Code: [{GITHUB_URL}]({GITHUB_URL})\n"
        "* Full research report: `docs/ADAPTIVERAG_REPORT.md` in the repository\n"
        "* Reproduction commands: `docs/reproducibility/reproduction.md`\n"
        "* Run the demo locally: `pip install -e \".[frontend]\" && "
        "streamlit run frontend/app.py`"
    )
