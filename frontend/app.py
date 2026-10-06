"""AdaptiveRAG Streamlit demo — entry point.

Run with::

    pip install -e ".[frontend]"
    streamlit run frontend/app.py

The app is a presentation layer over the frozen research project. It
reimplements no retrieval: live queries go through the project's own
``instantiate_components`` factory, and every benchmark number comes
from a recorded artifact (committed snapshot on a fresh clone, richer
on-disk records on the research machine).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_imports() -> None:
    """Make ``frontend.*`` and ``adaptive_rag`` importable.

    ``adaptive_rag`` is installed (``pip install -e .``), but the
    ``frontend`` package resolves only when the repository root is on
    ``sys.path`` — which is not guaranteed under ``streamlit run``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_imports()

import streamlit as st  # noqa: E402

from frontend.ui.pages import adaptive, benchmarks, compare, experiments  # noqa: E402
from frontend.ui.pages import findings, journey, overview, playground  # noqa: E402

PAGES: dict[str, tuple[str, object]] = {
    "Overview": ("The project, the problem, the strategies", overview),
    "Query Playground": ("Ask a live question over the corpus", playground),
    "Strategy Comparison": ("One query across strategies, side by side", compare),
    "Adaptive Decision": ("Watch the router analyze, decide and check", adaptive),
    "Benchmarks": ("Frozen results and the quality–cost frontier", benchmarks),
    "Experiments": ("The stored per-query runs behind the tables", experiments),
    "Research Journey": ("Fifteen phases, including the failures", journey),
    "Findings": ("What the evidence supports — and does not", findings),
}


def main() -> None:
    st.set_page_config(
        page_title="AdaptiveRAG — query-aware adaptive retrieval",
        page_icon="🔎",
        layout="wide",
    )
    st.sidebar.title("AdaptiveRAG")
    st.sidebar.caption("Query-aware adaptive retrieval — research demo")
    choice = st.sidebar.radio(
        "Navigate",
        list(PAGES.keys()),
        format_func=lambda name: f"{name} — {PAGES[name][0]}",
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Live results are retrieved on this machine right now. "
        "Benchmark numbers are frozen records, never live scores. "
        "Research complete · no optimal routing policy claimed."
    )
    PAGES[choice][1].render()


main()
