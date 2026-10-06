"""Presentation layer for the AdaptiveRAG Streamlit demo.

This package is a thin presentation layer over the frozen research
artifacts and the existing ``adaptive_rag`` retrieval interfaces. It
contains no retrieval algorithms of its own: every ranking, fusion and
scoring step stays in ``adaptive_rag``. The modules here only

* probe what this deployment can actually run (``availability``),
* load committed data and on-disk artifacts (``corpus``, ``benchmark``),
* build retrievers through the project's own factories (``service``),
* and render them (``charts``, ``components``, ``timeline``).
"""
