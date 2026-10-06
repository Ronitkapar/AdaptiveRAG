"""Phase-timeline access for the Research Journey page.

A thin, Streamlit-free wrapper over the committed
``frontend/data/phases.json`` snapshot so the journey content is
testable and the page stays a renderer.
"""

from __future__ import annotations

from typing import Any

from frontend.ui.benchmark import load_phase_timeline

TRACK_LABELS: dict[str, str] = {
    "foundation": "Foundation (Phases 1-5)",
    "adaptive": "Adaptive routing (Phases 6-8)",
    "post-dense": "Post-dense track (Phases 9-12)",
    "cheap-first": "Cheap-first track (Phases 13-15)",
}


def phases() -> list[dict[str, Any]]:
    """All 15 phases in order."""
    return load_phase_timeline()


def phases_by_track(track: str) -> list[dict[str, Any]]:
    """The phases belonging to one research track."""
    return [phase for phase in phases() if phase.get("track") == track]


def verdict_style(verdict: str) -> str:
    """A short display hint for a phase verdict."""
    return {
        "COMPLETE": "done",
        "PROMISING_SIGNAL": "signal",
        "FAILURE": "negative",
        "OUTCOME_B": "partial",
        "STOP": "stopped",
        "INSUFFICIENT_EVIDENCE": "inconclusive",
    }.get(verdict, "done")
