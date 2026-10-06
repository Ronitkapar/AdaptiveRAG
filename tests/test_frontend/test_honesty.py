"""Honesty guards: no faked results, no overclaiming.

Two structural guarantees the demo must keep:

1. Every frozen number in the committed snapshot names its source
   artifact, so a rendered figure can always be traced.
2. The presentation layer never uses the project's forbidden
   vocabulary (optimal routing, best strategy, proven, ...).
"""

from pathlib import Path

from adaptive_rag.config.paths import REPO_ROOT

FRONTEND_DIR = REPO_ROOT / "frontend"

FORBIDDEN_PHRASES = (
    "optimal routing",
    "optimal adaptive",
    "best retrieval strategy",
    "state-of-the-art",
    "always chooses the best",
    "proven adaptive",
    "production-ready routing",
)


def _frontend_text_files():
    return [
        path
        for path in FRONTEND_DIR.rglob("*")
        if path.suffix in {".py", ".json"}
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
    ]


def test_snapshot_numbers_all_have_sources():
    import json

    snapshot = json.loads((FRONTEND_DIR / "data" / "headline.json").read_text())
    for key in ("phase7_e1", "phase8_e1", "oracle_ceiling", "cheap_first_phase15"):
        assert snapshot[key].get("source"), f"{key} must name its source artifact"


def test_no_forbidden_claims_in_presentation_layer():
    # A forbidden phrase is a violation only when asserted, not when
    # explicitly denied ("no optimal routing policy claimed" is the
    # honest disclaimer, not an overclaim). A line counts as a denial
    # when a negation scopes over the phrase.
    negations = ("no ", "not ", "never ", "n't ", "without ", "does not")
    violations: list[str] = []
    for path in _frontend_text_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").lower().splitlines(), start=1
        ):
            for phrase in FORBIDDEN_PHRASES:
                if phrase not in line:
                    continue
                before = line.split(phrase)[0]
                # A negation anywhere before the phrase on the same line
                # scopes over it (all current denials are single-line).
                if any(neg in before for neg in negations):
                    continue
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {phrase!r}"
                )
    assert not violations, "forbidden overclaiming vocabulary: " + "; ".join(violations)


def test_live_vs_offline_labels_exist_in_components():
    text = (FRONTEND_DIR / "ui" / "components.py").read_text(encoding="utf-8")
    assert "LIVE QUERY RESULT" in text
    assert "FROZEN BENCHMARK RESULT" in text


def test_page_modules_import_without_side_effects():
    # Importing a page must define render() without executing Streamlit
    # calls (no app runtime exists under pytest).
    import importlib

    for module in (
        "overview",
        "playground",
        "compare",
        "adaptive",
        "benchmarks",
        "experiments",
        "journey",
        "findings",
    ):
        mod = importlib.import_module(f"frontend.ui.pages.{module}")
        assert callable(mod.render), f"{module}.render must exist"
