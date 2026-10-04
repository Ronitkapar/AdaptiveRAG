"""
tests.test_phase8_arm_comparison
---------------------------------
The Phase 8 before/after comparison driver (`scripts/compare_phase8_arms.py`).

The behaviour pinned here is the contamination guard, because that guard exists
because of a real failure: a Phase 8 run lost embedding calls on the reranker,
`metrics_retrieval.json` reported the surviving 59 queries as `n=59` beside
`trace_count=107` with no error, and the arm read as a complete result. Nothing
downstream noticed, because every consumer trusted the aggregate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_phase8_arms", REPO_ROOT / "scripts" / "compare_phase8_arms.py"
)
assert SPEC is not None and SPEC.loader is not None
compare_phase8_arms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_phase8_arms)

ContaminatedArm = compare_phase8_arms.ContaminatedArm


def _write_arm(
    suite: Path,
    system: str,
    *,
    n: int,
    failed: int,
    metrics: tuple[str, ...] = ("recall_at_5", "mrr", "total_latency_ms"),
) -> Path:
    """Materialise one arm: a run directory with `rows.jsonl` and `traces.jsonl`.

    Failed traces are appended *after* the good ones, which is what the real
    runner does, and they still occupy a row -- the property the guard's
    denominator depends on.
    """
    run_dir = suite / f"p8__E1_baseline_comparison__{system}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for index in range(n):
        query_id = f"q{index:03d}"
        value = 1.0 if index % 2 == 0 else 0.0
        rows.append(
            {
                "query_id": query_id,
                "system": system,
                **{metric: value for metric in metrics},
            }
        )
        traces.append({"query_id": query_id, "system": system, "status": "ok"})

    for index in range(failed):
        query_id = f"f{index:03d}"
        rows.append({"query_id": query_id, "system": system, "status": "failed"})
        traces.append(
            {
                "query_id": query_id,
                "system": system,
                "status": "retrieval_failed",
                "error": {
                    "error_type": "EmbeddingAPIError",
                    "stage": "retrieval",
                    "message": "AICredits embedding request failed: Connection error.",
                },
            }
        )

    (run_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "traces.jsonl").write_text(
        "".join(json.dumps(trace) + "\n" for trace in traces), encoding="utf-8"
    )
    return run_dir


def test_a_clean_arm_loads_and_keeps_its_rows(tmp_path):
    suite = tmp_path / "clean"
    _write_arm(suite, "bm25", n=8, failed=0)

    loaded = compare_phase8_arms.load_arm(suite, ["bm25"])

    assert list(loaded) == ["bm25"]
    assert len(loaded["bm25"]) == 8


def test_an_arm_with_failed_traces_is_refused(tmp_path):
    """The guard's whole purpose: a partial arm must not become a comparison."""
    suite = tmp_path / "partial"
    _write_arm(suite, "hybrid_rerank", n=59, failed=48)

    with pytest.raises(ContaminatedArm) as excinfo:
        compare_phase8_arms.load_arm(suite, ["hybrid_rerank"])

    message = str(excinfo.value)
    # The real failure's numbers, so the message cannot drift from the case it
    # exists to describe.
    assert "48 of 107" in message
    assert "59" in message


def test_the_refusal_names_the_shortfall_as_comparability_not_correctness(tmp_path):
    suite = tmp_path / "partial"
    _write_arm(suite, "dense", n=3, failed=1)

    with pytest.raises(ContaminatedArm) as excinfo:
        compare_phase8_arms.load_arm(suite, ["dense"])

    assert "not comparable to a full arm" in str(excinfo.value)


def test_allow_contaminated_records_the_shortfall_instead_of_dropping_it(tmp_path):
    suite = tmp_path / "partial"
    _write_arm(suite, "adaptive", n=33, failed=14)
    compare_phase8_arms.CONTAMINATION.clear()

    loaded = compare_phase8_arms.load_arm(
        suite, ["adaptive"], allow_contaminated=True
    )

    assert len(loaded["adaptive"]) == 47
    recorded = compare_phase8_arms.CONTAMINATION["arms"]
    assert len(recorded) == 1
    assert recorded[0]["failed"] == 14
    assert recorded[0]["evaluated"] == 33
    assert recorded[0]["total"] == 47


def test_a_missing_trace_file_is_not_treated_as_a_failure(tmp_path):
    """Absence of evidence about failures is not evidence of failures.

    An older or hand-built run directory may carry `rows.jsonl` alone. Refusing it
    would make the guard refuse well-formed input.
    """
    suite = tmp_path / "rows_only"
    run_dir = suite / "p8__E1_baseline_comparison__bm25"
    run_dir.mkdir(parents=True)
    (run_dir / "rows.jsonl").write_text(
        json.dumps({"query_id": "q1", "system": "bm25", "recall_at_5": 1.0}) + "\n",
        encoding="utf-8",
    )

    loaded = compare_phase8_arms.load_arm(suite, ["bm25"])

    assert len(loaded["bm25"]) == 1


def test_rows_from_a_different_system_are_refused(tmp_path):
    """A mis-globbed rows file would compare corpora that were never varied."""
    suite = tmp_path / "mislabelled"
    run_dir = _write_arm(suite, "bm25", n=4, failed=0)
    path = run_dir / "rows.jsonl"
    path.write_text(
        "".join(
            json.dumps({"query_id": f"q{i}", "system": "dense", "recall_at_5": 1.0})
            + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="carries system"):
        compare_phase8_arms.load_arm(suite, ["bm25"])


def test_ndcg_is_withheld_by_default_and_the_reason_is_recorded():
    """The exclusion is a measurement decision, so it must travel with the file."""
    assert "ndcg_at_5" not in compare_phase8_arms.DEFAULT_METRICS
    assert "ndcg_at_5" in compare_phase8_arms.PRE_REGISTERED_METRICS
    assert "spliced" in compare_phase8_arms.WITHHELD_METRICS["ndcg_at_5"]


def test_before_vs_after_pairs_each_system_with_itself(tmp_path):
    """The headline family is each arm against the same arm, not arm against arm."""
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_arm(before, "bm25", n=10, failed=0)
    _write_arm(after, "bm25", n=10, failed=0)

    report = compare_phase8_arms.before_vs_after(
        compare_phase8_arms.load_arm(before, ["bm25"]),
        compare_phase8_arms.load_arm(after, ["bm25"]),
        systems=["bm25"],
        metrics=["recall_at_5"],
        seed=1,
        n_resamples=64,
        confidence=0.95,
        alpha=0.05,
    )

    family = report["reports"]["recall_at_5"]
    assert len(family) == 1
    assert family[0]["treatment"] == "bm25@phase8_after"
    assert family[0]["baseline"] == "bm25@phase8_before"
    assert report["treatment_arm"] == "phase8_after"
