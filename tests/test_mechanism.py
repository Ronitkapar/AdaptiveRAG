"""Phase 11 mechanism analysis -- unit tests over synthetic records.

Pure functions only: no index, no provider, no artifact on disk. The tests pin
the frozen contract from `docs/phases/phase-11.md`: group boundaries reuse the
Phase 10 oracle rule unchanged, change attribution is exact set arithmetic,
and labels never leak beyond the change-attribution fields.
"""

from __future__ import annotations

import pytest

from adaptive_rag.evaluation.dispersion import EPSILON
from adaptive_rag.evaluation.mechanism import (
    ATTRIBUTION_DEPTH,
    GROUPS,
    TOP_K,
    assign_group,
    build_mechanism_record,
    dense_confidence,
    hybrid_change,
    separation_check,
)


def test_groups_and_epsilon_are_frozen():
    assert GROUPS == ("helps", "ties", "harms")
    assert EPSILON == 0.01
    assert TOP_K == 5
    assert ATTRIBUTION_DEPTH == 10


def test_assign_group_boundaries_match_phase10_oracle():
    assert assign_group(0.5) == "helps"
    assert assign_group(EPSILON) == "helps"
    assert assign_group(0.009) == "ties"
    assert assign_group(0.0) == "ties"
    assert assign_group(-0.009) == "ties"
    assert assign_group(-EPSILON) == "harms"
    assert assign_group(-0.5) == "harms"


def test_hybrid_change_attributes_entered_and_left():
    change = hybrid_change(
        dense_ids=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        hybrid_ids=["a", "x", "c", "f", "e", "b", "g", "h", "i", "j"],
        bm25_ids=["x", "f", "z", "a", "b", "c", "d", "e", "g", "h", "i"],
        relevant_docs=["x", "b", "d"],
    )
    assert change["entered_docs"] == ["f", "x"]
    assert change["left_docs"] == ["b", "d"]
    assert change["relevant_entered"] == ["x"]
    assert change["relevant_left"] == ["b", "d"]
    # x lives only in the BM25 pool; f sits in both top-10 pools while
    # outside the dense top-5, so fusion -- not BM25 alone -- promoted it.
    assert change["entered_source"] == {"x": "bm25_only", "f": "both"}


def test_hybrid_change_marks_unpersisted_depth_neither():
    change = hybrid_change(
        dense_ids=["a", "b", "c", "d", "e"],
        hybrid_ids=["a", "b", "c", "d", "q"],
        bm25_ids=["a", "b", "c", "d", "e"],
        relevant_docs=["q"],
    )
    assert change["entered_source"] == {"q": "neither"}
    assert change["relevant_entered"] == ["q"]


def test_dense_confidence_reuses_frozen_forms():
    confidence = dense_confidence([0.6, 0.5, 0.4, 0.3, 0.2])
    assert confidence["top1_score"] == pytest.approx(0.6)
    assert confidence["score_gap"] == pytest.approx((0.6 - 0.5) / 0.6)
    assert confidence["score_slope"] is not None
    assert confidence["score_slope"] < 0


def test_dense_confidence_empty_scores_are_none():
    assert dense_confidence([]) == {
        "top1_score": None,
        "score_gap": None,
        "score_slope": None,
    }


def _trace(ids, scores=None):
    scores = scores if scores is not None else [0.5] * len(ids)
    return {
        "retrieval": {
            "results": [
                {
                    "chunk_id": f"c{i}",
                    "score": s,
                    "metadata": {"document_id": doc},
                }
                for i, (doc, s) in enumerate(zip(ids, scores))
            ]
        }
    }


def _oracle_row(query_id="q", delta=1.0):
    return {
        "query_id": query_id,
        "corpus_arm": "after",
        "split": "calibration",
        "category": "factual",
        "oracle_escalate": delta >= EPSILON,
        "dense_recall_at_5": 0.0 if delta > 0 else 1.0,
        "hybrid_recall_at_5": 1.0 if delta > 0 else 0.0,
        "delta_recall_at_5": delta,
        "dense_mrr": 0.0,
        "hybrid_mrr": 1.0,
    }


def _signal_row():
    return {
        "distinct_doc_ratio": {"dense": 0.8},
        "jaccard": {"bm25|dense": 0.3},
        "union_concentration": 0.7,
        "top1_agreement": {"bm25|dense": False},
        "rank_correlation": {"bm25|dense": None},
    }


def test_build_record_help_state_and_label_lock():
    record = build_mechanism_record(
        oracle_row=_oracle_row("h", delta=1.0),
        signal_row=_signal_row(),
        dense_trace=_trace(["a", "b", "c", "d", "e"]),
        bm25_trace=_trace(["x", "a", "b", "c", "d"]),
        hybrid_trace=_trace(["x", "a", "b", "c", "d"]),
        relevant_docs=["x"],
    )
    assert record["group"] == "helps"
    assert record["state"] == "state2_uncertain_helpable"
    assert record["relevant_entered"] == ["x"]
    assert record["entered_source"] == {"x": "bm25_only"}
    assert record["ddr_dense"] == pytest.approx(0.8)


def test_build_record_harm_with_dense_evidence_is_state3():
    record = build_mechanism_record(
        oracle_row=_oracle_row("m", delta=-1.0),
        signal_row=_signal_row(),
        dense_trace=_trace(["a", "b", "c", "d", "e"]),
        bm25_trace=_trace(["x", "y", "b", "c", "d"]),
        hybrid_trace=_trace(["x", "y", "b", "c", "d"]),
        relevant_docs=["a"],
    )
    assert record["group"] == "harms"
    assert record["state"] == "state3_disagree_but_correct"
    assert record["relevant_left"] == ["a"]


def test_build_record_rejects_oracle_mismatch():
    row = _oracle_row("bad", delta=1.0)
    row["oracle_escalate"] = False  # tampered: delta says YES, label says NO
    with pytest.raises(ValueError):
        build_mechanism_record(
            oracle_row=row,
            signal_row=_signal_row(),
            dense_trace=_trace(["a"]),
            bm25_trace=_trace(["a"]),
            hybrid_trace=_trace(["a"]),
            relevant_docs=[],
        )


def test_separation_check_reports_direction_honestly():
    records = [
        {"group": "helps", "f": 0.1},
        {"group": "helps", "f": 0.2},
        {"group": "harms", "f": 0.8},
        {"group": "harms", "f": 0.9},
        {"group": "ties", "f": 0.5},
    ]
    (report,) = separation_check(records, ["f"])
    assert report["separates"] is True
    assert report["direction"] == "helps_below_harms"


def test_separation_check_overlap_is_not_separation():
    records = [
        {"group": "helps", "f": 0.1},
        {"group": "helps", "f": 0.9},
        {"group": "harms", "f": 0.5},
    ]
    (report,) = separation_check(records, ["f"])
    assert report["separates"] is False
    assert report["direction"] is None


def test_separation_check_skips_none_pairwise():
    records = [
        {"group": "helps", "f": None},
        {"group": "helps", "f": 0.2},
        {"group": "harms", "f": 0.8},
    ]
    (report,) = separation_check(records, ["f"])
    assert report["helps_values"] == [0.2]
    assert report["separates"] is True
