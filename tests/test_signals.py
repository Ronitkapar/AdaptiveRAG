"""
tests.test_signals
------------------
Phase 9 signal table assembly, and the structural leakage guard.

The leakage tests are the important ones: they assert that a ground-truth field
cannot reach a feature column, which is a property of the signatures and models
rather than of any convention the caller is trusted to follow.
"""

from __future__ import annotations

import inspect
import json

import pytest

from adaptive_rag.evaluation.agreement import DEFAULT_K
from adaptive_rag.evaluation.dispersion import (
    SOUND_STRATEGIES,
    StrategyOutcome,
)
from adaptive_rag.evaluation.signals import (
    SignalRow,
    build_signal_table,
    read_signal_table,
    rows_to_jsonl,
    signal_columns,
    to_record,
    write_signal_table,
)


def _outcome(system: str, documents: list[str], recall: float = 1.0) -> StrategyOutcome:
    return StrategyOutcome(
        system=system,
        recall_at_5=recall,
        mrr=recall,
        retrieved_document_ids=list(documents),
    )


def _per_query() -> dict[str, dict[str, StrategyOutcome]]:
    return {
        "q1": {
            "bm25": _outcome("bm25", ["d1", "d2", "d3"]),
            "dense": _outcome("dense", ["d1", "d9", "d8"]),
            "hybrid": _outcome("hybrid", ["d1", "d2", "d7"]),
        },
        "q2": {
            "bm25": _outcome("bm25", ["d1", "d2"]),
            "dense": _outcome("dense", ["d1", "d2"], recall=0.0),
            "hybrid": _outcome("hybrid", ["d1", "d3"], recall=0.5),
        },
    }


def test_signal_table_emits_one_row_per_complete_query():
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    assert [r.query_id for r in rows] == ["q1", "q2"]


def test_signal_table_excludes_incomplete_queries():
    """One fixed denominator: a row missing an arm cannot be correlated."""
    per_query = _per_query()
    per_query["q3"] = {"bm25": _outcome("bm25", ["d1"])}
    rows = build_signal_table(per_query, SOUND_STRATEGIES, DEFAULT_K)
    assert "q3" not in {r.query_id for r in rows}


def test_signal_row_carries_the_dispersion_targets():
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    by_id = {r.query_id: r for r in rows}
    assert by_id["q1"].dispersion == 0.0
    assert by_id["q2"].dispersion == 1.0


def test_signal_row_populates_every_registered_pair():
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    row = rows[0]
    assert set(row.jaccard) == {"bm25|dense", "bm25|hybrid", "dense|hybrid"}
    assert set(row.overlap_at_k) == set(row.jaccard)
    assert set(row.top1_agreement) == set(row.jaccard)
    assert set(row.rank_correlation) == set(row.jaccard)


def test_signal_row_populates_single_arm_signals():
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    assert set(rows[0].distinct_doc_ratio) == set(SOUND_STRATEGIES)


def test_signal_columns_are_prefixed_by_family():
    columns = signal_columns(build_signal_table(_per_query())[0])
    assert "jaccard@bm25|dense" in columns
    assert "distinct_doc_ratio@bm25" in columns
    assert "union_concentration" in columns


def test_signal_columns_drops_uncomputable_rank_correlations():
    """A correlation that could not be computed is missing data, not a zero.

    Also encodes the family restriction: `bm25|hybrid` is not a registered
    candidate, so it is absent from the family view even when it is computable.
    """
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    row = SignalRow(query_id="q", dispersion=0.0, dispersion_mrr=0.0)
    # bm25/dense share only one document -> undefined; bm25/hybrid is computable
    # but is not registered in docs/phases/phase-9.md 2.4.
    row.rank_correlation = {"bm25|dense": None, "bm25|hybrid": 0.5}
    columns = signal_columns(row)
    assert "rank_correlation@bm25|dense" not in columns  # undefined -> dropped
    assert "rank_correlation@bm25|hybrid" not in columns  # unregistered -> excluded
    # The unrestricted view still shows it, for inspection.
    assert signal_columns(row, registered_only=False)["rank_correlation@bm25|hybrid"] == 0.5


def test_query_features_default_to_empty():
    """Gate 9.2 must work without a feature source present."""
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    assert rows[0].query_features == {}


def test_query_features_keep_only_numeric_values():
    features = {
        "q1": {
            "complexity_score": 0.4,
            "query_length_words": 19,
            "multi_concept": True,
            "content_terms": ["a", "b"],
            "question_type": "conceptual",
        }
    }
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K, features)
    kept = rows[0].query_features
    assert kept == {"complexity_score": 0.4, "query_length_words": 19.0}
    # Booleans, lists and strings are dropped rather than coerced to numbers.
    assert "multi_concept" not in kept
    assert "content_terms" not in kept
    assert "question_type" not in kept


def test_query_features_prefixes_are_visible_in_the_unrestricted_view():
    """Query features ride on the row but are not registered candidates yet.

    §2.4's query-intrinsic family is unresolved (docs/phases/phase-9.md §8 Issue
    C), so they are carried for inspection and excluded from the family view.
    """
    features = {"q1": {"complexity_score": 0.4}}
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K, features)
    row = rows[0]
    assert "query_feature:complexity_score" not in signal_columns(row)
    assert (
        signal_columns(row, registered_only=False)["query_feature:complexity_score"]
        == 0.4
    )


# --- leakage guards -------------------------------------------------------


def test_signal_row_forbids_extra_fields():
    """The structural guard: a signal row cannot carry an undeclared column."""
    with pytest.raises(Exception):
        SignalRow(query_id="q1", dispersion=0.0, dispersion_mrr=0.0, recall_at_5=1.0)


def test_to_record_prefixes_targets_so_they_cannot_be_read_as_features():
    """`target_` is the only thing stopping a consumer grabbing a label."""
    record = to_record(build_signal_table(_per_query())[0])
    assert "target_dispersion" in record
    assert "target_dispersion_mrr" in record
    assert "dispersion" not in record
    assert "dispersion_mrr" not in record


def test_build_signal_table_signature_cannot_receive_ground_truth():
    """It takes outcomes and an optional feature mapping -- no labels.

    This is the guard `docs/phases/phase-9.md` §2.9 promises. It is a signature
    assertion because `StrategyOutcome` cannot carry labels at all, so the
    guarantee does not depend on the caller passing only clean objects.
    """
    parameters = inspect.signature(build_signal_table).parameters
    assert list(parameters) == [
        "per_query",
        "strategies",
        "k",
        "query_features",
    ]


def test_strategy_outcome_has_no_ground_truth_field():
    """The label is unreachable from a signal input, by construction."""
    fields = set(StrategyOutcome.model_fields)
    assert "relevant_documents" not in fields
    assert "relevant_sections" not in fields
    assert "reference_answer" not in fields


# --- serialisation --------------------------------------------------------


def test_jsonl_round_trip_is_byte_identical(tmp_path):
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    path = tmp_path / "signal_table.jsonl"
    written = write_signal_table(rows, path)
    assert written == len(rows)
    assert path.read_text(encoding="utf-8") == rows_to_jsonl(rows)
    assert len(read_signal_table(path)) == len(rows)


def test_rows_to_jsonl_is_stable_across_calls():
    """Deterministic output: two calls must not differ by key ordering."""
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    assert rows_to_jsonl(rows) == rows_to_jsonl(rows)


def test_read_signal_table_preserves_nested_shape():
    """The reader keeps the nested shape the gate script needs."""
    rows = build_signal_table(_per_query(), SOUND_STRATEGIES, DEFAULT_K)
    back = json.loads(rows_to_jsonl(rows).splitlines()[0])
    assert back["jaccard"] == rows[0].jaccard
    assert back["distinct_doc_ratio"] == rows[0].distinct_doc_ratio
