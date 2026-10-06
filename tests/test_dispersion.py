"""
tests.test_dispersion
---------------------
Phase 9 target construction: dispersion, the reranker decomposition, and the
Gate 3 control label.

All offline and deterministic. No index, no corpus, no provider.
"""

from __future__ import annotations

import pytest

from adaptive_rag.evaluation.dispersion import (
    EPSILON,
    SELECTABLE_STRATEGIES,
    SOUND_STRATEGIES,
    StrategyOutcome,
    complete_queries,
    dispersion,
    dispersion_table,
    is_tie_break_artefact,
    needed_split,
    outcome_from_row,
    per_query_outcomes,
    reranker_decomposition,
)


def _outcome(
    system: str,
    recall: float,
    mrr: float = 0.0,
    documents: list[str] | None = None,
) -> StrategyOutcome:
    return StrategyOutcome(
        system=system,
        recall_at_5=recall,
        mrr=mrr,
        retrieved_document_ids=documents or [],
    )


def _three(records: dict[str, float]) -> dict[str, StrategyOutcome]:
    return {name: _outcome(name, value) for name, value in records.items()}


def test_sound_strategies_exclude_the_reranker():
    """The reranker is excluded on purpose; see the module docstring and ADR-028."""
    assert "hybrid_rerank" not in SOUND_STRATEGIES
    assert set(SOUND_STRATEGIES) < set(SELECTABLE_STRATEGIES)
    assert "adaptive" not in SELECTABLE_STRATEGIES


def test_dispersion_is_zero_when_strategies_tie():
    result = dispersion("q1", _three({"bm25": 1.0, "dense": 1.0, "hybrid": 1.0}))
    assert result.dispersion == 0.0
    assert result.ties_at_max == 3


def test_dispersion_is_the_range_across_strategies():
    result = dispersion("q1", _three({"bm25": 1.0, "dense": 0.0, "hybrid": 0.5}))
    assert result.dispersion == 1.0
    assert result.ties_at_max == 1


def test_dispersion_records_per_strategy_scores():
    """Every aggregate must be re-derivable from the stored per-arm values."""
    result = dispersion("q1", _three({"bm25": 1.0, "dense": 0.5, "hybrid": 0.5}))
    assert result.recall_by_strategy == {"bm25": 1.0, "dense": 0.5, "hybrid": 0.5}
    assert max(result.recall_by_strategy.values()) - min(
        result.recall_by_strategy.values()
    ) == pytest.approx(result.dispersion)


def test_dispersion_mrr_uses_mrr_not_recall():
    """MRR dispersion must be computed from MRR, independently of recall."""
    outcomes = {
        "bm25": _outcome("bm25", 1.0, mrr=0.2),
        "dense": _outcome("dense", 1.0, mrr=0.9),
        "hybrid": _outcome("hybrid", 1.0, mrr=0.5),
    }
    # Recall ties at 1.0 for all three; MRR does not.
    result = dispersion("q1", outcomes)
    assert result.dispersion == 0.0
    assert result.dispersion_mrr == pytest.approx(0.7)


def test_missing_strategy_raises_rather_than_scoring_a_tie():
    """A missing arm must fail loudly, not enter the analysis as a zero."""
    with pytest.raises(ValueError, match="missing outcomes"):
        dispersion("q1", _three({"bm25": 1.0, "dense": 1.0}))


def test_needed_split_labels_a_lone_winner_needed():
    """Gate 3's label: nothing else within epsilon of the reference."""
    outcomes = _three({"bm25": 0.0, "dense": 1.0, "hybrid": 0.0})
    assert needed_split(outcomes, SOUND_STRATEGIES, "dense") == "needed"


def test_needed_split_labels_a_tie_sufficient():
    outcomes = _three({"bm25": 1.0, "dense": 1.0, "hybrid": 1.0})
    assert needed_split(outcomes, SOUND_STRATEGIES, "dense") == "sufficient"


def test_needed_split_respects_epsilon():
    """A strategy exactly epsilon below the reference counts as sufficient."""
    outcomes = _three({"bm25": 0.99, "dense": 1.0, "hybrid": 0.0})
    assert needed_split(outcomes, SOUND_STRATEGIES, "dense", epsilon=EPSILON) == (
        "sufficient"
    )


def test_needed_split_is_disjoint_from_carrying_headroom():
    """The pathology that motivates Phase 9, pinned as a test.

    A query where another strategy *beats* the reference cannot be labelled
    `needed`, because `needed` means nothing else comes within epsilon of it.
    This is why Gate 3's 8 positives all carried `oracle_gain == 0.0`.
    """
    # dense loses outright: another strategy is strictly better.
    outcomes = _three({"bm25": 1.0, "dense": 0.0, "hybrid": 1.0})
    assert needed_split(outcomes, SOUND_STRATEGIES, "dense") == "sufficient"
    assert max(o.recall_at_5 for o in outcomes.values()) > outcomes["dense"].recall_at_5


def test_tie_break_artefact_detects_near_binary_recall():
    outcomes = {
        **_three({"bm25": 1.0, "dense": 1.0, "hybrid": 1.0}),
        "hybrid_rerank": _outcome("hybrid_rerank", 1.0),
    }
    assert is_tie_break_artefact(outcomes) is True


def test_tie_break_artefact_false_on_real_dispersion():
    outcomes = {
        **_three({"bm25": 1.0, "dense": 0.0, "hybrid": 1.0}),
        "hybrid_rerank": _outcome("hybrid_rerank", 1.0),
    }
    assert is_tie_break_artefact(outcomes) is False


def test_reranker_decomposition_counts_overlap():
    """The counts overlap by construction; a query can satisfy two of them.

    `reranker_worst` below is dispersed among the sound three *and* has the
    reranker strictly last, so it is counted in both figures. This is why 26 and
    21 exceed the 34 dispersed queries on the real benchmark.
    """
    per_query = {
        # Dispersed only because the reranker drags; the sound three tie.
        "reranker_worst": {
            "bm25": _outcome("bm25", 1.0),
            "dense": _outcome("dense", 1.0),
            "hybrid": _outcome("hybrid", 1.0),
            "hybrid_rerank": _outcome("hybrid_rerank", 0.0),
        },
        # Dispersed among the sound three, reranker also strictly last.
        "both": {
            "bm25": _outcome("bm25", 1.0),
            "dense": _outcome("dense", 0.0),
            "hybrid": _outcome("hybrid", 1.0),
            "hybrid_rerank": _outcome("hybrid_rerank", 0.0),
        },
        # No dispersion at all.
        "tied": {
            "bm25": _outcome("bm25", 1.0),
            "dense": _outcome("dense", 1.0),
            "hybrid": _outcome("hybrid", 1.0),
            "hybrid_rerank": _outcome("hybrid_rerank", 1.0),
        },
    }
    result = reranker_decomposition(per_query)
    assert result.n_dispersed_four_arm == 2
    # Both dispersed queries have the reranker strictly below the best other.
    assert result.reranker_strictly_worst == 2
    assert result.reranker_strictly_best == 0
    # Only "both" is dispersed once the reranker is removed.
    assert result.dispersion_among_sound_only == 1


def test_per_query_outcomes_drops_failed_rows():
    """A failed row must not be scored as a zero.

    Phase 8 measured why: a contaminated arm read 0.6384 where its true value was
    0.352, a sign-flipping error.
    """
    records = [
        {"query_id": "q1", "system": "bm25", "status": "ok", "recall_at_5": 1.0, "mrr": 1.0},
        {"query_id": "q1", "system": "dense", "status": "error", "recall_at_5": 0.0, "mrr": 0.0},
    ]
    grouped = per_query_outcomes(records, SOUND_STRATEGIES)
    assert "dense" not in grouped["q1"]
    assert complete_queries(grouped, SOUND_STRATEGIES) == []


def test_per_query_outcomes_ignores_unrequested_strategies():
    records = [
        {"query_id": "q1", "system": "bm25", "status": "ok", "recall_at_5": 1.0, "mrr": 1.0},
        {"query_id": "q1", "system": "adaptive", "status": "ok", "recall_at_5": 1.0, "mrr": 1.0},
    ]
    grouped = per_query_outcomes(records, SOUND_STRATEGIES)
    assert set(grouped["q1"]) == {"bm25"}


def test_complete_queries_excludes_incomplete_and_sorts():
    per_query = {
        "q2": {s: _outcome(s, 1.0) for s in SOUND_STRATEGIES},
        "q1": {s: _outcome(s, 1.0) for s in SOUND_STRATEGIES},
        "q3": {"bm25": _outcome("bm25", 1.0)},
    }
    assert complete_queries(per_query, SOUND_STRATEGIES) == ["q1", "q2"]


def test_outcome_from_row_reads_only_the_declared_fields():
    record = {
        "query_id": "q1",
        "system": "bm25",
        "status": "ok",
        "recall_at_5": 0.5,
        "mrr": 0.25,
        "retrieved_document_ids": ["d1", "d2"],
        "retrieved_chunk_ids": ["d1::c1"],
        # A ground-truth field present in the row must be ignored.
        "relevant_documents": ["secret"],
    }
    outcome = outcome_from_row(record)
    assert outcome.retrieved_document_ids == ["d1", "d2"]
    assert not hasattr(outcome, "relevant_documents")
    assert "secret" not in outcome.model_dump()


def test_dispersion_table_is_deterministic_and_ordered():
    per_query = {
        "q2": {s: _outcome(s, 1.0) for s in SOUND_STRATEGIES},
        "q1": {
            "bm25": _outcome("bm25", 1.0),
            "dense": _outcome("dense", 0.0),
            "hybrid": _outcome("hybrid", 1.0),
        },
    }
    first = dispersion_table(per_query, SOUND_STRATEGIES)
    second = dispersion_table(per_query, SOUND_STRATEGIES)
    assert [r.query_id for r in first] == ["q1", "q2"]
    assert [r.dispersion for r in first] == [r.dispersion for r in second]
    assert first[0].dispersion == 1.0
    assert first[1].dispersion == 0.0


def test_strategy_outcome_forbids_extra_fields():
    """The leakage guard: a signal cannot carry an undeclared label field."""
    with pytest.raises(Exception):
        StrategyOutcome(
            system="bm25",
            recall_at_5=1.0,
            mrr=1.0,
            relevant_documents=["leaked"],
        )
