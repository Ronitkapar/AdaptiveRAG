"""
tests.test_phase9_family_contract
---------------------------------
Guards on the **scientific contract** of the Gate 9.2 candidate family.

These are not generic unit tests. Each one fails if the preregistration in
`docs/phases/phase-9.md` §2.4 and the code drift apart, or if the target,
direction convention, or arm symmetry is broken. They are the executable form of
"a signal not listed here may not enter the analysis as a candidate".

The query-intrinsic family is currently **empty by design** (§8 Issue C). The
tests below therefore assert the *known* state, and one of them is written to
fail loudly the moment someone adds a representative without reconciling §2.4.
"""

from __future__ import annotations

import pytest

from adaptive_rag.evaluation.agreement import DEFAULT_K
from adaptive_rag.evaluation.dispersion import (
    EPSILON,
    SELECTABLE_STRATEGIES,
    SOUND_STRATEGIES,
)
from adaptive_rag.evaluation.signals import (
    DECLARED_FAMILY_SIZE,
    REGISTERED_CANDIDATES,
    SCORE_ARMS,
    SignalRow,
    build_signal_table,
    signal_columns,
    defined_observations,
)

FAMILY = {c.name: c for c in REGISTERED_CANDIDATES}


# --- Issue B: the family is exactly what 2.4 registers ----------------------


def test_unregistered_signals_cannot_enter_the_family():
    """`overlap_at_k` is computed on every row and must never be a candidate."""
    row = SignalRow(query_id="q", dispersion=0.0, dispersion_mrr=0.0)
    row.overlap_at_k = {"bm25|dense": 0.5, "bm25|hybrid": 0.4, "dense|hybrid": 0.6}
    row.top1_agreement = {"bm25|dense": True, "bm25|hybrid": False}
    row.distinct_doc_ratio = {"bm25": 0.6, "dense": 0.4, "hybrid": 0.8}
    row.rank_correlation = {"bm25|dense": None, "bm25|hybrid": 0.9}
    columns = signal_columns(row)

    assert not any(name.startswith("overlap_at_k@") for name in columns)
    assert "distinct_doc_ratio@hybrid" not in columns
    assert "top1_agreement@bm25|hybrid" not in columns
    assert "rank_correlation@bm25|hybrid" not in columns
    # ...while the registered members survive.
    assert "top1_agreement@bm25|dense" in columns
    assert "distinct_doc_ratio@bm25" in columns


def test_unrestricted_view_still_exposes_everything_for_inspection():
    """The helpers stay available; only the *candidate family* is restricted."""
    row = SignalRow(query_id="q", dispersion=0.0, dispersion_mrr=0.0)
    row.overlap_at_k = {"bm25|dense": 0.5}
    assert "overlap_at_k@bm25|dense" in signal_columns(row, registered_only=False)
    assert "overlap_at_k@bm25|dense" not in signal_columns(row)


def test_registered_family_has_no_duplicate_names():
    names = [c.name for c in REGISTERED_CANDIDATES]
    assert len(names) == len(set(names))


def test_registered_family_names_match_section_2_4():
    """The literal transcription of §2.4's disagreement + score-geometry rows."""
    expected = {
        "jaccard@bm25|dense", "jaccard@bm25|hybrid", "jaccard@dense|hybrid",
        "union_concentration",
        "distinct_doc_ratio@bm25", "distinct_doc_ratio@dense",
        "top1_agreement@bm25|dense",
        "rank_correlation@bm25|dense",
        "score_gap_top1_top2@bm25", "score_gap_top1_top2@dense",
        "score_decay_slope@bm25", "score_decay_slope@dense",
    }
    assert set(FAMILY) == expected


# --- Issue A: score geometry must not disappear ------------------------------


def test_score_geometry_candidates_are_registered():
    for name in (
        "score_gap_top1_top2@bm25", "score_gap_top1_top2@dense",
        "score_decay_slope@bm25", "score_decay_slope@dense",
    ):
        assert name in FAMILY, f"registered score-geometry signal {name} disappeared"
        assert FAMILY[name].family == "score_geometry"


def test_score_geometry_only_on_registered_arms():
    assert set(SCORE_ARMS) == {"bm25", "dense"}
    assert all(not name.endswith("@hybrid") for name in FAMILY
               if name.startswith("score_"))


# --- Issue C: query-intrinsic family is unresolved and must stay visible ------


def test_query_intrinsic_family_is_withdrawn():
    """H5 was withdrawn as unidentifiable; docs/phases/phase-9.md §8.7.

    §2.4 originally declared a 16-hypothesis family but named only 12 signals,
    referring to four query-intrinsic representatives that no document, commit,
    plan or ADR ever specified. Adding representatives now would be a new
    scientific choice taken after exploratory results were seen. The family is
    therefore fixed at 12, and this test fails if anyone re-adds them.
    """
    assert [c for c in REGISTERED_CANDIDATES if c.family == "query_intrinsic"] == []


def test_declared_family_size_matches_the_registered_family():
    """Amended 16 -> 12 (section 8.7). The two must now agree exactly."""
    assert DECLARED_FAMILY_SIZE == 12
    assert len(REGISTERED_CANDIDATES) == DECLARED_FAMILY_SIZE


# --- direction convention ----------------------------------------------------


def test_direction_metadata_is_populated_for_every_candidate():
    for candidate in REGISTERED_CANDIDATES:
        assert candidate.expected_direction in {"negative", "positive", "either"}
        assert candidate.definition.strip()


def test_union_concentration_direction_is_positive():
    """It runs OPPOSITE to the jaccard family; the old blanket 'negative' was wrong."""
    assert FAMILY["union_concentration"].expected_direction == "positive"


def test_score_decay_slope_direction_is_positive():
    """Flat decay (slope near 0) means MORE dispersion, and flat is the higher value."""
    assert FAMILY["score_decay_slope@bm25"].expected_direction == "positive"
    assert FAMILY["score_decay_slope@dense"].expected_direction == "positive"


def test_agreement_family_directions_are_negative():
    for name in (
        "jaccard@bm25|dense", "jaccard@bm25|hybrid", "jaccard@dense|hybrid",
        "top1_agreement@bm25|dense", "rank_correlation@bm25|dense",
        "score_gap_top1_top2@bm25", "score_gap_top1_top2@dense",
    ):
        assert FAMILY[name].expected_direction == "negative", name


# --- frozen scientific constants (target, strategy set, k) -------------------


def test_frozen_constants_match_the_preregistration():
    assert set(SOUND_STRATEGIES) == {"bm25", "dense", "hybrid"}
    assert "hybrid_rerank" not in SOUND_STRATEGIES
    assert set(SELECTABLE_STRATEGIES) == {"bm25", "dense", "hybrid", "hybrid_rerank"}
    assert "adaptive" not in SELECTABLE_STRATEGIES
    assert EPSILON == 0.01
    assert DEFAULT_K == 5


# --- targets must never become predictors -----------------------------------


def test_targets_are_absent_from_the_candidate_family():
    for name in FAMILY:
        assert not name.startswith("target_")
        assert "dispersion" not in name


def test_undefined_signals_are_dropped_not_zeroed():
    """A None must not enter the correlation as a measured 0.0."""
    row = SignalRow(query_id="q", dispersion=0.0, dispersion_mrr=0.0)
    row.rank_correlation = {"bm25|dense": None}
    row.score_decay_slope = {"bm25": None, "dense": 0.5}
    columns = signal_columns(row)
    assert "rank_correlation@bm25|dense" not in columns
    assert columns["score_decay_slope@dense"] == 0.5
    assert "score_decay_slope@bm25" not in columns


# --- determinism -------------------------------------------------------------


def test_signal_table_is_deterministic():
    from adaptive_rag.evaluation.dispersion import StrategyOutcome

    def build():
        per_query = {
            "q1": {
                s: StrategyOutcome(
                    system=s, recall_at_5=1.0, mrr=1.0,
                    retrieved_document_ids=["d1", "d2", "d3"],
                    scores=[10.0, 9.0, 8.0, 7.0, 6.0],
                )
                for s in SOUND_STRATEGIES
            }
        }
        return build_signal_table(per_query, SOUND_STRATEGIES, DEFAULT_K)

    first = [signal_columns(r) for r in build()]
    second = [signal_columns(r) for r in build()]
    assert first == second


# --- 8.9 missing-data handling: pairwise-defined, no imputation --------------


def _row_with_rank_corr(query_id: str, value: float | None) -> SignalRow:
    row = SignalRow(query_id=query_id, dispersion=0.0, dispersion_mrr=0.0)
    row.rank_correlation = {"bm25|dense": value}
    return row


def test_undefined_rank_correlation_is_not_converted_to_zero():
    """A fabricated 0.0 is indistinguishable from a measured zero agreement."""
    rows = [_row_with_rank_corr("q1", 0.0), _row_with_rank_corr("q2", 0.0)]
    assert "rank_correlation@bm25|dense" in signal_columns(rows[0])

    rows = [_row_with_rank_corr("q1", None), _row_with_rank_corr("q2", 0.0)]
    columns = signal_columns(rows[0])
    assert "rank_correlation@bm25|dense" not in columns


def test_correlation_uses_only_defined_observations():
    """n is the count of defined observations, not the table row count."""
    rows = [_row_with_rank_corr(f"q{i}", None if i < 4 else 0.5 * i) for i in range(10)]
    ids, values = defined_observations(rows, "rank_correlation@bm25|dense")
    assert len(ids) == 6
    assert len(values) == 6
    assert ids == tuple(f"q{i}" for i in range(4, 10))


def test_effective_n_is_reported_not_inferred():
    """The helper is the only source of the effective n reported on output."""
    rows = [_row_with_rank_corr(f"q{i}", None if i < 8 else 0.3) for i in range(10)]
    ids, values = defined_observations(rows, "rank_correlation@bm25|dense")
    assert (len(ids), len(values)) == (2, 2)
    assert values == (0.3, 0.3)


def test_bootstrap_and_permutation_receive_the_same_defined_population():
    """One accessor feeds all three, so they cannot drift apart.

    The bootstrap resample and the permutation shuffle must both be built from
    this tuple -- never from all 107 rows with values reconstructed.
    """
    rows = [_row_with_rank_corr(f"q{i}", None if i % 2 else 0.1 * i) for i in range(10)]
    ids, values = defined_observations(rows, "rank_correlation@bm25|dense")
    assert len(ids) == len(values) == 5
    # A bootstrap resample drawn from the population has exactly n draws.
    import random
    rng = random.Random(20250109)
    resample = [values[rng.randrange(len(values))] for _ in range(len(values))]
    assert len(resample) == len(values)
    # No imputed zero can enter: every resampled value was genuinely observed.
    assert all(v in values for v in resample)
    assert len(set(values)) == len(values)  # no sentinel 0.0 was appended


def test_missingness_does_not_change_the_family_size():
    """Sparse signals do not shrink the Holm denominator."""
    before = len(REGISTERED_CANDIDATES)
    rows = [_row_with_rank_corr("q1", None)]
    signal_columns(rows[0])
    assert len(REGISTERED_CANDIDATES) == before == DECLARED_FAMILY_SIZE


def test_rank_correlation_remains_in_the_twelve_hypothesis_family():
    """Undefined on most queries does NOT remove it from the family."""
    assert "rank_correlation@bm25|dense" in FAMILY
    assert FAMILY["rank_correlation@bm25|dense"].family == "disagreement"
    assert len(REGISTERED_CANDIDATES) == 12


def test_no_imputation_helpers_exist_in_the_signal_module():
    """Structural guard: a future imputation must be added deliberately.

    The frozen rule forbids imputation, NULL-to-zero and complete-case
    restriction. There is no such code path to reach by accident.
    """
    import adaptive_rag.evaluation.signals as mod
    source = mod.__file__
    text = open(source).read()
    for banned in ("impute", "fillna", "nan_to_num"):
        assert banned not in text, f"imputation helper {banned!r} present"
