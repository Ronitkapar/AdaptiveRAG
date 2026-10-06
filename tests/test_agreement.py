"""
tests.test_agreement
--------------------
Phase 9 cross-strategy agreement signals.

Pure functions over ranked identifier lists -- no index, no corpus, no provider.
"""

from __future__ import annotations

import pytest

from adaptive_rag.errors import EvaluationError
from adaptive_rag.evaluation.agreement import (
    DEFAULT_K,
    SENSITIVITY_K,
    distinct_doc_ratio,
    jaccard,
    overlap_at_k,
    pair_agreement,
    rank_correlation,
    registered_pairs,
    score_decay_slope,
    score_gap_top1_top2,
    spearman,
    top1_agreement,
    union_concentration,
)


def test_default_k_is_five_and_frozen():
    """§2.5: k=5 is pre-registered and must not drift silently."""
    assert DEFAULT_K == 5
    assert SENSITIVITY_K == (3, 10)


def test_jaccard_identical_sets_is_one():
    assert jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_half_overlap():
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_jaccard_two_empty_sets_is_one_not_zero():
    """Two arms that retrieved nothing agreed completely about there being nothing.

    Scoring this 0.0 would make "both arms failed identically" look like the
    strongest possible evidence of strategy sensitivity -- exactly backwards.
    """
    assert jaccard([], []) == 1.0


def test_jaccard_respects_k_cutoff():
    """Documents past rank k must not affect the score."""
    a = ["a", "b", "c"]
    b = ["a", "b", "x"]
    assert jaccard(a, b, k=2) == pytest.approx(1.0)


def test_jaccard_collapses_duplicate_documents():
    """A repeated document occupies one slot, not two."""
    assert jaccard(["a", "a", "b"], ["a", "b"], k=3) == 1.0


def test_overlap_at_k_is_asymmetric():
    """`a`'s coverage by `b` differs from `b`'s coverage by `a`."""
    a = ["a", "b", "c"]
    b = ["a"]
    assert overlap_at_k(a, b) == pytest.approx(1 / 3)
    assert overlap_at_k(b, a) == 1.0


def test_distinct_doc_ratio_is_single_arm():
    """The one agreement-family signal needing no partner arm."""
    assert distinct_doc_ratio(["a", "a", "b", "c", "d"]) == pytest.approx(0.8)
    assert distinct_doc_ratio([]) == 1.0


def test_top1_agreement():
    assert top1_agreement(["a", "b"], ["a", "c"]) is True
    assert top1_agreement(["a"], ["b"]) is False
    assert top1_agreement([], ["b"]) is False


def test_spearman_perfect_and_reversed():
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_uses_midranks_for_ties():
    """Ties are pervasive here (recall@5 has 4 distinct values), so this matters.

    With mid-ranks, [1,1,2] and [1,1,2] correlate at exactly 1.0. An
    implementation that broke ties by position would report something else.
    """
    assert spearman([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)


def test_spearman_returns_none_on_constant_input():
    """Undefined, not 0.0 -- a zero would read as 'no relationship'."""
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) is None


def test_spearman_returns_none_on_degenerate_input():
    assert spearman([1.0], [1.0]) is None
    assert spearman([1.0, 2.0], [1.0]) is None


def test_spearman_is_permutation_invariant_under_ties():
    """Ranks must depend on the multiset, not the arrival order."""
    assert spearman([1.0, 1.0, 3.0], [1.0, 3.0, 3.0]) == pytest.approx(
        spearman([3.0, 1.0, 1.0], [3.0, 1.0, 3.0])
    )


def test_rank_correlation_over_shared_documents_only():
    """Correlates only over documents both arms retrieved."""
    assert rank_correlation(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)


def test_rank_correlation_none_below_two_shared():
    assert rank_correlation(["a"], ["a"]) is None


def test_union_concentration():
    result = union_concentration("q1", ["a", "b"], ["b", "c"])
    assert result.union_size == 3
    assert result.total_slots == 4
    assert result.union_concentration == pytest.approx(0.75)


def test_union_concentration_disjoint_arms_is_one():
    """Higher means MORE different -- the opposite of Jaccard.

    Pinned deliberately: the two are easy to conflate, and reading this column as
    an agreement measure would invert every conclusion drawn from it.
    """
    assert union_concentration("q1", ["a"], ["b"]).union_concentration == 1.0


def test_union_concentration_identical_arms_is_one_half():
    """The floor is 0.5, not 0.0: each arm's slots are counted separately."""
    assert union_concentration("q1", ["a"], ["a"]).union_concentration == 0.5


def test_union_concentration_moves_opposite_to_jaccard():
    """Two arms can be broad and agreed; the two columns separate those cases."""
    broad_agreed = union_concentration("q1", ["a", "b"], ["a", "c"])
    narrow_agreed = union_concentration("q1", ["a"], ["a"])
    assert broad_agreed.union_concentration > narrow_agreed.union_concentration
    # ...while Jaccard reads both as perfectly agreeing.
    assert jaccard(["a", "b"], ["a", "c"], k=2) == pytest.approx(1 / 3)
    assert jaccard(["a"], ["a"], k=2) == 1.0


def test_union_concentration_rejects_two_empty_arms():
    """The one place a guard is genuinely needed: an undefined ratio."""
    with pytest.raises(EvaluationError):
        union_concentration("q1", [], [])


def test_pair_agreement_bundles_every_signal():
    result = pair_agreement("q1", "bm25", ["a", "b"], "dense", ["a", "x"])
    assert result.query_id == "q1"
    assert result.system_a == "bm25"
    assert result.system_b == "dense"
    assert result.k == DEFAULT_K
    assert result.jaccard == pytest.approx(1 / 3)
    assert result.overlap_at_k == pytest.approx(0.5)
    assert result.top1_agreement is True
    assert result.rank_correlation is None


def test_registered_pairs_enumerates_each_pair_once():
    pairs = registered_pairs(("bm25", "dense", "hybrid"))
    assert pairs == (("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid"))
    assert len(pairs) == len(set(pairs))


def test_registered_pairs_is_order_stable():
    """Deterministic family membership -- set iteration must not decide it."""
    assert registered_pairs(("bm25", "dense")) == registered_pairs(("bm25", "dense"))


# --- score geometry (H4), registered in docs/phases/phase-9.md 2.4 ------------


def test_score_gap_normalises_by_the_top_score():
    """(s1 - s2) / |s1|, so BM25 and cosine scales stay comparable."""
    assert score_gap_top1_top2([10.0, 8.0]) == pytest.approx(0.2)
    assert score_gap_top1_top2([1.0, 0.8]) == pytest.approx(0.2)
    # Same *relative* shape, wildly different units -> same answer. This is the
    # whole reason for normalising: a raw gap would make BM25 dominate on units.
    assert score_gap_top1_top2([27.5, 24.75]) == pytest.approx(
        score_gap_top1_top2([1.0, 0.9])
    )


def test_score_gap_handles_negative_top_score():
    """`|s1|` in the denominator; the numerator is s1 minus s2.

    For a descending negative sequence s1 = -10, s2 = -12 the top score *is*
    above the second, so the normalised gap is +0.2, not -0.2. The denominator's
    absolute value is what keeps the magnitude scale-free; the numerator still
    reports which score is higher.
    """
    assert score_gap_top1_top2([-10.0, -12.0]) == pytest.approx(0.2)
    # And a top score below the second does go negative:
    assert score_gap_top1_top2([-10.0, -8.0]) == pytest.approx(-0.2)


def test_score_gap_none_when_fewer_than_two_scores():
    assert score_gap_top1_top2([5.0]) is None
    assert score_gap_top1_top2([]) is None


def test_score_gap_none_when_top_score_is_zero():
    """The ratio is undefined, not zero."""
    assert score_gap_top1_top2([0.0, 0.0]) is None


def test_score_gap_none_when_scores_are_none():
    assert score_gap_top1_top2([None, None]) is None


def test_score_gap_respects_k_cutoff():
    """Scores past rank k must not affect the margin."""
    assert score_gap_top1_top2([10.0, 9.0, 1.0], k=2) == pytest.approx(0.1)


def test_score_gap_drops_none_entries_rather_than_zeroing():
    """A missing score is missing data, not a score of 0.0.

    If `None` were coerced to 0.0 the top-2 would be [10.0, 0.0] and the gap 1.0.
    Dropping it leaves the two genuinely-present scores, giving 0.1.
    """
    assert score_gap_top1_top2([10.0, None, 9.0]) == pytest.approx(0.1)


def test_score_decay_slope_is_negative_for_monotonic_decay():
    assert score_decay_slope([10.0, 8.0, 6.0, 4.0, 2.0]) < 0


def test_score_decay_slope_is_near_zero_for_flat_scores():
    """Flat decay means a slope close to zero -- the 'more dispersion' case."""
    flat = score_decay_slope([5.0, 5.0, 5.0, 5.0, 5.0])
    assert flat == pytest.approx(0.0, abs=1e-12)
    steep = score_decay_slope([100.0, 1.0, 1.0, 1.0, 1.0])
    assert steep < flat


def test_score_decay_slope_flat_is_greater_than_steep():
    """Direction check: 'flat => more dispersion' requires flat to score higher."""
    flat = score_decay_slope([5.0, 5.0, 5.0])
    steep = score_decay_slope([50.0, 2.0, 1.0])
    assert flat > steep


def test_score_decay_slope_none_when_fewer_than_two_scores():
    assert score_decay_slope([5.0]) is None
    assert score_decay_slope([]) is None


def test_score_decay_slope_none_when_all_scores_missing():
    assert score_decay_slope([None, None, None]) is None


def test_score_decay_slope_uses_only_top_k():
    """The rank-6 outlier must not enter the fit; k has to bound it."""
    five = score_decay_slope([10.0, 9.0, 8.0, 7.0, 6.0, -999.0], k=5)
    without = score_decay_slope([10.0, 9.0, 8.0, 7.0, 6.0], k=5)
    assert five == pytest.approx(without)
    assert five < 0  # a decaying sequence still slopes downward
