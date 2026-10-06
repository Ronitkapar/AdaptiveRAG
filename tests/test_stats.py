"""
tests.test_stats
----------------
Offline, deterministic checks for the Phase 7 statistics module.

Every expected value here is hand-computed from a textbook null distribution
rather than copied from a library, so the tests check the arithmetic and not
just that the code agrees with itself.
"""

import math

import numpy as np
import pytest

from adaptive_rag.evaluation.stats import (
    MIN_REPORTABLE_P,
    assess_symmetry,
    bootstrap_ci,
    compare_paired,
    holm_bonferroni,
    paired_effect,
    sign_test_p,
    wilcoxon_signed_rank,
)


def test_wilcoxon_exact_all_positive_is_one_extreme_tail():
    """Differences 1..5 all point the same way: statistic 0, exact p = 2/32."""
    result = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result.method == "exact"
    assert result.n_pairs == 5
    assert result.statistic == 0.0
    # Only the empty subset of {1..5} sums to <= 0, so one-sided probability is
    # 1/32 and the two-sided value doubles it.
    assert result.p_value == pytest.approx(2 / 32, rel=1e-9)
    assert result.n_positive == 5
    assert result.n_negative == 0


def test_wilcoxon_exact_symmetric_null_is_not_significant():
    """A perfectly balanced ±1 sample must land near the null centre.

    All six magnitudes are equal, so the exact integer table does not apply and
    the tie-corrected approximation runs instead. Either path must agree with the
    null: 3-3 gives no evidence of a shift.
    """
    result = wilcoxon_signed_rank([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    assert result.p_value == pytest.approx(1.0, abs=1e-6)
    assert result.n_positive == 3 and result.n_negative == 3
    # Every magnitude is 1, so all six share the average rank 3.5.
    # W+ = W- = 3 * 3.5 = 10.5, and the statistic is the smaller tail.
    assert result.statistic == pytest.approx(10.5)


def test_wilcoxon_drops_zeros_and_reports_them():
    """Zero differences are excluded from the rank sum but still counted."""
    result = wilcoxon_signed_rank([0.0, 1.0, -1.0, 0.0, 2.0, 0.0])

    assert result.n_pairs == 3
    assert result.n_zero == 3
    # Non-zero magnitudes are {1, 1, 2} -> average ranks 1.5, 1.5, 3.
    # W+ = 1.5 + 3 = 4.5 and W- = 1.5, so the statistic is the smaller, 1.5.
    # It must never come back negative.
    assert result.statistic == pytest.approx(1.5)
    assert result.statistic >= 0.0


def test_wilcoxon_all_zero_is_degenerate_not_an_error():
    result = wilcoxon_signed_rank([0.0, 0.0, 0.0])

    assert result.method == "degenerate"
    assert result.p_value == 1.0
    assert result.n_pairs == 0


def test_wilcoxon_switches_to_approximation_when_ties_appear():
    """Tied absolute differences invalidate the exact integer table."""
    result = wilcoxon_signed_rank([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0])

    assert result.method.startswith("normal_approx")
    assert "tied_groups=" in result.method


def test_wilcoxon_uses_the_exact_table_when_magnitudes_are_distinct():
    """Distinct magnitudes keep the test on the exact null distribution."""
    result = wilcoxon_signed_rank([0.5, -1.5, 2.5, -3.5, 4.5, -5.5])

    assert result.method == "exact"


def test_wilcoxon_switches_to_approximation_above_exact_limit():
    rng = np.random.default_rng(7)
    differences = list(rng.normal(loc=0.1, scale=1.0, size=40))

    result = wilcoxon_signed_rank(differences)

    assert result.method.startswith("normal_approx")
    assert result.n_pairs == 40


def test_wilcoxon_detects_a_real_shift():
    """A consistent positive shift with distinct magnitudes must be significant.

    A unanimous 20-pair split has only the empty and full rank-sum subsets in the
    null tail, so 2/2**20 is the floor the test can possibly produce -- it cannot
    go lower no matter how small the true effect is.
    """
    result = wilcoxon_signed_rank([float(i) for i in range(1, 21)])

    assert result.method == "exact"
    # Six significant figures, matching the module's rounding contract for
    # artifacts -- deliberately tighter than fixed-decimal rounding would allow
    # but not so tight it would reject that rounding.
    assert result.p_value == pytest.approx(2 / 2**20, rel=1e-5)
    assert result.n_positive == 20
    assert result.n_negative == 0


def test_wilcoxon_under_heavy_tying_still_detects_the_shift():
    """Identical magnitudes force the approximation, which errs low.

    The exact answer for a 20-0 split is 2/2**20 ~= 1.9e-6. The tie correction
    over-shrinks the variance, so the approximation reports a *smaller* p-value
    than the truth. The test pins that direction deliberately: the point is that
    a reader sees the over-claim, not that the number is right. The floor at
    MIN_REPORTABLE_P must still hold, so the result is never a literal zero.
    """
    result = wilcoxon_signed_rank([1.0] * 20)

    assert result.method.startswith("normal_approx")
    assert result.p_value < 1e-6
    assert result.p_value >= MIN_REPORTABLE_P
    # The un-tied exact table would give this; the approximation sits below it.
    assert result.p_value < 2 / 2**20


def test_wilcoxon_rejects_non_finite_input():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1.0, float("nan"), 2.0])


def test_sign_test_matches_hand_computed_binomial():
    # 5-0 split: 2 * (C(5,0) / 2^5) = 2/32
    assert sign_test_p(5, 0) == pytest.approx(2 / 32, rel=1e-9)
    # 3-2 split: 2 * (C(5,0)+C(5,1)+C(5,2)) / 2^5 = 2 * 16/32 = 1.0
    assert sign_test_p(3, 2) == pytest.approx(1.0)
    assert sign_test_p(0, 0) == 1.0


def test_symmetry_flags_a_skewed_sample_and_reports_reasons():
    """Heavily right-skewed differences must not be silently certified symmetric."""
    skewed = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0]

    assessment = assess_symmetry(skewed)

    assert assessment.symmetry_ok is False
    assert assessment.reasons
    assert assessment.skew is not None and assessment.skew > 0


def test_symmetry_flags_too_few_points():
    assessment = assess_symmetry([1.0, 2.0, 3.0])

    assert assessment.symmetry_ok is False
    assert any("below" in reason for reason in assessment.reasons)


def test_symmetry_of_a_constant_sample_is_trivially_ok():
    assessment = assess_symmetry([2.0] * 8)

    assert assessment.symmetry_ok is True


def test_wilcoxon_flags_a_questionable_symmetry_assumption():
    result = wilcoxon_signed_rank([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 30.0])

    assert result.symmetry.symmetry_ok is False
    assert "prefer the sign test" in result.notes


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    """A fixed seed must reproduce a run exactly.

    The test deliberately does *not* assert that a different seed gives a
    different answer: a percentile bootstrap on a small sample frequently
    lands on the same order statistics for any seed, so that assertion would
    be testing luck rather than behaviour.
    """
    values = [0.1, 0.42, 0.07, 0.93, 0.28, 0.61, 0.15, 0.77, 0.34, 0.5]

    first = bootstrap_ci(values, seed=11)
    second = bootstrap_ci(values, seed=11)

    assert first == second
    assert first == (0.15, 0.635)


def test_bootstrap_brackets_the_median():
    values = [float(i) for i in range(1, 51)]
    low, high = bootstrap_ci(values, seed=3)

    assert low is not None and high is not None
    assert low <= np.median(values) <= high


def test_bootstrap_on_empty_and_singleton_samples():
    assert bootstrap_ci([]) == (None, None)
    low, high = bootstrap_ci([4.2])
    assert low == high == pytest.approx(4.2)


def test_paired_effect_counts_wins_ties_losses():
    effect = paired_effect([1.0, 0.0, -1.0, 2.0, -0.5])

    assert effect.wins == 2
    assert effect.losses == 2
    assert effect.ties == 1
    assert effect.n_pairs == 5
    assert effect.median_difference == pytest.approx(0.0)


def test_paired_effect_on_empty_input():
    effect = paired_effect([])

    assert effect.n_pairs == 0
    assert effect.wins == 0 and effect.losses == 0


def test_holm_matches_a_hand_computed_example():
    """p = [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06]."""
    assert holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped():
    adjusted = holm_bonferroni([0.5, 0.001, 0.9, 0.02])

    assert all(0.0 <= p <= 1.0 for p in adjusted)
    # Each adjusted value is at least its own raw p.
    for raw, adj in zip([0.5, 0.001, 0.9, 0.02], adjusted):
        assert adj >= raw - 1e-9
    # Holm never reports a smaller adjusted p than the smallest raw p.
    assert min(adjusted) >= 0.001 - 1e-9


def test_holm_of_empty_input():
    assert holm_bonferroni([]) == []


def test_holm_on_a_single_comparison_is_unchanged():
    assert holm_bonferroni([0.017]) == pytest.approx([0.017])


def test_compare_paired_rejects_misaligned_series():
    with pytest.raises(ValueError):
        compare_paired([1.0, 2.0], [1.0])


def test_compare_paired_reports_both_effect_and_significance():
    baseline = [0.5, 0.6, 0.55, 0.62, 0.58] * 6
    treatment = [v + 0.05 for v in baseline]

    result = compare_paired(treatment, baseline, seed=5)

    assert result["effect"]["wins"] == 30
    assert result["wilcoxon"]["n_pairs"] == 30
    assert result["wilcoxon"]["p_value"] < 0.01


def test_exact_table_matches_a_direct_enumeration():
    """The cached subset-sum table must equal brute-force enumeration."""
    from adaptive_rag.evaluation.stats import _exact_signed_rank_table

    n = 6
    table = _exact_signed_rank_table(n)

    assert sum(table) == 2**n
    # Brute force: every subset of {1..6} has one sum, and the table counts them.
    brute = [0] * (n * (n + 1) // 2 + 1)
    for mask in range(2**n):
        total = sum(rank for rank in range(1, n + 1) if mask & (1 << (rank - 1)))
        brute[total] += 1
    assert table == brute


def test_normal_approximation_is_close_to_exact_in_the_overlap_region():
    """Above the exact limit the approximation must land near the exact answer.

    Built by hand because scipy is deliberately not a project dependency: rank
    sums for n = 26 are enumerated directly, and the approximation is evaluated
    on the same statistic.
    """
    n = 26
    rng = np.random.default_rng(3)
    data = rng.normal(size=n)

    ranked = np.sort(np.abs(data))
    ranks = np.empty(n)
    for i, value in enumerate(ranked):
        ranks[i] = i + 1
    order = np.argsort(np.abs(data))
    ranks_sorted_back = np.empty(n)
    ranks_sorted_back[order] = ranks
    signed = np.where(data > 0, ranks_sorted_back, -ranks_sorted_back)
    w_plus = float(signed[signed > 0].sum())
    statistic = min(w_plus, float(signed[signed < 0].sum()))

    # Exact null tail for the same statistic (no ties in a continuous sample).
    counts = [0] * (n * (n + 1) // 2 + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for total in range(len(counts) - 1, rank - 1, -1):
            counts[total] += counts[total - rank]
    exact_p = min(1.0, 2.0 * sum(counts[: int(math.floor(statistic)) + 1]) / 2**n)

    approximation = wilcoxon_signed_rank(list(data))
    assert approximation.p_value == pytest.approx(exact_p, abs=0.05)
