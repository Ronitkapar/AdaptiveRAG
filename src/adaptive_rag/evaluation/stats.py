"""
evaluation.stats
----------------
Paired statistical tests and effect sizes for Phase 7 cross-arm comparisons.

Implemented on stdlib + numpy only. SciPy is deliberately absent: the project
holds to "do not introduce unnecessary dependencies", and the handful of tests
Phase 7 needs are each a page of well-specified arithmetic. Everything here is
deterministic -- bootstrap resampling is seeded, so a re-run of the analysis
over the same raw rows reproduces the same numbers byte for byte.

Two rules shape this module, both from the Phase 7 brief:

1. Assumptions are *checked before a test is applied*, not assumed. Every
   Wilcoxon result carries the skewness/kurtosis of its difference distribution
   and a `symmetry_ok` verdict, because the signed-rank test assumes the
   differences are symmetric about their median. Where that verdict is
   negative, the distribution-free sign test is reported alongside so a reader
   never has to take the assumption on trust.

2. A p-value alone is not a result. Every comparison reports the paired effect
   (median difference) with a bootstrap confidence interval and a win/tie/loss
   tally, because with a benchmark this size a "significant" difference of a
   few thousandths is not an effect worth acting on.

No composite quality score is computed anywhere. Quality and efficiency stay on
separate axes and are compared separately.
"""

import math
from typing import Any, Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

STATS_VERSION = "phase7_stats_v1"

# Signed-rank p-values below this are treated as the floor rather than as a
# claim about the exact value; the exact table can represent far more precision
# than a 100-query benchmark has any right to imply.
MIN_REPORTABLE_P = 1e-9

# Above this many non-zero pairs the exact signed-rank table is replaced by the
# normal approximation with tie and continuity corrections.
EXACT_MAX_N = 25

# Rule-of-thumb symmetry verdict. A paired-difference sample this small cannot
# support a strong distributional claim, so the flags are deliberately lenient
# about skew and strict about sample size.
SKEW_THRESHOLD = 0.5
EXCESS_KURTOSIS_THRESHOLD = 2.0
MIN_N_FOR_SYMMETRY = 10


class SymmetryAssessment(BaseModel):
    """Distributional check performed before a signed-rank test is reported."""

    model_config = ConfigDict(extra="forbid")

    n: int
    skew: float | None
    excess_kurtosis: float | None
    symmetry_ok: bool
    reasons: list[str] = Field(default_factory=list)


class WilcoxonResult(BaseModel):
    """Outcome of a two-sided paired signed-rank test."""

    model_config = ConfigDict(extra="forbid")

    n_pairs: int
    n_zero: int
    n_positive: int
    n_negative: int
    statistic: float
    method: str
    p_value: float
    sign_test_p: float
    mean_rank_difference: float
    symmetry: SymmetryAssessment
    notes: str = ""


class PairedEffect(BaseModel):
    """Size of a paired difference, independent of any significance claim."""

    model_config = ConfigDict(extra="forbid")

    n_pairs: int
    median_difference: float
    mean_difference: float
    wins: int
    ties: int
    losses: int
    ci_low: float
    ci_high: float
    confidence: float = 0.95


def assess_symmetry(values: Sequence[float]) -> SymmetryAssessment:
    """Judge whether a difference sample is plausibly symmetric about its median.

    Reported, never enforced: the caller decides what to do with the verdict.
    """
    data = np.asarray(list(values), dtype=float)
    n = int(data.size)
    if n < 3:
        return SymmetryAssessment(
            n=n,
            skew=None,
            excess_kurtosis=None,
            symmetry_ok=False,
            reasons=["fewer than 3 non-zero differences"],
        )

    centered = data - data.mean()
    std = float(data.std())  # population sd, matching the usual moment estimators
    reasons: list[str] = []
    if std == 0.0:
        return SymmetryAssessment(
            n=n,
            skew=None,
            excess_kurtosis=None,
            symmetry_ok=True,
            reasons=["all differences identical"],
        )

    skew = float(np.mean(centered**3) / std**3)
    excess_kurtosis = float(np.mean(centered**4) / std**4 - 3.0)

    if abs(skew) > SKEW_THRESHOLD:
        reasons.append(f"|skew|={abs(skew):.3f} exceeds {SKEW_THRESHOLD}")
    if abs(excess_kurtosis) > EXCESS_KURTOSIS_THRESHOLD:
        reasons.append(
            f"|excess kurtosis|={abs(excess_kurtosis):.3f} exceeds "
            f"{EXCESS_KURTOSIS_THRESHOLD}"
        )
    if n < MIN_N_FOR_SYMMETRY:
        reasons.append(f"n={n} below {MIN_N_FOR_SYMMETRY}; moments are unstable")

    return SymmetryAssessment(
        n=n,
        skew=round(skew, 4),
        excess_kurtosis=round(excess_kurtosis, 4),
        symmetry_ok=not reasons,
        reasons=reasons,
    )


def _round_sig(value: float, digits: int = 6) -> float:
    """Round to significant figures, not decimal places.

    Fixed-decimal rounding is actively wrong for p-values and confidence
    intervals: round(2.3e-7, 6) is 0.0, so a very strong result would be written
    to disk as an absence of evidence, and round(1.907e-6, 6) loses a digit the
    exact table computed. Significant-figure rounding keeps small magnitudes
    intact while still making artifacts byte-stable across runs.
    """
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, -int(math.floor(math.log10(abs(value)))) + (digits - 1))


def _average_ranks(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Average ranks of |values| (1-based), plus the tie-correction term.

    Zero differences are dropped before ranking: under the null they contribute
    rank zero, so including them would only dilute the statistic.

    The returned correction is sum(t^3 - t) over tied groups, which is the term
    the normal approximation subtracts from the rank-sum variance. It is 0
    exactly when every |difference| is distinct, which is the condition under
    which the exact null distribution is valid.
    """
    n = int(values.size)
    if n == 0:
        return np.asarray([], dtype=float), 0
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(n, dtype=float)

    correction = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        group_size = j - i + 1
        if group_size > 1:
            correction += group_size**3 - group_size
        i = j + 1

    return ranks, correction


def _exact_signed_rank_table(n: int) -> list[int]:
    """Count subsets of {1..n} by rank sum -- the exact null distribution."""
    max_sum = n * (n + 1) // 2
    counts = [0] * (max_sum + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for total in range(max_sum, rank - 1, -1):
            counts[total] += counts[total - rank]
    return counts


def sign_test_p(n_positive: int, n_negative: int) -> float:
    """Exact two-sided binomial sign test, computed without SciPy.

    The sign test makes no distributional assumption at all, so it is the honest
    companion whenever the signed-rank symmetry verdict is negative.
    """
    n = n_positive + n_negative
    if n == 0:
        return 1.0
    k = min(n_positive, n_negative)
    # Two-sided exact binomial: 2 * P(X <= k), capped at 1.
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def wilcoxon_signed_rank(differences: Sequence[float]) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test over per-query differences.

    The exact null distribution is used when it is both affordable and valid
    (n <= 25 and no tied absolute differences). Ties make the exact table
    inapplicable -- average ranks land off the integer grid -- so the normal
    approximation with tie and continuity corrections takes over and the method
    used is recorded on the result.

    Known limitation, stated rather than hidden: the tie-corrected normal
    approximation is unreliable in *both* directions under heavy tying. For a
    sample of identical magnitudes split 20-0 the exact value is 2/2**20 ~= 1.9e-6
    while the approximation returns ~2.4e-7 -- more significant than the truth,
    because the tie correction over-shrinks the variance. Whenever `method`
    reports an approximation, the p-value should be read as an
    order-of-magnitude statement. Phase 7's metric series are continuous recall
    and latency values, which almost never tie this heavily; `assess_symmetry`
    and the reported sign test are the honest cross-checks.
    """
    data = np.asarray(list(differences), dtype=float)
    if data.size and not np.all(np.isfinite(data)):
        raise ValueError("differences must all be finite")

    n_zero = int(np.count_nonzero(data == 0.0))
    nonzero = data[data != 0.0]
    n = int(nonzero.size)
    n_positive = int(np.count_nonzero(nonzero > 0.0))

    symmetry = assess_symmetry(nonzero)

    if n == 0:
        return WilcoxonResult(
            n_pairs=0,
            n_zero=int(data.size),
            n_positive=0,
            n_negative=0,
            statistic=0.0,
            method="degenerate",
            p_value=1.0,
            sign_test_p=1.0,
            mean_rank_difference=0.0,
            symmetry=symmetry,
            notes="every paired difference is zero; the systems are indistinguishable "
            "on this metric",
        )

    ranks, tie_correction = _average_ranks(np.abs(nonzero))
    positive = nonzero > 0.0
    # Both tail totals are accumulated as magnitudes. The signed-rank statistic is
    # the smaller of the two, which is why W- is summed from -ranks but compared
    # as a positive quantity: reporting the signed sum directly would return a
    # negative statistic whenever the negative ranks dominate.
    w_plus = float(ranks[positive].sum())
    w_minus = float(ranks[~positive].sum())
    statistic = min(w_plus, w_minus)
    mean_rank_difference = float(ranks[positive].mean() if positive.any() else 0.0)

    if tie_correction == 0 and n <= EXACT_MAX_N:
        table = _exact_signed_rank_table(n)
        total = 2**n
        # `statistic` is the smaller of the two tail totals, so it sits at or
        # below the symmetric midpoint n(n+1)/4 and the lower tail is the one to use.
        cutoff = int(math.floor(statistic))
        p_value = min(1.0, 2.0 * sum(table[: cutoff + 1]) / total)
        method = "exact"
    else:
        mu = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 24.0
        if variance <= 0:
            # Under extreme tying the correction can drive the variance to (or
            # past) zero. The approximation is undefined here, so report no
            # evidence rather than an overflowed zero, which would read as
            # overwhelming significance the test never established.
            p_value = 1.0
            method = f"normal_approx_degenerate(tied_groups={tie_correction})"
        else:
            # Continuity correction moves the observed tail toward the null mean.
            # z goes negative whenever the statistic sits within half a rank of
            # mu, which is the middle of the null -- there is no evidence there,
            # so the test returns 1 rather than an inflated tail probability.
            z = (abs(statistic - mu) - 0.5) / math.sqrt(variance)
            p_value = 1.0 if z <= 0.0 else math.erfc(z / math.sqrt(2.0))
            method = f"normal_approx(tied_groups={tie_correction})"

    p_value = max(p_value, MIN_REPORTABLE_P)
    return WilcoxonResult(
        n_pairs=n,
        n_zero=n_zero,
        n_positive=n_positive,
        n_negative=n - n_positive,
        statistic=round(statistic, 4),
        method=method,
        p_value=_round_sig(p_value),
        sign_test_p=_round_sig(sign_test_p(n_positive, n - n_positive)),
        mean_rank_difference=round(mean_rank_difference, 4),
        symmetry=symmetry,
        notes=(
            "symmetry assumption not supported; prefer the sign test"
            if not symmetry.symmetry_ok
            else ""
        ),
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap confidence interval for a statistic of `values`.

    Seeded, so the interval is reproducible. Returns (None, None) for an empty
    sample rather than inventing a degenerate interval.
    """
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return None, None
    if data.size == 1:
        value = float(statistic(data))
        return value, value

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(n_resamples, data.size))
    replicates = np.asarray(statistic(data[idx], axis=1), dtype=float)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(replicates, alpha))
    high = float(np.quantile(replicates, 1.0 - alpha))
    return low, high


def paired_effect(
    differences: Sequence[float], *, confidence: float = 0.95, seed: int = 0
) -> PairedEffect:
    """Magnitude of a paired difference, with a bootstrap CI on the median."""
    data = np.asarray(list(differences), dtype=float)
    if data.size == 0:
        return PairedEffect(
            n_pairs=0,
            median_difference=0.0,
            mean_difference=0.0,
            wins=0,
            ties=0,
            losses=0,
            ci_low=0.0,
            ci_high=0.0,
            confidence=confidence,
        )

    low, high = bootstrap_ci(data, confidence=confidence, seed=seed)
    n_ties = int(np.count_nonzero(data == 0.0))
    return PairedEffect(
        n_pairs=int(data.size),
        median_difference=round(float(np.median(data)), 6),
        mean_difference=round(float(data.mean()), 6),
        wins=int(np.count_nonzero(data > 0.0)),
        ties=n_ties,
        losses=int(np.count_nonzero(data < 0.0)),
        ci_low=_round_sig(low) if low is not None else 0.0,
        ci_high=_round_sig(high) if high is not None else 0.0,
        confidence=confidence,
    )


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved.

    Phase 7 generates many comparisons per metric across arms and ablations;
    reporting each one at alpha=0.05 uncorrected would overstate the evidence.
    Adjusted values are made monotone non-decreasing along the sorted order and
    capped at 1.
    """
    values = [float(p) for p in pvalues]
    m = len(values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return [round(p, 6) for p in adjusted]


def compare_paired(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Full paired comparison of two systems' per-query metric series.

    Both series must be aligned by position: the caller is responsible for
    ordering them by the same query ids, which the suite guarantees by writing
    raw rows in dataset order.
    """
    if len(treatment) != len(baseline):
        raise ValueError(
            f"paired comparison needs aligned series, got {len(treatment)} and "
            f"{len(baseline)}"
        )
    differences = [float(t) - float(b) for t, b in zip(treatment, baseline)]
    return {
        "wilcoxon": wilcoxon_signed_rank(differences).model_dump(mode="json"),
        "effect": paired_effect(differences, seed=seed).model_dump(mode="json"),
    }
