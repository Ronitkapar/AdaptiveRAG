"""
evaluation.statistics
---------------------
The paired-statistics driver for Phase 7: which test, on which pairs, with what
correction, and why.

`evaluation.stats` computes the tests. This module decides *when* each one is
admissible, aligns the pairs, and reports the family of comparisons. It is the
first production caller of `stats.py`, and the reason it exists as a separate
layer is that a test is only defensible next to the argument for its own
applicability. `wilcoxon_signed_rank` is correct arithmetic; whether it is the
right test for a 12-second rate-limited embedding call in a latency difference
is a question about the data, and that question is answered here rather than by
convention.

================================================================================
TEST SELECTION -- THE REASONING, STATED ONCE AND RECORDED IN EVERY ARTIFACT
================================================================================

The choice is per metric *kind*, not one test for the whole report:

**Quality metrics** (recall@k, precision@k, hit@k, MRR, nDCG@k) are the case
where a signed-rank test is admissible. They are paired by construction -- the
same query, two systems -- which is exactly the design Wilcoxon assumes, and
they live in [0, 1] with no structural skew: a system either found the
relevant document or it did not, and swapping which one is a symmetric
redistribution rather than a heavy tail. The test is still *checked* rather than
assumed: `assess_symmetry` runs on the differences and a negative verdict
replaces the signed-rank p-value with the exact sign test. Quality values also
tie heavily (recall is a ratio of small integers, and a lot of queries score
exactly 0.0 or 1.0), which is precisely the regime `stats.wilcoxon_signed_rank`
documents as making its tie-corrected normal approximation unreliable. So the
sign test is reported beside every signed-rank p-value, and when the exact table
was not usable the result's `method` field says so -- an approximation is never
reported as exact.

**Latency metrics are the opposite case, and by construction.** The frozen cost
table records a 12.0 s hybrid outlier -- one rate-limited embedding call -- that
inflates that arm's stdev to 1373 ms while leaving its median at 456 ms. A
per-query latency difference inherits that right tail: the distribution is
skewed, so the symmetry assumption the signed-rank test needs does not hold, and
`assess_symmetry` will (correctly) refuse it. The sign test is therefore the
reported test for latency, and it is chosen up front rather than as a fallback,
because the skew is a property of the workload and not a surprise discovered
after the fact. Two consequences are honoured throughout:

  * the **median** is the reported centre, with a bootstrap CI on the median
    difference, never the mean -- a mean over a right-skewed sample is a
    statement about the 12-second call and not about the system; and
  * the **mean is still printed**, next to median, p95 and max, precisely so a
    reader can see how far the tail pulls it. Hiding the mean is as much a
    distortion as reporting only the mean.

**Cost and token metrics** get the sign test for a second reason: they are
non-negative, heavy-tailed and frequently exactly tied. Two systems that issued
the identical prompt have an identical `estimated_cost_usd`; those differences
are zero, and zero differences are dropped before ranking, so the signed-rank
statistic is computed over an ever-shrinking sample while the tie correction
grows. `stats.wilcoxon_signed_rank` says outright that under heavy tying the
approximation can be *anti*-conservative. The sign test, which assumes nothing
about shape, is the honest test here.

**Holm-Bonferroni** is applied within each metric across the family of
comparisons being made for it. Phase 7 generates many comparisons per metric
(five systems, or seven ablation variants), and reporting each at alpha=0.05
uncorrected overstates the evidence by roughly the size of the family. Both the
raw and the adjusted p-value are reported, and the family size is recorded, so
a reader can see what was corrected rather than having to trust it.

================================================================================
PAIRING
================================================================================

Comparisons are paired by `query_id`, and the pairing is **asserted, not
assumed**. `stats.compare_paired` documents that the caller is responsible for
ordering both series by the same query ids; this module checks that the id sets
are equal, rejects a duplicate `query_id` within one system, and raises rather
than comparing position-wise lists that happen to be the same length. Two
different queries in the same position would otherwise produce a clean,
confident, meaningless p-value.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.errors import EvaluationError
from adaptive_rag.evaluation.analysis import (
    COST_COLUMNS,
    LATENCY_COLUMNS,
    QUALITY_METRICS,
    distribution,
    row_field,
)
from adaptive_rag.evaluation.stats import (
    bootstrap_ci,
    compare_paired,
    holm_bonferroni,
    wilcoxon_signed_rank,
)

STATISTICS_VERSION = "phase7_statistics_v1"

# Recorded on every comparison and on the report. A bootstrap interval without
# the seed and the resample count that produced it is not reproducible, and a
# re-run of the analysis that produced a different interval would be
# indistinguishable from a bug.
DEFAULT_SEED = 20250101
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05

# `stats.paired_effect` fixes the resample count at its own signature default,
# which is what `compare_paired` uses. When a caller asks for a different count
# the interval is recomputed with `bootstrap_ci` directly, so the count recorded
# below is always the count that was actually used.
STATS_DEFAULT_RESAMPLES = 10_000

MetricKind = Literal["quality", "latency", "cost"]


class PairingError(EvaluationError):
    """Raised when two series cannot be paired by `query_id`.

    A pairing error is never recoverable by reordering: the two arms measured
    different query sets, or one arm recorded a query twice, and either way the
    per-query differences behind the test would not be differences of the same
    queries. Raising is the only honest response -- `stats.compare_paired`
    requires aligned series and would happily pair whatever it is handed.
    """


# --------------------------------------------------------------------------
# metric kinds and the selection rule
# --------------------------------------------------------------------------


def metric_kind(metric: str) -> MetricKind:
    """Which family a metric column belongs to.

    Read from `evaluation.analysis`'s own column lists, so a column cannot be a
    latency metric in the analysis tables and a quality metric here.
    """
    if metric in QUALITY_METRICS:
        return "quality"
    if metric in LATENCY_COLUMNS:
        return "latency"
    if metric in COST_COLUMNS:
        return "cost"
    raise ValueError(
        f"{metric!r} is not a known Phase 7 metric column; expected one of the "
        f"quality, latency or cost columns declared in evaluation.analysis"
    )


TEST_SELECTION_RULE: dict[str, Any] = {
    "quality": {
        "test": "paired Wilcoxon signed-rank",
        "admissible_when": "the paired difference distribution is plausibly symmetric",
        "checked_by": "adaptive_rag.evaluation.stats.assess_symmetry",
        "fallback": (
            "when symmetry is not supported, the exact two-sided sign test is the "
            "reported p-value"
        ),
        "why": (
            "Quality metrics are paired by construction and bounded in [0,1] with "
            "no structural tail, which is the regime the signed-rank test assumes. "
            "The assumption is still measured rather than trusted."
        ),
        "caveat": (
            "Quality values tie heavily (recall is a ratio of small integers, and "
            "many queries score exactly 0.0 or 1.0). When the exact null table is "
            "unavailable -- n > 25, or any tied absolute difference -- the p-value "
            "comes from the tie-corrected normal approximation, which "
            "stats.wilcoxon_signed_rank documents as order-of-magnitude in both "
            "directions under heavy tying. The sign test is reported alongside "
            "every signed-rank p-value for exactly that reason, and an "
            "approximation is never described as exact."
        ),
    },
    "latency": {
        "test": "exact two-sided sign test",
        "admissible_when": "always -- the sign test assumes no distributional shape",
        "checked_by": "adaptive_rag.evaluation.stats.assess_symmetry (reported, not enforced)",
        "fallback": "not needed; the sign test is the primary test for this family",
        "why": (
            "Latency is right-skewed by construction here, not incidentally. The "
            "frozen cost table in RoutingConfig.strategy_cost_ms records a 12.0 s "
            "hybrid outlier -- a single rate-limited embedding call -- against a "
            "455.97 ms median. A per-query latency difference inherits that tail, "
            "so the signed-rank symmetry assumption fails and the test is refused "
            "on its own stated criterion."
        ),
        "reporting": (
            "The median is the reported centre, with a bootstrap CI on the median "
            "paired difference. The mean is still reported beside median, p95 and "
            "max so the tail's effect on it is visible rather than hidden."
        ),
    },
    "cost": {
        "test": "exact two-sided sign test",
        "admissible_when": "always -- the sign test assumes no distributional shape",
        "checked_by": "adaptive_rag.evaluation.stats.assess_symmetry (reported, not enforced)",
        "fallback": "not needed; the sign test is the primary test for this family",
        "why": (
            "Cost and token columns are non-negative and frequently exactly tied: "
            "two systems that built the same context have an identical "
            "estimated_cost_usd. Zero differences are dropped before ranking, so "
            "the signed-rank sample shrinks as the ties grow, and "
            "stats.wilcoxon_signed_rank documents that its tie-corrected normal "
            "approximation can then be anti-conservative."
        ),
        "reporting": "median and p95 with a bootstrap CI on the median difference",
    },
    "family_correction": {
        "method": "Holm-Bonferroni step-down",
        "family": "every comparison made for one metric",
        "implemented_by": "adaptive_rag.evaluation.stats.holm_bonferroni",
        "why": (
            "Phase 7 makes many comparisons per metric (five systems in E1, seven "
            "variants in E3). Reporting each at alpha=0.05 uncorrected overstates "
            "the evidence by about the size of the family. Raw and adjusted "
            "p-values are both reported, with the family size."
        ),
    },
    "pairing": (
        "by query_id, asserted: equal id sets and no duplicate ids per system, "
        "otherwise PairingError"
    ),
}


def select_test(metric: str, differences: Sequence[float]) -> dict[str, Any]:
    """Decide the test for one metric's paired differences, and say why.

    Returns the chosen test, the reason, the symmetry verdict, and -- for the
    quality family -- both p-values: the signed-rank one and the sign-test one.
    Selecting the sign test for latency and cost is not a fallback path, it is
    the primary choice, and the reason travels with the number.
    """
    kind = metric_kind(metric)
    data = [float(d) for d in differences]
    if kind == "quality":
        wilcoxon = wilcoxon_signed_rank(data)
        if wilcoxon.symmetry.symmetry_ok:
            selected = "wilcoxon_signed_rank"
            reason = (
                "the paired difference distribution is plausibly symmetric "
                f"(skew={wilcoxon.symmetry.skew}, excess_kurtosis="
                f"{wilcoxon.symmetry.excess_kurtosis}), so the signed-rank test is "
                "admissible"
            )
        else:
            selected = "sign_test"
            reason = (
                "symmetry is not supported ("
                + "; ".join(wilcoxon.symmetry.reasons)
                + "), so the distribution-free exact sign test is reported instead"
            )
        is_signed_rank = selected == "wilcoxon_signed_rank"
        p_value = wilcoxon.p_value if is_signed_rank else wilcoxon.sign_test_p
        return {
            "metric": metric,
            "metric_kind": kind,
            "selected_test": selected,
            "reason": reason,
            "p_value": p_value,
            # Only the signed-rank p-value carries the signed-rank method string.
            # Attaching `normal_approx(...)` to a *sign test* p-value would
            # describe the wrong computation, which is the mislabelling this
            # module exists to prevent.
            "p_value_method": (
                wilcoxon.method if is_signed_rank else "exact binomial sign test"
            ),
            "sign_test_p": wilcoxon.sign_test_p,
            "wilcoxon_p_value": wilcoxon.p_value,
            "symmetry": wilcoxon.symmetry.model_dump(mode="json"),
            "n_positive": wilcoxon.n_positive,
            "n_negative": wilcoxon.n_negative,
            "n_zero": wilcoxon.n_zero,
            "statistic": wilcoxon.statistic,
            "notes": wilcoxon.notes,
        }
    return {
        "metric": metric,
        "metric_kind": kind,
        "selected_test": "sign_test",
        "reason": (
            f"{kind} metrics are non-negative and right-skewed or heavily tied by "
            "construction, so the paired signed-rank symmetry assumption is not "
            "met; the exact sign test assumes no shape and is reported as primary. "
            f"See TEST_SELECTION_RULE['{kind}']."
        ),
        "p_value": None,  # filled by the caller from the paired result
        "p_value_method": "exact binomial sign test",
        "sign_test_p": None,
        "wilcoxon_p_value": None,
        "symmetry": None,
        "n_positive": None,
        "n_negative": None,
        "n_zero": None,
        "statistic": None,
        "notes": "",
    }


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------


def index_by_query(rows: Iterable[Any]) -> dict[str, Any]:
    """`query_id -> row` for one system's rows, rejecting duplicate ids.

    A repeated `query_id` is a hard error rather than a last-one-wins merge: the
    suite appends to `traces.jsonl` on a resume, so a duplicated id means the
    row set describes the same query twice, and keeping either copy silently
    halves or doubles that query's weight in every mean.
    """
    indexed: dict[str, Any] = {}
    for row in rows:
        query_id = row_field(row, "query_id")
        if query_id is None:
            raise PairingError("a row has no query_id; it cannot be paired")
        key = str(query_id)
        if key in indexed:
            raise PairingError(
                f"duplicate query_id {key!r} within one system; the row set "
                "describes the same query more than once, so every paired "
                "difference would be miscounted"
            )
        indexed[key] = row
    return indexed


def pair_metric(
    treatment_rows: Sequence[Any],
    baseline_rows: Sequence[Any],
    *,
    metric: str,
    treatment: str,
    baseline: str,
) -> dict[str, Any]:
    """Two systems' per-query values for one metric, aligned by `query_id`.

    Alignment is verified, never assumed. The two id sets must be equal and the
    values are returned in `query_id` order, so the pairing does not depend on
    either export having been written in the same order -- a property the row
    export cannot promise, since a resumed run appends.

    `None` values are reported separately rather than dropped: a latency column
    is `None` on a run that failed before that clock started, and subtracting
    nothing for it would silently change which queries are compared.
    """
    treatment_index = index_by_query(treatment_rows)
    baseline_index = index_by_query(baseline_rows)

    only_treatment = sorted(set(treatment_index) - set(baseline_index))
    only_baseline = sorted(set(baseline_index) - set(treatment_index))
    if only_treatment or only_baseline:
        raise PairingError(
            f"{treatment!r} and {baseline!r} were not run on the same queries for "
            f"metric {metric!r}: "
            f"{len(only_treatment)} only in {treatment!r} "
            f"(e.g. {only_treatment[:3]}), "
            f"{len(only_baseline)} only in {baseline!r} "
            f"(e.g. {only_baseline[:3]}). A paired test over different query sets "
            "is not a paired test."
        )

    aligned: list[str] = sorted(treatment_index)
    paired: list[tuple[float, float]] = []
    missing_treatment: list[str] = []
    missing_baseline: list[str] = []
    for query_id in aligned:
        treatment_value = row_field(treatment_index[query_id], metric)
        baseline_value = row_field(baseline_index[query_id], metric)
        if not isinstance(treatment_value, (int, float)) or isinstance(
            treatment_value, bool
        ):
            missing_treatment.append(query_id)
        if not isinstance(baseline_value, (int, float)) or isinstance(
            baseline_value, bool
        ):
            missing_baseline.append(query_id)
        if (
            isinstance(treatment_value, (int, float))
            and not isinstance(treatment_value, bool)
            and isinstance(baseline_value, (int, float))
            and not isinstance(baseline_value, bool)
        ):
            paired.append((float(treatment_value), float(baseline_value)))

    return {
        "query_ids": aligned,
        "pairs": paired,
        "n_pairs": len(paired),
        "n_aligned_queries": len(aligned),
        "n_missing_treatment": len(missing_treatment),
        "n_missing_baseline": len(missing_baseline),
        "missing_treatment_query_ids": missing_treatment[:20],
        "missing_baseline_query_ids": missing_baseline[:20],
    }


# --------------------------------------------------------------------------
# one comparison
# --------------------------------------------------------------------------


def _summary(values: Sequence[float]) -> dict[str, Any]:
    """n / mean / median / p95 / stdev / min / max for one series.

    p95 is the shared nearest-rank estimator (`evaluation.base.percentile` via
    `analysis.distribution`), so the p95 in a statistics artifact and the p95 in
    an analysis table are the same number computed the same way.
    """
    return distribution(values)


def compare_metric(
    treatment_rows: Sequence[Any],
    baseline_rows: Sequence[Any],
    *,
    metric: str,
    treatment: str,
    baseline: str,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """One metric, one treatment versus one baseline, paired by query.

    Produces the series summaries, the paired differences, the effect size with
    a seeded bootstrap CI, the test chosen by `select_test` with its reasoning,
    and a `p_value` that is the *selected* test's -- not the signed-rank one
    when the sign test was chosen.

    `compare_paired` from `evaluation.stats` is the workhorse: it runs the
    signed-rank test and the paired effect together, which is the pairing of
    those two the module was written for. When `n_resamples` differs from the
    count `stats.paired_effect` fixes internally, the interval is recomputed
    with `bootstrap_ci` so the recorded resample count is the one that produced
    the interval.
    """
    paired = pair_metric(
        treatment_rows,
        baseline_rows,
        metric=metric,
        treatment=treatment,
        baseline=baseline,
    )
    pairs = paired["pairs"]
    treatment_values = [t for t, _ in pairs]
    baseline_values = [b for _, b in pairs]
    differences = [t - b for t, b in pairs]

    test = select_test(metric, differences)
    if not pairs:
        comparison = {
            "metric": metric,
            "metric_kind": test["metric_kind"],
            "treatment": treatment,
            "baseline": baseline,
            "n_pairs": 0,
            "n_aligned_queries": paired["n_aligned_queries"],
            "n_missing_treatment": paired["n_missing_treatment"],
            "n_missing_baseline": paired["n_missing_baseline"],
            "test": {**test, "p_value": None, "sign_test_p": None},
            "effect": None,
            "treatment_summary": _summary(treatment_values),
            "baseline_summary": _summary(baseline_values),
            "difference_summary": _summary(differences),
            "p_value": None,
            "adjusted_p_value": None,
            "significant": False,
            "alpha": alpha,
            "note": "no query carried a value for this metric in both systems",
        }
        return comparison

    paired_result = compare_paired(treatment_values, baseline_values, seed=seed)
    wilcoxon = paired_result["wilcoxon"]
    effect = dict(paired_result["effect"])
    if n_resamples != STATS_DEFAULT_RESAMPLES and differences:
        low, high = bootstrap_ci(
            differences,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        # Six significant figures, matching `stats._round_sig`'s convention for a
        # value `paired_effect` produced itself, so a recomputed interval and a
        # default-count interval are formatted the same way.
        effect["ci_low"] = float(f"{low:.6g}") if low is not None else 0.0
        effect["ci_high"] = float(f"{high:.6g}") if high is not None else 0.0
        effect["confidence"] = confidence

    if test["selected_test"] == "wilcoxon_signed_rank":
        p_value = wilcoxon["p_value"]
        method = wilcoxon["method"]
    else:
        p_value = wilcoxon["sign_test_p"]
        method = test["p_value_method"]

    test = {
        **test,
        "p_value": p_value,
        "p_value_method": method,
        "sign_test_p": wilcoxon["sign_test_p"],
        "symmetry": wilcoxon["symmetry"],
    }

    return {
        "metric": metric,
        "metric_kind": test["metric_kind"],
        "treatment": treatment,
        "baseline": baseline,
        "n_pairs": paired["n_pairs"],
        "n_aligned_queries": paired["n_aligned_queries"],
        "n_missing_treatment": paired["n_missing_treatment"],
        "n_missing_baseline": paired["n_missing_baseline"],
        "missing_treatment_query_ids": paired["missing_treatment_query_ids"],
        "missing_baseline_query_ids": paired["missing_baseline_query_ids"],
        "test": test,
        "effect": effect,
        "treatment_summary": _summary(treatment_values),
        "baseline_summary": _summary(baseline_values),
        "difference_summary": _summary(differences),
        "p_value": p_value,
        "adjusted_p_value": None,
        "significant": None,
        "alpha": alpha,
        "bootstrap_seed": seed,
        "bootstrap_resamples": n_resamples,
        "bootstrap_confidence": confidence,
        "note": (
            "mean is reported for visibility only; the effect estimate and its "
            "interval are on the median, because this metric family is not "
            "mean-stable"
            if test["metric_kind"] in ("latency", "cost")
            else ""
        ),
    }


# --------------------------------------------------------------------------
# a family of comparisons
# --------------------------------------------------------------------------


def _group_rows(rows: Sequence[Any], group_column: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        key = row_field(row, group_column)
        if key is None:
            continue
        grouped.setdefault(str(key), []).append(row)
    return grouped


def paired_comparisons(
    rows: Sequence[Any],
    *,
    groups: Sequence[str],
    baseline: str,
    metrics: Sequence[str],
    group_column: str = "system",
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Every (group, metric) comparison against one baseline, Holm-corrected.

    This is the single driver both E1 (`groups` = systems) and E2..E5
    (`groups` = variants) call, because they are the same computation over the
    same rows: each arm measured on the same queries, each compared to a
    baseline, each metric corrected for the size of its own family. The
    difference between them is which strings name the arms, and pretending
    otherwise would duplicate the pairing and the correction.

    Correction is applied **within each metric** across the groups compared for
    it. The family is not "all comparisons in the report": that would treat a
    latency comparison as evidence about recall, and would make the adjusted
    p-value depend on how many studies happened to be in the run. Family size is
    recorded per metric so the correction is checkable.
    """
    grouped = _group_rows(rows, group_column)
    if baseline not in grouped:
        raise PairingError(
            f"baseline {baseline!r} has no rows; there is nothing to compare against"
        )

    ordered = [name for name in groups if name in grouped]
    absent = [name for name in groups if name not in grouped]
    treatments = [name for name in ordered if name != baseline]

    by_metric: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
    comparisons: list[dict[str, Any]] = []
    for metric in metrics:
        for name in treatments:
            comparison = compare_metric(
                grouped[name],
                grouped[baseline],
                metric=metric,
                treatment=name,
                baseline=baseline,
                seed=seed,
                n_resamples=n_resamples,
                confidence=confidence,
                alpha=alpha,
            )
            by_metric[metric].append(comparison)
            comparisons.append(comparison)

    for metric, family in by_metric.items():
        pvalues = [c["p_value"] for c in family if c["p_value"] is not None]
        adjusted = holm_bonferroni(pvalues) if pvalues else []
        cursor = 0
        for comparison in family:
            if comparison["p_value"] is None:
                continue
            comparison["adjusted_p_value"] = adjusted[cursor]
            comparison["significant"] = adjusted[cursor] < alpha
            comparison["family_size"] = len(pvalues)
            cursor += 1

    return {
        "statistics_version": STATISTICS_VERSION,
        "group_column": group_column,
        "baseline": baseline,
        "groups_requested": list(groups),
        "groups_present": ordered,
        "groups_absent": absent,
        "metrics": list(metrics),
        "alpha": alpha,
        "bootstrap_seed": seed,
        "bootstrap_resamples": n_resamples,
        "bootstrap_confidence": confidence,
        "n_rows": len(rows),
        "test_selection_rule": TEST_SELECTION_RULE,
        "family_correction_note": (
            "Holm-Bonferroni is applied within each metric across the groups "
            "compared for it; the family size is on every comparison. Raw and "
            "adjusted p-values are both reported."
        ),
        "by_metric": by_metric,
        "comparisons": comparisons,
    }


def compare_systems(
    rows: Sequence[Any],
    *,
    systems: Sequence[str],
    baseline: str,
    metrics: Sequence[str],
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """E1's statistics: every system against the named baseline, paired."""
    return paired_comparisons(
        rows,
        groups=systems,
        baseline=baseline,
        metrics=metrics,
        group_column="system",
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
        alpha=alpha,
    )


def compare_variants(
    rows: Sequence[Any],
    *,
    variants: Sequence[str],
    baseline: str,
    metrics: Sequence[str],
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """E2..E5's statistics: every ablation variant against its own baseline.

    Identical machinery to `compare_systems` under a different name for the
    grouping, because an ablation variant *is* a `system` value in the row
    export. Kept as a named entry point so the CLI reads as the study it is
    reporting rather than as a re-parameterised E1.
    """
    return paired_comparisons(
        rows,
        groups=variants,
        baseline=baseline,
        metrics=metrics,
        group_column="system",
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
        alpha=alpha,
    )


def compare_all_pairs(
    rows: Sequence[Any],
    *,
    systems: Sequence[str],
    metrics: Sequence[str],
    group_column: str = "system",
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Every unordered pair of systems, Holm-corrected within each metric.

    This is the pre-registered Phase 8 family, written into
    `docs/phases/phase-8.md` §2.2 before the new corpus existed: five systems,
    **all** C(5,2) = 10 pairs via `itertools.combinations`, on
    `recall_at_5`, `mrr`, `ndcg_at_5`, and `total_latency_ms`, with
    `family_size` recorded per metric.

    The difference from `paired_comparisons` is exactly two things -- which pairs
    are enumerated, and what family the correction is applied over -- so
    `compare_metric` is reused unchanged. A baseline-anchored family of four
    comparisons answers "is anything better than bm25"; this answers "which of
    these five orderings are distinguishable from which", which is the question a
    before/after corpus study actually needs.

    Pair *direction* is arbitrary in principle and fixed in practice: within each
    pair the later system in `systems` is the treatment, so
    `systems=("bm25", "dense", ...)` reports `dense`-vs-`bm25` for the pairs it
    shares with `compare_systems`. Every test is two-sided, so this choice cannot
    change a verdict -- only which system the signed difference summary reads
    from.

    A system absent from the rows is **not** silently dropped: it is listed in
    `pairs_absent` and every pair that would have included it is reported with
    `n_pairs = 0` rather than being deleted from the family, because dropping
    pairs after the fact is precisely what a pre-registration forbids.
    """
    requested = list(systems)
    if len(set(requested)) != len(requested):
        raise PairingError(
            f"systems contains a duplicate name: {requested}. Each system must "
            "appear once, or a pair would be counted twice."
        )

    grouped = _group_rows(rows, group_column)
    present = [name for name in requested if name in grouped]
    absent = [name for name in requested if name not in grouped]
    if len(present) < 2:
        raise PairingError(
            f"all-pairs comparison needs at least 2 systems with rows; "
            f"{len(present)} of {len(requested)} are present "
            f"(absent: {absent})"
        )

    by_metric: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
    comparisons: list[dict[str, Any]] = []
    pair_log: list[dict[str, Any]] = []
    for metric in metrics:
        for first, second in combinations(present, 2):
            if first in absent or second in absent:
                continue
            # `combinations` yields the earlier system first, so the pair is
            # reported later-against-earlier. That is the direction a reader
            # expects -- "is the arm added after bm25 better than bm25" -- and it
            # makes `systems=("bm25", "dense", ...)` behave like the
            # baseline-anchored family for the pairs the two have in common.
            # The p-value is two-sided, so the direction cannot change the verdict;
            # it only decides which system the signed difference summary reads from.
            treatment, baseline = second, first
            comparison = compare_metric(
                grouped[treatment],
                grouped[baseline],
                metric=metric,
                treatment=treatment,
                baseline=baseline,
                seed=seed,
                n_resamples=n_resamples,
                confidence=confidence,
                alpha=alpha,
            )
            by_metric[metric].append(comparison)
            comparisons.append(comparison)
            pair_log.append({"metric": metric, "treatment": treatment, "baseline": baseline})

    for metric, family in by_metric.items():
        pvalues = [c["p_value"] for c in family if c["p_value"] is not None]
        adjusted = holm_bonferroni(pvalues) if pvalues else []
        cursor = 0
        for comparison in family:
            if comparison["p_value"] is None:
                continue
            comparison["adjusted_p_value"] = adjusted[cursor]
            comparison["significant"] = adjusted[cursor] < alpha
            # Two family sizes, because they answer different questions and a
            # reader who cannot tell them apart cannot tell what the p-value was
            # corrected against: `family_size_requested` is what the
            # pre-registration fixed (C(n_requested, 2)) and never changes, while
            # `family_size` is what Holm actually divided by, and shrinks when a
            # pair had no evaluable value.
            comparison["family_size"] = len(pvalues)
            comparison["family_size_requested"] = (
                len(requested) * (len(requested) - 1) // 2
            )
            cursor += 1

    return {
        "statistics_version": STATISTICS_VERSION,
        "group_column": group_column,
        "comparison_design": "all unordered pairs",
        "pair_enumeration": (
            "itertools.combinations over the systems given, in order; within each "
            "pair the later system is the treatment"
        ),
        "systems_requested": requested,
        "systems_present": present,
        "systems_absent": absent,
        "n_systems_present": len(present),
        # Evaluable pairs, so this agrees with the comparison count. The requested
        # count is on every comparison as `family_size_requested`.
        "n_pairs_per_metric": len(present) * (len(present) - 1) // 2,
        "n_pairs_requested_per_metric": len(requested) * (len(requested) - 1) // 2,
        "metrics": list(metrics),
        "alpha": alpha,
        "bootstrap_seed": seed,
        "bootstrap_resamples": n_resamples,
        "bootstrap_confidence": confidence,
        "n_rows": len(rows),
        "test_selection_rule": TEST_SELECTION_RULE,
        "family_correction_note": (
            "Holm-Bonferroni is applied within each metric across ALL "
            "C(n,2) pairs, not against a baseline subset; family_size and "
            "family_size_requested are both on every comparison. This family is "
            "pre-registered in docs/phases/phase-8.md section 2.2 and is "
            "post-hoc with respect to Phase 7's E1 point estimates, which were "
            "observed before it was specified."
        ),
        "preregistration": (
            "docs/phases/phase-8.md section 2 -- written before the new corpus "
            "existed; post-hoc/exploratory within this program because Phase 7's "
            "ordering motivated it"
        ),
        "pairs": pair_log,
        "by_metric": by_metric,
        "comparisons": comparisons,
    }


class StatisticsSettings(BaseModel):
    """The reproducibility record carried on every statistics artifact."""

    model_config = ConfigDict(extra="forbid")

    statistics_version: str = STATISTICS_VERSION
    seed: int = DEFAULT_SEED
    n_resamples: int = DEFAULT_N_RESAMPLES
    confidence: float = DEFAULT_CONFIDENCE
    alpha: float = DEFAULT_ALPHA
    pairing_rule: str = "by query_id, asserted (equal id sets, no duplicates)"
    family_correction: str = "Holm-Bonferroni step-down, within each metric"
    bootstrap_statistic: str = "median of the paired differences"
    metric_kinds: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_metrics(
        cls, metrics: Sequence[str], **overrides: Any
    ) -> "StatisticsSettings":
        """Settings for a run over `metrics`, recording each one's family."""
        return cls(
            metric_kinds={metric: metric_kind(metric) for metric in metrics},
            **overrides,
        )


def statistics_settings(
    metrics: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """The reproducibility record, as a plain JSON-ready mapping."""
    return StatisticsSettings.from_metrics(
        metrics,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
        alpha=alpha,
    ).model_dump(mode="json")


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "STATISTICS_VERSION",
    "TEST_SELECTION_RULE",
    "PairingError",
    "StatisticsSettings",
    "compare_all_pairs",
    "compare_metric",
    "compare_systems",
    "compare_variants",
    "index_by_query",
    "metric_kind",
    "pair_metric",
    "paired_comparisons",
    "select_test",
    "statistics_settings",
]
