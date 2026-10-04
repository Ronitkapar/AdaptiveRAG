"""
tests.test_statistics
---------------------
Offline, deterministic checks for the Phase 7 paired-statistics driver.

Two things are being verified here, and they are different. The *arithmetic* is
`tests.test_stats`'s job -- this module checks it was given the right inputs
(paired by `query_id`, verified rather than assumed) and that it chose the right
test for each metric family. The p-values themselves are not asserted against
hand-computed values, because a test that pins a p-value pins the data as much
as the test; what is pinned is the *decision*: which test ran, why, and that an
approximation is never described as exact.

Every seed is fixed and every bootstrap count is small, so the whole file is
byte-reproducible on any machine.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Sequence

import pytest

from adaptive_rag.evaluation.statistics import (
    DEFAULT_ALPHA,
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    TEST_SELECTION_RULE,
    PairingError,
    compare_all_pairs,
    compare_metric,
    compare_systems,
    compare_variants,
    index_by_query,
    metric_kind,
    pair_metric,
    paired_comparisons,
    select_test,
    statistics_settings,
)
from adaptive_rag.evaluation.stats import holm_bonferroni, sign_test_p

FAST = {"n_resamples": 400, "seed": 12345}


def make_row(query_id, system, **values):
    row = {"query_id": query_id, "system": system, "category": "factual", "split": "test"}
    row.update(values)
    return row


def paired_rows(n=10, *, latency_skew=False, cost_flat=False):
    """A baseline and a treatment over the same `n` queries.

    Quality rises by a fixed 0.1 per query for the treatment. Latency is
    identical except for a single 12 000 ms sample when `latency_skew` is set --
    the documented rate-limited embedding call. Cost is identical across the two
    systems when `cost_flat` is set, which is the heavily-tied case the sign test
    exists for.
    """
    baseline = []
    treatment = []
    for i in range(n):
        baseline.append(
            make_row(
                f"q{i:02d}",
                "bm25",
                recall_at_5=0.1 * (i + 1),
                mrr=0.05 * (i + 1),
                total_latency_ms=100.0 + 2.0 * i if not latency_skew
                else (12000.0 if i == 3 else 100.0 + 2.0 * i),
                estimated_cost_usd=0.001 if cost_flat else 0.001 + 0.0001 * i,
            )
        )
        treatment.append(
            make_row(
                f"q{i:02d}",
                "adaptive",
                recall_at_5=0.1 * (i + 1) + 0.1,
                mrr=0.05 * (i + 1) + 0.05,
                total_latency_ms=100.0 + 2.0 * i if not latency_skew
                else (12000.0 if i == 3 else 100.0 + 2.0 * i),
                estimated_cost_usd=0.001 if cost_flat else 0.002 + 0.0001 * i,
            )
        )
    return baseline, treatment


# --------------------------------------------------------------------------
# metric classification
# --------------------------------------------------------------------------


def test_metric_kind_classifies_every_phase7_column():
    assert metric_kind("recall_at_5") == "quality"
    assert metric_kind("mrr") == "quality"
    assert metric_kind("ndcg_at_5") == "quality"
    assert metric_kind("total_latency_ms") == "latency"
    assert metric_kind("routing_latency_ms") == "latency"
    assert metric_kind("estimated_cost_usd") == "cost"
    assert metric_kind("context_tokens") == "cost"


def test_metric_kind_rejects_a_column_it_does_not_own():
    """Guessing a family for an unknown column is how a latency series gets
    signed-rank tested as though it were a recall."""
    with pytest.raises(ValueError, match="not a known Phase 7 metric column"):
        metric_kind("some_new_column")


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------


def test_pair_metric_raises_when_the_two_arms_saw_different_queries():
    """Positional pairing over different id sets produces a confident, meaningless
    p-value. The pairing is asserted, not assumed."""
    baseline, treatment = paired_rows(n=5)
    truncated = [row for row in baseline if row["query_id"] != "q00"]

    with pytest.raises(PairingError) as excinfo:
        pair_metric(
            treatment, truncated, metric="recall_at_5",
            treatment="adaptive", baseline="bm25",
        )

    message = str(excinfo.value)
    assert "not run on the same queries" in message
    assert "q00" in message


def test_pair_metric_raises_on_a_duplicate_query_id_within_one_system():
    """A resumed run appends to `traces.jsonl`, so a repeated id is a real risk.
    Keeping either copy silently reweights that query."""
    duplicated = [
        make_row("q0", "adaptive", recall_at_5=0.5),
        make_row("q0", "adaptive", recall_at_5=0.6),
    ]

    with pytest.raises(PairingError, match="duplicate query_id"):
        index_by_query(duplicated)


def test_pair_metric_aligns_by_id_not_by_position():
    """Reversing one arm's export order must not change a single number."""
    baseline, treatment = paired_rows(n=6)

    forwards = pair_metric(
        treatment, baseline, metric="recall_at_5",
        treatment="adaptive", baseline="bm25",
    )
    backwards = pair_metric(
        list(reversed(treatment)), list(reversed(baseline)), metric="recall_at_5",
        treatment="adaptive", baseline="bm25",
    )

    assert forwards["pairs"] == backwards["pairs"]
    assert forwards["query_ids"] == sorted(forwards["query_ids"])


def test_pair_metric_counts_missing_values_instead_of_dropping_them():
    """A `None` latency is a clock that never started, not a zero-millisecond one,
    and it must not quietly change which queries are compared."""
    baseline, treatment = paired_rows(n=4)
    treatment[2]["total_latency_ms"] = None

    paired = pair_metric(
        treatment, baseline, metric="total_latency_ms",
        treatment="adaptive", baseline="bm25",
    )

    assert paired["n_aligned_queries"] == 4
    assert paired["n_pairs"] == 3
    assert paired["n_missing_treatment"] == 1
    assert paired["missing_treatment_query_ids"] == ["q02"]


# --------------------------------------------------------------------------
# test selection
# --------------------------------------------------------------------------


def test_quality_uses_the_signed_rank_test_when_the_differences_are_symmetric():
    """Symmetric, centred differences: the signed-rank assumption holds and the
    signed-rank test is the one reported."""
    differences = [0.02, -0.01, 0.03, 0.04, -0.02, 0.01, 0.05, -0.03, 0.02, 0.01]

    result = select_test("recall_at_5", differences)

    assert result["metric_kind"] == "quality"
    assert result["selected_test"] == "wilcoxon_signed_rank"
    assert result["symmetry"]["symmetry_ok"] is True
    assert "symmetric" in result["reason"]
    assert result["p_value"] == result["wilcoxon_p_value"]
    # The sign test is still reported beside it, because quality values tie
    # heavily and the exact table is then unavailable.
    assert result["sign_test_p"] is not None


def test_quality_falls_back_to_the_sign_test_when_symmetry_fails():
    """A right-skewed quality difference is refused by the signed-rank test on its
    own stated criterion, and the sign test becomes the reported p-value."""
    differences = [0.5] * 9 + [-0.01]

    result = select_test("recall_at_5", differences)

    assert result["metric_kind"] == "quality"
    assert result["symmetry"]["symmetry_ok"] is False
    assert result["selected_test"] == "sign_test"
    assert result["p_value"] == result["sign_test_p"]
    assert result["p_value"] != result["wilcoxon_p_value"]
    assert "sign test" in result["reason"]


def test_latency_uses_the_sign_test_by_construction_not_as_a_fallback():
    """Latency is right-skewed by design here, so the sign test is the primary
    choice rather than something discovered after a failed assumption check."""
    differences = [10.0] * 9 + [-5000.0]

    result = select_test("total_latency_ms", differences)

    assert result["metric_kind"] == "latency"
    assert result["selected_test"] == "sign_test"
    assert "non-negative and right-skewed" in result["reason"]
    assert "12.0 s" in TEST_SELECTION_RULE["latency"]["why"]


def test_cost_uses_the_sign_test_because_cost_ties_exactly():
    """Two systems that built the same context have identical cost, so the
    signed-rank sample shrinks as the ties grow."""
    result = select_test("estimated_cost_usd", [0.0, 0.0, 0.0, 0.0])

    assert result["metric_kind"] == "cost"
    assert result["selected_test"] == "sign_test"
    assert "tied" in result["reason"]


def test_an_approximation_is_never_reported_as_exact():
    """The one terminology rule the statistics module exists to enforce: a
    tie-corrected normal approximation is not an exact null distribution."""
    # 30 differences with a tie group: beyond EXACT_MAX_N, so the approximation
    # takes over, and it must say so in the method string.
    differences = [0.1 * (i % 5) for i in range(30)]
    quality = select_test("recall_at_5", differences)
    assert quality["symmetry"]["symmetry_ok"] is True
    assert quality["p_value_method"].startswith("normal_approx")
    assert "exact" not in quality["p_value_method"]

    # And when the sign test is the reported test, the method string describes
    # the sign test rather than the signed-rank computation that was skipped.
    skewed = select_test("recall_at_5", [0.5] * 29 + [-0.01])
    assert skewed["selected_test"] == "sign_test"
    assert skewed["p_value_method"] == "exact binomial sign test"
    # stats rounds a p-value to six significant figures, so compare with a
    # relative tolerance rather than for exact float equality.
    assert skewed["p_value"] == pytest.approx(sign_test_p(29, 1), rel=1e-5)


# --------------------------------------------------------------------------
# one comparison
# --------------------------------------------------------------------------


def test_compare_metric_reports_both_series_and_a_median_effect():
    """The treatment is 0.1 above the baseline on every query, so every paired
    difference is exactly 0.1 and the median difference is 0.1."""
    baseline, treatment = paired_rows(n=10)

    result = compare_metric(
        treatment, baseline, metric="recall_at_5",
        treatment="adaptive", baseline="bm25", **FAST,
    )

    assert result["metric"] == "recall_at_5"
    assert result["metric_kind"] == "quality"
    assert result["n_pairs"] == 10
    assert result["treatment_summary"]["mean"] == pytest.approx(0.65)
    assert result["baseline_summary"]["mean"] == pytest.approx(0.55)
    assert result["difference_summary"]["mean"] == pytest.approx(0.1)
    assert result["effect"]["median_difference"] == pytest.approx(0.1)
    assert result["effect"]["wins"] == 10
    assert result["effect"]["losses"] == 0
    assert result["effect"]["ties"] == 0
    assert result["effect"]["ci_low"] == pytest.approx(0.1)
    assert result["effect"]["ci_high"] == pytest.approx(0.1)
    assert result["bootstrap_seed"] == FAST["seed"]
    assert result["bootstrap_resamples"] == FAST["n_resamples"]


def test_compare_metric_keeps_the_latency_tail_visible_next_to_the_median():
    """The 12 s sample drags the mean far above the median; both are reported, and
    the effect estimate stays on the median."""
    baseline, treatment = paired_rows(n=10, latency_skew=True)

    result = compare_metric(
        treatment, baseline, metric="total_latency_ms",
        treatment="adaptive", baseline="bm25", **FAST,
    )

    assert result["metric_kind"] == "latency"
    assert result["test"]["selected_test"] == "sign_test"
    assert result["treatment_summary"]["mean"] > result["treatment_summary"]["median"]
    assert result["treatment_summary"]["max"] == 12000.0
    assert result["treatment_summary"]["p95"] == 12000.0
    assert result["effect"]["median_difference"] == 0.0
    assert "median" in result["note"]


def test_compare_metric_is_reproducible_for_a_fixed_seed_and_resample_count():
    """Two runs of the same comparison with the same seed agree to the last digit;
    a different seed is allowed to move the interval, which is why it is recorded."""
    baseline, treatment = paired_rows(n=8)

    def run(**overrides):
        return compare_metric(
            treatment, baseline, metric="mrr",
            treatment="adaptive", baseline="bm25",
            **{**FAST, **overrides},
        )

    first, second = run(), run()
    assert first["effect"] == second["effect"]
    assert first["p_value"] == second["p_value"]

    # A different resample count is honoured, and recorded as the count used.
    finer = run(n_resamples=2000)
    assert finer["bootstrap_resamples"] == 2000
    assert finer["effect"]["ci_low"] == pytest.approx(finer["effect"]["median_difference"])


def test_compare_metric_with_no_usable_pair_says_so_instead_of_scoring_zero():
    baseline, treatment = paired_rows(n=3)
    for row in treatment:
        row["mrr"] = None

    result = compare_metric(
        treatment, baseline, metric="mrr",
        treatment="adaptive", baseline="bm25", **FAST,
    )

    assert result["n_pairs"] == 0
    assert result["p_value"] is None
    assert result["significant"] is False
    assert "no query carried a value" in result["note"]


# --------------------------------------------------------------------------
# Holm-Bonferroni
# --------------------------------------------------------------------------


def test_holm_adjustment_is_monotone_and_never_below_the_raw_value():
    """Sorted by raw p, the adjusted values must not decrease and must not be
    smaller than the raw ones -- both are what makes Holm a valid step-down."""
    raw = [0.001, 0.008, 0.02, 0.04, 0.6]

    adjusted = holm_bonferroni(raw)

    assert adjusted == sorted(adjusted), "adjusted values must be non-decreasing"
    for original, corrected in zip(raw, adjusted):
        assert corrected >= original
        assert corrected <= 1.0
    # Hand-computed for m=5: 5*0.001, then max(that, 4*0.008), then max(.., 3*0.02),
    # then max(.., 2*0.04), then max(.., 1*0.6).
    assert adjusted == pytest.approx([0.005, 0.032, 0.06, 0.08, 0.6])


def test_holm_adjustment_applies_per_metric_family_not_across_the_report():
    """Each metric's family is the set of groups compared for it, so a latency
    comparison is never counted as evidence about recall."""
    rows = []
    for system, offset in (("bm25", 0.0), ("a", 0.1), ("b", 0.2), ("c", 0.03)):
        for i in range(8):
            rows.append(
                make_row(
                    f"q{i:02d}", system,
                    recall_at_5=0.4 + 0.01 * i + offset,
                    total_latency_ms=100.0 + i + offset * 10,
                )
            )

    report = compare_systems(
        rows,
        systems=("bm25", "a", "b", "c"),
        baseline="bm25",
        metrics=("recall_at_5", "total_latency_ms"),
        **FAST,
    )

    for metric, family in report["by_metric"].items():
        assert len(family) == 3
        for comparison in family:
            assert comparison["metric"] == metric
            assert comparison["family_size"] == 3
            assert comparison["adjusted_p_value"] >= comparison["p_value"]
            assert comparison["significant"] == (comparison["adjusted_p_value"] < DEFAULT_ALPHA)


def test_reported_raw_and_adjusted_p_values_are_both_present():
    """A reader must be able to see what the correction changed."""
    rows = []
    for system, offset in (("bm25", 0.0), ("a", 0.4), ("b", 0.42)):
        for i in range(8):
            rows.append(
                make_row(f"q{i:02d}", system, recall_at_5=0.4 + 0.01 * i + offset)
            )

    report = compare_systems(
        rows, systems=("bm25", "a", "b"), baseline="bm25",
        metrics=("recall_at_5",), **FAST,
    )

    for comparison in report["comparisons"]:
        assert comparison["p_value"] is not None
        assert comparison["adjusted_p_value"] is not None
        assert comparison["significant"] in (True, False)
    adjusted = [c["adjusted_p_value"] for c in report["comparisons"]]
    raw = [c["p_value"] for c in report["comparisons"]]
    assert any(a > r for a, r in zip(adjusted, raw)), "correction changed nothing here"


# --------------------------------------------------------------------------
# a family of comparisons
# --------------------------------------------------------------------------


def test_paired_comparisons_records_the_full_reproducibility_context():
    baseline, treatment = paired_rows(n=6)

    report = compare_systems(
        baseline + treatment, systems=("bm25", "adaptive"), baseline="bm25",
        metrics=("recall_at_5",), **FAST,
    )

    assert report["statistics_version"] == "phase7_statistics_v1"
    assert report["baseline"] == "bm25"
    assert report["bootstrap_seed"] == FAST["seed"]
    assert report["bootstrap_resamples"] == FAST["n_resamples"]
    assert report["alpha"] == DEFAULT_ALPHA
    assert report["test_selection_rule"] == TEST_SELECTION_RULE
    assert "within each metric" in report["family_correction_note"]
    assert json.loads(json.dumps(report)) == report


def test_paired_comparisons_reports_absent_groups_without_guessing_a_baseline():
    baseline, treatment = paired_rows(n=6)

    report = compare_systems(
        baseline + treatment, systems=("bm25", "adaptive", "hybrid"),
        baseline="bm25", metrics=("recall_at_5",), **FAST,
    )

    assert report["groups_absent"] == ["hybrid"]
    assert [c["treatment"] for c in report["comparisons"]] == ["adaptive"]


def test_paired_comparisons_refuses_a_baseline_with_no_rows():
    baseline, _ = paired_rows(n=4)

    with pytest.raises(PairingError, match="nothing to compare against"):
        compare_systems(
            baseline, systems=("bm25",), baseline="adaptive",
            metrics=("recall_at_5",), **FAST,
        )


def test_compare_variants_is_the_same_computation_over_a_variant_grid():
    """E3's arms are `system` values like any other, so the driver is shared and
    a variant grid cannot get different pairing or correction from E1's."""
    rows = []
    for variant, offset in (("full", 0.0), ("without_entity", 0.05),
                            ("without_lexical", 0.01)):
        for i in range(8):
            rows.append(
                make_row(f"q{i:02d}", variant, recall_at_5=0.4 + 0.01 * i + offset)
            )

    report = compare_variants(
        rows, variants=("full", "without_entity", "without_lexical"),
        baseline="full", metrics=("recall_at_5",), **FAST,
    )

    assert report["group_column"] == "system"
    assert {c["treatment"] for c in report["comparisons"]} == {
        "without_entity",
        "without_lexical",
    }
    assert all(c["baseline"] == "full" for c in report["comparisons"])


def test_identical_systems_produce_no_evidence_rather_than_a_small_p_value():
    """With every difference zero the paired test is degenerate, and a degenerate
    comparison must read as 'no evidence', not as significance."""
    rows = [make_row(f"q{i:02d}", system, recall_at_5=0.5)
            for i in range(8) for system in ("bm25", "adaptive")]

    report = compare_systems(
        rows, systems=("bm25", "adaptive"), baseline="bm25",
        metrics=("recall_at_5",), **FAST,
    )

    comparison = report["comparisons"][0]
    assert comparison["effect"]["ties"] == 8
    assert comparison["effect"]["median_difference"] == 0.0
    assert comparison["p_value"] == 1.0
    assert comparison["significant"] is False


def test_statistics_settings_classify_every_requested_metric():
    settings = statistics_settings(
        ("recall_at_5", "total_latency_ms", "estimated_cost_usd"),
        seed=7, n_resamples=1000, alpha=0.01,
    )

    assert settings["seed"] == 7
    assert settings["n_resamples"] == 1000
    assert settings["alpha"] == 0.01
    assert settings["metric_kinds"] == {
        "recall_at_5": "quality",
        "total_latency_ms": "latency",
        "estimated_cost_usd": "cost",
    }
    assert "median of the paired differences" in settings["bootstrap_statistic"]
    assert settings["seed"] == DEFAULT_SEED or settings["seed"] == 7
    assert DEFAULT_N_RESAMPLES == 10000


# --------------------------------------------------------------------------
# E2/E3 ablation comparison structure regression tests
# --------------------------------------------------------------------------


def _make_ablation_rows(variants: Sequence[str], n: int = 10, base: float = 0.4) -> list[dict[str, Any]]:
    """Synthetic rows for ablation variants, each differing by a fixed offset."""
    rows = []
    for idx, variant in enumerate(variants):
        offset = idx * 0.05  # A=0.0, B=0.05, C=0.10, etc.
        for i in range(n):
            rows.append(make_row(f"q{i:02d}", variant, recall_at_5=base + 0.01 * i + offset))
    return rows


def test_e2_escalation_ablation_compares_chained_pairs_only():
    """E2 is a component-addition ablation: A -> B -> C.
    The meaningful pairs are A-vs-B (cost of measuring sufficiency)
    and B-vs-C (value of acting on it). C-vs-A must NOT be computed.
    """
    from adaptive_rag.evaluation.analysis import DEFAULT_ESCALATION_VARIANTS
    from scripts.run_phase7_analysis import build_statistics

    variants = list(DEFAULT_ESCALATION_VARIANTS)  # ["A_no_sufficiency_no_escalation", "B_sufficiency_no_escalation", "C_sufficiency_bounded_escalation"]
    rows = _make_ablation_rows(variants, n=10)
    grouped = {"E2_escalation_ablation": rows}

    stats = build_statistics(
        grouped,
        systems=("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"),
        baseline="bm25",
        metrics=("recall_at_5",),
        seed=12345,
        n_resamples=400,
        confidence=0.95,
        alpha=0.05,
    )

    e2_report = stats["reports"]["E2"]
    # Should have exactly 2 comparisons: A-vs-B and B-vs-C
    assert len(e2_report["comparisons"]) == 2
    pairs = {(c["baseline"], c["treatment"]) for c in e2_report["comparisons"]}
    expected = {
        ("A_no_sufficiency_no_escalation", "B_sufficiency_no_escalation"),
        ("B_sufficiency_no_escalation", "C_sufficiency_bounded_escalation"),
    }
    assert pairs == expected, f"Expected chained pairs {expected}, got {pairs}"
    # C-vs-A must not be present
    assert ("A_no_sufficiency_no_escalation", "C_sufficiency_bounded_escalation") not in pairs
    assert ("C_sufficiency_bounded_escalation", "A_no_sufficiency_no_escalation") not in pairs
    # Family size is 2 (two chained comparisons per metric)
    for c in e2_report["comparisons"]:
        assert c["family_size"] == 2
    # Baseline recorded as chained
    assert e2_report["baseline"] == "chained (A-vs-B, B-vs-C)"


def test_e3_feature_ablation_compares_all_against_full():
    """E3 is a leave-one-out ablation: each without_* variant is compared
    against the 'full' baseline. No chained pairs.
    """
    from adaptive_rag.evaluation.analysis import DEFAULT_FEATURE_VARIANTS
    from scripts.run_phase7_analysis import build_statistics

    # Use all feature variants from the actual analysis module
    variants = list(DEFAULT_FEATURE_VARIANTS)  # ["full", "without_lexical", "without_semantic", "without_entity", "without_complexity", "without_question_type", "without_multi_concept"]
    rows = _make_ablation_rows(variants, n=10)
    grouped = {"E3_feature_ablation": rows}

    stats = build_statistics(
        grouped,
        systems=("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"),
        baseline="bm25",
        metrics=("recall_at_5",),
        seed=12345,
        n_resamples=400,
        confidence=0.95,
        alpha=0.05,
    )

    e3_report = stats["reports"]["E3"]
    # Should have 6 comparisons: each without_* vs full (6 feature groups)
    assert len(e3_report["comparisons"]) == 6
    for c in e3_report["comparisons"]:
        assert c["baseline"] == "full"
        assert c["treatment"].startswith("without_")
    assert e3_report["baseline"] == "full"
    assert e3_report["groups_absent"] == []


# --------------------------------------------------------------------------
# Phase 8 all-pairs comparison family
# --------------------------------------------------------------------------


def _five_arm_rows(offsets: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """The five Phase 8 systems over the same 8 queries."""
    offsets = offsets or {
        "bm25": 0.0,
        "dense": 0.12,
        "hybrid": 0.08,
        "hybrid_rerank": 0.04,
        "adaptive": 0.10,
    }
    return [
        make_row(
            f"q{i:02d}",
            system,
            recall_at_5=0.4 + 0.01 * i + offset,
            total_latency_ms=100.0 + i + offset * 10,
        )
        for system, offset in offsets.items()
        for i in range(8)
    ]


def test_compare_all_pairs_enumerates_every_unordered_pair():
    """Five systems must give all C(5,2) = 10 pairs, not a baseline subset.

    The pre-registered family (`docs/phases/phase-8.md` section 2.2) is the whole
    point: E1 compared each arm against `bm25` only, which answers "is anything
    better than bm25" and not "which of these five are distinguishable".
    """
    systems = ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")

    report = compare_all_pairs(
        _five_arm_rows(),
        systems=systems,
        metrics=("recall_at_5", "total_latency_ms"),
        **FAST,
    )

    assert report["n_systems_present"] == 5
    assert report["n_pairs_per_metric"] == 10
    assert len(report["by_metric"]["recall_at_5"]) == 10
    assert len(report["by_metric"]["total_latency_ms"]) == 10
    assert len(report["comparisons"]) == 20

    unordered = {
        frozenset((c["treatment"], c["baseline"])) for c in report["comparisons"]
    }
    expected = {frozenset(pair) for pair in combinations(systems, 2)}
    assert unordered == expected
    assert len(unordered) == 10, "each unordered pair must appear exactly once"


def test_compare_all_pairs_corrects_within_the_ten_pair_family():
    """Holm runs across all 10 pairs of a metric, and says how many it divided by.

    `family_size_requested` is pinned separately from `family_size` because they
    differ whenever a pair had no evaluable value, and a reader who cannot tell
    those apart cannot tell what the p-value was corrected against.
    """
    report = compare_all_pairs(
        _five_arm_rows(),
        systems=("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"),
        metrics=("recall_at_5",),
        **FAST,
    )

    for comparison in report["comparisons"]:
        assert comparison["family_size"] == 10
        assert comparison["family_size_requested"] == 10
        assert comparison["adjusted_p_value"] >= comparison["p_value"]
        assert comparison["significant"] == (
            comparison["adjusted_p_value"] < DEFAULT_ALPHA
        )


def test_compare_all_pairs_records_the_disclosure_it_runs_under():
    """The family is pre-registered but post-hoc, and the artifact must say so.

    Phase 7's E1 ordering was seen before this family was specified, so a reader
    must not be able to mistake a pre-registered test for an anticipated one.
    """
    report = compare_all_pairs(
        _five_arm_rows(),
        systems=("bm25", "dense"),
        metrics=("recall_at_5",),
        **FAST,
    )

    assert "docs/phases/phase-8.md section 2" in report["preregistration"]
    assert "post-hoc" in report["preregistration"]
    assert "post-hoc" in report["family_correction_note"]


def test_compare_all_pairs_reports_an_absent_system_rather_than_dropping_its_pairs():
    """A missing system is recorded, not silently pruned.

    Dropping pairs after the fact is exactly what a pre-registration forbids, so
    the omission has to be visible in the artifact.
    """
    rows = _five_arm_rows()
    present = [r for r in rows if r["system"] != "adaptive"]

    report = compare_all_pairs(
        present,
        systems=("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"),
        metrics=("recall_at_5",),
        **FAST,
    )

    assert report["systems_absent"] == ["adaptive"]
    assert report["systems_present"] == ["bm25", "dense", "hybrid", "hybrid_rerank"]
    # 4 systems present -> C(4,2) = 6 evaluable pairs, against a requested 10.
    assert report["n_pairs_per_metric"] == 6
    assert len(report["comparisons"]) == 6
    for comparison in report["comparisons"]:
        assert comparison["family_size"] == 6
        assert comparison["family_size_requested"] == 10


def test_compare_all_pairs_rejects_a_duplicate_system_name():
    """A duplicate would count one unordered pair twice, inflating the family."""
    with pytest.raises(PairingError, match="duplicate"):
        compare_all_pairs(
            _five_arm_rows(),
            systems=("bm25", "dense", "bm25"),
            metrics=("recall_at_5",),
            **FAST,
        )


def test_compare_all_pairs_needs_two_present_systems():
    with pytest.raises(PairingError, match="at least 2 systems"):
        compare_all_pairs(
            _five_arm_rows(),
            systems=("bm25",),
            metrics=("recall_at_5",),
            **FAST,
        )


def test_compare_all_pairs_shares_pairing_and_correction_with_the_baseline_driver():
    """The all-pairs family must not be a second implementation of the test.

    `compare_metric` is reused unchanged, so the one pair both drivers share has
    to come out identical -- if this ever diverges, the difference is in pair
    enumeration or the correction family, which is the only thing that is
    supposed to differ.
    """
    rows = _five_arm_rows()
    systems = ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")

    all_pairs = compare_all_pairs(
        rows, systems=systems, metrics=("recall_at_5",), **FAST
    )
    anchored = compare_systems(
        rows, systems=systems, baseline="bm25", metrics=("recall_at_5",), **FAST
    )

    def _find(report, treatment):
        return next(
            c for c in report["comparisons"]
            if c["treatment"] == treatment and c["baseline"] == "bm25"
        )

    shared = _find(all_pairs, "dense")
    reference = _find(anchored, "dense")
    for field in ("p_value", "test", "n_pairs", "effect"):
        assert shared[field] == reference[field], field
    # Only the family the correction is applied over may differ.
    assert shared["family_size"] == 10
    assert reference["family_size"] == 4
