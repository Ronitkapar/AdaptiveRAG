"""
Offline tests for Gate 3, `scripts/gate3_signal_exhaustion.py`.

The failures these guard against are the ones that would make a null result look
like a result: a label defined on the wrong split, an AUC that cannot detect a
perfect signal, a permutation test that leaves the fitted model untouched, and
Holm correction applied in the wrong order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.gate3_signal_exhaustion import (
    ALPHA,
    AUC_FLOOR,
    CATEGORICAL_LEVELS,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    POSITIVE_LABEL,
    QUERY_FEATURES,
    UNIVARIATE_COLUMNS,
    RidgeLogistic,
    auc_score,
    bootstrap_auc_ci,
    constant_columns,
    decision_criteria,
    design_matrix,
    evaluate_arm_gate,
    evaluate_gate,
    hanley_mcneil_se,
    holm_bonferroni,
    minimum_detectable_auc,
    permutation_p_value,
    read_observable_features,
    univariate_screen,
)


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_auc_is_one_for_a_perfect_separation_and_zero_when_inverted():
    assert auc_score([1, 1, 0, 0], [3.0, 4.0, 1.0, 2.0]) == 1.0
    assert auc_score([1, 1, 0, 0], [1.0, 2.0, 3.0, 4.0]) == 0.0


def test_auc_counts_ties_as_half_so_a_degenerate_feature_is_chance():
    """A constant feature must read exactly 0.5, not 0 or 1 by tie order."""
    assert auc_score([1, 1, 0, 0], [1.0, 1.0, 1.0, 1.0]) == 0.5


def test_auc_is_nan_when_one_class_is_absent():
    assert np.isnan(auc_score([1, 1, 1], [1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# The measuring model
# ---------------------------------------------------------------------------


def test_ridge_logistic_learns_a_separable_direction():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = RidgeLogistic().fit(x, y)
    assert model.coef_[0] > 0
    assert model.converged_
    assert model.decision_function(np.array([[5.0]]))[0] > model.decision_function(
        np.array([[-5.0]])
    )[0]


def test_ridge_logistic_is_deterministic_under_a_fixed_input():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5))
    y = (x[:, 0] > 0).astype(int)
    assert np.allclose(RidgeLogistic().fit(x, y).coef_, RidgeLogistic().fit(x, y).coef_)


def test_class_weight_balancing_actually_moves_the_boundary():
    """Without balancing, 2 positives against 44 negatives fit the intercept and
    every test AUC is 0.5 by construction -- the failure mode that would
    silently manufacture a null result."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(46, 3))
    y = np.zeros(46, dtype=int)
    y[:2] = 1
    x[:2] += 3.0  # positives are separated
    balanced = RidgeLogistic(class_weight="balanced").fit(x, y)
    unbalanced = RidgeLogistic(class_weight="none").fit(x, y)
    assert np.max(np.abs(balanced.coef_)) > np.max(np.abs(unbalanced.coef_))


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def test_holm_is_monotone_in_rank_and_never_below_the_raw_p():
    corrections = holm_bonferroni([0.01, 0.04, 0.03])
    adjusted = [c["p_holm"] for c in corrections]
    assert all(a >= r for a, r in zip(adjusted, [0.01, 0.04, 0.03]))
    by_rank = sorted(corrections, key=lambda c: c["rank"])
    assert by_rank[0]["p_holm"] <= by_rank[1]["p_holm"] <= by_rank[2]["p_holm"]


def test_holm_rejects_only_the_earliest_of_a_moderate_family():
    """One p=0.01 in a family of 10 must not survive Holm; alone it would."""
    assert not any(c["significant_holm"] for c in holm_bonferroni([0.01] + [0.5] * 9))
    assert holm_bonferroni([0.01])[0]["significant_holm"]


def test_holm_is_at_least_as_conservative_as_the_smallest_raw_p():
    assert holm_bonferroni([0.001, 0.9, 0.9])[0]["p_holm"] >= 0.001 * 3


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


def test_hanley_mcneil_se_shrinks_as_the_effect_grows():
    assert hanley_mcneil_se(0.9, 6, 54) < hanley_mcneil_se(0.5, 6, 54)


def test_power_is_low_for_a_weak_effect_and_high_for_a_strong_one():
    """The sign convention matters: a true AUC of 0.6 with 6 positives cannot
    produce a CI excluding chance, so power must be low there."""
    result = minimum_detectable_auc(6, 54)
    grid = {row["true_auc"]: row["power_ci_excludes_0_5"] for row in result["grid"]}
    assert grid[0.6] < 0.25
    assert grid[0.9] > 0.95
    assert grid[0.6] < 0.80 < grid[0.9]
    assert result["min_detectable_auc_at_80pct_power"] > 0.6


def test_minimum_detectable_auc_is_higher_for_a_rarer_class():
    assert minimum_detectable_auc(2, 45)["min_detectable_auc_at_80pct_power"] > (
        minimum_detectable_auc(60, 600)["min_detectable_auc_at_80pct_power"]
    )
# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def test_permutation_test_finds_a_real_signal():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 4))
    labels = (matrix[:, 0] + rng.normal(scale=0.2, size=40) > 0).astype(int)
    fit_idx = np.arange(0, 20)
    eval_idx = np.arange(20, 40)
    model = RidgeLogistic().fit(matrix[fit_idx], labels[fit_idx])
    observed = auc_score(labels[eval_idx], model.decision_function(matrix[eval_idx]))
    result = permutation_p_value(
        matrix, labels, fit_idx, eval_idx, observed, n_resamples=500, seed=3
    )
    assert result["p_value"] < 0.05


def test_permutation_test_finds_nothing_in_pure_noise():
    """The null must come back null. If this fails, a negative Gate 3 result
    would be an artefact of a broken test rather than evidence."""
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(60, 4))
    labels = np.array([1] * 12 + [0] * 48)
    rng.shuffle(labels)
    fit_idx = np.arange(0, 30)
    eval_idx = np.arange(30, 60)
    model = RidgeLogistic().fit(matrix[fit_idx], labels[fit_idx])
    observed = auc_score(labels[eval_idx], model.decision_function(matrix[eval_idx]))
    result = permutation_p_value(
        matrix, labels, fit_idx, eval_idx, observed, n_resamples=1000, seed=5
    )
    assert result["p_value"] > 0.05


def test_permutation_test_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(40, 3))
    labels = (matrix[:, 1] > 0).astype(int)
    fit_idx = np.arange(0, 20)
    eval_idx = np.arange(20, 40)
    model = RidgeLogistic().fit(matrix[fit_idx], labels[fit_idx])
    observed = auc_score(labels[eval_idx], model.decision_function(matrix[eval_idx]))
    kwargs = dict(n_resamples=300, seed=13)
    first = permutation_p_value(matrix, labels, fit_idx, eval_idx, observed, **kwargs)
    second = permutation_p_value(matrix, labels, fit_idx, eval_idx, observed, **kwargs)
    assert first["p_value"] == second["p_value"]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_under_a_fixed_seed():
    labels = [1] * 8 + [0] * 40
    scores = list(np.linspace(0.0, 1.0, 48))
    first = bootstrap_auc_ci(labels, scores, n_resamples=200, seed=4)
    second = bootstrap_auc_ci(labels, scores, n_resamples=200, seed=4)
    assert first == second

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------


def test_no_feature_column_is_empty_and_names_are_unique():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert all(column for column in FEATURE_COLUMNS)


def test_the_categorical_one_hot_block_matches_its_declared_levels():
    expected = {
        f"{name}={level}"
        for name, levels in CATEGORICAL_LEVELS.items()
        for level in levels
    }
    assert expected.issubset(set(FEATURE_COLUMNS))


def test_every_univariate_feature_belongs_to_a_declared_group():
    declared = {name for names in FEATURE_GROUPS.values() for name in names}
    assert set(UNIVARIATE_COLUMNS).issubset(declared)


def test_design_matrix_imputes_from_the_rows_it_is_built_from():
    rows = {"q0": {"a": 1.0}, "q1": {"a": None}, "q2": {"a": 5.0}}
    matrix = design_matrix(rows, ["q0", "q1", "q2"], columns=("a",))
    # Median of {1, 5} is 3.0; a missing value must not silently become zero.
    assert matrix[1, 0] == 3.0


def test_design_matrix_flags_constant_columns_instead_of_hiding_them():
    rows = {"q0": {"a": 2.0, "b": 1.0}, "q1": {"a": 2.0, "b": 4.0}}
    matrix = design_matrix(rows, ["q0", "q1"], columns=("a", "b"))
    assert constant_columns(matrix, ["a", "b"]) == ["a"]


def test_read_observable_features_yields_no_evaluation_metric(tmp_path):
    """A feature row must contain nothing that reveals an oracle outcome; that
    is the boundary that keeps Gate 3 from importing its own answer."""
    trace = {
        "example_id": "q0",
        "category": "factual",
        "routing": {
            "initial_strategy": "hybrid",
            "initial_result_count": 10,
            "features": dict(
                {name: 1 for name in QUERY_FEATURES},
                question_type="compare",
                multi_concept=True,
            ),
            "sufficiency": {
                "result_count": 10,
                "coverage": 0.8,
                "top1_coverage": 0.4,
                "score": 0.7,
            },
            "decision": {
                "confidence": 0.3,
                "evidence": [
                    {"strategy": "hybrid", "score": 2.0},
                    {"strategy": "bm25", "score": 1.0},
                ],
            },
        },
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    record = read_observable_features(path)["q0"]
    assert record["coverage"] == 0.8
# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def _arm(**overrides) -> dict:
    primary = {
        "auc": 0.80,
        "ci_lower": 0.62,
        "ci_excludes_0_5": True,
        "permutation": {"p_value": 0.01},
    }
    screen = {"n_significant_holm": 2}
    primary.update(overrides.pop("primary", {}))
    screen.update(overrides.pop("screen", {}))
    block = {
        "primary_out_of_sample": primary,
        "univariate_screen": screen,
        "gate": {},
    }
    block.update(overrides)
    return block


def test_arm_gate_passes_only_when_every_pre_registered_condition_holds():
    assert evaluate_arm_gate(_arm())["passed"]


def test_arm_gate_fails_when_the_auc_is_below_the_deployability_floor():
    gate = evaluate_arm_gate(_arm(primary={"auc": 0.55}))
    assert not gate["passed"]
    assert not gate["conditions"]["auc_at_or_above_floor"]["pass"]


def test_arm_gate_fails_when_no_feature_survives_holm():
    gate = evaluate_arm_gate(_arm(screen={"n_significant_holm": 0}))
    assert not gate["passed"]
    assert not gate["conditions"]["a_feature_survives_holm"]["pass"]


def test_arm_gate_fails_when_the_permutation_p_is_not_significant():
    gate = evaluate_arm_gate(_arm(primary={"permutation": {"p_value": 0.4}}))
    assert not gate["conditions"]["permutation_p_below_alpha"]["pass"]


def test_gate_requires_replication_across_both_arms():
    strong = _arm()
    weak = _arm(
        primary={
            "auc": 0.52,
            "ci_lower": 0.3,
            "ci_excludes_0_5": False,
            "permutation": {"p_value": 0.4},
        },
        screen={"n_significant_holm": 0},
    )
    strong["gate"] = evaluate_arm_gate(strong)
    weak["gate"] = evaluate_arm_gate(weak)
    verdict = evaluate_gate({"after": strong, "before": weak})
    assert verdict["verdict"] == "no_out_of_sample_signal"
    assert not verdict["replication_holds"]


def test_gate_passes_only_when_both_arms_clear_every_condition():
    first = _arm()
    second = _arm()
    first["gate"] = evaluate_arm_gate(first)
    second["gate"] = evaluate_arm_gate(second)
    verdict = evaluate_gate({"after": first, "before": second})
    assert verdict["verdict"] == "signal_detected"
    assert verdict["replication_holds"]


def test_pre_registered_criteria_state_the_thresholds_the_gate_enforces():
    criteria = decision_criteria()
    assert criteria["auc_floor"] == AUC_FLOOR
    assert criteria["alpha"] == ALPHA
    assert criteria["fit_split"] == "calibration"
    assert criteria["eval_split"] == "test"
    assert "not as 'no signal exists'" in criteria["interpretation_rule"]


def test_the_pre_registered_rule_forbids_a_learned_router_on_failure():
    """The decision the user asked for, pinned: a negative gate must not
    license a learned router just because the rule-based one failed."""
    weak = _arm(
        primary={
            "auc": 0.52,
            "ci_lower": 0.3,
            "ci_excludes_0_5": False,
            "permutation": {"p_value": 0.4},
        },
        screen={"n_significant_holm": 0},
    )
    weak["gate"] = evaluate_arm_gate(weak)
    verdict = evaluate_gate({"after": weak, "before": weak})
    assert "must not be attempted" in verdict["permits"]
    assert "not identifiable" in verdict["permits"]


def test_univariate_screen_reports_effect_sizes_for_every_feature():
    """Effect sizes are primary here, so the screen must not filter to
    survivors only."""
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(40, 3))
    labels = np.array([1] * 8 + [0] * 32)
    screen = univariate_screen(
        matrix,
        labels,
        ("f0", "f1", "f2"),
        {"query": ("f0", "f1", "f2")},
        n_resamples=200,
        seed=1,
    )
    assert screen["family_size"] == 3
    assert len(screen["features"]) == 3
    assert all("oriented_auc" in row for row in screen["features"])
    assert screen["correction"] == "holm_bonferroni"

def test_bootstrap_ci_brackets_the_point_estimate():
    labels = [1] * 8 + [0] * 40
    scores = list(np.linspace(0.0, 1.0, 48))
    point = auc_score(labels, scores)
    ci = bootstrap_auc_ci(labels, scores, n_resamples=500, seed=4)
    assert ci["lower"] <= point <= ci["upper"]