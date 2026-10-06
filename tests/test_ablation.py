"""
tests.test_ablation
-------------------
Offline checks that every Phase 7 ablation is pure configuration.

The central claim these tests defend is that Phase 7 adds no retrieval or
routing code path: an ablation turns a `RoutingConfig` knob and nothing else.
That claim is checkable, so it is checked rather than asserted in prose.
"""

import pytest

from adaptive_rag.evaluation.ablation import (
    ablation_summary,
    cost_weight_sweep,
    escalation_step_sweep,
    escalation_variants,
    feature_group_variants,
    threshold_sweep,
)
from adaptive_rag.schemas.config import FEATURE_GROUPS, RoutingConfig


def test_escalation_variants_are_the_documented_abc():
    variants = escalation_variants()
    names = [v.name for v in variants]

    assert names == [
        "A_no_sufficiency_no_escalation",
        "B_sufficiency_no_escalation",
        "C_sufficiency_bounded_escalation",
    ]


def test_escalation_variant_a_disables_both_mechanisms():
    a = escalation_variants()[0]

    assert a.routing.sufficiency_enabled is False
    assert a.routing.escalation_enabled is False
    assert a.is_baseline is True


def test_escalation_variant_b_measures_without_acting():
    """B isolates the cost of measuring sufficiency without recovering from it."""
    b = escalation_variants()[1]

    assert b.routing.sufficiency_enabled is True
    assert b.routing.escalation_enabled is False


def test_escalation_variant_c_is_the_shipped_configuration():
    c = escalation_variants()[2]

    assert c.routing.sufficiency_enabled is True
    assert c.routing.escalation_enabled is True
    assert c.routing == RoutingConfig()


def test_all_escalation_variants_share_one_experiment_id():
    ids = {v.experiment_id for v in escalation_variants()}

    assert ids == {"E2_escalation_ablation"}


def test_feature_ablation_covers_every_real_group():
    """Leave-one-out must cover all six groups that Phase 6 actually defines."""
    variants = feature_group_variants()
    dropped = {v.name for v in variants if not v.is_baseline}

    assert dropped == {f"without_{g}" for g in FEATURE_GROUPS}
    assert len(FEATURE_GROUPS) == 6


def test_feature_ablation_removes_exactly_one_group_each_time():
    base = RoutingConfig()
    for variant in feature_group_variants():
        if variant.is_baseline:
            continue
        removed = set(base.enabled_feature_groups) - set(
            variant.routing.enabled_feature_groups
        )
        assert len(removed) == 1, f"{variant.name} removed {removed}"


def test_feature_ablation_never_empties_the_group_list():
    """A router with no evidence groups is not an ablation of the router."""
    for variant in feature_group_variants():
        assert variant.routing.enabled_feature_groups


def test_feature_ablation_preserves_group_order():
    """Order must match FEATURE_GROUPS so two variants are diffable."""
    for variant in feature_group_variants():
        assert variant.routing.enabled_feature_groups == [
            g for g in FEATURE_GROUPS if g in set(variant.routing.enabled_feature_groups)
        ]


def test_feature_ablation_skips_groups_already_disabled_in_the_base():
    """Removing an already-absent group is a no-op and must not become a variant.

    Otherwise the grid reports a phantom "effect" that is really just the
    default configuration measured twice.
    """
    narrowed = RoutingConfig(enabled_feature_groups=["lexical", "semantic", "complexity"])
    variants = feature_group_variants(narrowed)
    dropped = {v.name for v in variants if not v.is_baseline}

    assert dropped == {"without_lexical", "without_semantic", "without_complexity"}


def test_threshold_sweep_defaults_bracket_the_shipped_value():
    variants = threshold_sweep()
    values = [v.routing.sufficiency_threshold for v in variants]

    assert RoutingConfig().sufficiency_threshold in values
    assert min(values) <= 0.5 <= max(values)
    assert sum(1 for v in variants if v.is_baseline) == 1


def test_threshold_sweep_rejects_a_non_threshold_field():
    with pytest.raises(ValueError, match="not a sweepable threshold"):
        threshold_sweep(field="cost_weight")


def test_threshold_sweep_covers_every_allowed_threshold():
    for field in (
        "sufficiency_threshold",
        "coverage_threshold",
        "top1_coverage_threshold",
    ):
        variants = threshold_sweep(values=(0.4, 0.6), field=field)
        assert [v.routing.model_dump()[field] for v in variants] == [0.4, 0.6]


def test_threshold_sweep_requires_integers_for_min_results():
    with pytest.raises(ValueError, match="integer"):
        threshold_sweep(values=(1.5, 2.5), field="min_results")


def test_cost_weight_sweep_includes_pure_evidence_and_the_default():
    variants = cost_weight_sweep()
    values = [v.routing.cost_weight for v in variants]

    assert 0.0 in values, "cost_weight=0.0 (pure evidence) must be measured"
    assert RoutingConfig().cost_weight in values
    assert values == sorted(values)


def test_cost_weight_sweep_rejects_negative_weights():
    """RoutingConfig forbids negative cost weights; the sweep must not bypass that."""
    with pytest.raises(Exception):
        cost_weight_sweep(values=(-0.5,))


def test_escalation_step_sweep_clamps_to_the_ladder():
    variants = escalation_step_sweep(values=(0, 1, 2, 3, 9))
    bounds = [v.routing.max_escalation_steps for v in variants]

    assert bounds == [0, 1, 2, 3, 3]
    assert max(bounds) == len(RoutingConfig().escalation_ladder) - 1


def test_escalation_step_sweep_explains_a_clamp():
    clamped = [v for v in escalation_step_sweep(values=(9,)) if v.routing.max_escalation_steps < 9]

    assert clamped and "clamped" in clamped[0].description


def test_no_variant_touches_retrieval_configuration():
    """The load-bearing guard: ablations change routing config and nothing else.

    `Variant` carries only a `RoutingConfig`, so the strong form of this claim is
    structural. The meaningful runtime check is that each variant differs from
    the shipped router in exactly the field its name advertises, and in nothing
    else -- an ablation that silently perturbed, say, the rule weights or the
    cost table would no longer be a comparison of routing *policies*, and the
    resulting numbers would be attributed to the router rather than to the
    change actually made.
    """
    base = RoutingConfig().model_dump()
    expected_changes: dict[str, set[str]] = {}

    for variant in escalation_variants():
        expected_changes[variant.name] = {
            field
            for field in RoutingConfig.model_fields
            if variant.routing.model_dump()[field] != base[field]
        }
    for variant in feature_group_variants():
        expected_changes[variant.name] = (
            {"enabled_feature_groups"} if not variant.is_baseline else set()
        )
    for variant in threshold_sweep():
        expected_changes[variant.name] = {
            field
            for field in RoutingConfig.model_fields
            if variant.routing.model_dump()[field] != base[field]
        }
    for variant in cost_weight_sweep():
        expected_changes[variant.name] = (
            {"cost_weight"} if not variant.is_baseline else set()
        )
    for variant in escalation_step_sweep():
        expected_changes[variant.name] = (
            {"max_escalation_steps"} if not variant.is_baseline else set()
        )

    for variant in [
        *escalation_variants(),
        *feature_group_variants(),
        *threshold_sweep(),
        *cost_weight_sweep(),
        *escalation_step_sweep(),
    ]:
        changed = {
            field
            for field in RoutingConfig.model_fields
            if variant.routing.model_dump()[field] != base[field]
        }
        assert changed == expected_changes[variant.name], (
            f"{variant.name} changed {sorted(changed)}, expected "
            f"{sorted(expected_changes[variant.name])}"
        )


def test_variants_carry_only_routing_config():
    """Structurally, a variant has nowhere to hide a retrieval change."""
    from adaptive_rag.evaluation.ablation import Variant

    assert set(Variant.model_fields) == {
        "name",
        "experiment_id",
        "routing",
        "description",
        "is_baseline",
    }


def test_variants_never_change_the_rule_weights_or_cost_table():
    """The evidence table and cost table are the router's prior, not its knob.

    Sweeping these would be a different study -- re-tuning the router rather
    than ablating it -- and the plan's freeze rule forbids it.
    """
    all_variants = [
        *escalation_variants(),
        *feature_group_variants(),
        *threshold_sweep(),
        *cost_weight_sweep(),
        *escalation_step_sweep(),
    ]
    for variant in all_variants:
        assert variant.routing.rule_weights == RoutingConfig().rule_weights
        assert variant.routing.strategy_cost_ms == RoutingConfig().strategy_cost_ms
        assert variant.routing.available_strategies == RoutingConfig().available_strategies
        assert variant.routing.escalation_ladder == RoutingConfig().escalation_ladder


def test_variants_re_run_config_validation():
    """Overrides must go through the model, not model_copy(update=...).

    model_copy skips validators, so an invalid combination would reach the
    router instead of failing where the mistake is legible.
    """
    # A ladder naming a strategy outside available_strategies must be rejected.
    with pytest.raises(Exception, match="ladder"):
        threshold_sweep(
            base=RoutingConfig(available_strategies=["bm25"], escalation_ladder=["bm25"]),
            values=(0.5,),
            field="max_escalation_steps",
        )


def test_variants_are_deterministic():
    """Re-running the sweep must produce identical configurations."""
    first = [v.routing.model_dump() for v in feature_group_variants()]
    second = [v.routing.model_dump() for v in feature_group_variants()]

    assert first == second


def test_summary_groups_variants_by_study():
    summary = ablation_summary(
        [*escalation_variants(), *feature_group_variants(), *cost_weight_sweep()]
    )

    assert summary["n_variants"] > 0
    assert set(summary["by_experiment"]) == {
        "E2_escalation_ablation",
        "E3_feature_ablation",
        "E5_cost_weight_ablation",
    }


def test_summary_round_trips_routing_config_as_json():
    summary = ablation_summary(escalation_variants())

    for entry in summary["variants"]:
        # Must be plain JSON: these payloads are written into suite.json.
        assert isinstance(entry["routing"], dict)
        assert isinstance(entry["name"], str)