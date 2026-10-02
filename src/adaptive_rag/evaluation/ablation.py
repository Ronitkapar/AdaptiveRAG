"""
evaluation.ablation
--------------------
Phase 7 ablation variants, expressed as configuration and nothing else.

Every ablation in the Phase 7 brief -- sufficiency, escalation, router feature
groups, thresholds, cost weight -- maps onto a field that Phase 6 already
exposes on `RoutingConfig`. That is the point of this module: it makes the
"no code path is added, only a knob is turned" claim mechanically checkable.
`test_ablation.py` asserts that no variant ever touches `RetrievalConfig`, so a
future contributor cannot smuggle a behavioural change into an ablation that is
advertised as configuration-only.

Variants are *declarations*. Nothing here runs retrieval, opens an index, or
knows what a retriever is. A variant is a `RoutingConfig` you can hand to the
existing `build_experiment_config`.
"""

from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.schemas.config import FEATURE_GROUPS, RoutingConfig

ABLATION_VERSION = "phase7_ablation_v1"


class Variant(BaseModel):
    """One named configuration point in an ablation grid.

    `experiment_id` follows the Phase 7 brief's scheme (E2_escalation_ablation,
    E3_feature_ablation, ...) so a variant name is traceable back to the study
    section that motivated running it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    experiment_id: str
    routing: RoutingConfig
    description: str
    # True for the variant that represents the unmodified system, which every
    # other variant in the same grid is compared against.
    is_baseline: bool = False


def _with(base: RoutingConfig, **overrides: object) -> RoutingConfig:
    """Copy `base` with routing fields replaced, re-running validation.

    Using model_copy(update=...) would skip the validators, so an invalid
    combination (a ladder naming an unavailable strategy, say) would reach the
    router instead of failing here where the mistake is still legible.
    """
    return RoutingConfig(**{**base.model_dump(), **overrides})


def escalation_variants(base: RoutingConfig | None = None) -> list[Variant]:
    """The A/B/C sufficiency-and-escalation ablation (study section 7.4).

    A: no sufficiency check and no escalation -- routing happens once, so the
       system is a pure query-classifier. This is the floor: it shows what
       adaptive routing contributes before any recovery mechanism.
    B: sufficiency check but no escalation -- the check runs and its verdict is
       recorded but nothing is done about it. Isolates the value of acting on
       sufficiency from the value of measuring it.
    C: the shipped configuration -- measure and act, bounded to
       `max_escalation_steps`.
    """
    base = base or RoutingConfig()
    experiment_id = "E2_escalation_ablation"
    return [
        Variant(
            name="A_no_sufficiency_no_escalation",
            experiment_id=experiment_id,
            routing=_with(
                base, sufficiency_enabled=False, escalation_enabled=False
            ),
            description=(
                "Route once and stop: neither the sufficiency check nor bounded "
                "escalation runs. Isolates query-aware routing alone."
            ),
            is_baseline=True,
        ),
        Variant(
            name="B_sufficiency_no_escalation",
            experiment_id=experiment_id,
            routing=_with(base, sufficiency_enabled=True, escalation_enabled=False),
            description=(
                "Sufficiency is computed and recorded but never acted on. "
                "Isolates the cost of measuring sufficiency without recovering from it."
            ),
        ),
        Variant(
            name="C_sufficiency_bounded_escalation",
            experiment_id=experiment_id,
            routing=base.model_copy(),
            description=(
                "The shipped configuration: measure sufficiency and escalate "
                "conditionally, bounded by max_escalation_steps."
            ),
        ),
    ]


def feature_group_variants(
    base: RoutingConfig | None = None,
    *,
    groups: Iterable[str] = FEATURE_GROUPS,
) -> list[Variant]:
    """Leave-one-out ablation over the router's real feature groups (7.5).

    Only groups that actually exist are ablated. `FEATURE_GROUPS` is the
    authoritative list, so a group added to Phase 6 later is swept
    automatically rather than silently omitted -- and a group named here that
    does not exist is a hard error, not a no-op, because a variant that removes
    nothing would masquerade as an ablation result.
    """
    base = base or RoutingConfig()
    experiment_id = "E3_feature_ablation"

    variants = [
        Variant(
            name="full",
            experiment_id=experiment_id,
            routing=base.model_copy(),
            description="All router feature groups enabled (the shipped router).",
            is_baseline=True,
        )
    ]
    for group in FEATURE_GROUPS:
        remaining = [g for g in base.enabled_feature_groups if g != group]
        if group not in base.enabled_feature_groups:
            # Already disabled in the base config: dropping it again is a no-op
            # and would report a phantom effect.
            continue
        if not remaining:
            # RoutingConfig rejects an empty group list; a router with no
            # evidence groups is not an ablation of the router, it is a
            # different (and undefined) system.
            continue
        variants.append(
            Variant(
                name=f"without_{group}",
                experiment_id=experiment_id,
                routing=_with(base, enabled_feature_groups=remaining),
                description=f"Leave-one-out: the '{group}' signal group is disabled.",
            )
        )
    return variants


def threshold_sweep(
    base: RoutingConfig | None = None,
    *,
    values: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    field: str = "sufficiency_threshold",
    experiment_id: str = "E4_threshold_calibration",
) -> list[Variant]:
    """One-parameter sweep over a sufficiency threshold (7.6).

    Deliberately restricted to the *sufficiency* thresholds the repository
    already defines. The brief says not to invent scoring formulas where the
    repository states the policy, and `RoutingConfig`'s four threshold fields are
    that policy.
    """
    base = base or RoutingConfig()
    allowed = {
        "sufficiency_threshold",
        "coverage_threshold",
        "top1_coverage_threshold",
        "min_results",
        "max_escalation_steps",
    }
    if field not in allowed:
        raise ValueError(f"{field!r} is not a sweepable threshold; allowed: {sorted(allowed)}")

    variants: list[Variant] = []
    for value in values:
        if field == "min_results" and not isinstance(value, int):
            raise ValueError("min_results must be swept with integer values")
        variants.append(
            Variant(
                name=f"{field}={value}",
                experiment_id=experiment_id,
                routing=_with(base, **{field: value}),
                description=f"Sweep {field} = {value}.",
                is_baseline=(value == getattr(base, field)),
            )
        )
    return variants


def cost_weight_sweep(
    base: RoutingConfig | None = None,
    *,
    values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[Variant]:
    """Sweep the quality-vs-cost knob (7.7).

    0.0 is pure evidence: the router spends whatever the signals suggest. Higher
    values subtract `cost_weight * cost(strategy) / max_cost` from each
    strategy's score, biasing toward cheaper retrievals. No new scoring formula
    is introduced -- `RuleBasedRouter` already implements exactly this, and this
    function only names the settings worth measuring.
    """
    base = base or RoutingConfig()
    experiment_id = "E5_cost_weight_ablation"

    regimes = {
        0.0: "Quality: routing ignores measured cost and follows evidence alone.",
        0.25: "Balanced: the shipped default.",
        0.5: "Efficiency-leaning: measured cost is weighted equally with a strong signal.",
        1.0: "Efficiency-first: cost penalty equals the weight of a full [0,1] signal.",
    }

    variants: list[Variant] = []
    for value in values:
        variants.append(
            Variant(
                name=f"cost_weight={value}",
                experiment_id=experiment_id,
                routing=_with(base, cost_weight=value),
                description=regimes.get(value, f"Sweep cost_weight = {value}."),
                is_baseline=(value == base.cost_weight),
            )
        )
    return variants


def escalation_step_sweep(
    base: RoutingConfig | None = None,
    *,
    values: Sequence[int] = (0, 1, 2, 3),
) -> list[Variant]:
    """Sweep the escalation bound (7.6).

    Values are clamped to the ladder length: a bound past the last rung would
    promise escalation steps the ladder cannot supply, which `RoutingConfig`
    rejects outright.
    """
    base = base or RoutingConfig()
    max_steps = len(base.escalation_ladder) - 1
    experiment_id = "E4_threshold_calibration"

    variants: list[Variant] = []
    for value in values:
        clamped = max(0, min(int(value), max_steps))
        variants.append(
            Variant(
                name=f"max_escalation_steps={clamped}",
                experiment_id=experiment_id,
                routing=_with(base, max_escalation_steps=clamped),
                description=(
                    f"Allow at most {clamped} escalation step(s)"
                    + (f" (requested {value} clamped to the ladder length)."
                       if clamped != value
                       else ".")
                ),
                is_baseline=(clamped == base.max_escalation_steps),
            )
        )
    return variants


def ablation_summary(variants: Sequence[Variant]) -> dict[str, object]:
    """Group variants by study for reporting."""
    grouped: dict[str, list[str]] = {}
    for variant in variants:
        grouped.setdefault(variant.experiment_id, []).append(variant.name)
    return {
        "ablation_version": ABLATION_VERSION,
        "n_variants": len(variants),
        "by_experiment": grouped,
        "variants": [
            {
                "name": v.name,
                "experiment_id": v.experiment_id,
                "description": v.description,
                "is_baseline": v.is_baseline,
                "routing": v.routing.model_dump(mode="json"),
            }
            for v in variants
        ],
    }