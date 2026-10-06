"""
routing.rule_based
------------------
Phase 6 rule-based router: a weighted, transparent strategy scorer.

For every available strategy the router computes

```text
score = Σ (weight[group] × signal[group])  −  cost_weight × normalised_cost
```

and selects the highest score. Nothing is learned and nothing is opaque: each
strategy's score decomposes into the exact group contributions that produced it,
which are recorded on the `RoutingDecision` so a decision can be explained
without re-running the router.

`confidence` is the normalized margin between the winner and the runner-up. It is
a *routing-strength* signal, not a calibrated probability, and it deliberately
does not participate in the sufficiency decision -- sufficiency is a separate
mechanism that looks at retrieved evidence instead (see routing/sufficiency.py).

Determinism: signals come only from the analyzer, weights only from configuration,
and ties break on `STRATEGY_ORDER`, never on dict or input ordering.
"""

from adaptive_rag.errors import ConfigurationError
from adaptive_rag.schemas import (
    FEATURE_GROUPS,
    STRATEGY_ORDER,
    QueryFeatures,
    RoutingConfig,
    RoutingDecision,
    StrategyEvidence,
    StrategyName,
)

ROUTER_VERSION = "rule_based_v1"

# Per-strategy support for the analyzer's 10-way question classification. This is
# the one place a categorical feature becomes per-strategy support, so it is
# spelled out rather than buried in arithmetic.
QUESTION_TYPE_SUPPORT: dict[str, dict[str, float]] = {
    # A bare `define` is an exact terminology lookup: BM25's home turf.
    "bm25": {
        "what": 0.6, "how": 0.2, "why": 0.0, "which": 0.7, "who": 0.8,
        "when": 0.7, "where": 0.7, "compare": 0.0, "define": 1.0, "other": 0.3,
    },
    # Explanatory questions are what a dense encoder exists for.
    "dense": {
        "what": 0.6, "how": 0.8, "why": 1.0, "which": 0.3, "who": 0.2,
        "when": 0.2, "where": 0.2, "compare": 0.4, "define": 0.2, "other": 0.6,
    },
    # Fusion earns its cost when a query mixes lexical and conceptual needs or
    # spans several concepts.
    "hybrid": {
        "what": 0.6, "how": 0.7, "why": 0.7, "which": 0.5, "who": 0.4,
        "when": 0.4, "where": 0.4, "compare": 1.0, "define": 0.5, "other": 0.7,
    },
    # Reranking is a precision stage, supported by hard multi-concept queries
    # rather than by conceptual phrasing.
    "hybrid_rerank": {
        "what": 0.4, "how": 0.5, "why": 0.4, "which": 0.3, "who": 0.3,
        "when": 0.3, "where": 0.3, "compare": 0.7, "define": 0.3, "other": 0.4,
    },
}


def _clip(value: float) -> float:
    """Clamp to [0, 1]."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


class RuleBasedRouter:
    """Selects a retrieval strategy by weighted evidence over query features."""

    version = ROUTER_VERSION

    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or RoutingConfig()

    def route(self, features: QueryFeatures, *, top_k: int) -> RoutingDecision:
        """Return the structured routing decision for analyzed query features."""
        available = list(self.config.available_strategies)
        if not available:
            raise ConfigurationError("No retrieval strategy is available to route to")

        signals = self._signals(features)
        costs = self._normalised_costs(available)
        enabled = [g for g in FEATURE_GROUPS if g in self.config.enabled_feature_groups]
        disabled = [g for g in FEATURE_GROUPS if g not in enabled]

        evidence = [
            self._score_strategy(strategy, features, signals, costs, enabled)
            for strategy in available
        ]

        # Deterministic ordering: score descending, then STRATEGY_ORDER, so a tie
        # always resolves identically regardless of insertion order.
        ranked = sorted(
            evidence, key=lambda e: (-e.score, STRATEGY_ORDER.index(e.strategy))
        )
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        return RoutingDecision(
            strategy=winner.strategy,
            confidence=self._confidence(winner.score, runner_up),
            evidence=evidence,
            candidate_k=self.config.candidate_k,
            final_top_k=top_k,
            reranking_required=winner.strategy == "hybrid_rerank",
            router_kind="rule_based",
            router_version=self.version,
            analyzer_version=features.analyzer_version,
            feature_groups_used=list(enabled),
            feature_groups_disabled=list(disabled),
            cost_weight=self.config.cost_weight,
            metadata={
                "signals": {k: round(v, 6) for k, v in sorted(signals.items())},
                "normalised_cost": {k: round(v, 6) for k, v in sorted(costs.items())},
                "score_ranking": [
                    {"strategy": e.strategy, "score": round(e.score, 6)} for e in ranked
                ],
                "question_type": features.question_type,
            },
        )

    # --- scoring internals --------------------------------------------------

    def _signals(self, features: QueryFeatures) -> dict[str, float]:
        """Normalize query features into the router's signal groups, each [0, 1].

        Ratios are scaled toward the unit interval so one weight table is
        meaningful across groups: an entity indicator on a short query is far more
        informative than the same ratio on a long one, and raw ratios would not
        express that. `question_type` is categorical and has no single numeric
        value, so it is absent here and resolved per strategy in
        `_score_strategy`; reporting a stand-in number would be misleading.
        """
        return {
            "lexical": _clip(features.lexical_density),
            "semantic": _clip(features.semantic_ratio * 4.0),
            "entity": _clip(features.entity_ratio * 2.0),
            "complexity": _clip(features.complexity_score),
            "multi_concept": _clip((features.concept_count - 1) / 2.0),
        }

    def _normalised_costs(self, available: list[str]) -> dict[str, float]:
        """Scale measured strategy latencies to [0, 1] against the priciest option.

        Normalizing against the *available* set rather than a global constant keeps
        the cost term meaningful when a run only offers cheap strategies: with
        BM25 alone, cost cannot discriminate between them and correctly does not.
        """
        costs = self.config.strategy_cost_ms
        observed = [float(costs.get(s, 0.0)) for s in available]
        highest = max(observed) if observed else 0.0
        if highest <= 0.0:
            return {s: 0.0 for s in available}
        return {s: float(costs.get(s, 0.0)) / highest for s in available}

    def _score_strategy(
        self,
        strategy: StrategyName,
        features: QueryFeatures,
        signals: dict[str, float],
        costs: dict[str, float],
        enabled: list[str],
    ) -> StrategyEvidence:
        """Score one strategy as the sum of its enabled weighted contributions."""
        weights = self.config.rule_weights.get(strategy, {})
        contributions: dict[str, float] = {}
        total = 0.0
        for group in enabled:
            weight = float(weights.get(group, 0.0))
            if weight == 0.0:
                continue
            if group == "question_type":
                value = QUESTION_TYPE_SUPPORT.get(strategy, {}).get(
                    features.question_type, 0.0
                )
            else:
                value = signals.get(group, 0.0)
            contribution = weight * value
            contributions[group] = round(contribution, 6)
            total += contribution

        cost_penalty = self.config.cost_weight * costs.get(strategy, 0.0)
        return StrategyEvidence(
            strategy=strategy,
            score=round(total - cost_penalty, 6),
            cost_penalty=round(cost_penalty, 6),
            contributions=contributions,
        )

    @staticmethod
    def _confidence(best: float, runner_up: StrategyEvidence | None) -> float:
        """Normalized margin over the runner-up, in [0, 1].

        Not a calibrated probability, and never the sole trigger for escalation:
        sufficiency is judged from retrieved evidence by a separate mechanism. A
        single available strategy yields 0.0, because there was no evidence to
        discriminate on.
        """
        if runner_up is None:
            return 0.0
        margin = best - runner_up.score
        span = abs(best) + abs(runner_up.score)
        if span <= 0.0:
            return 0.0
        return round(_clip(margin / span), 6)