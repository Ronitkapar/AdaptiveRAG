"""
evaluation.routing
------------------
Phase 6 routing metrics: strategy distribution, escalation behaviour, and the
routing cost, computed from persisted traces like every other evaluator.

Routing-specific metrics are emitted **only** when at least one trace carries a
`routing` payload. A fixed-strategy run therefore produces a byte-identical
metrics file, which is what keeps Phase 2-5 baselines comparable across the
introduction of Phase 6.

Deliberately absent: the adaptive-vs-fixed comparison harness and the ablation
framework. Those belong to Phase 7, which owns the broader evaluation program;
this module reports only what a single adaptive run needs to describe itself.
"""

from typing import Sequence

from adaptive_rag.evaluation.base import mean, metric
from adaptive_rag.schemas import (
    EvaluationConfig,
    EvaluationExample,
    EvaluationReport,
    ExperimentTrace,
    MetricValue,
)

EVALUATOR_VERSION = "routing_eval_v1"

STRATEGY_ORDER = ("bm25", "dense", "hybrid", "hybrid_rerank")


def _distribution(values: Sequence[str], order: Sequence[str]) -> dict[str, dict[str, float]]:
    """Counts and shares per strategy, always reporting every known strategy.

    Zero-count strategies stay in the table so a distribution cannot mislead by
    omission: "hybrid_rerank: 0%" is itself an observation that matters, since an
    over-eager reranker is the failure mode this metric exists to catch.
    """
    n = len(values)
    dist: dict[str, dict[str, float]] = {}
    for strategy in order:
        count = sum(1 for v in values if v == strategy)
        dist[strategy] = {"count": count, "share": round(count / n, 4) if n else 0.0}
    return dist
class RoutingEvaluator:
    """Aggregates Phase 6 routing behaviour from traces."""

    name = "routing"
    version = EVALUATOR_VERSION

    def evaluate(
        self,
        traces: Sequence[ExperimentTrace],
        examples: Sequence[EvaluationExample] | None = None,
        config: EvaluationConfig | None = None,
    ) -> EvaluationReport:
        routed = [t for t in traces if t.routing is not None]

        # A run with no routing at all (every fixed-strategy baseline) contributes
        # no metrics, so its artifacts stay byte-identical to before Phase 6.
        if not routed:
            return EvaluationReport(
                evaluator=self.name,
                evaluator_version=self.version,
                metrics=[],
                aggregates={"routed_traces": 0},
            )

        n = len(routed)
        initial = [t.routing.initial_strategy for t in routed if t.routing]
        final = [t.routing.final_strategy for t in routed if t.routing]
        escalated = [
            bool(t.routing.escalation and t.routing.escalation.escalated) for t in routed
        ]
        confidences = [t.routing.decision.confidence for t in routed if t.routing]
        stages = [t.routing.stage_count for t in routed if t.routing]
        routing_latencies = [t.routing.routing_latency_ms for t in routed if t.routing]
        insufficient = [
            t.routing.sufficiency is not None and not t.routing.sufficiency.sufficient
            for t in routed
        ]
        rerank_count = sum(
            1 for t in routed if t.routing and t.routing.final_strategy.endswith("_rerank")
        )
        # Total retrieval latency split by escalation outcome, so the price of
        # escalating stays visible instead of being averaged away.
        escalated_latency = [
            t.retrieval_latency_ms
            for t, esc in zip(routed, escalated)
            if esc and t.retrieval_latency_ms is not None
        ]
        settled_latency = [
            t.retrieval_latency_ms
            for t, esc in zip(routed, escalated)
            if not esc and t.retrieval_latency_ms is not None
        ]

        metrics: list[MetricValue] = [
            metric("escalation_rate", mean([float(e) for e in escalated]), n=n,
                   version=self.version),
            metric("rerank_rate", rerank_count / n, n=n, version=self.version,
                   notes="queries whose final stage included second-stage scoring"),
            metric("insufficient_initial_rate", mean([float(i) for i in insufficient]),
                   n=n, version=self.version),
            metric("routing_confidence_mean", mean(confidences), n=len(confidences),
                   version=self.version,
                   notes="normalized score margin; not a calibrated probability"),
            metric("retrieval_stages_mean", mean([float(s) for s in stages]), n=len(stages),
                   version=self.version),
            metric("routing_latency_ms_mean", mean(routing_latencies), n=len(routing_latencies),
                   version=self.version, notes="analysis + routing overhead per query"),
            metric("escalated_retrieval_latency_ms_mean", mean(escalated_latency),
                   n=len(escalated_latency), version=self.version),
            metric("non_escalated_retrieval_latency_ms_mean", mean(settled_latency),
                   n=len(settled_latency), version=self.version),
        ]

        per_example = {
            t.example_id: {
                "initial_strategy": t.routing.initial_strategy,
                "final_strategy": t.routing.final_strategy,
                "escalated": bool(t.routing.escalation and t.routing.escalation.escalated),
                "confidence": t.routing.decision.confidence,
                "stage_count": t.routing.stage_count,
                "sufficient": (
                    t.routing.sufficiency.sufficient if t.routing.sufficiency else None
                ),
                "retrieval_latency_ms": t.retrieval_latency_ms,
            }
            for t in routed
            if t.routing
        }

        return EvaluationReport(
            evaluator=self.name,
            evaluator_version=self.version,
            metrics=metrics,
            aggregates={
                "routed_traces": n,
                "initial_strategy_distribution": _distribution(initial, STRATEGY_ORDER),
                "final_strategy_distribution": _distribution(final, STRATEGY_ORDER),
                "per_example": per_example,
            },
        )