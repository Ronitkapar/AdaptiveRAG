"""
evaluation.efficiency
---------------------
Latency, token, and estimated-cost measurements derived purely from raw traces.
"""

from typing import Any, Sequence

from adaptive_rag.config.pricing import PRICING_VERSION, calculate_cost_usd
from adaptive_rag.evaluation.base import Evaluator, mean, metric, percentile
from adaptive_rag.schemas import (
    EvaluationConfig,
    EvaluationExample,
    EvaluationReport,
    ExperimentTrace,
    MetricValue,
)

EVALUATOR_VERSION = "efficiency_eval_v1"


class EfficiencyEvaluator(Evaluator):
    """Aggregates latency, token usage, and estimated cost from traces."""

    name = "efficiency"
    version = EVALUATOR_VERSION

    def __init__(self, generation_model: str = "openai/gpt-oss-120b", embedding_model: str = "text-embedding-3-large"):
        self.generation_model = generation_model
        self.embedding_model = embedding_model

    def evaluate(
        self,
        traces: Sequence[ExperimentTrace],
        examples: Sequence[EvaluationExample] | None = None,
        config: EvaluationConfig | None = None,
    ) -> EvaluationReport:
        retrieval_latencies = [
            t.retrieval_latency_ms
            for t in traces
            if t.retrieval_latency_ms is not None
        ]
        generation_latencies = [
            t.generation_latency_ms
            for t in traces
            if t.generation_latency_ms is not None
        ]
        total_latencies = [t.total_latency_ms for t in traces if t.total_latency_ms is not None]

        input_tokens = sum(t.usage.input_tokens for t in traces if t.usage is not None)
        output_tokens = sum(t.usage.output_tokens for t in traces if t.usage is not None)
        total_tokens = sum(t.usage.total_tokens for t in traces if t.usage is not None)
        costs = [t.estimated_cost_usd for t in traces if t.estimated_cost_usd is not None]

        metrics: list[MetricValue] = [
            metric(
                "retrieval_latency_ms_mean",
                mean(retrieval_latencies),
                n=len(retrieval_latencies),
                version=self.version,
            ),
            metric(
                "retrieval_latency_ms_p50",
                percentile(retrieval_latencies, 50),
                n=len(retrieval_latencies),
                version=self.version,
            ),
            metric(
                "retrieval_latency_ms_p95",
                percentile(retrieval_latencies, 95),
                n=len(retrieval_latencies),
                version=self.version,
            ),
            metric(
                "generation_latency_ms_mean",
                mean(generation_latencies),
                n=len(generation_latencies),
                version=self.version,
            ),
            metric(
                "generation_latency_ms_p50",
                percentile(generation_latencies, 50),
                n=len(generation_latencies),
                version=self.version,
            ),
            metric(
                "generation_latency_ms_p95",
                percentile(generation_latencies, 95),
                n=len(generation_latencies),
                version=self.version,
            ),
            metric(
                "total_latency_ms_mean",
                mean(total_latencies),
                n=len(total_latencies),
                version=self.version,
            ),
            metric(
                "input_tokens_total",
                float(input_tokens),
                n=len(traces),
                version=self.version,
            ),
            metric(
                "output_tokens_total",
                float(output_tokens),
                n=len(traces),
                version=self.version,
            ),
            metric(
                "total_tokens_total",
                float(total_tokens),
                n=len(traces),
                version=self.version,
            ),
            metric(
                "estimated_cost_usd_total",
                round(sum(costs), 6) if costs else None,
                n=len(costs),
                version=self.version,
                notes=f"generation model {self.generation_model}; pricing {PRICING_VERSION}",
            ),
            metric(
                "estimated_cost_usd_per_query",
                round(sum(costs) / len(costs), 6) if costs else None,
                n=len(costs),
                version=self.version,
            ),
            metric(
                "estimated_cost_usd_per_1000_queries",
                round(1000 * sum(costs) / len(costs), 4) if costs else None,
                n=len(costs),
                version=self.version,
            ),
        ]

        aggregates: dict[str, Any] = {
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "pricing_version": PRICING_VERSION,
            "pricing_table": {
                self.generation_model: calculate_cost_usd(
                    self.generation_model, input_tokens=1_000_000
                )
            },
        }

        return EvaluationReport(
            evaluator=self.name,
            evaluator_version=self.version,
            metrics=metrics,
            aggregates=aggregates,
        )
