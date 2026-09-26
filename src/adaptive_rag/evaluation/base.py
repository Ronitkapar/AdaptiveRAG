"""
evaluation.base
---------------
Evaluator protocol and metric helpers.
Evaluation is an independent observer over persisted traces.
"""

from typing import Any, Protocol, Sequence

from adaptive_rag.schemas import (
    ErrorInfo,
    EvaluationConfig,
    EvaluationExample,
    EvaluationReport,
    ExperimentTrace,
    MetricValue,
)


class Evaluator(Protocol):
    """Protocol for reusable evaluator components."""

    name: str
    version: str

    def evaluate(
        self,
        traces: Sequence[ExperimentTrace],
        examples: Sequence[EvaluationExample] | None = None,
        config: EvaluationConfig | None = None,
    ) -> EvaluationReport:
        """Compute metrics over raw experiment traces."""
        ...


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None for an empty sequence."""
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile of a numeric sequence."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * len(ordered))) - 1))
    return ordered[idx]


def metric(
    name: str,
    value: float | None,
    *,
    n: int,
    version: str,
    k: int | None = None,
    notes: str = "",
) -> MetricValue:
    """Build a MetricValue, rounding floats consistently for stable artifacts."""
    rounded = round(value, 4) if isinstance(value, float) else value
    return MetricValue(name=name, value=rounded, k=k, n=n, version=version, notes=notes)


def failed_traces(traces: Sequence[ExperimentTrace]) -> list[ExperimentTrace]:
    """Traces whose retrieval or generation stage failed."""
    return [t for t in traces if t.error is not None]


def count_errors(traces: Sequence[ExperimentTrace]) -> list[ErrorInfo]:
    """All recorded error records in a trace set."""
    return [t.error for t in traces if t.error is not None]


def build_error_breakdown(traces: Sequence[ExperimentTrace]) -> dict[str, Any]:
    """Per-stage/per-type error counts so failures are visible in metrics."""
    breakdown: dict[str, dict[str, int]] = {}
    for trace in traces:
        if trace.error is None:
            continue
        stage = trace.error.stage
        breakdown.setdefault(stage, {})
        breakdown[stage][trace.error.error_type] = (
            breakdown[stage].get(trace.error.error_type, 0) + 1
        )
    return {
        "total_traces": len(traces),
        "failed_traces": len(failed_traces(traces)),
        "by_stage_and_type": breakdown,
    }
