"""
evaluation.retrieval
--------------------
Retrieval-quality metrics computed from raw experiment traces.
Implements Recall@k, Precision@k, Hit@k, MRR, and nDCG@k over document and
(rule-derived) chunk relevance.
"""

import math
from typing import Sequence

from adaptive_rag.evaluation.base import Evaluator, build_error_breakdown, mean, metric
from adaptive_rag.evaluation.dataset import relevant_chunk_ids
from adaptive_rag.schemas import (
    EvaluationConfig,
    EvaluationExample,
    EvaluationReport,
    ExperimentTrace,
    MetricValue,
)

EVALUATOR_VERSION = "retrieval_eval_v1"


class RetrievalEvaluator(Evaluator):
    """Computes retrieval metrics over traces that contain a retrieval stage."""

    name = "retrieval"
    version = EVALUATOR_VERSION

    def evaluate(
        self,
        traces: Sequence[ExperimentTrace],
        examples: Sequence[EvaluationExample] | None = None,
        config: EvaluationConfig | None = None,
    ) -> EvaluationReport:
        eval_config = config or EvaluationConfig()
        k_grid = eval_config.k_grid
        example_by_id = {ex.example_id: ex for ex in (examples or [])}

        paired: list[tuple[ExperimentTrace, EvaluationExample]] = [
            (t, example_by_id[t.example_id])
            for t in traces
            if t.retrieval is not None and t.example_id in example_by_id
        ]

        metrics: list[MetricValue] = []
        per_example: dict[str, dict[str, float]] = {}

        for k in k_grid:
            metrics.append(
                metric(
                    "recall_at_k",
                    mean([self._recall_at_k(t, ex, k) for t, ex in paired]),
                    k=k,
                    n=len(paired),
                    version=self.version,
                    notes="document-level relevance",
                )
            )
            metrics.append(
                metric(
                    "precision_at_k",
                    mean([self._precision_at_k(t, ex, k) for t, ex in paired]),
                    k=k,
                    n=len(paired),
                    version=self.version,
                    notes="chunk-level relevance derived from document labels",
                )
            )
            metrics.append(
                metric(
                    "hit_at_k",
                    mean([1.0 if self._recall_at_k(t, ex, k) > 0 else 0.0 for t, ex in paired]),
                    k=k,
                    n=len(paired),
                    version=self.version,
                    notes="any relevant document retrieved",
                )
            )
            metrics.append(
                metric(
                    "ndcg_at_k",
                    mean([self._ndcg_at_k(t, ex, k) for t, ex in paired]),
                    k=k,
                    n=len(paired),
                    version=self.version,
                )
            )

        metrics.append(
            metric(
                "mrr",
                mean([self._reciprocal_rank(t, ex) for t, ex in paired]),
                n=len(paired),
                version=self.version,
            )
        )
        metrics.append(
            metric(
                "retrieved_results_mean",
                mean([len(t.retrieval.results) for t, _ in paired if t.retrieval]),
                n=len(paired),
                version=self.version,
            )
        )

        for trace, example in paired:
            per_example[trace.example_id] = {
                "recall_at_5": self._recall_at_k(trace, example, 5),
                "mrr": self._reciprocal_rank(trace, example),
            }

        return EvaluationReport(
            evaluator=self.name,
            evaluator_version=self.version,
            metrics=metrics,
            aggregates={
                "n_examples": len(paired),
                "k_grid": k_grid,
                "errors": build_error_breakdown(traces),
                "per_example": per_example,
            },
        )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _ranked_doc_ids(trace: ExperimentTrace, k: int) -> list[str]:
        if trace.retrieval is None:
            return []
        return [r.metadata.document_id for r in trace.retrieval.results[:k]]

    def _recall_at_k(self, trace: ExperimentTrace, example: EvaluationExample, k: int) -> float:
        relevant = set(example.relevant_documents)
        if not relevant:
            return 0.0
        retrieved = set(self._ranked_doc_ids(trace, k))
        return len(relevant & retrieved) / len(relevant)

    def _precision_at_k(self, trace: ExperimentTrace, example: EvaluationExample, k: int) -> float:
        if trace.retrieval is None or k <= 0:
            return 0.0
        top = trace.retrieval.results[:k]
        if not top:
            return 0.0
        relevant_ids = relevant_chunk_ids(example, trace)
        hits = sum(1 for r in top if r.chunk_id in relevant_ids)
        return hits / len(top)

    def _reciprocal_rank(self, trace: ExperimentTrace, example: EvaluationExample) -> float:
        relevant = set(example.relevant_documents)
        if not relevant or trace.retrieval is None:
            return 0.0
        for result in trace.retrieval.results:
            if result.metadata.document_id in relevant:
                return 1.0 / result.rank
        return 0.0

    def _ndcg_at_k(self, trace: ExperimentTrace, example: EvaluationExample, k: int) -> float:
        if trace.retrieval is None:
            return 0.0
        relevant_ids = relevant_chunk_ids(example, trace)
        gains = [1.0 if r.chunk_id in relevant_ids else 0.0 for r in trace.retrieval.results[:k]]
        if not any(gains):
            return 0.0
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal_hits = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0
