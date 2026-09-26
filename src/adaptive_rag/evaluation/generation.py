"""
evaluation.generation
---------------------
Generation-quality metrics: deterministic lexical overlap plus optional LLM judge.
Failures and empty answers are reported, never silently scored as zero-quality successes.
"""

import re
from collections import Counter
from typing import Any, Sequence

from adaptive_rag.config.pricing import PRICING_VERSION, calculate_cost_usd
from adaptive_rag.evaluation.base import Evaluator, count_errors, mean, metric
from adaptive_rag.evaluation.judge import JUDGE_METRICS, GroqLLMJudge
from adaptive_rag.schemas import (
    EvaluationConfig,
    EvaluationExample,
    EvaluationReport,
    ExperimentTrace,
    MetricValue,
)

EVALUATOR_VERSION = "generation_eval_v1"
TOKEN_REGEX = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for lexical metrics."""
    return TOKEN_REGEX.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between generated and reference answers."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure via longest common subsequence (stdlib only)."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0

    prev = [0] * (len(ref) + 1)
    for token in pred:
        cur = [0] * (len(ref) + 1)
        for j, ref_token in enumerate(ref, start=1):
            cur[j] = prev[j - 1] + 1 if token == ref_token else max(prev[j], cur[j - 1])
        prev = cur

    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


class GenerationEvaluator(Evaluator):
    """Computes lexical generation metrics and optional LLM-judge scores."""

    name = "generation"
    version = EVALUATOR_VERSION

    def __init__(self, judge: GroqLLMJudge | None = None):
        self._judge = judge

    def evaluate(
        self,
        traces: Sequence[ExperimentTrace],
        examples: Sequence[EvaluationExample] | None = None,
        config: EvaluationConfig | None = None,
    ) -> EvaluationReport:
        eval_config = config or EvaluationConfig()
        example_by_id = {ex.example_id: ex for ex in (examples or [])}

        answered = [
            (t, example_by_id[t.example_id])
            for t in traces
            if t.generation is not None and t.example_id in example_by_id
        ]

        metrics: list[MetricValue] = [
            metric(
                "answer_count",
                float(len(answered)),
                n=len(traces),
                version=self.version,
                notes="traces with a generated answer",
            ),
            metric("empty_answer_rate", self._empty_rate(traces), n=len(traces), version=self.version),
            metric(
                "generation_failure_rate",
                self._stage_failure_rate(traces, "generation"),
                n=len(traces),
                version=self.version,
            ),
            metric(
                "token_f1",
                mean([token_f1(t.generation.answer, ex.reference_answer) for t, ex in answered]),
                n=len(answered),
                version=self.version,
                notes="lexical overlap vs reference answer",
            ),
            metric(
                "rouge_l",
                mean([rouge_l(t.generation.answer, ex.reference_answer) for t, ex in answered]),
                n=len(answered),
                version=self.version,
            ),
            metric(
                "citation_coverage",
                self._citation_coverage(answered),
                n=len(answered),
                version=self.version,
                notes="answers citing at least one source chunk",
            ),
        ]

        aggregates: dict[str, Any] = {
            "n_examples": len(traces),
            "n_answered": len(answered),
            "judge": {"enabled": False, "pricing_version": PRICING_VERSION},
            "errors": [e.model_dump(mode="json") for e in count_errors(traces)],
        }

        if eval_config.enable_llm_judge:
            judge_metrics, judge_info = self._run_judge(answered, eval_config)
            metrics.extend(judge_metrics)
            aggregates["judge"] = judge_info

        return EvaluationReport(
            evaluator=self.name,
            evaluator_version=self.version,
            metrics=metrics,
            aggregates=aggregates,
        )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _empty_rate(traces: Sequence[ExperimentTrace]) -> float | None:
        answered = [t for t in traces if t.generation is not None]
        if not answered:
            return None
        empties = sum(1 for t in answered if not t.generation.answer.strip())
        return empties / len(answered)

    @staticmethod
    def _stage_failure_rate(traces: Sequence[ExperimentTrace], stage: str) -> float | None:
        if not traces:
            return None
        failures = sum(1 for t in traces if t.error is not None and t.error.stage == stage)
        return failures / len(traces)

    @staticmethod
    def _citation_coverage(
        answered: Sequence[tuple[ExperimentTrace, EvaluationExample]],
    ) -> float | None:
        if not answered:
            return None
        cited = sum(1 for t, _ in answered if t.generation and t.generation.source_chunk_ids)
        return cited / len(answered)

    def _run_judge(
        self,
        answered: Sequence[tuple[ExperimentTrace, EvaluationExample]],
        eval_config: EvaluationConfig,
    ) -> tuple[list[MetricValue], dict[str, Any]]:
        judge = self._judge or GroqLLMJudge(
            model=eval_config.judge_model,
            prompt_version=eval_config.judge_prompt_version,
            use_cache=eval_config.use_judge_cache,
        )

        collected: dict[str, list[float]] = {name: [] for name in JUDGE_METRICS}
        for trace, example in answered:
            assert trace.generation is not None
            context_text = ""
            if trace.retrieval is not None:
                context_text = "\n\n".join(r.text for r in trace.retrieval.results)
            scores = judge.judge(
                query=trace.query,
                generated_answer=trace.generation.answer,
                reference_answer=example.reference_answer,
                retrieved_context=context_text,
            )
            for name in JUDGE_METRICS:
                value = scores.get(name)
                if isinstance(value, (int, float)):
                    collected[name].append(float(value))

        metrics = [
            metric(
                f"judge_{name}",
                mean(values),
                n=len(values),
                version=eval_config.judge_prompt_version,
                notes=f"LLM judge score 1-5 ({eval_config.judge_model})",
            )
            for name, values in collected.items()
        ]

        judge_info = {
            "enabled": True,
            "model": eval_config.judge_model,
            "prompt_version": eval_config.judge_prompt_version,
            "calls": judge.usage.get("calls", 0),
            "cache_hits": judge.usage.get("cache_hits", 0),
            "input_tokens": judge.usage.get("input_tokens", 0),
            "output_tokens": judge.usage.get("output_tokens", 0),
            "estimated_cost_usd": calculate_cost_usd(
                eval_config.judge_model,
                input_tokens=judge.usage.get("input_tokens", 0),
                output_tokens=judge.usage.get("output_tokens", 0),
            ),
            "pricing_version": PRICING_VERSION,
        }
        return metrics, judge_info
