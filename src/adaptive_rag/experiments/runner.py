"""
experiments.runner
------------------
Retrieval-strategy-agnostic experiment runner.
Executes the runtime system over an evaluation dataset, persists raw traces,
and hands traces to independent evaluators for aggregate metrics.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adaptive_rag.config.hashing import canonical_json
from adaptive_rag.config.paths import EXPERIMENTS_DIR
from adaptive_rag.config.pricing import PRICING_VERSION, calculate_cost_usd
from adaptive_rag.errors import AdaptiveRAGError
from adaptive_rag.generation.base import Generator
from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.schemas import (
    ErrorInfo,
    EvaluationExample,
    ExperimentConfig,
    ExperimentTrace,
    GenerationRequest,
    ReferenceInfo,
    RetrievalResponse,
)


def _git_commit() -> str | None:
    """Best-effort HEAD commit for run manifests."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ExperimentRunner:
    """Runs a retrieval/generation system over a benchmark and records traces."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator | None = None,
        context_builder: ContextBuilder | None = None,
        evaluators: Sequence[Any] | None = None,
        output_root: Path = EXPERIMENTS_DIR,
    ):
        self.retriever = retriever
        self.generator = generator
        self.context_builder = context_builder or ContextBuilder()
        self.evaluators = list(evaluators or [])
        self.output_root = output_root

    def run(
        self,
        config: ExperimentConfig,
        dataset: Sequence[EvaluationExample],
        *,
        resume: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute the experiment and write all artifacts to disk."""
        experiment_id = run_id or f"{_timestamp()}-{config.name}"
        run_dir = self.output_root / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)

        traces_path = run_dir / "traces.jsonl"
        traces: list[ExperimentTrace] = []
        completed_ids: set[str] = set()

        if resume and traces_path.is_file():
            with open(traces_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        trace = ExperimentTrace.model_validate_json(line)
                        traces.append(trace)
                        completed_ids.add(trace.example_id)

        pending = [ex for ex in dataset if ex.example_id not in completed_ids]
        with open(traces_path, "a", encoding="utf-8") as f:
            for example in pending:
                trace = self._run_example(config, example, experiment_id)
                traces.append(trace)
                f.write(canonical_json(trace.model_dump(mode="json")) + "\n")
                f.flush()

        (run_dir / "config.json").write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest = self._build_manifest(config, experiment_id, traces)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        reports = [ev.evaluate(traces, dataset, config.evaluation) for ev in self.evaluators]
        metrics: dict[str, Any] = {}
        for report in reports:
            payload = report.model_dump(mode="json")
            metrics[report.evaluator] = payload
            (run_dir / f"metrics_{report.evaluator}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "report.md").write_text(
            self._render_report(config, manifest, reports), encoding="utf-8"
        )

        return {
            "experiment_id": experiment_id,
            "run_dir": str(run_dir),
            "trace_count": len(traces),
            "metrics": metrics,
            "manifest": manifest,
        }

    def _run_example(
        self,
        config: ExperimentConfig,
        example: EvaluationExample,
        experiment_id: str,
    ) -> ExperimentTrace:
        """Execute retrieve → context → generate for one example, capturing all failures."""
        trace = ExperimentTrace(
            trace_id=f"{experiment_id}:{example.example_id}",
            experiment_id=experiment_id,
            example_id=example.example_id,
            query=example.query,
            category=example.category,
            reference=ReferenceInfo(
                reference_answer=example.reference_answer,
                relevant_documents=example.relevant_documents,
                relevant_sections=example.relevant_sections,
                relevant_chunks=example.relevant_chunks,
            ),
            status="ok",
            config_hash=config.config_hash,
            corpus_version=config.corpus_version,
        )

        # 1. Retrieval — failures recorded, never converted to "no results"
        retrieval: RetrievalResponse | None = None
        try:
            retrieval = self.retriever.retrieve(
                query=example.query,
                top_k=config.retrieval.top_k,
            )
            trace.retrieval = retrieval
            trace.retrieval_latency_ms = retrieval.retrieval_metadata.latency_ms
        except Exception as exc:
            trace.status = "retrieval_failed"
            trace.error = ErrorInfo(
                stage="retrieval",
                error_type=type(exc).__name__,
                message=str(exc)[:1000],
            )
            return trace

        # 2. Context construction
        try:
            request: GenerationRequest = self.context_builder.build(retrieval)
            trace.context_chunk_ids = [c.chunk_id for c in request.context]
            trace.context_tokens = int(request.metadata.get("context_tokens", 0))
            trace.dropped_chunk_ids = list(request.metadata.get("dropped_chunk_ids", []))
        except Exception as exc:
            trace.status = "generation_failed"
            trace.error = ErrorInfo(
                stage="context",
                error_type=type(exc).__name__,
                message=str(exc)[:1000],
            )
            return trace

        if self.generator is None:
            trace.status = "empty"
            return trace

        # 3. Generation — failures recorded as failed, never as empty answers
        try:
            result = self.generator.generate(request)
            trace.generation = result
            trace.generation_latency_ms = result.latency_ms
            trace.usage = result.usage
            trace.estimated_cost_usd = calculate_cost_usd(
                config.generation.model,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            )
            trace.status = "ok" if result.answer.strip() else "empty"
        except Exception as exc:
            trace.status = "generation_failed"
            trace.error = ErrorInfo(
                stage="generation",
                error_type=type(exc).__name__,
                message=str(exc)[:1000],
            )
            return trace

        if trace.retrieval_latency_ms is not None and trace.generation_latency_ms is not None:
            trace.total_latency_ms = trace.retrieval_latency_ms + trace.generation_latency_ms
        return trace


    def _build_manifest(
        self,
        config: ExperimentConfig,
        experiment_id: str,
        traces: list[ExperimentTrace],
    ) -> dict[str, Any]:
        """Build the self-describing run manifest for reproducibility."""
        manifest_status = {"ok": 0, "empty": 0, "retrieval_failed": 0, "generation_failed": 0}
        for trace in traces:
            manifest_status[trace.status] = manifest_status.get(trace.status, 0) + 1

        return {
            "experiment_id": experiment_id,
            "config_name": config.name,
            "config_hash": config.config_hash,
            "corpus_version": config.corpus_version,
            "retrieval_method": config.retrieval.retrieval_method,
            "trace_count": len(traces),
            "status_counts": manifest_status,
            "error_types": sorted(
                {t.error.error_type for t in traces if t.error is not None}
            ),
            "git_commit": _git_commit(),
            "pricing_version": PRICING_VERSION,
            "component_versions": config.component_versions,
        }

    def _render_report(
        self,
        config: ExperimentConfig,
        manifest: dict[str, Any],
        reports: list[Any],
    ) -> str:
        """Render a human-readable markdown summary of the experiment."""
        lines = [
            f"# Experiment {manifest.get('experiment_id', config.experiment_id)}",
            "",
            f"- Config: `{config.name}` (hash `{config.config_hash}`)",
            f"- Retrieval: `{config.retrieval.retrieval_method}` top_k={config.retrieval.top_k}",
            f"- Corpus: `{config.corpus_version}`",
            f"- Traces: {manifest.get('trace_count', 0)}",
            f"- Status: {manifest.get('status_counts', {})}",
            f"- Judge enabled: {config.evaluation.enable_llm_judge}",
            "",
            "## Metrics",
            "",
        ]
        for report in reports:
            lines.append(f"### {report.evaluator} ({report.evaluator_version})")
            for metric in report.metrics:
                lines.append(
                    f"- {metric.name}"
                    + (f"@k={metric.k}" if metric.k is not None else "")
                    + f": {metric.value} (n={metric.n})"
                )
            lines.append("")
        return "\n".join(lines)
