"""
scripts.run_experiment
----------------------
CLI entrypoint for the Phase 2 fixed Dense-RAG baseline experiment.
Builds the experiment configuration, instantiates components from .env settings,
executes the dataset through ExperimentRunner, and persists trace artifacts.
"""

import argparse
import logging
import sys
from pathlib import Path

from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.paths import EVALUATION_DIR
from adaptive_rag.errors import MissingCredentialError
from adaptive_rag.evaluation import (
    EfficiencyEvaluator,
    GenerationEvaluator,
    GroqLLMJudge,
    RetrievalEvaluator,
    load_evaluation_dataset,
)
from adaptive_rag.experiments import (
    build_experiment_config,
    instantiate_components,
)
from adaptive_rag.experiments.runner import ExperimentRunner

logger = logging.getLogger("run_experiment")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed Dense-RAG baseline experiment.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVALUATION_DIR / "dense_eval_v1.jsonl",
        help="Evaluation dataset JSONL path",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dense_baseline_v1",
        help="Experiment configuration name",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit experiment run id (default: timestamped)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run id instead of starting over",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable the LLM judge (lexical metrics only, fully offline)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Groq model id for the LLM judge",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Loading evaluation dataset from %s", args.dataset)
    dataset = load_evaluation_dataset(args.dataset)
    logger.info("Loaded %d evaluation examples", len(dataset))

    evaluation_kwargs: dict = {}
    if args.no_judge:
        from adaptive_rag.schemas import EvaluationConfig

        evaluation_kwargs["evaluation"] = EvaluationConfig(enable_llm_judge=False)

    config = build_experiment_config(name=args.name, **evaluation_kwargs)
    if args.no_judge is False and args.judge_model != config.evaluation.judge_model:
        config = config.model_copy(
            update={
                "evaluation": config.evaluation.model_copy(
                    update={"judge_model": args.judge_model}
                )
            }
        )
    logger.info(
        "Experiment config: name=%s corpus=%s top_k=%d judge=%s",
        config.name,
        config.corpus_version,
        config.retrieval.top_k,
        config.evaluation.enable_llm_judge,
    )

    try:
        (
            embedding_model,
            vector_store,
            retriever,
            context_builder,
            generator,
            _chunker,
            _ingestion_pipeline,
        ) = instantiate_components(config)
    except MissingCredentialError as exc:
        logger.error("Missing credentials: %s", exc)
        return 2

    if vector_store.count() == 0:
        logger.error(
            "Vector store collection '%s' is empty. Run scripts/build_index.py first.",
            config.index.collection_name,
        )
        return 1

    runner = ExperimentRunner(
        retriever=retriever,
        generator=generator,
        context_builder=context_builder,
        evaluators=[
            RetrievalEvaluator(),
            GenerationEvaluator(
                judge=None
                if args.no_judge
                else GroqLLMJudge(model=args.judge_model, use_cache=True),
            ),
            EfficiencyEvaluator(generation_model=config.generation.model),
        ],
    )

    logger.info("Running experiment over %d examples...", len(dataset))
    summary = runner.run(config, dataset, resume=args.resume, run_id=args.run_id)

    logger.info("=" * 60)
    logger.info("Experiment complete: %s", summary["experiment_id"])
    logger.info("Run directory: %s", summary["run_dir"])
    logger.info("Traces: %d", summary["trace_count"])
    logger.info("Status counts: %s", summary["manifest"].get("status_counts", {}))
    for evaluator_name, payload in summary["metrics"].items():
        logger.info("--- %s ---", evaluator_name)
        for metric in payload.get("metrics", []):
            logger.info(
                "  %s%s = %s (n=%s)",
                metric.get("name"),
                f"@k={metric.get('k')}" if metric.get("k") is not None else "",
                metric.get("value"),
                metric.get("n"),
            )
    logger.info("=" * 60)

    retrieval_top5 = next(
        (
            m.get("value")
            for m in summary["metrics"].get("retrieval", {}).get("metrics", [])
            if m.get("name") == "recall_at_k" and m.get("k") == 5
        ),
        None,
    )
    logger.info("Baseline headline — Recall@5: %s", retrieval_top5)
    try:
        vector_store.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
