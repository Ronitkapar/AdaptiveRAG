"""
scripts.run_experiment
----------------------
CLI entrypoint for fixed retrieval baseline experiments (dense, BM25, or hybrid).
Builds the experiment configuration, instantiates components from settings,
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
    RoutingEvaluator,
    load_evaluation_dataset,
)
from adaptive_rag.experiments import (
    build_experiment_config,
    instantiate_components,
)
from adaptive_rag.experiments.runner import ExperimentRunner

logger = logging.getLogger("run_experiment")

# Default experiment name per retrieval strategy, used when --name is left at its
# default so hybrid never lands in the dense run directory.
DEFAULT_RUN_NAMES = {
    "dense": "dense_baseline_v1",
    "bm25": "bm25_baseline_v1",
    "hybrid": "hybrid_baseline_v1",
    "dense_rerank": "dense_rerank_v1",
    "bm25_rerank": "bm25_rerank_v1",
    "hybrid_rerank": "hybrid_rerank_v1",
    "adaptive": "adaptive_baseline_v1",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed retrieval baseline experiment.")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["dense", "bm25", "hybrid", "adaptive"],
        default="dense",
        help="Retrieval strategy to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVALUATION_DIR / "dense_eval_v1.jsonl",
        help="Evaluation dataset JSONL path",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment configuration name (default: <strategy>_baseline_v1)",
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
        "--no-generation",
        action="store_true",
        help="Skip answer generation (retrieval-only evaluation, fully offline)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Groq model id for the LLM judge",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Add a second-stage cross-encoder over the first-stage candidates",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Cross-encoder model id for --rerank",
    )
    parser.add_argument(
        "--reranker-revision",
        type=str,
        default=None,
        help="Pinned model revision for reproducibility (recorded in config.json)",
    )
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=None,
        help="First-stage candidate depth fed to the reranker (default 20)",
    )
    parser.add_argument(
        "--rerank-device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Execution provider selection for --rerank (default auto)",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=None,
        help="Scoring batch size for --rerank (default 16)",
    )
    parser.add_argument(
        "--rerank-max-length",
        type=int,
        default=None,
        help="Tokenizer truncation length for --rerank (default 512)",
    )
    parser.add_argument(
        "--rerank-fallback",
        action="store_true",
        help="Return un-reranked candidates when the reranker fails (opt-in, "
        "visible as rerank_fallback in every trace)",
    )

    # --- Phase 6 adaptive routing flags -------------------------------------
    parser.add_argument(
        "--routing-strategies",
        type=str,
        default=None,
        help="Comma-separated strategies the adaptive router may select "
        "(default bm25,dense,hybrid,hybrid_rerank). Restricting this also "
        "restricts which indexes are built, so a BM25-only run needs no vector store",
    )
    parser.add_argument(
        "--routing-ladder",
        type=str,
        default=None,
        help="Comma-separated escalation ladder, weakest first "
        "(default bm25,dense,hybrid,hybrid_rerank)",
    )
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Route once and never escalate (ablation baseline)",
    )
    parser.add_argument(
        "--no-sufficiency",
        action="store_true",
        help="Skip the retrieval-sufficiency check entirely (ablation baseline)",
    )
    parser.add_argument(
        "--routing-cost-weight",
        type=float,
        default=None,
        help="Quality-vs-cost trade-off: 0.0 is pure evidence, higher values "
        "penalize costlier strategies more (default 0.25)",
    )
    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=None,
        help="Sufficiency score above which retrieval is accepted (default 0.5)",
    )
    parser.add_argument(
        "--disable-feature-group",
        action="append",
        default=None,
        choices=["lexical", "semantic", "entity", "complexity", "question_type", "multi_concept"],
        help="Remove one routing signal group; repeatable (signal ablation)",
    )
    parser.add_argument(
        "--max-escalation-steps",
        type=int,
        default=None,
        help="Upper bound on escalations per query (default 1)",
    )
    args = parser.parse_args()

    # Reranking is a *routing* decision under Phase 6, not a fixed flag: the router
    # selects `hybrid_rerank` only when the evidence supports it. Combining the two
    # would silently override the router, so it is rejected rather than ignored.
    if args.retriever == "adaptive" and args.rerank:
        parser.error(
            "--rerank cannot be combined with --retriever adaptive: second-stage "
            "scoring is one of the strategies the adaptive router selects"
        )

    setup_logging()
    logger.info("Loading evaluation dataset from %s", args.dataset)
    dataset = load_evaluation_dataset(args.dataset)
    logger.info("Loaded %d evaluation examples", len(dataset))

    evaluation_kwargs: dict = {}
    if args.no_judge:
        from adaptive_rag.schemas import EvaluationConfig

        evaluation_kwargs["evaluation"] = EvaluationConfig(enable_llm_judge=False)

    from adaptive_rag.schemas import RetrievalConfig, RoutingConfig

    routing_fields: dict = {}
    if args.routing_strategies is not None:
        routing_fields["available_strategies"] = [
            s.strip() for s in args.routing_strategies.split(",") if s.strip()
        ]
    if args.routing_ladder is not None:
        routing_fields["escalation_ladder"] = [
            s.strip() for s in args.routing_ladder.split(",") if s.strip()
        ]
    if args.no_escalation:
        routing_fields["escalation_enabled"] = False
    if args.no_sufficiency:
        routing_fields["sufficiency_enabled"] = False
    if args.routing_cost_weight is not None:
        routing_fields["cost_weight"] = args.routing_cost_weight
    if args.sufficiency_threshold is not None:
        routing_fields["sufficiency_threshold"] = args.sufficiency_threshold
    if args.disable_feature_group:
        from adaptive_rag.schemas import FEATURE_GROUPS

        routing_fields["enabled_feature_groups"] = [
            g for g in FEATURE_GROUPS if g not in set(args.disable_feature_group)
        ]
    if args.max_escalation_steps is not None:
        routing_fields["max_escalation_steps"] = args.max_escalation_steps

    # When the strategy set is narrowed but no ladder is given explicitly, the
    # ladder must narrow with it. The config validator rejects a ladder naming an
    # unavailable strategy, and that check is correct: a rung the router could
    # never execute would be a promise the pipeline cannot keep. Narrowing here
    # keeps the default ladder consistent with --routing-strategies.
    if routing_fields.get("available_strategies") is not None and (
        routing_fields.get("escalation_ladder") is None
    ):
        routing_fields["escalation_ladder"] = list(routing_fields["available_strategies"])
    if "escalation_ladder" in routing_fields and "max_escalation_steps" not in routing_fields:
        # A one-rung ladder has nowhere to escalate to, so the step bound must drop
        # with it rather than silently keeping a now-unreachable bound of 1.
        routing_fields["max_escalation_steps"] = max(
            0, len(routing_fields["escalation_ladder"]) - 1
        )

    method = f"{args.retriever}_rerank" if args.rerank else args.retriever
    retrieval_fields: dict = {"retrieval_method": method}
    if args.rerank:
        retrieval_fields["rerank_enabled"] = True
        if args.reranker_model is not None:
            retrieval_fields["rerank_model_id"] = args.reranker_model
        if args.reranker_revision is not None:
            retrieval_fields["rerank_model_revision"] = args.reranker_revision
        if args.rerank_candidate_k is not None:
            retrieval_fields["rerank_candidate_k"] = args.rerank_candidate_k
            # --rerank-candidate-k must also deepen the fused pool for hybrid, or
            # the reranker would be silently capped by the fusion depth.
            if method == "hybrid_rerank":
                retrieval_fields["candidate_k"] = args.rerank_candidate_k
        if args.rerank_device is not None:
            retrieval_fields["rerank_device"] = args.rerank_device
        if args.rerank_batch_size is not None:
            retrieval_fields["rerank_batch_size"] = args.rerank_batch_size
        if args.rerank_max_length is not None:
            retrieval_fields["rerank_max_length"] = args.rerank_max_length
        if args.rerank_fallback:
            retrieval_fields["rerank_fallback"] = True
    retrieval_kwargs: dict = {"retrieval": RetrievalConfig(**retrieval_fields)}
    if method == "adaptive":
        try:
            routing_kwargs: dict = {"routing": RoutingConfig(**routing_fields)}
        except Exception as exc:
            logger.error("Invalid routing configuration: %s", exc)
            return 2

    default_name = DEFAULT_RUN_NAMES[method]
    # Historical default was "dense_baseline_v1"; keep the remap so a dense-named
    # run never collects a non-dense strategy's traces.
    requested_name = args.name if args.name is not None else default_name
    experiment_name = (
        default_name
        if requested_name == "dense_baseline_v1" and method != "dense"
        else requested_name
    )

    config = build_experiment_config(
        name=experiment_name, **evaluation_kwargs, **retrieval_kwargs, **routing_kwargs
    )
    if args.no_judge is False and args.judge_model != config.evaluation.judge_model:
        config = config.model_copy(
            update={
                "evaluation": config.evaluation.model_copy(
                    update={"judge_model": args.judge_model}
                )
            }
        )
    logger.info(
        "Experiment config: name=%s corpus=%s retriever=%s top_k=%d judge=%s",
        config.name,
        config.corpus_version,
        config.retrieval.retrieval_method,
        config.retrieval.top_k,
        config.evaluation.enable_llm_judge,
    )

    try:
        (
            embedding_model,
            index_or_store,
            retriever,
            context_builder,
            generator,
            _chunker,
            _ingestion_pipeline,
        ) = instantiate_components(config)
    except MissingCredentialError as exc:
        logger.error("Missing credentials: %s", exc)
        return 2

    method = config.retrieval.retrieval_method

    if method == "adaptive":
        # The adaptive retriever composes several retrievers; check each one that
        # its index is populated, naming the strategy that is actually missing so
        # the failure is actionable instead of generic.
        for strategy, sub in retriever.retrievers.items():
            count = (
                sub.index.total_docs
                if getattr(sub, "method", "") == "bm25"
                else sub.vector_store.count()
            )
            if count == 0:
                logger.error(
                    "Strategy '%s' is available to the adaptive router but its index "
                    "is empty. Run scripts/build_bm25_index.py and/or "
                    "scripts/build_index.py first, or narrow --routing-strategies.",
                    strategy,
                )
                return 1
        logger.info(
            "Adaptive routing active: strategies=%s ladder=%s cost_weight=%.3f "
            "sufficiency=%s escalation=%s",
            config.routing.available_strategies,
            config.routing.escalation_ladder,
            config.routing.cost_weight,
            config.routing.sufficiency_enabled,
            config.routing.escalation_enabled,
        )
    else:
        is_bm25 = method in ("bm25", "bm25_rerank")
        # Hybrid returns the dense vector store in this slot; its BM25 index is held
        # by the composed retriever, so both sides are checked explicitly. With
        # reranking on, the composite sits one level deeper behind the wrapper.
        base_retriever = getattr(retriever, "base_retriever", retriever)
        doc_count = index_or_store.total_docs if is_bm25 else index_or_store.count()
        if doc_count == 0:
            if is_bm25:
                logger.error(
                    "BM25 index is empty. Run scripts/build_bm25_index.py first.",
                )
            else:
                logger.error(
                    "Vector store collection '%s' is empty. Run scripts/build_index.py first.",
                    config.index.collection_name,
                )
            return 1
        if method in ("hybrid", "hybrid_rerank"):
            if base_retriever.bm25_retriever.index.total_docs == 0:
                logger.error(
                    "Hybrid retrieval needs both indexes, but the BM25 index is empty. "
                    "Run scripts/build_bm25_index.py first."
                )
                return 1
            logger.info(
                "Hybrid retrieval active: dense collection '%s' (%d points) + BM25 index "
                "(%d docs), rrf_k=%d candidate_k=%d",
                config.index.collection_name,
                doc_count,
                base_retriever.bm25_retriever.index.total_docs,
                config.retrieval.rrf_k,
                config.retrieval.candidate_k,
            )

    if config.retrieval.rerank_enabled:
        logger.info(
            "Second-stage scoring active: model=%s revision=%s device=%s "
            "batch_size=%d candidate_k=%d top_k=%d fallback=%s",
            config.retrieval.rerank_model_id,
            config.retrieval.rerank_model_revision,
            config.retrieval.rerank_device,
            config.retrieval.rerank_batch_size,
            config.retrieval.rerank_candidate_k,
            config.retrieval.top_k,
            config.retrieval.rerank_fallback,
        )

    runner = ExperimentRunner(
        retriever=retriever,
        generator=None if args.no_generation else generator,
        context_builder=context_builder,
        evaluators=[
            RetrievalEvaluator(),
            RoutingEvaluator(),
            GenerationEvaluator(
                judge=None
                if (args.no_judge or args.no_generation)
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
        if hasattr(index_or_store, "close"):
            index_or_store.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
