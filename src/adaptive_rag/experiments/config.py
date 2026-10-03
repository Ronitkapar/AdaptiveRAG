"""
experiments.config
------------------
Experiment configuration assembly and corpus fingerprinting.
"""

import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from adaptive_rag import __version__
from adaptive_rag.chunking.structure_aware import StructureAwareChunker
from adaptive_rag.config.hashing import compute_config_hash, compute_file_sha256
from adaptive_rag.config.paths import PAPERS_MANIFEST_PATH, REPO_ROOT, RERANKER_DIR
from adaptive_rag.config.settings import settings as default_settings
from adaptive_rag.embeddings.aicredits import AICreditsEmbeddingModel
from adaptive_rag.errors import ConfigurationError, IndexUnavailableError
from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.generation.groq import GroqGenerator
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.indexing.bm25 import BM25Index
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.reranking import OnnxCrossEncoderReranker, Reranker
from adaptive_rag.retrieval.adaptive import AdaptiveRetriever
from adaptive_rag.retrieval.bm25 import BM25Retriever
from adaptive_rag.retrieval.dense import DenseRetriever
from adaptive_rag.retrieval.hybrid import HybridRetriever
from adaptive_rag.retrieval.reranked import RerankedRetriever
from adaptive_rag.routing import (
    EscalationPolicy,
    QueryFeatureAnalyzer,
    RuleBasedRouter,
    SufficiencyChecker,
)
from adaptive_rag.schemas import (
    ChunkingConfig,
    ContextConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    IndexConfig,
    IngestionConfig,
    RerankerConfig,
    RetrievalConfig,
    RoutingConfig,
)

# Component versions recorded in every experiment for reproducibility
COMPONENT_VERSIONS = {
    "adaptive_rag": __version__,
    "ingestion": IngestionConfig().ingestion_version,
    "chunking": ChunkingConfig().chunking_version,
    "embedding": EmbeddingConfig().embedding_pipeline_version,
    "index": IndexConfig().index_version,
    "retrieval": RetrievalConfig().retriever_version,
    "context": ContextConfig().context_template_version,
    "generation": GenerationConfig().generator_version,
    "generation_prompt": GenerationConfig().prompt_version,
    "evaluation": EvaluationConfig().evaluation_version,
}


def compute_corpus_version(manifest_path: Path = PAPERS_MANIFEST_PATH) -> str:
    """Fingerprint the corpus: manifest bytes plus every raw PDF hash."""
    manifest_bytes = manifest_path.read_bytes()
    parts = [str(compute_config_hash({"manifest": manifest_bytes.decode("utf-8", "ignore")}))]

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    for paper in manifest:
        pdf_path = REPO_ROOT / paper["local_path"]
        if pdf_path.is_file():
            parts.append(compute_file_sha256(pdf_path))
        else:
            parts.append(paper.get("sha256", "missing"))

    return "corpus_" + compute_config_hash({"files": parts})


def build_experiment_config(
    name: str = "dense_baseline_v1",
    *,
    experiment_id: str | None = None,
    ingestion: IngestionConfig | None = None,
    chunking: ChunkingConfig | None = None,
    embedding: EmbeddingConfig | None = None,
    index: IndexConfig | None = None,
    retrieval: RetrievalConfig | None = None,
    context: ContextConfig | None = None,
    generation: GenerationConfig | None = None,
    evaluation: EvaluationConfig | None = None,
    routing: RoutingConfig | None = None,
    corpus_version: str | None = None,
) -> ExperimentConfig:
    """Assemble a fully-specified, hash-stamped experiment configuration."""
    active_retrieval = retrieval or RetrievalConfig()
    comp_versions = dict(COMPONENT_VERSIONS)
    comp_versions["retrieval"] = active_retrieval.retriever_version
    if active_retrieval.rerank_enabled:
        comp_versions["reranker"] = RerankerConfig().reranker_version
    # Routing versions are recorded only for adaptive runs, mirroring how the
    # reranker is recorded only when enabled: a fixed-strategy run's manifest
    # must stay identical to what Phase 2-5 produced.
    active_routing = routing or RoutingConfig()
    if active_retrieval.retrieval_method == "adaptive":
        comp_versions["routing"] = active_routing.router_version
        comp_versions["routing_analyzer"] = active_routing.analyzer_version
        comp_versions["routing_sufficiency"] = active_routing.sufficiency_version
        comp_versions["routing_escalation"] = active_routing.escalation_version

    config = ExperimentConfig(
        experiment_id=experiment_id or name,
        name=name,
        corpus_version=corpus_version or compute_corpus_version(),
        ingestion=ingestion or IngestionConfig(),
        chunking=chunking or ChunkingConfig(),
        embedding=embedding or EmbeddingConfig(),
        index=index or IndexConfig(),
        retrieval=active_retrieval,
        context=context or ContextConfig(),
        generation=generation or GenerationConfig(),
        evaluation=evaluation or EvaluationConfig(),
        routing=active_routing,
        component_versions=comp_versions,
        config_hash="pending",
    )
    # Hash everything except the hash field itself, then stamp it
    hashed = compute_config_hash(config.model_dump(mode="json", exclude={"config_hash"}))
    return config.model_copy(update={"config_hash": hashed})


def describe_component_versions() -> dict[str, str]:
    """Component version strings for manifests."""
    return dict(COMPONENT_VERSIONS)


def build_reranker(config: ExperimentConfig) -> Reranker:
    """Build the second-stage scorer described by the retrieval configuration."""
    retrieval = config.retrieval
    return OnnxCrossEncoderReranker(
        model_id=retrieval.rerank_model_id,
        model_revision=retrieval.rerank_model_revision,
        device=retrieval.rerank_device,
        batch_size=retrieval.rerank_batch_size,
        max_length=retrieval.rerank_max_length,
        model_dir=RERANKER_DIR,
    )


def build_routing_components(
    config: ExperimentConfig,
) -> tuple[RuleBasedRouter, QueryFeatureAnalyzer, SufficiencyChecker, EscalationPolicy]:
    """Build the four Phase 6 collaborators described by an experiment config.

    Kept separate from `instantiate_components` so the routing layer can be built
    and tested without touching any index or provider.
    """
    routing = config.routing
    return (
        RuleBasedRouter(routing),
        QueryFeatureAnalyzer(),
        SufficiencyChecker(routing),
        EscalationPolicy(routing),
    )


def _shared_qdrant_client() -> Any:
    """Open one embedded Qdrant client for the whole component tree, or None.

    Local mode (`QDRANT_MODE=local`) takes an exclusive lock on the storage
    folder and refuses a second client in the same process. Every
    Qdrant-backed strategy builds its own `QdrantVectorStore`, so an adaptive
    run with more than one Qdrant-using strategy would try to open several
    clients and fail on the second one.

    That failure was invisible until Phase 7 because the only adaptive run in
    `experiments/` configured `available_strategies=['bm25']`, which opens no
    vector store at all. Any run naming more than one dense-capable strategy
    hit it.

    Sharing one client is safe here because the strategies only ever read from
    the same immutable collection, and `QdrantVectorStore` already accepts an
    injected client. Server mode is unaffected -- those clients are stateless
    handles -- so the caller always gets its own and no sharing is imposed.
    """
    settings = default_settings
    if settings.QDRANT_MODE == "server" and settings.QDRANT_URL:
        return None
    try:
        path = Path(settings.QDRANT_PATH)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    except Exception as exc:  # noqa: BLE001 -- normalized to the store's error
        raise IndexUnavailableError(
            f"Failed to initialize shared Qdrant client: {exc} "
            f"Storage path: {settings.QDRANT_PATH}. In local mode the commonest "
            f"cause is another process on this machine still holding the lock on "
            f"that folder (a previous run, a REPL, or a stray script) -- stop it "
            f"and retry; only switch to Qdrant server mode if a shared daemon is "
            f"actually what you want."
        ) from exc


def _build_bm25(
    config: ExperimentConfig, constituent: "RetrievalConfig", client: Any = None
) -> tuple[Any, Any]:
    """Build a real BM25 retriever plus the index it holds open.

    `client` is accepted and ignored so every strategy builder shares one
    signature; see `_shared_qdrant_client` for why the adaptive arm needs it.
    """
    bm25_index = BM25Index.load(expected_corpus_version=config.corpus_version)
    return (
        BM25Retriever(
            index=bm25_index,
            config=constituent,
            corpus_version=config.corpus_version,
            index_id="adaptiverag_bm25_v1",
        ),
        bm25_index,
    )


def _build_dense(
    config: ExperimentConfig, constituent: "RetrievalConfig", client: Any = None
) -> tuple[Any, Any]:
    """Build a real dense retriever plus the vector store it holds open.

    `client` is the process-wide Qdrant client; when supplied it is reused
    instead of opening a second embedded handle (see `_shared_qdrant_client`).
    """
    vector_store = QdrantVectorStore(config=config.index, client=client)
    return (
        DenseRetriever(
            embedding_model=AICreditsEmbeddingModel(config=config.embedding),
            vector_store=vector_store,
            config=constituent,
            corpus_version=config.corpus_version,
            index_id=config.index.collection_name,
        ),
        vector_store,
    )


def _build_hybrid(
    config: ExperimentConfig,
    constituent: "RetrievalConfig",
    rerank: bool,
    client: Any = None,
) -> tuple[Any, Any]:
    """Build real hybrid (optionally reranked) plus the vector store it holds open.

    The fused pool is deepened to the rerank depth when reranking, so
    `--rerank-candidate-k` cannot be silently capped by the fusion depth.

    `client` is the process-wide Qdrant client; see `_shared_qdrant_client`.
    """
    branch = constituent.model_copy(update={"score_threshold": None})
    hybrid_config = branch.model_copy(
        update={"retrieval_method": "hybrid", "retriever_version": "hybrid_v1"}
    )
    if rerank:
        branch = branch.model_copy(
            update={"candidate_k": config.retrieval.rerank_candidate_k}
        )
        hybrid_config = hybrid_config.model_copy(
            update={"candidate_k": config.retrieval.rerank_candidate_k}
        )

    bm25_index = BM25Index.load(expected_corpus_version=config.corpus_version)
    vector_store = QdrantVectorStore(config=config.index, client=client)
    hybrid = HybridRetriever(
        dense_retriever=DenseRetriever(
            embedding_model=AICreditsEmbeddingModel(config=config.embedding),
            vector_store=vector_store,
            config=branch,
            corpus_version=config.corpus_version,
            index_id=config.index.collection_name,
        ),
        bm25_retriever=BM25Retriever(
            index=bm25_index,
            config=branch,
            corpus_version=config.corpus_version,
            index_id="adaptiverag_bm25_v1",
        ),
        config=hybrid_config,
        corpus_version=config.corpus_version,
    )
    if not rerank:
        return hybrid, vector_store
    return (
        RerankedRetriever(
            base_retriever=hybrid,
            reranker=build_reranker(config),
            config=config.retrieval.model_copy(
                update={
                    "retrieval_method": "hybrid_rerank",
                    "rerank_enabled": True,
                    "retriever_version": "hybrid_rerank_v1",
                }
            ),
            corpus_version=config.corpus_version,
        ),
        vector_store,
    )


def instantiate_components(config: ExperimentConfig, client: Any = None):
    """Build the runtime components described by an experiment configuration.

    Returns (embedding_model, vector_store_or_index, retriever, context_builder, generator,
    chunker, ingestion_pipeline). Component construction is lazy where credentials
    are required. When `retrieval.rerank_enabled` is set, the first-stage retriever
    is wrapped in a `RerankedRetriever`; slot 1 still holds the index or vector
    store so callers can count and close it directly.

    For `retrieval_method="adaptive"`, the routing layer composes one retriever per
    strategy named in `routing.available_strategies` -- and only those, so a
    BM25-only adaptive run never builds, opens, or requires a vector store.
    """
    context_builder = ContextBuilder(
        context_config=config.context, generation_config=config.generation
    )
    generator = GroqGenerator(config=config.generation)
    chunker = StructureAwareChunker(config=config.chunking)
    ingestion_pipeline = IngestionPipeline(config=config.ingestion)

    if config.retrieval.retrieval_method == "adaptive":
        # Honour an injected client rather than always opening a second one. A
        # caller that builds several arms at once (the Phase 7 cost sweep) is
        # already holding the exclusive local-mode lock, so opening another here
        # fails outright with `AlreadyLocked`. Single-arm callers pass None and
        # still get a client of their own.
        shared_client = client if client is not None else _shared_qdrant_client()
        builders = {
            "bm25": lambda: _build_bm25(config, config.retrieval, shared_client),
            "dense": lambda: _build_dense(config, config.retrieval, shared_client),
            "hybrid": lambda: _build_hybrid(
                config, config.retrieval, rerank=False, client=shared_client
            ),
            "hybrid_rerank": lambda: _build_hybrid(
                config, config.retrieval, rerank=True, client=shared_client
            ),
        }
        retrievers: dict[str, Any] = {}
        primary_resource: Any = None
        for strategy in config.routing.available_strategies:
            if strategy not in builders:
                raise ConfigurationError(
                    f"Unsupported adaptive retrieval strategy: {strategy!r}"
                )
            retriever, resource = builders[strategy]()
            retrievers[strategy] = retriever
            if primary_resource is None:
                primary_resource = resource

        router, analyzer, checker, policy = build_routing_components(config)
        adaptive = AdaptiveRetriever(
            retrievers=retrievers,
            router=router,
            analyzer=analyzer,
            sufficiency_checker=checker,
            escalation_policy=policy,
            retrieval_config=config.retrieval,
            routing_config=config.routing,
            corpus_version=config.corpus_version,
        )
        return (
            None,
            primary_resource,
            adaptive,
            context_builder,
            generator,
            chunker,
            ingestion_pipeline,
        )

    reranking = config.retrieval.rerank_enabled

    def _maybe_wrap(retriever):
        if not reranking:
            return retriever
        return RerankedRetriever(
            base_retriever=retriever,
            reranker=build_reranker(config),
            config=config.retrieval,
            corpus_version=config.corpus_version,
        )

    if config.retrieval.retrieval_method in ("bm25", "bm25_rerank"):
        bm25_index = BM25Index.load(
            expected_corpus_version=config.corpus_version,
        )
        retriever = BM25Retriever(
            index=bm25_index,
            config=config.retrieval,
            corpus_version=config.corpus_version,
            index_id="adaptiverag_bm25_v1",
        )
        return (
            None,
            bm25_index,
            _maybe_wrap(retriever),
            context_builder,
            generator,
            chunker,
            ingestion_pipeline,
        )

    if config.retrieval.retrieval_method in ("hybrid", "hybrid_rerank"):
        # Hybrid fuses rankings, so a score threshold is meaningless across the two
        # branches: each constituent gets a threshold-free view of the config.
        constituent_config = config.retrieval.model_copy(update={"score_threshold": None})
        if reranking:
            # The fused pool must be deep enough to feed the second stage, otherwise
            # `--rerank-candidate-k` would be silently capped by the fusion depth.
            constituent_config = constituent_config.model_copy(
                update={"candidate_k": config.retrieval.rerank_candidate_k}
            )
        bm25_index = BM25Index.load(
            expected_corpus_version=config.corpus_version,
        )
        embedding_model = AICreditsEmbeddingModel(config=config.embedding)
        vector_store = QdrantVectorStore(config=config.index, client=client)
        dense_retriever = DenseRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            config=constituent_config,
            corpus_version=config.corpus_version,
            index_id=config.index.collection_name,
        )
        bm25_retriever = BM25Retriever(
            index=bm25_index,
            config=constituent_config,
            corpus_version=config.corpus_version,
            index_id="adaptiverag_bm25_v1",
        )
        hybrid_config = config.retrieval
        if reranking:
            hybrid_config = hybrid_config.model_copy(
                update={"candidate_k": config.retrieval.rerank_candidate_k}
            )
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            config=hybrid_config,
            corpus_version=config.corpus_version,
        )
        return (
            embedding_model,
            vector_store,
            _maybe_wrap(retriever),
            context_builder,
            generator,
            chunker,
            ingestion_pipeline,
        )

    embedding_model = AICreditsEmbeddingModel(config=config.embedding)
    vector_store = QdrantVectorStore(config=config.index, client=client)
    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        config=config.retrieval,
        corpus_version=config.corpus_version,
        index_id=config.index.collection_name,
    )
    return (
        embedding_model,
        vector_store,
        _maybe_wrap(retriever),
        context_builder,
        generator,
        chunker,
        ingestion_pipeline,
    )
