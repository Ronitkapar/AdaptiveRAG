"""
experiments.config
------------------
Experiment configuration assembly and corpus fingerprinting.
"""

import json
from pathlib import Path

from adaptive_rag import __version__
from adaptive_rag.chunking.structure_aware import StructureAwareChunker
from adaptive_rag.config.hashing import compute_config_hash, compute_file_sha256
from adaptive_rag.config.paths import PAPERS_MANIFEST_PATH, REPO_ROOT
from adaptive_rag.embeddings.aicredits import AICreditsEmbeddingModel
from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.generation.groq import GroqGenerator
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.retrieval.dense import DenseRetriever
from adaptive_rag.schemas import (
    ChunkingConfig,
    ContextConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    IndexConfig,
    IngestionConfig,
    RetrievalConfig,
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
    corpus_version: str | None = None,
) -> ExperimentConfig:
    """Assemble a fully-specified, hash-stamped experiment configuration."""
    config = ExperimentConfig(
        experiment_id=experiment_id or name,
        name=name,
        corpus_version=corpus_version or compute_corpus_version(),
        ingestion=ingestion or IngestionConfig(),
        chunking=chunking or ChunkingConfig(),
        embedding=embedding or EmbeddingConfig(),
        index=index or IndexConfig(),
        retrieval=retrieval or RetrievalConfig(),
        context=context or ContextConfig(),
        generation=generation or GenerationConfig(),
        evaluation=evaluation or EvaluationConfig(),
        component_versions=dict(COMPONENT_VERSIONS),
        config_hash="pending",
    )
    # Hash everything except the hash field itself, then stamp it
    hashed = compute_config_hash(config.model_dump(mode="json", exclude={"config_hash"}))
    return config.model_copy(update={"config_hash": hashed})


def describe_component_versions() -> dict[str, str]:
    """Component version strings for manifests."""
    return dict(COMPONENT_VERSIONS)


def instantiate_components(config: ExperimentConfig):
    """Build the runtime components described by an experiment configuration.

    Returns (embedding_model, vector_store, retriever, context_builder, generator,
    chunker, ingestion_pipeline). Component construction is lazy where credentials
    are required.
    """
    embedding_model = AICreditsEmbeddingModel(config=config.embedding)
    vector_store = QdrantVectorStore(config=config.index)
    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        config=config.retrieval,
        corpus_version=config.corpus_version,
        index_id=config.index.collection_name,
    )
    context_builder = ContextBuilder(config=config.context, generation_config=config.generation)
    generator = GroqGenerator(config=config.generation)
    chunker = StructureAwareChunker(config=config.chunking)
    ingestion_pipeline = IngestionPipeline(config=config.ingestion)
    return (
        embedding_model,
        vector_store,
        retriever,
        context_builder,
        generator,
        chunker,
        ingestion_pipeline,
    )
