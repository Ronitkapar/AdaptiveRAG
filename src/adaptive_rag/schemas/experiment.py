"""
schemas.experiment
------------------
Data contracts for evaluation benchmarks, traces, run configurations, and metrics.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.schemas.config import (
    ChunkingConfig,
    ContextConfig,
    EmbeddingConfig,
    EvaluationConfig,
    GenerationConfig,
    IndexConfig,
    IngestionConfig,
    RetrievalConfig,
)
from adaptive_rag.schemas.generation import GenerationResult, TokenUsage
from adaptive_rag.schemas.retrieval import RetrievalResponse


class SectionRef(BaseModel):
    """Reference to a relevant document section."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    section_path_prefix: list[str] = Field(default_factory=list)


class EvaluationExample(BaseModel):
    """A curated benchmark question with ground truth references."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    query: str
    reference_answer: str
    relevant_documents: list[str]
    relevant_sections: list[SectionRef] = Field(default_factory=list)
    relevant_chunks: list[str] = Field(default_factory=list)
    category: Literal[
        "factual",
        "terminology",
        "conceptual",
        "comparative",
        "multi_document",
        "fine_grained",
    ]
    requires_multi_hop: bool = False
    notes: str = ""
    dataset_version: str = "dense_eval_v1"


class ErrorInfo(BaseModel):
    """Structured failure record captured during pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["retrieval", "context", "generation", "evaluation"]
    error_type: str
    message: str


class ReferenceInfo(BaseModel):
    """Ground truth reference information attached to an experiment trace."""

    model_config = ConfigDict(extra="forbid")

    reference_answer: str
    relevant_documents: list[str]
    relevant_sections: list[SectionRef] = Field(default_factory=list)
    relevant_chunks: list[str] = Field(default_factory=list)


class ExperimentTrace(BaseModel):
    """Complete per-query audit record capturing all intermediate stages and latencies."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    experiment_id: str
    example_id: str
    query: str
    category: str
    retrieval: RetrievalResponse | None = None
    context_chunk_ids: list[str] = Field(default_factory=list)
    context_tokens: int = 0
    dropped_chunk_ids: list[str] = Field(default_factory=list)
    generation: GenerationResult | None = None
    reference: ReferenceInfo
    status: Literal["ok", "empty", "retrieval_failed", "generation_failed"]
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    # Second-stage scoring split (absent on non-reranked runs)
    candidate_generation_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    rerank_candidate_count: int | None = None
    rerank_result_count: int | None = None
    rerank_fallback: bool | None = None
    usage: TokenUsage | None = None
    estimated_cost_usd: float | None = None
    error: ErrorInfo | None = None
    config_hash: str
    corpus_version: str


class ExperimentConfig(BaseModel):
    """Full self-describing experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str
    corpus_version: str
    ingestion: IngestionConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig
    retrieval: RetrievalConfig
    context: ContextConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    component_versions: dict[str, str] = Field(default_factory=dict)
    config_hash: str


class MetricValue(BaseModel):
    """A single evaluated metric observation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None
    k: int | None = None
    n: int
    version: str
    notes: str = ""


class EvaluationReport(BaseModel):
    """Aggregated evaluation report produced by an Evaluator."""

    model_config = ConfigDict(extra="forbid")

    evaluator: str
    evaluator_version: str
    metrics: list[MetricValue] = Field(default_factory=list)
    aggregates: dict[str, Any] = Field(default_factory=dict)
