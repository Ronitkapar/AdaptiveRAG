"""
schemas.retrieval
-----------------
Data contracts for dense retrieval queries, ranked results, and responses.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.schemas.chunk import ChunkMetadata, ChunkProvenance
from adaptive_rag.schemas.routing import RoutingTrace


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its rank and similarity score."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: ChunkMetadata
    provenance: ChunkProvenance
    # Second-stage scoring provenance: `score` holds the rerank score when the
    # result was reordered, and these keep the first-stage signal intact.
    retrieval_score: float | None = None
    retrieval_rank: int | None = None


class RetrievalMetadata(BaseModel):
    """Observability metadata capturing retriever execution details."""

    model_config = ConfigDict(extra="forbid")

    top_k: int
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None
    retriever_version: str
    index_id: str
    corpus_version: str
    latency_ms: float
    search_latency_ms: float
    # Vector / dense retrieval fields (optional for lexical retrievers)
    embedding_model_id: str | None = None
    embedding_dim: int | None = None
    collection: str | None = None
    query_embedding_latency_ms: float | None = None
    # BM25 / lexical retrieval fields (optional for dense retrievers)
    k1: float | None = None
    b: float | None = None
    # Hybrid / fusion fields (optional for single-strategy retrievers)
    fusion_method: str | None = None
    rrf_k: int | None = None
    candidate_k: int | None = None
    dense_candidate_count: int | None = None
    bm25_candidate_count: int | None = None
    dense_latency_ms: float | None = None
    bm25_latency_ms: float | None = None
    fusion_latency_ms: float | None = None
    # Second-stage scoring fields (optional when no second stage runs)
    rerank_enabled: bool | None = None
    reranker_version: str | None = None
    reranker_model_id: str | None = None
    reranker_device: str | None = None
    rerank_candidate_k: int | None = None
    rerank_top_k: int | None = None
    candidate_generation_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    candidate_count: int | None = None
    result_count: int | None = None
    rerank_fallback: bool | None = None
    # Adaptive routing fields (present only when the adaptive pipeline ran). The
    # routing trace is carried here so it travels with the response and reaches the
    # experiment trace without the retriever holding mutable per-query state.
    routing: RoutingTrace | None = None
    adaptive_router_version: str | None = None
    adaptive_initial_strategy: str | None = None
    adaptive_final_strategy: str | None = None
    adaptive_stage_count: int | None = None
    adaptive_routing_latency_ms: float | None = None
    adaptive_initial_latency_ms: float | None = None
    adaptive_escalated: bool | None = None
    adaptive_sufficient: bool | None = None


class RetrievalResponse(BaseModel):
    """Complete response returned by a retriever."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    retrieval_method: Literal[
        "dense",
        "bm25",
        "hybrid",
        "dense_rerank",
        "bm25_rerank",
        "hybrid_rerank",
        "adaptive",
    ] = "dense"
    status: Literal["ok", "no_results"] = "ok"
    retrieval_metadata: RetrievalMetadata
