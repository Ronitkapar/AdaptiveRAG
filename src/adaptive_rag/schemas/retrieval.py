"""
schemas.retrieval
-----------------
Data contracts for dense retrieval queries, ranked results, and responses.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.schemas.chunk import ChunkMetadata, ChunkProvenance


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its rank and similarity score."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: ChunkMetadata
    provenance: ChunkProvenance


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


class RetrievalResponse(BaseModel):
    """Complete response returned by a retriever."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    retrieval_method: Literal["dense", "bm25", "hybrid"] = "dense"
    status: Literal["ok", "no_results"] = "ok"
    retrieval_metadata: RetrievalMetadata
