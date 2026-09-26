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
    embedding_model_id: str
    embedding_dim: int
    index_id: str
    collection: str
    corpus_version: str
    latency_ms: float
    query_embedding_latency_ms: float
    search_latency_ms: float


class RetrievalResponse(BaseModel):
    """Complete response returned by a retriever."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    retrieval_method: Literal["dense"] = "dense"
    status: Literal["ok", "no_results"] = "ok"
    retrieval_metadata: RetrievalMetadata
