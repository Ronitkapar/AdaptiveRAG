"""
fakes.py
--------
Deterministic offline fake components for unit testing without live API keys.
"""

import hashlib
import numpy as np
from typing import Any, Sequence

from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ContextChunk,
    GenerationRequest,
    GenerationResult,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
    TokenUsage,
)


class FakeEmbeddingModel:
    """Generates deterministic pseudo-random unit vectors based on text hash."""

    def __init__(self, dimension: int = 64, normalize: bool = True):
        self.dimension = dimension
        self.normalize = normalize
        self.model_id = "fake-embedding-model"
        self.config_hash = "fake_cfg_hash"
        self.call_count = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        # Hash query to seed a deterministic vector
        seed = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec.tolist()


class StubRetriever:
    """Controllable Retriever double: canned results, recorded calls, optional failure.

    Satisfies the `Retriever` protocol without any index, embedding model, or
    credential. Every call is recorded so tests can assert the `top_k` and
    filters a composite retriever passed down.
    """

    def __init__(
        self,
        method: str = "stub",
        results: Sequence[RetrievalResult] | None = None,
        raises: BaseException | None = None,
        latency_ms: float = 1.5,
        search_latency_ms: float = 0.75,
        index_id: str = "stub_index",
    ):
        self.method = method
        self.results = list(results or [])
        self.raises = raises
        self.latency_ms = latency_ms
        self.search_latency_ms = search_latency_ms
        self.index_id = index_id
        self.corpus_version = "stub_corpus"
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": filters}
        )
        if self.raises is not None:
            raise self.raises
        selected = self.results if top_k is None else self.results[:top_k]
        status = "ok" if selected else "no_results"
        meta = RetrievalMetadata(
            top_k=top_k if top_k is not None else len(selected),
            filters=filters,
            retriever_version=f"{self.method}_v1",
            index_id=self.index_id,
            corpus_version=self.corpus_version,
            latency_ms=self.latency_ms,
            search_latency_ms=self.search_latency_ms,
        )
        return RetrievalResponse(
            query=query,
            results=list(selected),
            retrieval_method=self.method,  # type: ignore[arg-type]
            status=status,
            retrieval_metadata=meta,
        )


class FakeGenerator:
    """Generates deterministic answers citing retrieved context sources."""

    def __init__(self, model_name: str = "fake-groq-model"):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1
        citations = []
        source_ids = []
        for ctx in request.context:
            citations.append(f"[Source {ctx.source_index}]")
            source_ids.append(ctx.chunk_id)

        cite_str = " ".join(citations) if citations else "[No sources cited]"
        answer = f"Synthesized answer for '{request.query}'. Verified against {cite_str}."

        return GenerationResult(
            answer=answer,
            source_chunk_ids=source_ids,
            model=self.model_name,
            usage=TokenUsage(input_tokens=150, output_tokens=40, total_tokens=190),
            latency_ms=45.2,
            status="ok",
            generator_version="fake_v1",
            prompt_version="fake_prompt_v1",
        )
