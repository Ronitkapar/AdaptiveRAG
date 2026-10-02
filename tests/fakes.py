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


class FakeReranker:
    """Deterministic `Reranker` double: canned scores, recorded calls, optional failure.

    Satisfies the `Reranker` protocol with no model, no ONNX session, and no
    network. Scores come from a fixed table (or a per-passage callable) so tests
    can assert exact ordering. Every call is recorded, including the batch sizes
    the wrapper passed down, so batching and call-count behaviour are testable.
    """

    def __init__(
        self,
        scores: Sequence[float] | None = None,
        *,
        score_fn: Any = None,
        constant_score: float = 0.0,
        raises: BaseException | None = None,
        version: str = "fake_reranker_v1",
        model_id: str = "fake-cross-encoder",
    ):
        self.scores = list(scores) if scores is not None else None
        self.score_fn = score_fn
        self.constant_score = constant_score
        self.raises = raises
        self.version = version
        self.model_id = model_id
        self.calls: list[dict[str, Any]] = []
        self.batch_sizes: list[int] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def scored_passage_count(self) -> int:
        return sum(call["batch_size"] for call in self.calls)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        batch_size = len(passages)
        self.calls.append(
            {"query": query, "batch_size": batch_size, "passages": list(passages)}
        )
        self.batch_sizes.append(batch_size)
        if self.raises is not None:
            raise self.raises
        if self.score_fn is not None:
            return [float(self.score_fn(p)) for p in passages]
        if self.scores is not None:
            return [float(s) for s in self.scores]
        return [float(self.constant_score) for _ in passages]


class FakeRouter:
    """Deterministic `Router` double returning canned decisions.

    Satisfies the `Router` protocol so the adaptive orchestration path can be
    tested without depending on the rule-based weight table. Every call is
    recorded, so tests can assert the features a decision was derived from --
    which is what proves the analyzer ran before the router.
    """

    def __init__(
        self,
        decision=None,
        raises: BaseException | None = None,
        version: str = "fake_router_v1",
    ):
        self.decision = decision
        self.raises = raises
        self.version = version
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def route(self, features, *, top_k: int):
        self.calls.append({"features": features, "top_k": top_k})
        if self.raises is not None:
            raise self.raises
        if self.decision is not None:
            return self.decision
        raise AssertionError("FakeRouter has no canned decision configured")


class FakeAnalyzer:
    """Deterministic `QueryAnalyzer` double returning fixed features.

    Lets escalation tests drive sufficiency from a chosen query without depending
    on what the real analyzer infers from the string.
    """

    def __init__(
        self,
        features=None,
        raises: BaseException | None = None,
        version: str = "fake_analyzer_v1",
    ):
        self.features = features
        self.raises = raises
        self.version = version
        self.calls: list[str] = []

    def analyze(self, query: str):
        self.calls.append(query)
        if self.raises is not None:
            raise self.raises
        if self.features is not None:
            return self.features
        raise AssertionError("FakeAnalyzer has no canned features configured")


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
