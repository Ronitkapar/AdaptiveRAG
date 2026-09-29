"""
retrieval.hybrid
----------------
Phase 4 hybrid retriever: composes two existing `Retriever` implementations and
fuses their rankings with Reciprocal Rank Fusion.

This module only orchestrates. It performs no tokenization, no embedding, no
index access, and no score mixing; the constituents own all of that. Constituent
exceptions propagate untouched, so a failing branch can never be silently
downgraded to a single-strategy answer.
"""

import time
from typing import Any

from adaptive_rag.errors import InvalidQueryError
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.retrieval.fusion import reciprocal_rank_fusion
from adaptive_rag.schemas import (
    RetrievalConfig,
    RetrievalMetadata,
    RetrievalResponse,
)


class HybridRetriever:
    """Rank-fusion retriever over two constituent retrievers."""

    method: str = "hybrid"

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: Retriever,
        config: RetrievalConfig | None = None,
        corpus_version: str = "corpus_v1",
        index_id: str = "hybrid_dense_bm25_v1",
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config or RetrievalConfig(
            retrieval_method="hybrid", retriever_version="hybrid_v1"
        )
        self.corpus_version = corpus_version
        self.index_id = index_id

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Retrieve candidates from both branches, fuse by rank, and truncate.

        `score_threshold` is deliberately not forwarded: a cosine threshold and
        a BM25 threshold live on incomparable scales, so applying one to both
        would silently truncate a single candidate list. The configured value is
        echoed into `RetrievalMetadata.score_threshold` for traceability only.
        `filters` are forwarded to both branches.
        """
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")

        k = top_k if top_k is not None else self.config.top_k
        candidate_k = self.config.candidate_k
        filt = filters if filters is not None else self.config.filters

        t0 = time.perf_counter()

        dense_response = self.dense_retriever.retrieve(
            query, top_k=candidate_k, filters=filt
        )
        bm25_response = self.bm25_retriever.retrieve(
            query, top_k=candidate_k, filters=filt
        )

        t_fuse_0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [dense_response.results, bm25_response.results],
            rrf_k=self.config.rrf_k,
        )
        selected = [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(fused[:k], start=1)
        ]
        t_fusion = (time.perf_counter() - t_fuse_0) * 1000.0
        total_latency = (time.perf_counter() - t0) * 1000.0

        dense_meta = dense_response.retrieval_metadata
        bm25_meta = bm25_response.retrieval_metadata

        status = "ok" if selected else "no_results"
        ret_meta = RetrievalMetadata(
            top_k=k,
            score_threshold=self.config.score_threshold,
            filters=filt,
            retriever_version=self.config.retriever_version,
            index_id=self.index_id,
            corpus_version=self.corpus_version,
            latency_ms=total_latency,
            search_latency_ms=dense_meta.search_latency_ms + bm25_meta.search_latency_ms,
            fusion_method="rrf",
            rrf_k=self.config.rrf_k,
            candidate_k=candidate_k,
            dense_candidate_count=len(dense_response.results),
            bm25_candidate_count=len(bm25_response.results),
            dense_latency_ms=dense_meta.latency_ms,
            bm25_latency_ms=bm25_meta.latency_ms,
            fusion_latency_ms=t_fusion,
        )

        return RetrievalResponse(
            query=query,
            results=selected,
            retrieval_method="hybrid",
            status=status,
            retrieval_metadata=ret_meta,
        )
