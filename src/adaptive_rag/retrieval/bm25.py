"""
retrieval.bm25
--------------
Okapi BM25 lexical retriever implementation.
Executes lexical queries over prebuilt BM25Index and returns structured RetrievalResponse.
Preserves native BM25 scores, ranks, metadata, and source provenance.
"""

import time
from typing import Any

from adaptive_rag.errors import InvalidQueryError
from adaptive_rag.indexing.bm25 import BM25Index
from adaptive_rag.schemas import (
    RetrievalConfig,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
)


class BM25Retriever:
    """Fixed-top-k lexical retriever querying BM25 inverted index."""

    method: str = "bm25"

    def __init__(
        self,
        index: BM25Index,
        config: RetrievalConfig | None = None,
        corpus_version: str = "corpus_v1",
        index_id: str = "adaptiverag_bm25_v1",
    ):
        self.index = index
        self.config = config or RetrievalConfig(retrieval_method="bm25", retriever_version="bm25_v1")
        self.corpus_version = corpus_version or index.corpus_version
        self.index_id = index_id

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Search BM25 inverted index and format structured response."""
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")

        k = top_k if top_k is not None else self.config.top_k
        thresh = score_threshold if score_threshold is not None else self.config.score_threshold
        filt = filters if filters is not None else self.config.filters

        t0 = time.perf_counter()

        # Execute lexical search on BM25Index
        t_search_0 = time.perf_counter()
        points = self.index.search(
            query=query,
            top_k=k,
            score_threshold=thresh,
            filters=filt,
        )
        t_search = (time.perf_counter() - t_search_0) * 1000.0
        total_latency = (time.perf_counter() - t0) * 1000.0

        # Format results preserving ranks, scores, metadata, and provenance
        results: list[RetrievalResult] = []
        for rank, p in enumerate(points, start=1):
            results.append(
                RetrievalResult(
                    chunk_id=p.chunk_id,
                    text=p.text,
                    score=p.score,
                    rank=rank,
                    metadata=p.metadata,
                    provenance=p.provenance,
                )
            )

        status = "ok" if results else "no_results"
        ret_meta = RetrievalMetadata(
            top_k=k,
            score_threshold=thresh,
            filters=filt,
            retriever_version=self.config.retriever_version,
            index_id=self.index_id,
            corpus_version=self.corpus_version,
            latency_ms=total_latency,
            search_latency_ms=t_search,
            k1=self.index.k1,
            b=self.index.b,
        )

        return RetrievalResponse(
            query=query,
            results=results,
            retrieval_method="bm25",
            status=status,
            retrieval_metadata=ret_meta,
        )
