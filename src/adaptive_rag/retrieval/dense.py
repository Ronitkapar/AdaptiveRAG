"""
retrieval.dense
---------------
Dense retriever implementation.
Embeds incoming query, searches Qdrant index, and returns structured RetrievalResponse.
Preserves similarity scores, ranks, and source provenance.
"""

import time
from typing import Any

from adaptive_rag.embeddings.base import EmbeddingModel
from adaptive_rag.errors import InvalidQueryError
from adaptive_rag.indexing.base import VectorStore
from adaptive_rag.schemas import (
    ChunkMetadata,
    ChunkProvenance,
    RetrievalConfig,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
)


class DenseRetriever:
    """Fixed-top-k dense retriever querying vector index."""

    method: str = "dense"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
        corpus_version: str = "corpus_v1",
        index_id: str = "adaptiverag_dense_v1",
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self.corpus_version = corpus_version
        self.index_id = index_id

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Embed query, search vector index, and format response."""
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")

        k = top_k if top_k is not None else self.config.top_k
        thresh = score_threshold if score_threshold is not None else self.config.score_threshold
        filt = filters if filters is not None else self.config.filters

        t0 = time.perf_counter()

        # 1. Query embedding
        t_emb_0 = time.perf_counter()
        query_vector = self.embedding_model.embed_query(query)
        t_emb = (time.perf_counter() - t_emb_0) * 1000.0

        # 2. Vector search
        t_search_0 = time.perf_counter()
        points = self.vector_store.search(
            vector=query_vector,
            top_k=k,
            score_threshold=thresh,
            filters=filt,
        )
        t_search = (time.perf_counter() - t_search_0) * 1000.0
        total_latency = (time.perf_counter() - t0) * 1000.0

        # 3. Format results preserving ranks and scores
        results: list[RetrievalResult] = []
        for rank, p in enumerate(points, start=1):
            payload = p.payload
            meta = ChunkMetadata(
                document_id=payload.get("document_id", ""),
                doc_title=payload.get("doc_title", ""),
                section_id=payload.get("section_id"),
                section_path=payload.get("section_path", []),
                headings=payload.get("headings", []),
                element_ids=payload.get("element_ids", []),
                element_types=payload.get("element_types", []),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                token_count=payload.get("token_count", 0),
                char_count=len(payload.get("text", "")),
            )
            prov = ChunkProvenance(
                document_id=payload.get("document_id", ""),
                pages=[p for p in (payload.get("page_start"), payload.get("page_end")) if p is not None],
                source_sha256=payload.get("source_sha256", ""),
            )
            results.append(
                RetrievalResult(
                    chunk_id=p.chunk_id,
                    text=payload.get("text", ""),
                    score=p.score,
                    rank=rank,
                    metadata=meta,
                    provenance=prov,
                )
            )

        status = "ok" if results else "no_results"
        ret_meta = RetrievalMetadata(
            top_k=k,
            score_threshold=thresh,
            filters=filt,
            retriever_version=self.config.retriever_version,
            embedding_model_id=self.embedding_model.model_id,
            embedding_dim=self.embedding_model.dimension,
            index_id=self.index_id,
            collection=getattr(self.vector_store, "collection_name", "default"),
            corpus_version=self.corpus_version,
            latency_ms=total_latency,
            query_embedding_latency_ms=t_emb,
            search_latency_ms=t_search,
        )

        return RetrievalResponse(
            query=query,
            results=results,
            retrieval_method="dense",
            status=status,
            retrieval_metadata=ret_meta,
        )
