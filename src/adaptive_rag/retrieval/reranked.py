"""
retrieval.reranked
------------------
Phase 5 second-stage retriever: wraps any first-stage `Retriever` with a
`Reranker` so the candidate ordering is re-scored by a cross-encoder.

This module only orchestrates. It performs no tokenization, no embedding, and no
index access; the base retriever owns candidate generation and the reranker owns
scoring. Text, chunk metadata, and provenance are carried through untouched —
the reranker contributes exactly one number per chunk.
"""

import time
from typing import Any, Sequence

from adaptive_rag.errors import InvalidQueryError, RerankingError
from adaptive_rag.reranking.base import Reranker
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.schemas import (
    RetrievalConfig,
    RetrievalResponse,
    RetrievalResult,
)

# First-stage method -> second-stage pipeline identity. Kept explicit so a
# manifest's retrieval_method names the full pipeline rather than just its head.
RERANKED_METHODS = {
    "dense": "dense_rerank",
    "bm25": "bm25_rerank",
    "hybrid": "hybrid_rerank",
}


def apply_rerank_scores(
    results: Sequence[RetrievalResult],
    scores: Sequence[float],
    top_k: int,
) -> list[RetrievalResult]:
    """Reorder first-stage results by rerank score and truncate to `top_k`.

    The sort key is `(-score, retrieval_rank, chunk_id)`: the pre-rerank rank and
    then the chunk id make the order total, so equal scores never depend on input
    order and repeated runs are stable.
    """
    scored = [
        result.model_copy(
            update={
                "score": float(score),
                "retrieval_score": result.score,
                "retrieval_rank": result.rank,
            }
        )
        for result, score in zip(results, scores, strict=True)
    ]
    scored.sort(key=lambda r: (-r.score, r.retrieval_rank or 0, r.chunk_id))
    return [
        result.model_copy(update={"rank": rank})
        for rank, result in enumerate(scored[:top_k], start=1)
    ]


class RerankedRetriever:
    """Second-stage retriever that re-scores a base retriever's candidates."""

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        config: RetrievalConfig | None = None,
        corpus_version: str = "corpus_v1",
        index_id: str | None = None,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.config = config or RetrievalConfig(
            retrieval_method="dense_rerank", retriever_version="dense_rerank_v1"
        )
        self.corpus_version = corpus_version
        self.method = RERANKED_METHODS.get(base_retriever.method, "")
        self.index_id = index_id or f"{getattr(base_retriever, 'index_id', 'base')}_rerank"

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Generate candidates at rerank depth, re-score them, and truncate.

        `score_threshold` is deliberately not forwarded: a first-stage score
        threshold and a second-stage ranker are different concerns, and applying
        one here would silently shrink the candidate pool before the ranker ever
        sees it. The configured value is echoed into
        `RetrievalMetadata.score_threshold` for traceability only. `filters` are
        forwarded to the base retriever.
        """
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")

        if not self.method:
            raise RerankingError(
                f"unsupported base retrieval method for second-stage scoring: "
                f"{self.base_retriever.method!r}"
            )

        k = top_k if top_k is not None else self.config.top_k
        candidate_k = self.config.rerank_candidate_k
        filt = filters if filters is not None else self.config.filters

        base_response = self.base_retriever.retrieve(
            query, top_k=candidate_k, score_threshold=None, filters=filt
        )
        candidates = base_response.results
        candidate_latency = base_response.retrieval_metadata.latency_ms

        # An empty candidate pool is a normal outcome, not a failure: no model
        # invocation happens at all, and no reranking latency is attributed.
        if not candidates:
            return self._build_response(
                query=query,
                results=[],
                base_response=base_response,
                top_k=k,
                candidate_k=candidate_k,
                filters=filt,
                candidate_latency=candidate_latency,
                rerank_latency=0.0,
                fallback=False,
            )

        t_rerank_0 = time.perf_counter()
        try:
            scores = list(self.reranker.score(query, [c.text for c in candidates]))
        except Exception:
            if not self.config.rerank_fallback:
                raise
            # Opt-in degradation: return un-reordered candidates, truncated to
            # top_k, with `score` left as the first-stage score so downstream
            # consumers see one consistent ranking. Marked in every trace.
            fallback_results = [
                result.model_copy(
                    update={
                        "rank": rank,
                        "retrieval_score": result.score,
                        "retrieval_rank": result.rank,
                    }
                )
                for rank, result in enumerate(candidates[:k], start=1)
            ]
            return self._build_response(
                query=query,
                results=fallback_results,
                base_response=base_response,
                top_k=k,
                candidate_k=candidate_k,
                filters=filt,
                candidate_latency=candidate_latency,
                rerank_latency=(time.perf_counter() - t_rerank_0) * 1000.0,
                fallback=True,
            )
        rerank_latency = (time.perf_counter() - t_rerank_0) * 1000.0

        if len(scores) != len(candidates):
            raise RerankingError(
                f"reranker returned {len(scores)} scores for "
                f"{len(candidates)} candidates"
            )

        selected = apply_rerank_scores(candidates, scores, k)
        return self._build_response(
            query=query,
            results=selected,
            base_response=base_response,
            top_k=k,
            candidate_k=candidate_k,
            filters=filt,
            candidate_latency=candidate_latency,
            rerank_latency=rerank_latency,
            fallback=False,
        )

    def _build_response(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
        base_response: RetrievalResponse,
        top_k: int,
        candidate_k: int,
        filters: dict[str, Any] | None,
        candidate_latency: float,
        rerank_latency: float,
        fallback: bool,
    ) -> RetrievalResponse:
        """Assemble the response, carrying the base metadata through unchanged."""
        base_meta = base_response.retrieval_metadata
        metadata = base_meta.model_copy(
            update={
                "top_k": top_k,
                "score_threshold": self.config.score_threshold,
                "filters": filters,
                "retriever_version": self.config.retriever_version,
                "index_id": self.index_id,
                "corpus_version": self.corpus_version,
                "latency_ms": candidate_latency + rerank_latency,
                "rerank_enabled": True,
                "reranker_version": self.reranker.version,
                "reranker_model_id": self.reranker.model_id,
                "reranker_device": getattr(self.reranker, "device", None),
                "rerank_candidate_k": candidate_k,
                "rerank_top_k": top_k,
                "candidate_generation_latency_ms": candidate_latency,
                "rerank_latency_ms": rerank_latency,
                "candidate_count": len(base_response.results),
                "result_count": len(results),
                "rerank_fallback": fallback,
            }
        )
        return RetrievalResponse(
            query=query,
            results=results,
            retrieval_method=self.method,  # type: ignore[arg-type]
            status="ok" if results else "no_results",
            retrieval_metadata=metadata,
        )