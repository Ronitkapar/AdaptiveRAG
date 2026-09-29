"""
retrieval.fusion
----------------
Reciprocal Rank Fusion (RRF), isolated from orchestration so the ranking
mathematics can be unit-tested without any retriever, index, or credential.

RRF combines rankings, never scores: only `1 / (rrf_k + rank)` terms are
summed, so incomparable score scales never meet. Deduplication is by
`chunk_id`, never by text.
"""

from typing import Sequence

from adaptive_rag.errors import ConfigurationError
from adaptive_rag.schemas import RetrievalResult

# Rank used for chunks that have no contribution from any list; never a real rank.
_MISSING_RANK = 1 << 30


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse ranked result lists with Reciprocal Rank Fusion (rank-based only).

    The first list is the preferred payload carrier: when a chunk appears in
    several lists, its `text` / `metadata` / `provenance` are taken from the
    first list that contains it. Only `score` is replaced, by the fused RRF
    score. `rank` is left to the caller, which assigns final positions after
    truncation.

    Ordering is deterministic: `(-rrf_score, best_single_source_rank, chunk_id)`.
    `chunk_id` is the final tie-break, so nothing depends on input or dict
    iteration order.
    """
    if rrf_k < 0:
        raise ConfigurationError("rrf_k must be non-negative")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    source: dict[str, RetrievalResult] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            cid = result.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if rank < best_rank.get(cid, _MISSING_RANK):
                best_rank[cid] = rank
            # The first list always wins as payload carrier; otherwise keep
            # whichever result was seen first.
            if cid not in source or results is ranked_lists[0]:
                source[cid] = result

    fused = [
        source[cid].model_copy(update={"score": score}) for cid, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r.score, best_rank[r.chunk_id], r.chunk_id))
    return fused
