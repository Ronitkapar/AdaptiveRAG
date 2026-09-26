"""
test_context_builder.py
-----------------------
Unit tests for ContextBuilder.
Verifies token budgets, whole-chunk retention, [Source N] formatting, and dropped chunk auditing.
"""

from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.schemas import (
    ChunkMetadata,
    ChunkProvenance,
    ContextConfig,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
)


def test_context_builder_formats_and_limits_chunks():
    results = []
    for i in range(1, 6):
        meta = ChunkMetadata(
            document_id="doc1",
            doc_title="Paper Title",
            section_path=["3. Evaluation"],
            headings=["Evaluation"],
            element_ids=[f"e{i}"],
            element_types=["paragraph"],
            token_count=50,
            char_count=200,
        )
        prov = ChunkProvenance(document_id="doc1", pages=[i], source_sha256="hash")
        results.append(
            RetrievalResult(
                chunk_id=f"c{i}",
                text=f"Passage text content for rank {i}",
                score=1.0 - (i * 0.1),
                rank=i,
                metadata=meta,
                provenance=prov,
            )
        )

    response = RetrievalResponse(
        query="what is evaluation?",
        results=results,
        retrieval_metadata=RetrievalMetadata(
            top_k=5,
            retriever_version="dense_v1",
            embedding_model_id="emb",
            embedding_dim=8,
            index_id="idx",
            collection="col",
            corpus_version="v1",
            latency_ms=10.0,
            query_embedding_latency_ms=5.0,
            search_latency_ms=5.0,
        ),
    )

    # Max chunks = 3
    builder = ContextBuilder(context_config=ContextConfig(max_chunks=3))
    req = builder.build(response)

    assert len(req.context) == 3
    assert req.context[0].source_index == 1
    assert req.context[0].chunk_id == "c1"
    assert req.context[2].source_index == 3
    assert req.context[2].chunk_id == "c3"
    assert "[Source 1" in req.context[0].text
    assert req.metadata["dropped_chunks"] == 2
    assert "c4" in req.metadata["dropped_chunk_ids"]
    assert "c5" in req.metadata["dropped_chunk_ids"]
