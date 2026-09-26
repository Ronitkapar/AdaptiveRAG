"""
test_retrieval.py
-----------------
Unit tests for DenseRetriever.
Verifies query embedding, vector search, rank preservation, and error handling.
"""

import pytest
from qdrant_client import QdrantClient

from adaptive_rag.errors import InvalidQueryError
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.retrieval.dense import DenseRetriever
from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingMetadata,
    IndexConfig,
    RetrievalConfig,
)
from tests.fakes import FakeEmbeddingModel


@pytest.fixture
def retriever_fixture():
    dim = 8
    model = FakeEmbeddingModel(dimension=dim)
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(config=IndexConfig(collection_name="retrieval_col"), client=client)
    store.ensure_collection(dim=dim, distance="cosine")

    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            document_id="doc1",
            text=f"Important research concept number {i}",
            metadata=ChunkMetadata(
                document_id="doc1",
                doc_title="Research Paper",
                section_path=["2. Methods"],
                headings=["Methods"],
                element_ids=[f"e{i}"],
                element_types=["paragraph"],
                token_count=10,
                char_count=35,
            ),
            provenance=ChunkProvenance(document_id="doc1", pages=[1], source_sha256="hash123"),
            chunking_metadata=ChunkingMetadata(chunking_version="v1", config_hash="cfg", ordinal=i),
        )
        for i in range(4)
    ]
    vectors = [model.embed_query(c.text) for c in chunks]
    store.upsert(chunks, vectors)

    retriever = DenseRetriever(
        embedding_model=model,
        vector_store=store,
        config=RetrievalConfig(top_k=2),
    )
    return retriever


def test_dense_retriever_returns_ranked_results(retriever_fixture):
    resp = retriever_fixture.retrieve("research concept number 0", top_k=2)

    assert resp.status == "ok"
    assert len(resp.results) == 2
    assert resp.results[0].rank == 1
    assert resp.results[1].rank == 2
    assert resp.results[0].score >= resp.results[1].score
    assert resp.results[0].chunk_id == "c0"
    assert resp.results[0].provenance.source_sha256 == "hash123"
    assert resp.retrieval_metadata.top_k == 2
    assert resp.retrieval_metadata.latency_ms > 0


def test_empty_query_raises_invalid_query_error(retriever_fixture):
    with pytest.raises(InvalidQueryError):
        retriever_fixture.retrieve("   ")
