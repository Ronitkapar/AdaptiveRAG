"""
test_indexing.py
----------------
Unit tests for QdrantVectorStore in in-memory mode.
Verifies upsert, cosine search, rank ordering, and dimension mismatch guards.
"""

import pytest
from qdrant_client import QdrantClient

from adaptive_rag.errors import IndexConfigMismatchError
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingMetadata,
    IndexConfig,
)


@pytest.fixture
def in_memory_store():
    client = QdrantClient(location=":memory:")
    config = IndexConfig(collection_name="test_collection")
    store = QdrantVectorStore(config=config, client=client)
    store.ensure_collection(dim=4, distance="cosine")
    return store


def test_upsert_and_search_contract(in_memory_store):
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            document_id="doc1",
            text=f"Sample text {i}",
            metadata=ChunkMetadata(
                document_id="doc1",
                doc_title="Paper Title",
                section_path=["1. Intro"],
                headings=["Intro"],
                element_ids=[f"e{i}"],
                element_types=["paragraph"],
                token_count=10,
                char_count=30,
            ),
            provenance=ChunkProvenance(document_id="doc1", pages=[1], source_sha256="hash"),
            chunking_metadata=ChunkingMetadata(chunking_version="v1", config_hash="cfg", ordinal=i),
        )
        for i in range(3)
    ]

    # Orthogonal vectors
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    upserted = in_memory_store.upsert(chunks, vectors)
    assert upserted == 3
    assert in_memory_store.count() == 3

    # Query matching vector 0
    results = in_memory_store.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c0"
    assert abs(results[0].score - 1.0) < 1e-4
    assert results[0].payload["text"] == "Sample text 0"


def test_dimension_mismatch_raises_guard_error():
    client = QdrantClient(location=":memory:")
    config = IndexConfig(collection_name="dim_guard_col")
    store = QdrantVectorStore(config=config, client=client)

    # First setup with dim 4
    store.ensure_collection(dim=4, distance="cosine")

    # Second setup with dim 8 should raise IndexConfigMismatchError
    with pytest.raises(IndexConfigMismatchError):
        store.ensure_collection(dim=8, distance="cosine", recreate=False)
