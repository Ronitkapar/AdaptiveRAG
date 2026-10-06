"""
test_embedding_cache.py
-----------------------
Unit tests for content-addressed SQLite embedding cache.
Verifies key derivation, hits/misses, and zero repeated API calls.
"""

from pathlib import Path
import pytest

from adaptive_rag.embeddings.cache import EmbeddingCache, derive_cache_key
from adaptive_rag.embeddings.pipeline import EmbeddingPipeline
from adaptive_rag.schemas import Chunk, ChunkMetadata, ChunkProvenance, ChunkingMetadata
from tests.fakes import FakeEmbeddingModel


def test_cache_key_derivation_deterministic():
    k1 = derive_cache_key("hello world", "model1", "host1", 3072, True, "v1")
    k2 = derive_cache_key("hello world", "model1", "host1", 3072, True, "v1")
    assert k1 == k2

    # Different text -> different key
    k3 = derive_cache_key("hello world 2", "model1", "host1", 3072, True, "v1")
    assert k1 != k3


def test_cache_put_get_roundtrip(tmp_path: Path):
    db_path = tmp_path / "cache.sqlite3"
    cache = EmbeddingCache(db_path=db_path)

    vec = [0.1, 0.2, 0.3, -0.4]
    key = "sample_key_123"

    assert cache.get(key) is None
    cache.put(cache_key=key, vector=vec, chunk_id="c1", model_id="m1")

    cached_vec = cache.get(key)
    assert cached_vec is not None
    assert len(cached_vec) == 4
    for a, b in zip(vec, cached_vec):
        assert abs(a - b) < 1e-5


def test_pipeline_avoids_repeated_api_calls(tmp_path: Path):
    db_path = tmp_path / "cache.sqlite3"
    cache = EmbeddingCache(db_path=db_path)
    model = FakeEmbeddingModel(dimension=16)
    pipeline = EmbeddingPipeline(model=model, cache=cache)

    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            document_id="d1",
            text=f"Paragraph {i} content text",
            metadata=ChunkMetadata(
                document_id="d1",
                doc_title="T",
                section_path=["S"],
                headings=["H"],
                element_ids=[f"e{i}"],
                element_types=["paragraph"],
                token_count=10,
                char_count=30,
            ),
            provenance=ChunkProvenance(document_id="d1", pages=[1], source_sha256="h"),
            chunking_metadata=ChunkingMetadata(chunking_version="v1", config_hash="cfg", ordinal=i),
        )
        for i in range(5)
    ]

    # First run: all misses, model should be called 5 times
    v1 = pipeline.embed_chunks(chunks)
    assert len(v1) == 5
    assert model.call_count == 5

    # Second run: 100% cache hits, model call count should remain 5!
    v2 = pipeline.embed_chunks(chunks)
    assert len(v2) == 5
    assert model.call_count == 5
