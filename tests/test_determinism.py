"""
tests.test_determinism
----------------------
Byte-stability guarantees: same PDF + same config always produce the same
Document, chunks, cache keys, and experiment configuration hashes.
"""

import json

from adaptive_rag.chunking.structure_aware import StructureAwareChunker
from adaptive_rag.config.hashing import canonical_json
from adaptive_rag.embeddings.cache import derive_cache_key
from adaptive_rag.experiments import build_experiment_config
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.schemas import IngestionConfig


def test_ingestion_is_byte_identical(fixture_pdf_path):
    pipeline = IngestionPipeline(IngestionConfig())
    meta = {
        "document_id": "det_doc",
        "title": "Determinism Check",
        "authors": ["A"],
        "year": 2024,
        "category": "dense_retrieval",
        "source_url": "https://example.com",
        "download_url": "https://example.com/pdf",
        "local_path": str(fixture_pdf_path),
        "description": "d",
        "concepts": ["c"],
        "sha256": None,
        "size_bytes": 1000,
    }
    first = pipeline.ingest_pdf(fixture_pdf_path, manifest_meta=meta)
    second = pipeline.ingest_pdf(fixture_pdf_path, manifest_meta=meta)
    assert canonical_json(first.model_dump(mode="json")) == canonical_json(
        second.model_dump(mode="json")
    )


def test_chunking_is_byte_identical(fixture_pdf_path):
    pipeline = IngestionPipeline(IngestionConfig())
    meta = {
        "document_id": "det_doc",
        "title": "Determinism Check",
        "authors": ["A"],
        "year": 2024,
        "category": "dense_retrieval",
        "source_url": "https://example.com",
        "download_url": "https://example.com/pdf",
        "local_path": str(fixture_pdf_path),
        "description": "d",
        "concepts": ["c"],
        "sha256": None,
        "size_bytes": 1000,
    }
    document = pipeline.ingest_pdf(fixture_pdf_path, manifest_meta=meta)
    chunker = StructureAwareChunker()

    first = [c.model_dump(mode="json") for c in chunker.chunk_document(document)]
    second = [c.model_dump(mode="json") for c in chunker.chunk_document(document)]
    assert first == second
    assert [c["chunk_id"] for c in first] == sorted(c["chunk_id"] for c in first)


def test_cache_keys_and_experiment_hashes_are_stable():
    key_a = derive_cache_key("hello", "m", "h", None, True, "v1")
    key_b = derive_cache_key("hello", "m", "h", None, True, "v1")
    key_c = derive_cache_key("hello", "m", "h", None, True, "v2")
    assert key_a == key_b
    assert key_a != key_c

    first = build_experiment_config(corpus_version="corpus_test")
    second = build_experiment_config(corpus_version="corpus_test")
    assert first.config_hash == second.config_hash

    payload = json.loads(canonical_json(first.model_dump(mode="json")))
    assert payload["config_hash"] == first.config_hash
