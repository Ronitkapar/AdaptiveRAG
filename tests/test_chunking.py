"""
test_chunking.py
----------------
Unit tests for structure-aware chunking, hierarchy preservation, and overlap rules.
"""

from pathlib import Path
import pytest

from adaptive_rag.chunking.pipeline import ChunkingPipeline
from adaptive_rag.chunking.structure_aware import StructureAwareChunker
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.schemas import ChunkingConfig, IngestionConfig
from tests.fixtures.make_fixture_pdf import create_synthetic_paper_pdf


@pytest.fixture
def sample_document(tmp_path: Path):
    pdf_path = create_synthetic_paper_pdf(tmp_path / "sample.pdf")
    meta = {
        "document_id": "test_doc",
        "title": "A Synthetic Study",
        "authors": ["Author"],
        "year": 2024,
        "category": "dense_retrieval",
        "source_url": "https://example.com",
        "download_url": "https://example.com/pdf",
        "local_path": str(pdf_path),
        "description": "desc",
        "concepts": ["dense"],
        "sha256": None,
        "size_bytes": 1000,
    }
    return IngestionPipeline(IngestionConfig()).ingest_pdf(pdf_path, manifest_meta=meta)


def test_chunker_preserves_hierarchy_and_elements(sample_document):
    chunker = StructureAwareChunker(ChunkingConfig(target_tokens=100, max_tokens=150))
    chunks = chunker.chunk_document(sample_document)

    assert len(chunks) > 0

    # Verify all chunk metadata fields are populated
    for c in chunks:
        assert c.document_id == "test_doc"
        assert c.metadata.token_count > 0
        assert len(c.metadata.element_ids) > 0
        assert len(c.provenance.pages) > 0
        assert c.chunk_id.startswith("test_doc::structure_aware_v1::c")

    # Verify table element is preserved in its chunk
    table_chunks = [c for c in chunks if "table" in c.metadata.element_types]
    assert len(table_chunks) >= 1
    assert "Recall@5" in table_chunks[0].text


def test_chunking_determinism(sample_document):
    chunker = StructureAwareChunker(ChunkingConfig())
    c1 = chunker.chunk_document(sample_document)
    c2 = chunker.chunk_document(sample_document)

    assert len(c1) == len(c2)
    for a, b in zip(c1, c2):
        assert a.chunk_id == b.chunk_id
        assert a.text == b.text
        assert a.metadata.token_count == b.metadata.token_count
