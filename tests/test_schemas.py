"""
test_schemas.py
--------------
Unit tests for data contracts, round-trip serialization, and schema constraints.
"""

from pydantic import ValidationError
import pytest

from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingConfig,
    ChunkingMetadata,
    Document,
    DocumentMetadata,
    Element,
    ElementContent,
    ExtractionReport,
    Page,
    Provenance,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
    Section,
)


def test_element_and_document_roundtrip():
    prov = Provenance(document_id="doc1", page=1, bbox=(10.0, 20.0, 100.0, 50.0))
    el = Element(
        element_id="doc1::p001::e0000",
        type="paragraph",
        position=0,
        content=ElementContent(text="Hello world"),
        provenance=prov,
    )
    page = Page(page_number=1, element_ids=[el.element_id], text_char_count=11)
    sec = Section(
        section_id="doc1::s0000",
        title="Introduction",
        path=["1. Introduction"],
        level=1,
        element_ids=[el.element_id],
        provenance=prov,
    )
    meta = DocumentMetadata(
        document_id="doc1",
        title="Test Paper",
        authors=["Alice", "Bob"],
        year=2024,
        category="foundational_rag",
        source_url="https://example.com",
        download_url="https://example.com/pdf",
        local_path="data/raw/doc1.pdf",
        description="A paper",
        concepts=["rag"],
        source_sha256="abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
        source_size_bytes=100000,
    )
    report = ExtractionReport(
        ingestion_version="v1",
        config_hash="cfg123",
        page_count=1,
        element_counts={"paragraph": 1},
        section_count=1,
        issues=[],
    )
    doc = Document(
        document_id="doc1",
        metadata=meta,
        pages=[page],
        sections=[sec],
        elements=[el],
        extraction_report=report,
    )

    raw_json = doc.model_dump_json()
    loaded = Document.model_validate_json(raw_json)
    assert loaded.document_id == "doc1"
    assert len(loaded.elements) == 1
    assert loaded.elements[0].content.text == "Hello world"
    assert loaded.sections[0].title == "Introduction"


def test_chunk_cannot_have_extra_fields():
    meta = ChunkMetadata(
        document_id="doc1",
        doc_title="Paper",
        section_path=["1. Intro"],
        headings=["Intro"],
        element_ids=["doc1::p001::e0000"],
        element_types=["paragraph"],
        token_count=10,
        char_count=45,
    )
    prov = ChunkProvenance(document_id="doc1", pages=[1], source_sha256="hash123")
    c_meta = ChunkingMetadata(chunking_version="v1", config_hash="cfg123", ordinal=0)

    # Valid chunk
    chunk = Chunk(
        chunk_id="doc1::v1::c00000",
        document_id="doc1",
        text="Sample chunk text",
        metadata=meta,
        provenance=prov,
        chunking_metadata=c_meta,
    )
    assert chunk.chunk_id == "doc1::v1::c00000"

    # Extra forbidden field should raise ValidationError
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="doc1::v1::c00000",
            document_id="doc1",
            text="text",
            metadata=meta,
            provenance=prov,
            chunking_metadata=c_meta,
            embedding=[0.1, 0.2],  # explicitly forbidden in Chunk model
        )


def test_retrieval_response_preserves_provenance():
    meta = ChunkMetadata(
        document_id="doc1",
        doc_title="Paper",
        section_path=["1. Intro"],
        headings=["Intro"],
        element_ids=["e1"],
        element_types=["paragraph"],
        token_count=10,
        char_count=45,
    )
    prov = ChunkProvenance(document_id="doc1", pages=[1], source_sha256="hash123")
    res = RetrievalResult(
        chunk_id="c1",
        text="text",
        score=0.88,
        rank=1,
        metadata=meta,
        provenance=prov,
    )
    resp = RetrievalResponse(
        query="what is rag?",
        results=[res],
        retrieval_metadata=RetrievalMetadata(
            top_k=5,
            retriever_version="dense_v1",
            embedding_model_id="text-embedding-3-large",
            embedding_dim=3072,
            index_id="idx1",
            collection="col1",
            corpus_version="corp1",
            latency_ms=15.2,
            query_embedding_latency_ms=10.1,
            search_latency_ms=5.1,
        ),
    )
    assert len(resp.results) == 1
    assert resp.results[0].provenance.source_sha256 == "hash123"
    assert resp.status == "ok"
