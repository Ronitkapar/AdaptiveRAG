"""
test_ingestion.py
-----------------
Unit tests for PDF extraction, normalization, and document structure validation.
"""

from pathlib import Path
import pytest

from adaptive_rag.ingestion.extractor import PdfplumberExtractor
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.schemas import IngestionConfig
from tests.fixtures.make_fixture_pdf import create_synthetic_paper_pdf


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "synthetic_paper.pdf"
    return create_synthetic_paper_pdf(pdf_path)


def test_extractor_extracts_raw_primitives(synthetic_pdf: Path):
    extractor = PdfplumberExtractor()
    raw_pages, pdf_info = extractor.extract_document(synthetic_pdf)

    assert len(raw_pages) == 2
    p1 = raw_pages[0]
    assert p1.page_number == 1
    assert len(p1.chars) > 50

    p2 = raw_pages[1]
    assert p2.page_number == 2
    assert len(p2.tables) == 1
    table = p2.tables[0]
    assert len(table.rows) == 3
    assert table.rows[0] == ["Method", "Recall@5", "MRR@10"]


def test_ingestion_pipeline_produces_valid_document(synthetic_pdf: Path):
    pipeline = IngestionPipeline(IngestionConfig())
    manifest_meta = {
        "document_id": "synthetic_doc",
        "title": "A Synthetic Study on Retrieval Architectures",
        "authors": ["Test Author"],
        "year": 2024,
        "category": "dense_retrieval",
        "source_url": "https://example.com",
        "download_url": "https://example.com/pdf",
        "local_path": str(synthetic_pdf),
        "description": "Synthetic paper for tests",
        "concepts": ["dense retrieval", "eval"],
        "sha256": None,
        "size_bytes": 1000,
    }

    doc = pipeline.ingest_pdf(synthetic_pdf, manifest_meta=manifest_meta)
    assert doc.document_id == "synthetic_doc"
    assert len(doc.pages) == 2
    assert len(doc.sections) >= 3

    # Check headings extracted
    h_titles = [s.title for s in doc.sections]
    assert any("Introduction" in t for t in h_titles)
    assert any("Experimental Results" in t for t in h_titles)

    # Check table element exists
    tables = [el for el in doc.elements if el.type == "table"]
    assert len(tables) == 1
    assert "Recall@5" in tables[0].content.text

    # Verify provenance on every element
    for el in doc.elements:
        assert el.provenance.page in (1, 2)
        assert el.provenance.document_id == "synthetic_doc"


def test_ingestion_determinism(synthetic_pdf: Path):
    pipeline = IngestionPipeline(IngestionConfig())
    meta = {
        "document_id": "synthetic_doc",
        "title": "A Synthetic Study on Retrieval Architectures",
        "authors": ["Test Author"],
        "year": 2024,
        "category": "dense_retrieval",
        "source_url": "https://example.com",
        "download_url": "https://example.com/pdf",
        "local_path": str(synthetic_pdf),
        "description": "Synthetic paper for tests",
        "concepts": ["dense retrieval"],
        "sha256": None,
        "size_bytes": 1000,
    }

    doc1 = pipeline.ingest_pdf(synthetic_pdf, manifest_meta=meta)
    doc2 = pipeline.ingest_pdf(synthetic_pdf, manifest_meta=meta)

    # Byte-identical serialization
    json1 = doc1.model_dump_json()
    json2 = doc2.model_dump_json()
    assert json1 == json2
