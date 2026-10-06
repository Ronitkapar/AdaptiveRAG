"""
ingestion.pipeline
------------------
High-level ingestion pipeline processing raw PDFs into canonical Documents.
Validates inputs, manages deterministic document artifact persistence, and generates audit reports.
"""

import json
from pathlib import Path
from typing import Any

from adaptive_rag.config.hashing import canonical_json, compute_file_sha256
from adaptive_rag.config.paths import DOCUMENTS_DIR, PAPERS_MANIFEST_PATH
from adaptive_rag.errors import CorpusIntegrityError, PdfExtractionError
from adaptive_rag.ingestion.extractor import PdfplumberExtractor
from adaptive_rag.ingestion.normalizer import normalize_pages_to_document
from adaptive_rag.schemas import Document, IngestionConfig


class IngestionPipeline:
    """Orchestrates PDF validation, raw extraction, normalization, and artifact persistence."""

    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()
        self.extractor = PdfplumberExtractor()

    def ingest_pdf(self, pdf_path: Path, manifest_meta: dict[str, Any]) -> Document:
        """Process a single PDF into a hierarchical Document."""
        if not pdf_path.is_file():
            raise PdfExtractionError(f"PDF file not found: {pdf_path}")

        # Check hash against manifest if present
        expected_sha = manifest_meta.get("sha256")
        if expected_sha:
            actual_sha = compute_file_sha256(pdf_path)
            if actual_sha != expected_sha:
                raise CorpusIntegrityError(
                    f"SHA256 mismatch for {pdf_path}: expected {expected_sha}, got {actual_sha}"
                )

        doc_id = manifest_meta["document_id"]
        try:
            raw_pages, pdf_info = self.extractor.extract_document(pdf_path)
        except Exception as exc:
            raise PdfExtractionError(f"Failed extracting PDF {pdf_path}: {exc}") from exc

        return normalize_pages_to_document(
            document_id=doc_id,
            raw_pages=raw_pages,
            pdf_info=pdf_info,
            manifest_meta=manifest_meta,
            config=self.config,
        )

    def save_document(self, doc: Document, out_dir: Path | None = None) -> Path:
        """Persist canonical Document JSON without non-deterministic metadata."""
        target_dir = out_dir or DOCUMENTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / f"{doc.document_id}.json"

        # Canonical json dump
        data = doc.model_dump(mode="json")
        dest_path.write_text(canonical_json(data), encoding="utf-8")
        return dest_path

    def ingest_corpus(
        self,
        manifest_path: Path = PAPERS_MANIFEST_PATH,
        out_dir: Path = DOCUMENTS_DIR,
        limit: int | None = None,
        only_id: str | None = None,
    ) -> list[Document]:
        """Ingest all or a subset of papers defined in the corpus manifest."""
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results: list[Document] = []

        papers_to_process = manifest
        if only_id:
            papers_to_process = [p for p in papers_to_process if p.get("document_id") == only_id]
        if limit:
            papers_to_process = papers_to_process[:limit]

        out_dir.mkdir(parents=True, exist_ok=True)
        summary_entries = []

        for paper in papers_to_process:
            doc_id = paper["document_id"]
            pdf_path = Path(paper["local_path"])
            doc = self.ingest_pdf(pdf_path=pdf_path, manifest_meta=paper)
            self.save_document(doc, out_dir=out_dir)
            results.append(doc)

            summary_entries.append({
                "document_id": doc_id,
                "pages": len(doc.pages),
                "sections": len(doc.sections),
                "elements": len(doc.elements),
                "element_counts": doc.extraction_report.element_counts,
                "issue_count": len(doc.extraction_report.issues),
            })

        # The summary is written beside the documents it describes, not to a fixed
        # location. Phase 8 ingests into its own directory while the Phase 7
        # artifacts stay in place as the study's "before" arm, and a summary
        # describing one corpus must not overwrite the other's.
        manifest_summary_path = out_dir.parent / "documents_manifest.json"
        manifest_summary_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_summary_path.write_text(
            canonical_json({
                "ingestion_version": self.config.ingestion_version,
                "document_count": len(results),
                "documents": summary_entries,
            }),
            encoding="utf-8",
        )

        return results
