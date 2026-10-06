#!/usr/bin/env python3
"""
ingest_corpus.py
----------------
Command-line interface to execute the document ingestion pipeline.
Converts Phase 1 raw PDFs into canonical hierarchical Document JSON artifacts.
"""

import argparse
import logging
import sys
from pathlib import Path

from adaptive_rag.config.paths import DOCUMENTS_DIR, PAPERS_MANIFEST_PATH
from adaptive_rag.config.logging import setup_logging
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.schemas import IngestionConfig

logger = logging.getLogger("ingest_corpus")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest research corpus PDFs into Document objects.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PAPERS_MANIFEST_PATH,
        help="Path to papers.json corpus manifest",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DOCUMENTS_DIR,
        help="Directory to save canonical document JSON artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of papers to ingest",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional specific document_id to ingest",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with non-zero code if any document contains extraction issues",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Initializing Ingestion Pipeline (pdfplumber backend)...")

    config = IngestionConfig()
    pipeline = IngestionPipeline(config=config)

    docs = pipeline.ingest_corpus(
        manifest_path=args.manifest,
        out_dir=args.out,
        limit=args.limit,
        only_id=args.only,
    )

    logger.info("=" * 60)
    logger.info("Ingestion Summary: Processed %d documents", len(docs))
    total_issues = 0
    for doc in docs:
        issues_cnt = len(doc.extraction_report.issues)
        total_issues += issues_cnt
        logger.info(
            "  - %-24s: %3d pages, %2d sections, %4d elements, %2d issues",
            doc.document_id,
            len(doc.pages),
            len(doc.sections),
            len(doc.elements),
            issues_cnt,
        )
    logger.info("=" * 60)

    if args.fail_on_issues and total_issues > 0:
        logger.error("Failed: %d total extraction issues detected with --fail-on-issues enabled", total_issues)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
