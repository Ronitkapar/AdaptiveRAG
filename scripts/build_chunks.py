#!/usr/bin/env python3
"""
build_chunks.py
---------------
CLI to execute structure-aware chunking over ingested Documents.
Persists deterministic chunk JSONL files and manifest statistics.
"""

import argparse
import logging
import sys
from pathlib import Path

from adaptive_rag.chunking.pipeline import ChunkingPipeline
from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.paths import CHUNKS_DIR, DOCUMENTS_DIR
from adaptive_rag.schemas import ChunkingConfig

logger = logging.getLogger("build_chunks")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structure-aware chunks from canonical Documents.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCUMENTS_DIR,
        help="Directory containing document JSON artifacts",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CHUNKS_DIR,
        help="Directory to save chunk JSONL artifacts",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=500,
        help="Target tokens per chunk",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum tokens before splitting a chunk",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=80,
        help="Token overlap carried from preceding chunk",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional specific document_id to chunk",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Initializing Chunking Pipeline (structure-aware)...")

    config = ChunkingConfig(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    pipeline = ChunkingPipeline(config=config)

    chunks_by_doc = pipeline.chunk_corpus(
        docs_dir=args.docs_dir,
        out_dir=args.out_dir,
        only_id=args.only,
    )

    total_chunks = sum(len(c) for c in chunks_by_doc.values())
    logger.info("=" * 60)
    logger.info("Chunking Summary: Generated %d chunks across %d documents", total_chunks, len(chunks_by_doc))
    for doc_id, doc_chunks in sorted(chunks_by_doc.items()):
        toks = [c.metadata.token_count for c in doc_chunks]
        avg_t = round(sum(toks) / len(toks), 1) if toks else 0
        logger.info(
            "  - %-24s: %3d chunks, total tokens=%5d, avg tokens=%3.1f",
            doc_id,
            len(doc_chunks),
            sum(toks),
            avg_t,
        )
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
