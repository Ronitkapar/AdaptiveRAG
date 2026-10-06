#!/usr/bin/env python3
"""
build_bm25_index.py
-------------------
CLI to build the Okapi BM25 lexical index from the canonical chunk corpus.
No credentials or external services required. Operates directly on chunk artifacts.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.paths import BM25_INDEX_PATH, CHUNKS_DIR
from adaptive_rag.experiments.config import compute_corpus_version
from adaptive_rag.indexing.bm25 import BM25Index
from adaptive_rag.schemas import Chunk, IndexConfig

logger = logging.getLogger("build_bm25_index")


def load_canonical_chunks(chunks_dir: Path = CHUNKS_DIR) -> list[Chunk]:
    """Read all canonical chunk JSONL files in chunks_dir."""
    chunk_files = sorted(list(chunks_dir.glob("*.chunks.jsonl")))
    chunks: list[Chunk] = []
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 index from canonical chunk corpus.")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNKS_DIR,
        help="Directory containing canonical chunk JSONL files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=BM25_INDEX_PATH,
        help="Path where BM25 index JSON should be written",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.2,
        help="BM25 k1 parameter (term saturation)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (document length normalization)",
    )
    parser.add_argument(
        "--corpus-arm",
        type=str,
        default=IndexConfig().corpus_arm,
        help=(
            "Corpus arm recorded in the index and checked on load. The Phase 8 "
            "arms share a corpus version, so this is the only thing that "
            "distinguishes an index built from the before arm from one built "
            "from the after arm."
        ),
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Reading canonical chunks from %s...", args.chunks_dir)
    chunks = load_canonical_chunks(args.chunks_dir)
    if not chunks:
        logger.error("No chunks found in %s! Run scripts/build_chunks.py first.", args.chunks_dir)
        return 1

    corpus_ver = compute_corpus_version()
    logger.info(
        "Found %d chunks across canonical corpus (corpus_version=%s, arm=%s)",
        len(chunks), corpus_ver, args.corpus_arm,
    )
    logger.info("Building BM25Index (k1=%.2f, b=%.2f)...", args.k1, args.b)

    index = BM25Index(
        k1=args.k1, b=args.b, corpus_version=corpus_ver, corpus_arm=args.corpus_arm
    )
    total_docs = index.build_from_chunks(
        chunks, corpus_version=corpus_ver, corpus_arm=args.corpus_arm
    )

    logger.info("Saving BM25 index to %s...", args.output_path)
    index.save(args.output_path)

    logger.info("=" * 60)
    logger.info("BM25 Indexing Complete!")
    logger.info("Total indexed documents: %d", total_docs)
    logger.info("Average document length: %.2f tokens", index.avgdl)
    logger.info("Vocabulary size: %d terms", len(index.inverted_index))
    logger.info("Output file size: %.2f KB", args.output_path.stat().st_size / 1024.0)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
