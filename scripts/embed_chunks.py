#!/usr/bin/env python3
"""
embed_chunks.py
---------------
CLI to execute embedding of chunk artifacts using AICredits text-embedding-3-large.
Uses SQLite cache to guarantee zero repeated API calls on unchanged chunks.
"""

import argparse
import logging
import sys
from pathlib import Path

from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.paths import CHUNKS_DIR, EMBEDDINGS_DIR
from adaptive_rag.embeddings.aicredits import AICreditsEmbeddingModel
from adaptive_rag.embeddings.cache import EmbeddingCache
from adaptive_rag.embeddings.pipeline import EmbeddingPipeline
from adaptive_rag.schemas import EmbeddingConfig

logger = logging.getLogger("embed_chunks")


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunks using AICredits text-embedding-3-large.")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNKS_DIR,
        help="Directory containing chunk JSONL files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help="Directory to persist embeddings JSONL files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding requests",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional specific document_id to embed",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Initializing Embedding Pipeline (AICredits text-embedding-3-large)...")

    config = EmbeddingConfig(batch_size=args.batch_size)
    model = AICreditsEmbeddingModel(config=config)
    cache = EmbeddingCache()
    pipeline = EmbeddingPipeline(model=model, cache=cache, config=config)

    logger.info("Cache contains %d cached vectors currently", cache.count())
    summary = pipeline.process_corpus_embeddings(
        chunks_dir=args.chunks_dir,
        out_dir=args.out_dir,
        only_id=args.only,
    )

    total_embedded = sum(summary.values())
    logger.info("=" * 60)
    logger.info("Embedding Summary: Embedded %d chunks across %d documents", total_embedded, len(summary))
    logger.info("Cache now contains %d total vectors", cache.count())
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
