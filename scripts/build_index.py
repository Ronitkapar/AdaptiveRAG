#!/usr/bin/env python3
"""
build_index.py
--------------
CLI to build the local Qdrant vector index from chunk and embedding JSONL artifacts.
Guarantees index configuration matching and records index metadata.
"""

import argparse
import logging
import sys
from pathlib import Path

from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.paths import CHUNKS_DIR, EMBEDDINGS_DIR
from adaptive_rag.indexing.pipeline import IndexingPipeline
from adaptive_rag.schemas import IndexConfig

logger = logging.getLogger("build_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Qdrant index from chunk and embedding artifacts.")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNKS_DIR,
        help="Directory containing chunk JSONL files",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help="Directory containing embedding JSONL files",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=(
            "Qdrant collection name. Defaults to IndexConfig.collection_name, "
            "which is the namespace the retrieval code reads -- a collection "
            "named here but not there produces an index nothing queries."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Force recreate collection if it already exists",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Initializing Indexing Pipeline (Qdrant vector store)...")

    config = IndexConfig(collection_name=args.collection or IndexConfig().collection_name)
    pipeline = IndexingPipeline(config=config)

    total_upserted = pipeline.build_index_from_artifacts(
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        recreate=args.recreate,
    )

    logger.info("=" * 60)
    logger.info(
        "Indexing Summary: Upserted %d points into collection '%s'",
        total_upserted,
        config.collection_name,
    )
    logger.info("Verified collection count: %d", pipeline.vector_store.count())
    logger.info("=" * 60)
    pipeline.vector_store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
