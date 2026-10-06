"""
config package initialization.
"""

from adaptive_rag.config.hashing import canonical_json, compute_config_hash, compute_file_sha256, compute_sha256
from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.paths import (
    BM25_INDEX_DIR,
    BM25_INDEX_PATH,
    CHUNKS_DIR,
    DATA_DIR,
    DOCS_DIR,
    DOCUMENTS_DIR,
    EMBEDDINGS_CACHE_PATH,
    EMBEDDINGS_DIR,
    EVAL_CACHE_PATH,
    EVALUATION_DIR,
    EXPERIMENTS_DIR,
    METADATA_DIR,
    PAPERS_MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    QDRANT_DIR,
    RAW_DATA_DIR,
    REPO_ROOT,
    STATS_DIR,
    STORAGE_DIR,
    ensure_directories,
    get_repo_root,
)
from adaptive_rag.config.pricing import MODEL_PRICING, PRICING_VERSION, calculate_cost_usd
from adaptive_rag.config.settings import Settings, settings

__all__ = [
    "BM25_INDEX_DIR",
    "BM25_INDEX_PATH",
    "CHUNKS_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "DOCUMENTS_DIR",
    "EMBEDDINGS_CACHE_PATH",
    "EMBEDDINGS_DIR",
    "EVALUATION_DIR",
    "EVAL_CACHE_PATH",
    "EXPERIMENTS_DIR",
    "METADATA_DIR",
    "MODEL_PRICING",
    "PAPERS_MANIFEST_PATH",
    "PRICING_VERSION",
    "PROCESSED_DATA_DIR",
    "QDRANT_DIR",
    "RAW_DATA_DIR",
    "REPO_ROOT",
    "STATS_DIR",
    "STORAGE_DIR",
    "Settings",
    "calculate_cost_usd",
    "canonical_json",
    "compute_config_hash",
    "compute_file_sha256",
    "compute_sha256",
    "ensure_directories",
    "get_repo_root",
    "settings",
    "setup_logging",
]
