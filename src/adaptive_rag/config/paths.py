"""
config.paths
------------
Filesystem path definitions and directory resolution for AdaptiveRAG.
All paths are resolved relative to the repository root.
"""

from pathlib import Path


def get_repo_root() -> Path:
    """Resolve the repository root directory."""
    # src/adaptive_rag/config/paths.py -> repo_root is 3 levels up
    return Path(__file__).resolve().parents[3]


REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCUMENTS_DIR = PROCESSED_DATA_DIR / "documents"
CHUNKS_DIR = PROCESSED_DATA_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DATA_DIR / "embeddings"
STATS_DIR = PROCESSED_DATA_DIR / "stats"
METADATA_DIR = DATA_DIR / "metadata"
EVALUATION_DIR = DATA_DIR / "evaluation"

STORAGE_DIR = REPO_ROOT / "storage"
EMBEDDINGS_CACHE_DIR = STORAGE_DIR / "embeddings"
QDRANT_DIR = STORAGE_DIR / "qdrant"
BM25_INDEX_DIR = STORAGE_DIR / "bm25"
EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "embeddings_cache.sqlite3"
EVAL_CACHE_PATH = STORAGE_DIR / "eval_cache.sqlite3"
BM25_INDEX_PATH = BM25_INDEX_DIR / "bm25_index.json"

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DOCS_DIR = REPO_ROOT / "docs"
PAPERS_MANIFEST_PATH = METADATA_DIR / "papers.json"


def ensure_directories() -> None:
    """Ensure all required runtime data and storage directories exist."""
    for path in (
        DOCUMENTS_DIR,
        CHUNKS_DIR,
        EMBEDDINGS_DIR,
        STATS_DIR,
        EVALUATION_DIR,
        EMBEDDINGS_CACHE_DIR,
        QDRANT_DIR,
        BM25_INDEX_DIR,
        EXPERIMENTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
