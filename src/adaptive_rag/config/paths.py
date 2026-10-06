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

# --- Phase 8 corpus arms ------------------------------------------------------
# Phase 8 is a before/after study of one change: two-column reading order. Both
# arms are built from the same PDFs and the same chunking config, differing ONLY in
# whether `_detect_columns` runs. That isolation is the whole point -- reusing the
# Phase 7 artifacts as the "before" arm would have confounded the column fix with an
# unrelated chunk-boundary difference (the Phase 7 indexes hold 713 chunks, the
# current documents chunk to 618), and no quality change could then be attributed.
#
# The Phase 7 artifacts in `PROCESSED_DATA_DIR`, `BM25_INDEX_PATH` and the
# `adaptiverag_dense_v1` collection are left exactly as they are. They are the
# historical record of what Phase 7 evaluated, not an arm of this study.
PHASE8_BEFORE_DATA_DIR = DATA_DIR / "processed_phase8_before"
PHASE8_BEFORE_DOCUMENTS_DIR = PHASE8_BEFORE_DATA_DIR / "documents"
PHASE8_BEFORE_CHUNKS_DIR = PHASE8_BEFORE_DATA_DIR / "chunks"
PHASE8_BEFORE_EMBEDDINGS_DIR = PHASE8_BEFORE_DATA_DIR / "embeddings"

PHASE8_AFTER_DATA_DIR = DATA_DIR / "processed_phase8_after"
PHASE8_AFTER_DOCUMENTS_DIR = PHASE8_AFTER_DATA_DIR / "documents"
PHASE8_AFTER_CHUNKS_DIR = PHASE8_AFTER_DATA_DIR / "chunks"
PHASE8_AFTER_EMBEDDINGS_DIR = PHASE8_AFTER_DATA_DIR / "embeddings"

STORAGE_DIR = REPO_ROOT / "storage"
EMBEDDINGS_CACHE_DIR = STORAGE_DIR / "embeddings"
QDRANT_DIR = STORAGE_DIR / "qdrant"
BM25_INDEX_DIR = STORAGE_DIR / "bm25"
RERANKER_DIR = STORAGE_DIR / "reranker"
EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "embeddings_cache.sqlite3"
EVAL_CACHE_PATH = STORAGE_DIR / "eval_cache.sqlite3"
BM25_INDEX_PATH = BM25_INDEX_DIR / "bm25_index.json"

# One lexical index per arm, beside the Phase 7 one. Separate files rather than a
# rebuild: overwriting would destroy the record of what Phase 7 evaluated, and one
# shared file cannot hold two corpora.
PHASE8_BEFORE_BM25_INDEX_PATH = BM25_INDEX_DIR / "bm25_index_phase8_before.json"
PHASE8_AFTER_BM25_INDEX_PATH = BM25_INDEX_DIR / "bm25_index_phase8_after.json"

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
        RERANKER_DIR,
        EXPERIMENTS_DIR,
        PHASE8_BEFORE_DOCUMENTS_DIR,
        PHASE8_BEFORE_CHUNKS_DIR,
        PHASE8_BEFORE_EMBEDDINGS_DIR,
        PHASE8_AFTER_DOCUMENTS_DIR,
        PHASE8_AFTER_CHUNKS_DIR,
        PHASE8_AFTER_EMBEDDINGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
