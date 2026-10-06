"""
embeddings.cache
----------------
SQLite-backed content-addressed embedding cache.
Prevents redundant external API calls and enforces configuration-based cache invalidation.
API keys and secrets are NEVER stored in the cache.
"""

import hashlib
import sqlite3
import struct
from pathlib import Path
from typing import Sequence

from adaptive_rag.config.hashing import compute_sha256
from adaptive_rag.config.paths import EMBEDDINGS_CACHE_PATH


def _pack_vector(vector: Sequence[float]) -> bytes:
    """Pack list of float32 values into binary bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack_vector(blob: bytes) -> list[float]:
    """Unpack binary bytes into list of float32 values."""
    num_floats = len(blob) // 4
    return list(struct.unpack(f"{num_floats}f", blob))


def derive_cache_key(
    text: str,
    model_id: str,
    base_url_host: str,
    dimensions: int | None,
    normalize: bool,
    pipeline_version: str,
) -> str:
    """Compute content-addressed cache key."""
    text_hash = compute_sha256(text)
    raw = (
        f"{pipeline_version}|{model_id}|{base_url_host}|"
        f"{dimensions}|{normalize}|{text_hash}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Persistent SQLite cache for embedding vectors."""

    def __init__(self, db_path: Path = EMBEDDINGS_CACHE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    chunk_id TEXT,
                    model_id TEXT,
                    dimension INTEGER,
                    text_sha256 TEXT,
                    vector BLOB
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> list[float] | None:
        """Retrieve a cached vector if present."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT vector FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
            if row:
                return _unpack_vector(row[0])
        return None

    def put(
        self,
        cache_key: str,
        vector: list[float],
        chunk_id: str = "",
        model_id: str = "",
        text_sha256: str = "",
    ) -> None:
        """Store an embedding vector into the cache."""
        blob = _pack_vector(vector)
        dim = len(vector)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                (cache_key, chunk_id, model_id, dimension, text_sha256, vector)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, chunk_id, model_id, dim, text_sha256, blob),
            )
            conn.commit()

    def count(self) -> int:
        """Total number of cached vectors."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM embedding_cache")
            return cur.fetchone()[0]
