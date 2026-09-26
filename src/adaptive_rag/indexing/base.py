"""
indexing.base
-------------
Protocol and data structures for vector store backends.
Enables swapping between embedded local Qdrant and remote server.
"""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from adaptive_rag.schemas import Chunk


@dataclass
class ScoredPoint:
    """A scored search match from the vector store."""

    chunk_id: str
    score: float
    payload: dict[str, Any]


@dataclass
class IndexMetadata:
    """Audit metadata recording the configuration that built an index."""

    index_id: str
    corpus_version: str
    chunking_version: str
    chunking_config_hash: str
    embedding_model_id: str
    embedding_dimension: int
    normalize: bool
    distance: str
    collection_name: str
    point_count: int


class VectorStore(Protocol):
    """Protocol for vector databases."""

    def ensure_collection(
        self,
        dim: int,
        distance: str = "cosine",
        recreate: bool = False,
    ) -> None:
        """Create or recreate a vector collection."""
        ...

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Upsert a batch of chunks with their corresponding vectors."""
        ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        """Perform vector similarity search."""
        ...

    def count(self) -> int:
        """Return total point count in collection."""
        ...

    def close(self) -> None:
        """Release underlying client connections or file locks."""
        ...
