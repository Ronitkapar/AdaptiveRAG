"""
embeddings.base
---------------
Embedding protocol and data structures.
Provider-agnostic interface keeping downstream components decoupled.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class ActiveEmbeddingInfo:
    """Metadata describing the active embedding model."""

    model_id: str
    dimension: int
    provider: str
    normalized: bool
    config_hash: str


class EmbeddingModel(Protocol):
    """Protocol for embedding generation models."""

    model_id: str
    dimension: int
    normalize: bool
    config_hash: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document texts into vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query into a vector."""
        ...
