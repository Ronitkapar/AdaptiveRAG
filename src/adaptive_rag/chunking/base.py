"""
chunking.base
-------------
Protocol definition for chunkers.
Allows future extension without modifying core baseline pipelines.
"""

from typing import Protocol, Sequence

from adaptive_rag.schemas import Chunk, ChunkingConfig, Document


class Chunker(Protocol):
    """Protocol for document chunking algorithms."""

    version: str
    config: ChunkingConfig

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Convert a hierarchical Document into structured Chunks."""
        ...
