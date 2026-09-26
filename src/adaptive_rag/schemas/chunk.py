"""
schemas.chunk
-------------
Data contracts for structure-aware document chunks.
Embeddings are deliberately excluded from this canonical representation.
"""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Contextual and structural metadata derived from document hierarchy."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    doc_title: str
    section_id: str | None = None
    section_path: list[str]
    headings: list[str]
    element_ids: list[str]
    element_types: list[str]
    page_start: int | None = None
    page_end: int | None = None
    token_count: int
    char_count: int


class ChunkProvenance(BaseModel):
    """Source provenance linking chunk back to Phase 1 corpus file."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    pages: list[int]
    source_sha256: str


class ChunkingMetadata(BaseModel):
    """Metadata regarding chunk generation, overlap, and boundary splitting."""

    model_config = ConfigDict(extra="forbid")

    chunking_version: str
    config_hash: str
    ordinal: int
    overlap_source_element_ids: list[str] = Field(default_factory=list)
    overlap_tokens: int = 0
    split_of_element_id: str | None = None
    part_index: int | None = None
    part_count: int | None = None
    oversized_atomic: bool = False
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A semantically coherent group of elements within document hierarchy."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str  # e.g. f"{document_id}::{chunking_version}::c{ordinal:05d}"
    document_id: str
    text: str
    metadata: ChunkMetadata
    provenance: ChunkProvenance
    chunking_metadata: ChunkingMetadata
