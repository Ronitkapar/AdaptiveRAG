"""
schemas.document
----------------
Hierarchical Document representation and element models.
Sections and Pages reference canonical Elements without data duplication.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

ElementType = Literal[
    "paragraph",
    "heading",
    "figure",
    "table",
    "equation",
    "caption",
    "list",
    "unknown",
]


class Provenance(BaseModel):
    """Source location and physical origin of an extracted element."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # (x0, top, x1, bottom) PDF points
    source: str = "pdfplumber"


class ElementContent(BaseModel):
    """Unified structured content representation."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    structured_data: dict[str, Any] = Field(default_factory=dict)
    content_format: Literal["text", "markdown", "table_rows", "mixed"] = "text"


class Element(BaseModel):
    """Canonical atomic element extracted from a document."""

    model_config = ConfigDict(extra="forbid")

    element_id: str  # e.g. f"{doc_id}::p{page:03d}::e{position:04d}"
    type: ElementType
    position: int  # monotonic document-wide reading order
    content: ElementContent
    provenance: Provenance
    caption_element_id: str | None = None
    parent_element_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    """Logical section node within a hierarchical document."""

    model_config = ConfigDict(extra="forbid")

    section_id: str  # e.g. f"{doc_id}::s{ordinal:04d}"
    title: str
    path: list[str]  # Ancestor titles, e.g. ["2. Related Work", "2.1 Dense Retrieval"]
    level: int  # 1 for top-level, 2 for subsection, etc.
    number: str | None = None  # e.g. "2.1"
    element_ids: list[str] = Field(default_factory=list)  # Direct canonical element ownership
    subsection_ids: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    provenance: Provenance


class Page(BaseModel):
    """Physical page record referencing canonical elements."""

    model_config = ConfigDict(extra="forbid")

    page_number: int
    width: float | None = None
    height: float | None = None
    element_ids: list[str] = Field(default_factory=list)
    text_char_count: int = 0
    issues: list[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Corpus bibliographic metadata and extraction origin."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    authors: list[str]
    year: int
    category: str
    venue: str | None = None
    source_url: str
    download_url: str
    local_path: str
    description: str
    concepts: list[str]
    source_sha256: str
    source_size_bytes: int
    pdf_title: str | None = None
    pdf_producer: str | None = None
    pdf_creation_date: str | None = None


class ExtractionIssue(BaseModel):
    """Structured extraction failure, warning, or quality observation."""

    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "EMPTY_PAGE",
        "NO_TEXT_LAYER",
        "LOW_TEXT_DENSITY",
        "TABLE_DETECTION_FAILED",
        "CAPTION_UNMATCHED",
        "HEADING_HIERARCHY_GAP",
        "EQUATION_UNCERTAIN",
        "FIGURE_REGION_AMBIGUOUS",
        "METADATA_MISSING",
        "PDF_OPEN_FAILED",
    ]
    severity: Literal["info", "warning", "error"]
    message: str
    page: int | None = None
    element_id: str | None = None


class ExtractionReport(BaseModel):
    """Audit summary of document extraction quality and issues."""

    model_config = ConfigDict(extra="forbid")

    ingestion_version: str
    config_hash: str
    page_count: int
    element_counts: dict[str, int] = Field(default_factory=dict)
    section_count: int
    issues: list[ExtractionIssue] = Field(default_factory=list)


class Document(BaseModel):
    """Canonical hierarchical representation of an ingested PDF paper."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    schema_version: str = "document_v1"
    metadata: DocumentMetadata
    pages: list[Page]
    sections: list[Section]
    elements: list[Element]  # Canonical ordered element store
    extraction_report: ExtractionReport
