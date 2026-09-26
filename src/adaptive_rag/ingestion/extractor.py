"""
ingestion.extractor
-------------------
PDF text, character, table, and image extraction protocol and implementation.
Extracts raw page primitives without performing semantic interpretation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pdfplumber


@dataclass
class RawChar:
    text: str
    fontname: str
    size: float
    is_bold: bool
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass
class RawTable:
    bbox: tuple[float, float, float, float]  # (x0, top, x1, bottom)
    rows: list[list[str | None]]


@dataclass
class RawImage:
    bbox: tuple[float, float, float, float]
    name: str | None
    width: float
    height: float


@dataclass
class RawPage:
    page_number: int
    width: float
    height: float
    chars: list[RawChar] = field(default_factory=list)
    tables: list[RawTable] = field(default_factory=list)
    images: list[RawImage] = field(default_factory=list)
    rects: list[dict[str, Any]] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    curves: list[dict[str, Any]] = field(default_factory=list)


class PdfExtractor(Protocol):
    """Protocol for extracting raw physical page structures from a PDF."""

    name: str
    version: str

    def extract_document(self, pdf_path: Path) -> tuple[list[RawPage], dict[str, Any]]:
        """Extract raw pages and document metadata."""
        ...


class PdfplumberExtractor:
    """PDF extractor implementation powered by pdfplumber."""

    name: str = "pdfplumber"
    version: str = "pdfplumber_0.11"

    def extract_document(self, pdf_path: Path) -> tuple[list[RawPage], dict[str, Any]]:
        raw_pages: list[RawPage] = []
        doc_metadata: dict[str, Any] = {}

        with pdfplumber.open(pdf_path) as pdf:
            doc_metadata = dict(pdf.metadata or {})

            for idx, page in enumerate(pdf.pages, start=1):
                raw_page = RawPage(
                    page_number=idx,
                    width=float(page.width),
                    height=float(page.height),
                )

                # 1. Chars
                chars = page.chars or []
                for c in chars:
                    fontname = str(c.get("fontname", ""))
                    is_bold = any(
                        sub in fontname.lower()
                        for sub in ("bold", "black", "heavy", "semibold")
                    )
                    raw_page.chars.append(
                        RawChar(
                            text=c.get("text", ""),
                            fontname=fontname,
                            size=float(c.get("size", 0.0)),
                            is_bold=is_bold,
                            x0=float(c.get("x0", 0.0)),
                            top=float(c.get("top", 0.0)),
                            x1=float(c.get("x1", 0.0)),
                            bottom=float(c.get("bottom", 0.0)),
                        )
                    )

                # 2. Tables
                try:
                    tables = page.find_tables() or []
                    for t in tables:
                        bbox = (
                            float(t.bbox[0]),
                            float(t.bbox[1]),
                            float(t.bbox[2]),
                            float(t.bbox[3]),
                        )
                        rows = t.extract() or []
                        raw_page.tables.append(RawTable(bbox=bbox, rows=rows))
                except Exception:
                    # Table detection errors are reported downstream
                    pass

                # 3. Images
                for img in page.images or []:
                    try:
                        x0 = float(img.get("x0", 0.0))
                        top = float(img.get("top", 0.0))
                        x1 = float(img.get("x1", 0.0))
                        bottom = float(img.get("bottom", 0.0))
                        raw_page.images.append(
                            RawImage(
                                bbox=(x0, top, x1, bottom),
                                name=str(img.get("name", "")),
                                width=float(img.get("width", 0.0)),
                                height=float(img.get("height", 0.0)),
                            )
                        )
                    except (ValueError, TypeError):
                        continue

                # 4. Drawings / vectors
                raw_page.rects = list(page.rects or [])
                raw_page.lines = list(page.lines or [])
                raw_page.curves = list(page.curves or [])

                raw_pages.append(raw_page)

        return raw_pages, doc_metadata
