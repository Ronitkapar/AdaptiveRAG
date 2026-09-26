"""
ingestion.normalizer
--------------------
Converts raw page primitives into a canonical hierarchical Document.
Determines reading order, detects headings/sections, extracts tables/equations/figures,
and builds section paths with full provenance.
"""

from collections import Counter
import re
from typing import Any

from adaptive_rag.config.hashing import compute_config_hash
from adaptive_rag.ingestion.extractor import RawChar, RawPage
from adaptive_rag.schemas import (
    Document,
    DocumentMetadata,
    Element,
    ElementContent,
    ElementType,
    ExtractionIssue,
    ExtractionReport,
    IngestionConfig,
    Page,
    Provenance,
    Section,
)

HEADING_ARABIC_REGEX = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)[\.\)]?\s+(\S.*)$")
HEADING_ROMAN_REGEX = re.compile(r"^([IVX]{1,4})\.\s+([A-Z][a-z].*)$")
HEADING_APPENDIX_REGEX = re.compile(r"^([A-Z])\.\s+([A-Z][a-z].*)$")
REFERENCE_LINE_REGEX = re.compile(r"^\[?\d{1,4}\]?\s*\(?(19|20)\d{2}[a-z]?[\.\)]?\s")
URLISH_REGEX = re.compile(r"(https?://|arxiv:|doi:|urlhttp)", re.IGNORECASE)
CAPTION_REGEX = re.compile(
    r"^(Figure|Fig\.|Table|Algorithm)\s*([0-9]+|[IVXLCDM]+)[:.]?\s*(.*)$",
    re.IGNORECASE,
)
MATH_CHARS = set("=≠≈±×÷∑∏∫√∂∇≤≥∈∉⊂⊃∪∩∧∨¬∀∃αβγδεζηθικλμνξπρστυφχψω")


def _looks_numeric_noise(text: str) -> bool:
    """Detect labels from inside figures/tables (mostly digits and symbols)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    numeric = sum(1 for c in chars if c.isdigit() or not (c.isalpha() or c in ".,'-"))
    return (numeric / len(chars)) > 0.45


def _has_many_single_letter_words(text: str) -> bool:
    """Detect figure axis-label runs such as 'R g RAG-TokB-1'."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if len(words) < 3:
        return False
    single = sum(1 for w in words if len(w.strip(".,:;()")) <= 1)
    return (single / len(words)) > 0.4


def classify_heading(text: str) -> tuple[int, str] | None:
    """Return (heading_level, heading_number) when text is a reliable heading.

    Numeric figure labels, equation fragments, bibliography entries, and other
    extraction noise are explicitly rejected rather than promoted to sections.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 100:
        return None
    if URLISH_REGEX.search(stripped) or REFERENCE_LINE_REGEX.match(stripped):
        return None
    if _looks_numeric_noise(stripped) or _is_math_line(stripped):
        return None
    if _has_many_single_letter_words(stripped):
        return None

    match_arabic = HEADING_ARABIC_REGEX.match(stripped)
    if match_arabic and match_arabic.group(2)[:1].isupper():
        number = match_arabic.group(1)
        return (1 + number.count("."), number)

    # Roman/letter labels must be short titles ("II. Related Work"), never
    # bibliography author lists ("H. Wallach, H. Larochelle, ...").
    if len(stripped) <= 40:
        for pattern in (HEADING_ROMAN_REGEX, HEADING_APPENDIX_REGEX):
            match = pattern.match(stripped)
            if match:
                return (1, match.group(1))

    return None



def _cluster_chars_into_lines(chars: list[RawChar]) -> list[list[RawChar]]:
    """Cluster raw characters into sorted lines by vertical position."""
    if not chars:
        return []

    sorted_chars = sorted(chars, key=lambda c: (round(c.top, 1), c.x0))
    lines: list[list[RawChar]] = []
    current_line: list[RawChar] = []
    current_top = sorted_chars[0].top

    for c in sorted_chars:
        if abs(c.top - current_top) <= 3.0:
            current_line.append(c)
        else:
            if current_line:
                current_line.sort(key=lambda ch: ch.x0)
                lines.append(current_line)
            current_line = [c]
            current_top = c.top

    if current_line:
        current_line.sort(key=lambda ch: ch.x0)
        lines.append(current_line)

    return lines


def _line_text(line: list[RawChar]) -> str:
    """Combine character text, inserting spaces where horizontal gaps occur."""
    if not line:
        return ""
    parts = []
    prev_x1 = line[0].x0
    for ch in line:
        if ch.x0 - prev_x1 > 2.5:
            parts.append(" ")
        parts.append(ch.text)
        prev_x1 = ch.x1
    return "".join(parts).strip()


def _is_math_line(text: str) -> bool:
    """Heuristic check for mathematical formula lines."""
    if not text:
        return False
    math_count = sum(1 for ch in text if ch in MATH_CHARS)
    return math_count >= 2 or ("=" in text and any(m in text for m in ("cos", "sin", "log", "||", "·")))


def normalize_pages_to_document(
    document_id: str,
    raw_pages: list[RawPage],
    pdf_info: dict[str, Any],
    manifest_meta: dict[str, Any],
    config: IngestionConfig,
) -> Document:
    """Normalize raw extracted pages into a canonical hierarchical Document."""
    issues: list[ExtractionIssue] = []
    canonical_elements: list[Element] = []
    pages: list[Page] = []
    all_sections: list[dict[str, Any]] = []

    # Estimate document-wide body font size
    font_sizes = [round(ch.size, 1) for rp in raw_pages for ch in rp.chars if ch.text.strip()]
    body_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 10.0

    root_section: dict[str, Any] = {
        "section_id": f"{document_id}::s0000",
        "title": "Document Frontmatter",
        "level": 1,
        "path": ["Document Frontmatter"],
        "number": None,
        "element_ids": [],
        "subsection_ids": [],
        "page_start": 1 if raw_pages else None,
        "page_end": 1 if raw_pages else None,
        "provenance": Provenance(document_id=document_id, page=1),
    }
    all_sections.append(root_section)
    active_section_stack = [root_section]

    el_idx = 0
    sec_idx = 1

    for rp in raw_pages:
        page_elem_ids: list[str] = []
        page_char_count = sum(len(c.text) for c in rp.chars)

        if not rp.chars and not rp.images and not rp.tables:
            issues.append(
                ExtractionIssue(
                    code="EMPTY_PAGE",
                    severity="warning",
                    message=f"Page {rp.page_number} has no text, images, or tables.",
                    page=rp.page_number,
                )
            )
        elif page_char_count < config.min_page_chars_warning:
            issues.append(
                ExtractionIssue(
                    code="LOW_TEXT_DENSITY",
                    severity="info",
                    message=f"Page {rp.page_number} has low text density ({page_char_count} chars).",
                    page=rp.page_number,
                )
            )

        table_bboxes = [t.bbox for t in rp.tables]
        lines = _cluster_chars_into_lines(rp.chars)
        current_para_lines: list[list[RawChar]] = []

        def flush_paragraph():
            nonlocal el_idx, current_para_lines
            if not current_para_lines:
                return
            para_text = " ".join(_line_text(l) for l in current_para_lines if _line_text(l)).strip()
            if not para_text:
                current_para_lines = []
                return

            first_char = current_para_lines[0][0]
            last_char = current_para_lines[-1][-1]
            bbox = (first_char.x0, first_char.top, last_char.x1, last_char.bottom)
            prov = Provenance(document_id=document_id, page=rp.page_number, bbox=bbox)

            el_type: ElementType = "paragraph"
            if _is_math_line(para_text) and len(current_para_lines) <= 2:
                el_type = "equation"
            elif para_text.startswith(("- ", "• ", "* ")) or (
                len(para_text) > 2 and para_text[0].isdigit() and para_text[1:3] in (". ", ") ")
            ):
                el_type = "list"

            el_id = f"{document_id}::p{rp.page_number:03d}::e{el_idx:04d}"
            el_idx += 1

            elem = Element(
                element_id=el_id,
                type=el_type,
                position=len(canonical_elements),
                content=ElementContent(text=para_text),
                provenance=prov,
            )
            canonical_elements.append(elem)
            page_elem_ids.append(el_id)
            active_section_stack[-1]["element_ids"].append(el_id)
            active_section_stack[-1]["page_end"] = rp.page_number
            current_para_lines = []

        for line in lines:
            if not line:
                continue
            l_text = _line_text(line)
            if not l_text:
                continue

            line_top, line_x0 = line[0].top, line[0].x0
            if any(tb[0] <= line_x0 <= tb[2] and tb[1] <= line_top <= tb[3] for tb in table_bboxes):
                continue

            is_bold = any(c.is_bold for c in line)
            max_size = max(c.size for c in line)

            is_heading = False
            heading_level = 1
            heading_num = None

            classified = classify_heading(l_text)
            if classified is not None:
                is_heading = True
                heading_level, heading_num = classified
            elif (
                is_bold
                and max_size >= body_size * 1.05
                and len(l_text) < 100
                and not _looks_numeric_noise(l_text)
                and not _is_math_line(l_text)
                and not REFERENCE_LINE_REGEX.match(l_text)
                and not URLISH_REGEX.search(l_text)
            ):
                is_heading = True
                heading_level = 1 if max_size >= body_size * 1.2 else 2


            if is_heading:
                flush_paragraph()
                sec_id = f"{document_id}::s{sec_idx:04d}"
                sec_idx += 1

                while len(active_section_stack) > 1 and active_section_stack[-1]["level"] >= heading_level:
                    active_section_stack.pop()

                parent_path = active_section_stack[-1]["path"] if active_section_stack else []
                current_path = (
                    parent_path + [l_text] if parent_path != ["Document Frontmatter"] else [l_text]
                )

                h_bbox = (line[0].x0, line[0].top, line[-1].x1, line[-1].bottom)
                h_prov = Provenance(document_id=document_id, page=rp.page_number, bbox=h_bbox)
                h_el_id = f"{document_id}::p{rp.page_number:03d}::e{el_idx:04d}"
                el_idx += 1

                h_elem = Element(
                    element_id=h_el_id,
                    type="heading",
                    position=len(canonical_elements),
                    content=ElementContent(text=l_text),
                    provenance=h_prov,
                    metadata={"heading_level": heading_level, "heading_number": heading_num},
                )
                canonical_elements.append(h_elem)
                page_elem_ids.append(h_el_id)

                new_sec = {
                    "section_id": sec_id,
                    "title": l_text,
                    "level": heading_level,
                    "path": current_path,
                    "number": heading_num,
                    "element_ids": [h_el_id],
                    "subsection_ids": [],
                    "page_start": rp.page_number,
                    "page_end": rp.page_number,
                    "provenance": h_prov,
                }
                active_section_stack[-1]["subsection_ids"].append(sec_id)
                active_section_stack.append(new_sec)
                all_sections.append(new_sec)
                continue

            match_caption = CAPTION_REGEX.match(l_text)
            if match_caption:
                flush_paragraph()
                c_bbox = (line[0].x0, line[0].top, line[-1].x1, line[-1].bottom)
                c_prov = Provenance(document_id=document_id, page=rp.page_number, bbox=c_bbox)
                c_id = f"{document_id}::p{rp.page_number:03d}::e{el_idx:04d}"
                el_idx += 1

                c_elem = Element(
                    element_id=c_id,
                    type="caption",
                    position=len(canonical_elements),
                    content=ElementContent(text=l_text),
                    provenance=c_prov,
                    metadata={"caption_label": match_caption.group(1)},
                )
                canonical_elements.append(c_elem)
                page_elem_ids.append(c_id)
                active_section_stack[-1]["element_ids"].append(c_id)
                continue

            current_para_lines.append(line)

        flush_paragraph()

        # Tables
        for t in rp.tables:
            t_id = f"{document_id}::p{rp.page_number:03d}::e{el_idx:04d}"
            el_idx += 1
            md_lines = []
            for row in t.rows:
                cells = [str(c or "").replace("\n", " ").strip() for c in row]
                md_lines.append("| " + " | ".join(cells) + " |")

            table_text = "\n".join(md_lines)
            t_prov = Provenance(document_id=document_id, page=rp.page_number, bbox=t.bbox)
            t_elem = Element(
                element_id=t_id,
                type="table",
                position=len(canonical_elements),
                content=ElementContent(
                    text=table_text,
                    structured_data={
                        "rows": t.rows,
                        "n_rows": len(t.rows),
                        "n_cols": len(t.rows[0]) if t.rows else 0,
                    },
                    content_format="table_rows",
                ),
                provenance=t_prov,
            )
            canonical_elements.append(t_elem)
            page_elem_ids.append(t_id)
            active_section_stack[-1]["element_ids"].append(t_id)

        # Images
        for img in rp.images:
            f_id = f"{document_id}::p{rp.page_number:03d}::e{el_idx:04d}"
            el_idx += 1
            f_prov = Provenance(document_id=document_id, page=rp.page_number, bbox=img.bbox)
            f_elem = Element(
                element_id=f_id,
                type="figure",
                position=len(canonical_elements),
                content=ElementContent(
                    text=f"[Figure: {img.name or 'embedded'}]",
                    structured_data={
                        "kind": "raster",
                        "name": img.name,
                        "width": img.width,
                        "height": img.height,
                    },
                ),
                provenance=f_prov,
            )
            canonical_elements.append(f_elem)
            page_elem_ids.append(f_id)
            active_section_stack[-1]["element_ids"].append(f_id)

        pages.append(
            Page(
                page_number=rp.page_number,
                width=rp.width,
                height=rp.height,
                element_ids=page_elem_ids,
                text_char_count=page_char_count,
            )
        )

    sections = [Section(**s) for s in all_sections]
    el_counts = Counter(el.type for el in canonical_elements)
    cfg_hash = compute_config_hash(config)
    report = ExtractionReport(
        ingestion_version=config.ingestion_version,
        config_hash=cfg_hash,
        page_count=len(raw_pages),
        element_counts=dict(el_counts),
        section_count=len(sections),
        issues=issues,
    )

    doc_meta = DocumentMetadata(
        document_id=document_id,
        title=manifest_meta.get("title", ""),
        authors=manifest_meta.get("authors", []),
        year=manifest_meta.get("year", 2024),
        category=manifest_meta.get("category", ""),
        venue=manifest_meta.get("venue"),
        source_url=manifest_meta.get("source_url", ""),
        download_url=manifest_meta.get("download_url", ""),
        local_path=manifest_meta.get("local_path", ""),
        description=manifest_meta.get("description", ""),
        concepts=manifest_meta.get("concepts", []),
        source_sha256=str(manifest_meta.get("sha256") or ""),
        source_size_bytes=manifest_meta.get("size_bytes", 0),
        pdf_title=str(pdf_info.get("Title") or ""),
        pdf_producer=str(pdf_info.get("Producer") or ""),
        pdf_creation_date=str(pdf_info.get("CreationDate") or ""),
    )

    return Document(
        document_id=document_id,
        metadata=doc_meta,
        pages=pages,
        sections=sections,
        elements=canonical_elements,
        extraction_report=report,
    )
