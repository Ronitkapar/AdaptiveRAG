"""
ingestion.normalizer
--------------------
Converts raw page primitives into a canonical hierarchical Document.
Determines reading order, detects headings/sections, extracts tables/equations/figures,
and builds section paths with full provenance.
"""

from collections import Counter
import math
import re
from typing import Any, NamedTuple

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



# --------------------------------------------------------------------------
# two-column detection
# --------------------------------------------------------------------------
#
# The 3.0 pt rule below merges every character that shares a baseline. On a
# two-column page both columns share their baselines, so each visual "line" holds
# the end of one column's sentence and the start of the next; sorting it by `x0`
# and concatenating interleaves the two columns mid-sentence. Every threshold
# below exists to make detection *conservative*: the cost of a false positive is a
# page split down a seam that was never a gutter, which scrambles reading order,
# while the cost of a miss is the status quo. Where the two conflict, this takes
# the miss.

# Character coverage is projected onto the x-axis in bins of this width. An
# occupancy count, not a per-character loop, so one stray glyph or a hyphen at a
# column edge cannot fabricate or bridge a gutter.
_COLUMN_BIN_PT = 1.0
# Vertical band height for the persistence test. Small enough that a title and a
# column of body text are separate bands.
_COLUMN_ROW_PT = 4.0
# Two-column papers set a gutter at roughly 12-24 pt at 10 pt body type. A
# narrower gap is not a gutter; single-column text has none at all.
_COLUMN_MIN_GUTTER_PT = 6.0
# The gutter sits at the page centre. A generous 25% of page width is allowed
# because margins are not symmetric on every paper, and a tighter bound would
# miss a page whose binding margin is wide.
_COLUMN_MAX_GUTTER_OFFSET_FRAC = 0.25
# "Persists across most of the page's vertical extent": at least this fraction of
# the page's occupied rows must be empty across the candidate band. A title, a
# figure, or a table that happens to leave a hole does not reach it; a gutter
# does, on every body row.
_COLUMN_MIN_VERTICAL_COVERAGE = 0.5
# Layout analysis needs enough occupied rows for the coverage fraction to mean
# anything.
_COLUMN_MIN_OCCUPIED_ROWS = 12
# A "column" narrower than this is a fragment of a figure or a table cell, not a
# column of text.
_COLUMN_MIN_COLUMN_WIDTH_FRAC = 0.15
# ...and one holding fewer chars than this is not carrying the page either.
_COLUMN_MIN_COLUMN_CHAR_FRAC = 0.10


def _column_index(x: float, ranges: list[tuple[float, float]]) -> int:
    """Which column a horizontal position belongs to.

    `ranges` are contiguous and cover the text extent, so a position inside one is
    found directly; a position past the last edge (a glyph in the right margin)
    clamps to the nearest column rather than falling off the end.
    """
    for index, (lo, hi) in enumerate(ranges):
        if lo <= x < hi:
            return index
    if x < ranges[0][0]:
        return 0
    return len(ranges) - 1


def _detect_columns(
    chars: list[RawChar], page_width: float
) -> list[tuple[float, float]] | None:
    """Column x-ranges for a page, or `None` when it is single-column.

    Character coverage is projected onto the x-axis **row by row**, and a
    candidate gutter is a band of bins that a large fraction of the page's
    occupied rows individually leave empty. A candidate must be **wide** (>= 6 pt
    of nothing), **near the page centre** (within 25% of `page_width`), and
    **vertically persistent** (empty across at least half of the page's occupied
    rows). Candidates that survive all three become column boundaries; the ranges
    between them are returned.

    Counting emptiness per row rather than on a union projection is what lets the
    first page of a two-column paper be detected. Its title and author block span
    the full width, so a union projection sees no interior gap at all and the page
    reads as single-column -- which is precisely how a first page ends up
    interleaved.

    Every column that results must be wide enough and hold enough characters to
    be a column of text, no column may be split by a further gutter of its own,
    and the page must have enough occupied rows for the persistence fraction to be
    meaningful. Otherwise `None`, and the caller falls back to the single-column
    path unchanged.

    Every refusal here is deliberate. Detection runs once per page over the whole
    corpus, and a false positive scrambles a page's reading order in a way that is
    far harder to notice -- and far worse -- than the interleaving it was meant to
    fix.
    """
    if not chars or page_width <= 0:
        return None

    text_x0 = min(c.x0 for c in chars)
    text_x1 = max(c.x1 for c in chars)
    top_min = min(c.top for c in chars)
    top_max = max(c.bottom for c in chars)
    if text_x1 <= text_x0 or top_max <= top_min:
        return None

    n_bins = int(math.ceil(text_x1 - text_x0)) + 1
    n_rows = int(math.ceil((top_max - top_min) / _COLUMN_ROW_PT)) + 1
    grid = [bytearray(n_bins) for _ in range(n_rows)]

    def _bin(x: float) -> int:
        return max(0, min(n_bins - 1, int(math.floor(x - text_x0))))

    def _row(y: float) -> int:
        return max(0, min(n_rows - 1, int(math.floor((y - top_min) / _COLUMN_ROW_PT))))

    for c in chars:
        left, right = _bin(c.x0), _bin(c.x1)
        first, last = _row(c.top), _row(c.bottom)
        for row_index in range(first, last + 1):
            row = grid[row_index]
            for bin_index in range(left, right + 1):
                row[bin_index] = 1

    occupied_rows = [i for i, row in enumerate(grid) if any(row)]
    if len(occupied_rows) < _COLUMN_MIN_OCCUPIED_ROWS:
        return None

    # Emptiness is counted **per row**, not on a union projection. A union
    # projection is the wrong instrument here and fails on exactly the pages that
    # matter most: the first page of a two-column paper carries a full-width title
    # and author block, and a single row spanning the gutter is enough to erase
    # the interior gap that every body row below it has. Counting, for each bin,
    # how many rows leave it empty -- restricted to bins interior to that row's
    # own extent -- recovers the gutter on those pages.
    empty_count = [0] * n_bins
    page_first_bin, page_last_bin = n_bins, -1
    for row_index in occupied_rows:
        row = grid[row_index]
        filled = [i for i in range(n_bins) if row[i]]
        if len(filled) < 2:
            continue
        row_first, row_last = filled[0], filled[-1]
        page_first_bin = min(page_first_bin, row_first)
        page_last_bin = max(page_last_bin, row_last)
        for bin_index in range(row_first + 1, row_last):
            if not row[bin_index]:
                empty_count[bin_index] += 1
    if page_last_bin <= page_first_bin:
        return None

    total_rows = len(occupied_rows)
    centre = page_width / 2.0
    max_offset = _COLUMN_MAX_GUTTER_OFFSET_FRAC * page_width
    boundaries: list[float] = []
    bin_index = page_first_bin
    while bin_index < page_last_bin:
        if empty_count[bin_index] / total_rows < _COLUMN_MIN_VERTICAL_COVERAGE:
            bin_index += 1
            continue
        gap_start = bin_index
        while (
            bin_index < page_last_bin
            and empty_count[bin_index] / total_rows >= _COLUMN_MIN_VERTICAL_COVERAGE
        ):
            bin_index += 1
        gap_end = bin_index  # exclusive
        gap_x0 = text_x0 + gap_start
        gap_x1 = text_x0 + gap_end
        if gap_x1 - gap_x0 < _COLUMN_MIN_GUTTER_PT:
            continue
        midpoint = (gap_x0 + gap_x1) / 2.0
        if abs(midpoint - centre) > max_offset:
            continue
        boundaries.append(midpoint)

    if not boundaries:
        return None

    edges = [text_x0, *sorted(boundaries), text_x1]
    ranges = [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo]
    if len(ranges) < 2:
        return None

    total_chars = len(chars)
    per_column = [0] * len(ranges)
    for c in chars:
        per_column[_column_index((c.x0 + c.x1) / 2.0, ranges)] += 1
    for index, (lo, hi) in enumerate(ranges):
        if (hi - lo) < _COLUMN_MIN_COLUMN_WIDTH_FRAC * page_width:
            return None
        if per_column[index] < _COLUMN_MIN_COLUMN_CHAR_FRAC * total_chars:
            return None

    if _has_nested_gutter(chars, page_width, ranges):
        return None

    return ranges


def _has_nested_gutter(chars: list[RawChar], page_width: float, ranges: list[tuple[float, float]]) -> bool:
    """True when a detected column is itself split by a gutter-sized seam.

    Without this check the detector can cut a real column in half. A stray glyph or
    a figure rule that intrudes into the gutter on only some rows leaves the band
    *beside* it empty on every row, so that wider band passes every other test --
    and the boundary lands in the middle of a column, splitting each sentence
    across two streams and losing half the page's reading order.

    The re-detection is deliberately shallow. Recursing until the leaves stop
    would let one stubborn seam disqualify a page whose columns are genuine; one
    level is enough to catch a column that is really two, which is the case that
    matters. Coordinates are translated so the sub-probe sees its own extent as a
    page of its own width, which is what keeps the centre-offset test meaningful.
    """
    for lo, hi in ranges:
        inside = [c for c in chars if lo <= (c.x0 + c.x1) / 2.0 <= hi]
        if len(inside) < _COLUMN_MIN_OCCUPIED_ROWS * 4:
            continue
        offset = lo
        shifted = [
            RawChar(
                text=c.text,
                fontname=c.fontname,
                size=c.size,
                is_bold=c.is_bold,
                x0=c.x0 - offset,
                x1=c.x1 - offset,
                top=c.top,
                bottom=c.bottom,
            )
            for c in inside
        ]
        nested = _detect_columns(shifted, hi - lo)
        if nested is not None and len(nested) >= 2:
            return True
    return False


def _cluster_lines_flat(chars: list[RawChar]) -> list[list[RawChar]]:
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


def _stream_chars(stream: list[list[RawChar]]) -> list[RawChar]:
    """Flatten a column's per-line character runs back into one character list.

    Re-clustering the flattened stream with the unchanged 3.0 pt rule is what
    makes a split line read as a line again: the two halves of one visual line
    share a baseline, so they recombine into a single line rather than two.
    """
    return [char for line in stream for char in line]


def _cluster_chars_into_lines(
    chars: list[RawChar],
    column_ranges: list[tuple[float, float]] | None = None,
) -> list[list[RawChar]]:
    """Lines in reading order, discarding the paragraph boundaries between them."""
    lines, _breaks = _cluster_lines_with_breaks(chars, column_ranges)
    return lines


def _cluster_lines_with_breaks(
    chars: list[RawChar],
    column_ranges: list[tuple[float, float]] | None = None,
) -> tuple[list[list[RawChar]], set[int]]:
    """Cluster characters into lines in reading order, honouring page columns.

    Returns the lines plus the set of line indices at which a paragraph must be
    broken. On a multi-column page that is every column transition: the last
    paragraph of the left column and the first of the right column are unrelated
    text, and joining them into one paragraph reinstates the very splice this
    function exists to remove. Splitting the lines is necessary but not
    sufficient -- without the break the two fragments are still concatenated.

    With `column_ranges=None` this is exactly `_cluster_lines_flat` -- the
    single-column path, unchanged, so a page that is not two-column produces
    byte-identical output to the pre-Phase-8 normalizer.

    With column ranges, each visual line is split **only if it spans a gutter**:
    it carries characters in two or more columns *and* leaves a real horizontal
    gap between them. That is the defect's own signature, tested directly, rather
    than a proxy for it.

    Two proxies were tried first and both are wrong. Comparing the line's extent
    to the column boundaries fails on the first page of a two-column paper, where a
    margin glyph puts the left column's geometric edge far outside the text block
    and every legitimate body line then looks like it does not reach it -- and an
    unsplit line is precisely the corruption being fixed. Deciding from font or
    text size fails too: the same page mixes 8 pt footnotes with 10 pt body text on
    a shared baseline.

    A line confined to one column, and a full-width element -- a centred title, a
    wide figure, a table row -- leave no gutter-sized gap between their parts, so
    they are emitted **intact**. Splitting a title at the gutter would break a
    heading in half and lose a section boundary, and there is nothing to gain: a
    full-width line read left to right is already in the order the column streams
    emit.

    Reading order is column by column, left to right, each top to bottom.
    """
    if column_ranges is None:
        return _cluster_lines_flat(chars), set()
    if not chars:
        return [], set()

    streams: list[list[list[RawChar]]] = [[] for _ in column_ranges]

    for line in _cluster_lines_flat(chars):
        parts: list[list[RawChar]] = [[] for _ in column_ranges]
        for c in line:
            parts[_column_index((c.x0 + c.x1) / 2.0, column_ranges)].append(c)

        occupied = [i for i, part in enumerate(parts) if part]
        if len(occupied) > 1 and _spans_gutter(parts, occupied):
            for index, part in enumerate(parts):
                if part:
                    part.sort(key=lambda ch: ch.x0)
                    streams[index].append(part)
            continue

        # No gutter on this line: intact, in the stream of the column holding most
        # of it, so a figure confined to the left column stays in the left column
        # at its own vertical position.
        counts = [len(part) for part in parts]
        streams[counts.index(max(counts))].append(line)

    lines: list[list[RawChar]] = []
    breaks: set[int] = set()
    for stream in streams:
        if not stream:
            continue
        if lines:
            # Whatever paragraph was open at the end of the previous column ends
            # here. Its last line and this column's first line are on the same
            # page but belong to different text.
            breaks.add(len(lines))
        lines.extend(_cluster_lines_flat(_stream_chars(stream)))
    return lines, breaks


def _spans_gutter(parts: list[list[RawChar]], occupied: list[int]) -> bool:
    """True when a line's column parts are separated by a real horizontal gap.

    The gap tested is the one between neighbouring occupied parts, and it must
    clear the same threshold that made the band a gutter in the first place. A
    space inside one column is a few points wide and does not clear it; the gutter
    between two columns is many, and does.
    """
    for left_index, right_index in zip(occupied, occupied[1:]):
        left_part = parts[left_index]
        right_part = parts[right_index]
        left_max = max(ch.x1 for ch in left_part)
        right_min = min(ch.x0 for ch in right_part)
        if right_min - left_max >= _COLUMN_MIN_GUTTER_PT:
            return True
    return False


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
        column_ranges = _detect_columns(rp.chars, rp.width)
        lines, paragraph_breaks = _cluster_lines_with_breaks(rp.chars, column_ranges)
        if column_ranges is not None:
            issues.append(
                ExtractionIssue(
                    code="COLUMN_LAYOUT_DETECTED",
                    severity="info",
                    message=(
                        f"Page {rp.page_number} split into {len(column_ranges)} columns "
                        f"at x={[(round(lo, 1), round(hi, 1)) for lo, hi in column_ranges]} "
                        "to preserve reading order."
                    ),
                    page=rp.page_number,
                )
            )
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

        for line_index, line in enumerate(lines):
            if line_index in paragraph_breaks:
                # The left column ended here and the right column begins. The open
                # paragraph belongs to the column that just ended.
                flush_paragraph()
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
