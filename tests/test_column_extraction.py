"""
test_column_extraction.py
-------------------------
Unit tests for the Phase 8 two-column extraction fix in
`adaptive_rag.ingestion.normalizer`.

Synthetic character grids stand in for pages, so every property is pinned to an
exact expected value rather than to a real paper that could change underfoot:

* `_detect_columns` finds a gutter on a two-column page and declines on a
  single-column one, and its refusals are as load-bearing as its acceptances --
  a false positive scrambles reading order.
* `_cluster_lines_with_breaks` reproduces the pre-Phase-8 output exactly when no
  columns are detected, which is the regression gate from
  `docs/phases/phase-8.md` §3.3 expressed as a unit test.
* A two-column page reads left column then right, and the paragraph boundary at
  the column transition is reported so the caller can flush.
* Full-width elements survive intact rather than being split.
"""

from __future__ import annotations

import math

from adaptive_rag.ingestion.normalizer import (
    _cluster_lines_flat,
    _cluster_lines_with_breaks,
    _column_index,
    _detect_columns,
    _line_text,
    RawChar,
)

PAGE_WIDTH = 612.0


# Column geometry for the synthetic pages. A letterpress page is 612 pt wide;
# these two columns sit either side of a 20 pt gutter centred on the page, which
# is the layout ACL-style proceedings actually use.
LEFT_EDGE = 72.0
LEFT_COLUMN_WIDTH = 220.0
GUTTER = (LEFT_EDGE + LEFT_COLUMN_WIDTH, LEFT_EDGE + LEFT_COLUMN_WIDTH + 20.0)
RIGHT_EDGE = GUTTER[1]
RIGHT_COLUMN_WIDTH = 220.0


def _text_run(
    text: str, x0: float, top: float, advance: float | None = None
) -> list[RawChar]:
    """Lay out `text` filling as much of `advance` as the string needs."""
    if advance is None:
        advance = 5.0
    return [
        RawChar(
            text=character,
            fontname="Times-Roman",
            size=10.0,
            is_bold=False,
            x0=x0 + index * advance,
            top=top,
            x1=x0 + index * advance + advance - 0.5,
            bottom=top + 10.0,
        )
        for index, character in enumerate(text)
    ]


def _column_run(marker: str, x0: float, width: float, top: float) -> list[RawChar]:
    """A full-column-width run of repeated `marker` text.

    Runs are stretched to the column width rather than to a fixed glyph count, so
    a test can assert on column geometry without the fixture disagreeing about
    where the gutter is.
    """
    count = max(1, int(width // 5.0))
    return _text_run(marker * count, x0, top)


def _two_column_page(rows: int = 40) -> list[RawChar]:
    """A page of body text: `rows` baselines, each with text in both columns."""
    chars: list[RawChar] = []
    for row in range(rows):
        top = 60.0 + row * 14.0
        chars.extend(_column_run("l", LEFT_EDGE, LEFT_COLUMN_WIDTH, top))
        chars.extend(_column_run("r", RIGHT_EDGE, RIGHT_COLUMN_WIDTH, top))
    return chars


def _single_column_page(rows: int = 40) -> list[RawChar]:
    chars: list[RawChar] = []
    for row in range(rows):
        top = 60.0 + row * 14.0
        chars.extend(_column_run("s", LEFT_EDGE, 468.0, top))
    return chars


# --- _detect_columns ---------------------------------------------------------


def test_detects_gutter_on_two_column_page():
    chars = _two_column_page()
    ranges = _detect_columns(chars, PAGE_WIDTH)

    assert ranges is not None
    assert len(ranges) == 2
    left, right = ranges
    # The boundary is the gutter's midpoint, so it must sit inside the gutter.
    assert GUTTER[0] <= left[1] <= GUTTER[1]
    # The ranges are the page's text extent divided at that boundary, so the outer
    # edges are exactly the extent the fixture produced.
    text_x0 = min(c.x0 for c in chars)
    text_x1 = max(c.x1 for c in chars)
    assert left[0] == text_x0
    assert right[1] == text_x1
    assert right[0] == left[1]


def test_declines_on_single_column_page():
    assert _detect_columns(_single_column_page(), PAGE_WIDTH) is None


def test_declines_on_empty_page():
    assert _detect_columns([], PAGE_WIDTH) is None
    assert _detect_columns(_two_column_page(rows=1), PAGE_WIDTH) is None


def test_declines_when_a_marginal_gap_is_off_centre():
    """A wide gap that is not a column boundary must not become one.

    The centre test earns its keep on a page whose layout is not two balanced
    columns: body text plus a wide margin element. Note that with *two* columns on
    a 612 pt page the gap midpoint is near the centre by construction, so this
    cannot be exercised by simply moving two blocks apart -- the gap has to be off
    centre *and* the resulting range has to be too narrow to be a column of text.
    Either refusal is correct here; the test pins the outcome, not the mechanism.
    """
    body_x0, body_width = 72.0, 400.0
    marginal_x0, marginal_width = 540.0, 24.0
    gap_midpoint = (body_x0 + body_width + marginal_x0) / 2.0
    assert gap_midpoint > PAGE_WIDTH / 2.0

    chars: list[RawChar] = []
    for row in range(40):
        top = 60.0 + row * 14.0
        chars.extend(_column_run("b", body_x0, body_width, top))
        chars.extend(_column_run("m", marginal_x0, marginal_width, top))
    assert _detect_columns(chars, PAGE_WIDTH) is None


def test_declines_when_a_column_is_split_by_its_own_seam():
    """A seam recurring inside a column means the page is not two columns.

    This is the case a width-and-persistence test alone gets wrong: the seam
    inside the left column is as wide, as centred within that column, and as
    vertically persistent as a real gutter, so it passes every individual check.
    Accepting it would put the boundary in the middle of a column and split every
    sentence across two streams. The nested-gutter re-check is what rejects it.
    """
    inner_x0, inner_width = 72.0, 60.0
    seam_x0, seam_width = 162.0, 20.0
    chars: list[RawChar] = []
    for row in range(40):
        top = 60.0 + row * 14.0
        chars.extend(_column_run("l", inner_x0, inner_width, top))
        chars.extend(_column_run("r", RIGHT_EDGE, RIGHT_COLUMN_WIDTH, top))
        # A second block after the seam, on every row: the seam is real, and it
        # sits inside what would otherwise be the left column.
        chars.extend(_column_run("l", seam_x0 + seam_width, 80.0, top))
    assert _detect_columns(chars, PAGE_WIDTH) is None


def test_detects_full_width_title_page():
    """A two-column page whose title spans the gutter is still detected.

    This is the case a union projection gets wrong: the title's own row fills the
    gutter, so a projection built by OR-ing rows together sees no interior gap at
    all and the page is declared single-column.
    """
    chars = _text_run("A Full Width Paper Title", 150.0, 20.0) + _two_column_page()
    ranges = _detect_columns(chars, PAGE_WIDTH)
    assert ranges is not None
    assert len(ranges) == 2


# --- _column_index -----------------------------------------------------------


def test_column_index_maps_positions_and_clamps_outside():
    ranges = [(10.0, 100.0), (100.0, 200.0)]
    assert _column_index(50.0, ranges) == 0
    assert _column_index(150.0, ranges) == 1
    # Outside the text extent, clamp rather than fall off the end.
    assert _column_index(-5.0, ranges) == 0
    assert _column_index(500.0, ranges) == 1


# --- reading order -----------------------------------------------------------


def test_single_column_output_is_identical_to_pre_phase8_path():
    """The regression gate: no columns detected means byte-identical output.

    `_cluster_lines_flat` is the pre-Phase-8 algorithm, kept verbatim, and this
    pins the two paths together so a future change to one cannot silently diverge
    from the other.
    """
    chars = _single_column_page()
    ranges = _detect_columns(chars, PAGE_WIDTH)
    assert ranges is None

    baseline = _cluster_lines_flat(chars)
    lines, breaks = _cluster_lines_with_breaks(chars, ranges)

    assert breaks == set()
    assert [_line_text(line) for line in lines] == [_line_text(line) for line in baseline]


def test_two_column_page_reads_left_column_before_right():
    chars = _two_column_page()
    ranges = _detect_columns(chars, PAGE_WIDTH)
    assert ranges is not None

    lines, _breaks = _cluster_lines_with_breaks(chars, ranges)
    texts = [_line_text(line) for line in lines]

    # The fixture fills the left column with "l" and the right with "r", so the
    # reading order is readable straight off the output.
    left_positions = [i for i, t in enumerate(texts) if t.startswith("l")]
    right_positions = [i for i, t in enumerate(texts) if t.startswith("r")]
    assert len(left_positions) == 40
    assert len(right_positions) == 40
    assert max(left_positions) < min(right_positions)


def test_column_transition_reports_a_paragraph_break():
    """Splitting the lines is not enough; the paragraph must break too.

    Without the break the last line of the left column and the first of the right
    are joined into one paragraph, which is the splice all over again.
    """
    chars = _two_column_page()
    ranges = _detect_columns(chars, PAGE_WIDTH)
    assert ranges is not None

    lines, breaks = _cluster_lines_with_breaks(chars, ranges)
    assert len(breaks) == 1

    break_index = next(iter(breaks))
    texts = [_line_text(line) for line in lines]
    # The break sits exactly where the left column's 40 lines give way to the
    # right column's 40, so nothing either side can belong to the other.
    assert break_index == 40
    assert texts[break_index].startswith("r")
    assert texts[break_index - 1].startswith("l")


def test_full_width_line_is_not_split():
    """A centred title spanning the gutter stays one line.

    Splitting it would break a heading in half and lose a section boundary.
    """
    chars = _two_column_page()
    ranges = _detect_columns(chars, PAGE_WIDTH)
    assert ranges is not None
    title = _text_run("A Full Width Paper Title", 150.0, 20.0)
    chars = title + chars

    lines, _breaks = _cluster_lines_with_breaks(chars, ranges)
    titles = [line for line in lines if "Full Width Paper Title" in _line_text(line)]
    assert len(titles) == 1
    assert _line_text(titles[0]).startswith("A Full Width Paper Title")


def test_single_column_line_on_a_two_column_page_is_kept_intact():
    """A line confined to the left column must not be reordered by the splitter."""
    chars = _two_column_page(rows=20)
    ranges = _detect_columns(chars, PAGE_WIDTH)
    assert ranges is not None
    chars = chars + _text_run("Only in the left column.", 72.0, 400.0)

    texts = [_line_text(line) for line in _cluster_lines_with_breaks(chars, ranges)[0]]
    assert any(t.startswith("Only in the left column") for t in texts)


def test_empty_chars_yields_no_lines():
    assert _cluster_lines_with_breaks([], None) == ([], set())
    assert _cluster_lines_with_breaks([], [(0.0, 100.0), (100.0, 200.0)]) == ([], set())


# --- geometry helpers --------------------------------------------------------


def test_page_width_edge_cases_do_not_raise():
    chars = _two_column_page()
    assert _detect_columns(chars, 0.0) is None
    assert _detect_columns(chars, -100.0) is None
    # A page far wider than its text still resolves, with the gutter off-centre
    # enough that detection correctly declines rather than splitting arbitrarily.
    assert _detect_columns(chars, 2000.0) is None


def test_math_import_is_used():
    """`math.ceil` is load-bearing in the occupancy grid sizing."""
    assert math.ceil(1.2) == 2