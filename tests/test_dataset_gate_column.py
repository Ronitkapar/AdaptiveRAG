"""
tests.test_dataset_gate_column
------------------------------
Offline tests for G7, the opt-in column-contiguity check in
`scripts/validate_phase7_dataset.py`.

G4 asks "is this quote a verbatim substring of the chunk text?". On a
two-column page that is not enough: the chunker reads both columns in one pass,
so a span that leaves the bottom of the left column and lands in the right one
is still a contiguous substring of the flattened text -- it just says something
the paper never said. These tests build tiny two-column PDFs in `tmp_path`,
flatten them into chunk text exactly as the chunker would, and assert the
verdicts: a quote inside one column is verified, one that crosses the break is
`contaminated`, a single-column page is never flagged, and a page the analyser
cannot split confidently is `indeterminate` rather than an accusation.

Nothing here touches the gitignored corpus or `data/raw/`: every PDF, chunk,
manifest, dataset, and ledger is written under `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pdfplumber
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from scripts.validate_phase7_dataset import (
    RAW_ABSENT_MESSAGE,
    main,
    page_column_runs,
    run_gate,
)
from tests.fixtures.make_fixture_pdf import create_synthetic_paper_pdf

DOC_SINGLE = "doc_single"
DOC_TWO_COLUMN = "doc_two_column"
DOC_UNSPLITTABLE = "doc_unsplittable"

# --- fixture geometry ---------------------------------------------------------
#
# Two columns on a letter page with a wide gutter, the right column's baselines
# offset by 4pt so each column forms its own character clusters (which is what a
# real two-column PDF looks like to pdfplumber, and what makes the flattened
# text interleave the two columns line by line).
LEFT_X, RIGHT_X, FIRST_BASELINE, ROW_PITCH, COLUMN_OFFSET = 54.0, 330.0, 720.0, 14.0, 4.0
ROWS = 12
THREE_COLUMN_X = (54.0, 234.0, 414.0)

LEFT_LINE = f"left column line {{i}} alpha beta gamma delta epsilon zeta eta theta"
RIGHT_LINE = f"right head marker {{i}} iota kappa lambda mu nu xi omicron pi"
THREE_COLUMN_LINE = f"col{{k}} line{{i}} alpha beta gamma delta"

# A quote that sits wholly inside the left column.
CLEAN_QUOTE = "left column line 3 alpha beta gamma delta epsilon"
# The last words of left line 0 followed by the first words of right line 0 --
# contiguous in the flattened page text, in no single column.
CROSSING_QUOTE = "epsilon zeta eta theta right head marker 0 iota"
# Same construction on the three-column page, which cannot be split at all.
UNSPLITTABLE_QUOTE = "alpha beta gamma delta col1 line0 alpha"

ANSWER_TERM = "epsilon"


def _write_two_column_pdf(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    page = canvas.Canvas(str(dest), pagesize=letter)
    page.setFont("Helvetica", 9)
    for i in range(ROWS):
        page.drawString(LEFT_X, FIRST_BASELINE - ROW_PITCH * i, LEFT_LINE.format(i=i))
        page.drawString(
            RIGHT_X,
            FIRST_BASELINE - COLUMN_OFFSET - ROW_PITCH * i,
            RIGHT_LINE.format(i=i),
        )
    page.showPage()
    page.save()
    return dest


def _write_three_column_pdf(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    page = canvas.Canvas(str(dest), pagesize=letter)
    page.setFont("Helvetica", 9)
    for i in range(ROWS):
        for k, x in enumerate(THREE_COLUMN_X):
            page.drawString(
                x,
                FIRST_BASELINE - COLUMN_OFFSET * k - ROW_PITCH * i,
                THREE_COLUMN_LINE.format(k=k, i=i),
            )
    page.showPage()
    page.save()
    return dest


def _flattened_page_text(pdf_path: Path, page_no: int) -> str:
    """Page text in reading order, i.e. what the chunker puts in a chunk.

    Deliberately the same reconstruction the gate uses, so the chunk text in
    these tests really does contain both columns flattened together -- which is
    the whole reason G4 cannot see a column break.
    """
    with pdfplumber.open(pdf_path) as pdf:
        return page_column_runs(pdf.pages[page_no - 1])["flat"]


def _write_chunks(chunks_dir: Path, entries: list[tuple[str, str, str]]) -> Path:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    by_document: dict[str, list[dict[str, Any]]] = {}
    for document_id, chunk_id, text in entries:
        by_document.setdefault(document_id, []).append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": text,
                "metadata": {"section_path": ["1 Body"]},
                "provenance": {"document_id": document_id, "pages": [1]},
            }
        )
    for document_id, records in by_document.items():
        with open(chunks_dir / f"{document_id}.chunks.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return chunks_dir


def _write_manifest(path: Path, document_ids: list[str]) -> Path:
    path.write_text(
        json.dumps([{"document_id": document_id} for document_id in document_ids]),
        encoding="utf-8",
    )
    return path


def _example(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": "p1",
        "query": "What does the left column say about the marker?",
        "reference_answer": "The left column reports the marker value.",
        "relevant_documents": [DOC_TWO_COLUMN],
        "relevant_sections": [
            {"document_id": DOC_TWO_COLUMN, "section_path_prefix": ["1 Body"]}
        ],
        "relevant_chunks": [],
        "category": "factual",
        "requires_multi_hop": False,
        "notes": "",
        "dataset_version": "column_test_ds",
        "split": "calibration",
    }
    record.update(overrides)
    return record


def _ledger(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": "p1",
        "split": "calibration",
        "relevant_documents": [DOC_TWO_COLUMN],
        "evidence": [
            {
                "document_id": DOC_TWO_COLUMN,
                "chunk_id": f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001",
                "quote": CLEAN_QUOTE,
            }
        ],
        "answer_terms": [ANSWER_TERM],
        "curator_notes": "The left column states the marker plainly.",
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _pair(**overrides: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One calibration record plus one test record (S4 needs both splits)."""
    second_example = _example(example_id="p2", split="test")
    second_ledger = _ledger(example_id="p2", split="test")
    return (
        [overrides.pop("example", _example()), second_example],
        [overrides.pop("ledger", _ledger()), second_ledger],
    )


def _column_record(report: dict[str, Any], example_id: str = "p1") -> dict[str, Any]:
    return next(r for r in report["records"] if r["example_id"] == example_id)["column_check"]


def _checks(report: dict[str, Any]) -> set[str]:
    return {failure["check"] for failure in report["failures"]}


def _reasons(report: dict[str, Any]) -> str:
    return " | ".join(failure["reason"] for failure in report["failures"])


@pytest.fixture()
def sources(tmp_path: Path) -> dict[str, Path]:
    """Three source PDFs: two-column, single-column, and unsplittable."""
    return {
        DOC_TWO_COLUMN: _write_two_column_pdf(tmp_path / "raw" / f"{DOC_TWO_COLUMN}.pdf"),
        DOC_SINGLE: create_synthetic_paper_pdf(tmp_path / "raw" / f"{DOC_SINGLE}.pdf"),
        DOC_UNSPLITTABLE: _write_three_column_pdf(
            tmp_path / "raw" / f"{DOC_UNSPLITTABLE}.pdf"
        ),
    }


@pytest.fixture()
def corpus(tmp_path: Path, sources: dict[str, Path]) -> tuple[Path, Path]:
    """Chunk corpus (flattened two-column text) plus a matching manifest."""
    chunks_dir = _write_chunks(
        tmp_path / "chunks",
        [
            (
                DOC_TWO_COLUMN,
                f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001",
                _flattened_page_text(sources[DOC_TWO_COLUMN], 1),
            ),
            (
                DOC_UNSPLITTABLE,
                f"{DOC_UNSPLITTABLE}::structure_aware_v1::c00001",
                _flattened_page_text(sources[DOC_UNSPLITTABLE], 1),
            ),
            (
                DOC_SINGLE,
                f"{DOC_SINGLE}::structure_aware_v1::c00001",
                _flattened_page_text(sources[DOC_SINGLE], 1),
            ),
        ],
    )
    manifest = _write_manifest(
        tmp_path / "papers.json",
        [DOC_TWO_COLUMN, DOC_SINGLE, DOC_UNSPLITTABLE],
    )
    return chunks_dir, manifest


def _run(
    tmp_path: Path,
    corpus: tuple[Path, Path],
    examples: list[dict[str, Any]],
    ledgers: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    chunks_dir, manifest = corpus
    dataset = _write_jsonl(tmp_path / "ds.jsonl", examples)
    ledger = _write_jsonl(tmp_path / "lg.jsonl", ledgers)
    kwargs.setdefault("min_per_category", 1)
    kwargs.setdefault("raw_dir", tmp_path / "raw")
    return run_gate(
        dataset,
        ledger,
        chunks_dir=chunks_dir,
        manifest_path=manifest,
        **kwargs,
    )


# --- layout analysis ----------------------------------------------------------


def test_two_column_fixture_is_recognised_as_two_column(sources: dict[str, Path]) -> None:
    """The fixture really is two columns; otherwise nothing below means anything."""
    with pdfplumber.open(sources[DOC_TWO_COLUMN]) as pdf:
        runs = page_column_runs(pdf.pages[0])

    assert runs["layout"] == "two_column"
    assert len(runs["columns"]) == 3  # left, right, gutter-spanning lines
    assert CROSSING_QUOTE in runs["flat"]
    assert all(CROSSING_QUOTE not in run for run in runs["columns"])
    assert CLEAN_QUOTE in runs["columns"][0]


def test_single_column_fixture_is_not_split(sources: dict[str, Path]) -> None:
    """A page with no gutter has no columns to contaminate."""
    with pdfplumber.open(sources[DOC_SINGLE]) as pdf:
        layouts = [page_column_runs(page)["layout"] for page in pdf.pages]

    assert layouts == ["single", "single"]


def test_three_column_fixture_is_indeterminate(sources: dict[str, Path]) -> None:
    """Three columns are two columns plus a seam; the seam must not be guessed at."""
    with pdfplumber.open(sources[DOC_UNSPLITTABLE]) as pdf:
        runs = page_column_runs(pdf.pages[0])

    assert runs["layout"] == "indeterminate"
    assert runs["info"]["nested_gutters"] >= 1


# --- G7 verdicts --------------------------------------------------------------


def test_quote_inside_one_column_is_verified(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    """The happy path: a quote contained in one column passes G7 and G4."""
    examples, ledgers = _pair()
    report = _run(tmp_path, corpus, examples, ledgers, check_column_contiguity=True)

    assert report["column_check"] == "enabled"
    assert report["summary"]["gate_open"] is True
    assert _column_record(report)["status"] == "verified"
    assert _column_record(report)["evidence"][0]["status"] == "verified"
    assert report["column_check_counts"] == {
        "quotes": 2,
        "verified": 2,
        "contaminated": 0,
        "indeterminate": 0,
    }


def test_quote_spanning_column_break_is_contaminated(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    """A spliced quote passes G4 verbatim and must still fail the gate."""
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_TWO_COLUMN,
                "chunk_id": f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001",
                "quote": CROSSING_QUOTE,
            }
        ],
        answer_terms=["marker"],
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers, check_column_contiguity=True)

    assert report["column_check_counts"]["contaminated"] == 1
    assert "G7" in _checks(report)
    assert report["summary"]["gate_open"] is False
    assert report["summary"]["failed_example_ids"] == ["p1"]
    assert _column_record(report)["status"] == "contaminated"

    # The reason must carry enough for a human to find the offending evidence.
    reason = _reasons(report)
    assert "never inside a single column" in reason
    assert CROSSING_QUOTE in reason
    assert f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001" in reason
    # G4 still passed -- this is exactly the hole being closed.
    assert "G4" not in _checks(report)


def test_single_column_quote_is_not_flagged(tmp_path: Path, corpus: tuple[Path, Path]) -> None:
    """A single-column source has no column break to cross, so G7 stays silent."""
    quote = "This paper evaluates dense retrieval pipelines."
    ledger = _ledger(
        relevant_documents=[DOC_SINGLE],
        evidence=[
            {
                "document_id": DOC_SINGLE,
                "chunk_id": f"{DOC_SINGLE}::structure_aware_v1::c00001",
                "quote": quote,
            }
        ],
        answer_terms=["dense retrieval pipelines"],
    )
    example = _example(
        relevant_documents=[DOC_SINGLE],
        relevant_sections=[{"document_id": DOC_SINGLE, "section_path_prefix": ["1 Body"]}],
    )
    examples, ledgers = _pair(example=example, ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers, check_column_contiguity=True)

    assert report["column_check_counts"]["contaminated"] == 0
    assert report["column_check_counts"]["indeterminate"] == 0
    assert report["summary"]["gate_open"] is True


def test_unsplittable_page_is_indeterminate_not_contaminated(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    """When the page cannot be split confidently the check must not accuse."""
    ledger = _ledger(
        relevant_documents=[DOC_UNSPLITTABLE],
        evidence=[
            {
                "document_id": DOC_UNSPLITTABLE,
                "chunk_id": f"{DOC_UNSPLITTABLE}::structure_aware_v1::c00001",
                "quote": UNSPLITTABLE_QUOTE,
            }
        ],
        answer_terms=["alpha"],
    )
    example = _example(
        relevant_documents=[DOC_UNSPLITTABLE],
        relevant_sections=[
            {"document_id": DOC_UNSPLITTABLE, "section_path_prefix": ["1 Body"]}
        ],
    )
    examples, ledgers = _pair(example=example, ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers, check_column_contiguity=True)

    assert UNSPLITTABLE_QUOTE in _flattened_page_text(
        Path(report["column_check_detail"]["raw_dir"]) / f"{DOC_UNSPLITTABLE}.pdf", 1
    )
    assert report["column_check_counts"]["contaminated"] == 0
    # p1's quote sits on the unsplittable page (indeterminate); p2's sits on
    # the clean two-column page (verified).
    assert report["column_check_counts"]["indeterminate"] == 1
    assert report["column_check_counts"]["verified"] == 1
    assert _column_record(report)["status"] == "indeterminate"
    assert "G7" not in _checks(report)
    assert report["summary"]["gate_open"] is True


def test_chunk_without_page_provenance_is_indeterminate(
    tmp_path: Path, sources: dict[str, Path]
) -> None:
    """A chunk with no `provenance` block must not crash the gate or be flagged."""
    chunks_dir = _write_chunks(
        tmp_path / "chunks",
        [
            (
                DOC_TWO_COLUMN,
                f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001",
                _flattened_page_text(sources[DOC_TWO_COLUMN], 1),
            )
        ],
    )
    records_path = chunks_dir / f"{DOC_TWO_COLUMN}.chunks.jsonl"
    lines = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record.pop("provenance")
        lines.append(json.dumps(record))
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = _write_manifest(tmp_path / "papers.json", [DOC_TWO_COLUMN])
    examples, ledgers = _pair()
    report = _run(
        tmp_path,
        (chunks_dir, manifest),
        examples,
        ledgers,
        check_column_contiguity=True,
    )

    assert report["column_check_counts"]["indeterminate"] == 2
    assert report["column_check_counts"]["contaminated"] == 0
    assert "no page provenance" in _column_record(report)["evidence"][0]["detail"]
    assert report["summary"]["gate_open"] is True


# --- the check is opt-in ------------------------------------------------------


def test_check_is_off_by_default(tmp_path: Path, corpus: tuple[Path, Path]) -> None:
    """Without the flag nothing changes and nothing is claimed."""
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_TWO_COLUMN,
                "chunk_id": f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001",
                "quote": CROSSING_QUOTE,
            }
        ],
        answer_terms=["marker"],
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["column_check"] == "disabled"
    assert report["column_check_counts"] == {
        "quotes": 0,
        "verified": 0,
        "contaminated": 0,
        "indeterminate": 0,
    }
    assert report["summary"]["gate_open"] is True
    assert _column_record(report)["status"] == "disabled"
    assert "passed" not in json.dumps(report["column_check"])


def test_disabled_is_never_reported_as_a_pass(tmp_path: Path, corpus: tuple[Path, Path]) -> None:
    """A skipped check and a clean check must not look alike in the artifact."""
    examples, ledgers = _pair()
    off = _run(tmp_path / "off", corpus, examples, ledgers, raw_dir=tmp_path / "raw")
    on = _run(
        tmp_path / "on",
        corpus,
        examples,
        ledgers,
        check_column_contiguity=True,
        raw_dir=tmp_path / "raw",
    )

    assert off["column_check"] != on["column_check"]
    assert on["column_check"] == "enabled"
    assert off["column_check"] == "disabled"
    assert off["column_check_detail"]["raw_corpus_present"] is True


def test_absent_raw_corpus_exits_non_zero_with_rebuild_message(
    tmp_path: Path, corpus: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """No raw PDFs means no G7 verdict; the run must fail and say how to fix it."""
    chunks_dir, manifest = corpus
    examples, ledgers = _pair()
    dataset = _write_jsonl(tmp_path / "ds.jsonl", examples)
    ledger = _write_jsonl(tmp_path / "lg.jsonl", ledgers)
    missing_raw = tmp_path / "no_raw_corpus"
    missing_raw.mkdir()
    artifact = tmp_path / "gate.json"

    exit_code = main([
        "--dataset", str(dataset),
        "--ledger", str(ledger),
        "--chunks-dir", str(chunks_dir),
        "--manifest", str(manifest),
        "--raw-dir", str(missing_raw),
        "--out", str(artifact),
        "--check-column-contiguity",
        "--json-only",
    ])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "download_corpus.py" in stderr
    assert "column-contiguity" in stderr
    report = json.loads(artifact.read_text(encoding="utf-8"))
    assert report["column_check"] == "indeterminate"
    assert report["column_check_detail"]["raw_corpus_present"] is False
    assert report["column_check_counts"] == {
        "quotes": 0,
        "verified": 0,
        "contaminated": 0,
        "indeterminate": 0,
    }
    assert report["summary"]["gate_open"] is False
    assert "G7" in _checks(report)


def test_absent_raw_corpus_message_names_the_rebuild_chain() -> None:
    message = RAW_ABSENT_MESSAGE.format(raw_dir="data/raw")
    assert "download_corpus.py" in message
    assert "--check-column-contiguity" in message


# --- non-vacuity --------------------------------------------------------------


def test_gate_is_not_vacuous_across_a_column_break(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    """Two quotes differing only in crossing the break must get opposite verdicts.

    Without this, a G7 that ignored its input and always answered "verified"
    would pass every other test in this file.
    """
    clean = _ledger()
    crossing = _ledger(
        evidence=[
            {
                "document_id": DOC_TWO_COLUMN,
                "chunk_id": f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001",
                "quote": CROSSING_QUOTE,
            }
        ],
        answer_terms=["marker"],
    )
    good_examples, good_ledgers = _pair(ledger=clean)
    bad_examples, bad_ledgers = _pair(ledger=crossing)

    good = _run(
        tmp_path / "good",
        corpus,
        good_examples,
        good_ledgers,
        check_column_contiguity=True,
        raw_dir=tmp_path / "raw",
    )
    bad = _run(
        tmp_path / "bad",
        corpus,
        bad_examples,
        bad_ledgers,
        check_column_contiguity=True,
        raw_dir=tmp_path / "raw",
    )

    assert good["summary"]["gate_open"] is True
    assert bad["summary"]["gate_open"] is False
    assert good["column_check_counts"]["verified"] == 2
    assert bad["column_check_counts"]["contaminated"] == 1
    # Both quotes are verbatim substrings of the same chunk text, so G4 sees no
    # difference at all -- only the column analysis does.
    chunk_text = _flattened_page_text(tmp_path / "raw" / f"{DOC_TWO_COLUMN}.pdf", 1)
    assert CLEAN_QUOTE in chunk_text and CROSSING_QUOTE in chunk_text


# --- the other checks are untouched -------------------------------------------


def test_g4_still_rejects_a_fabricated_quote_with_the_column_check_on(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_TWO_COLUMN,
                "chunk_id": f"{DOC_TWO_COLUMN}::structure_aware_v1::c00001",
                "quote": "left column line 3 alpha beta gamma delta omega",
            }
        ]
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers, check_column_contiguity=True)

    assert "G4" in _checks(report)
    assert "not a verbatim substring" in _reasons(report)
    assert "G7" not in _checks(report)


def test_g5_still_rejects_an_ungrounded_answer_with_the_column_check_on(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    ledger = _ledger(answer_terms=["epsilon", "reciprocal rank fusion"])
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers, check_column_contiguity=True)

    assert "G5" in _checks(report)
    assert "G7" not in _checks(report)


def test_g3_and_g6_still_fire_with_the_column_check_on(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    example = _example(
        relevant_sections=[
            {"document_id": DOC_TWO_COLUMN, "section_path_prefix": ["9 Nonexistent"]}
        ]
    )
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_TWO_COLUMN,
                "chunk_id": f"{DOC_TWO_COLUMN}::structure_aware_v1::c09999",
                "quote": CLEAN_QUOTE,
            }
        ]
    )
    examples, ledgers = _pair(example=example, ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers, check_column_contiguity=True)

    assert {"G3", "G6"} <= _checks(report)
    assert report["column_check_counts"]["contaminated"] == 0
