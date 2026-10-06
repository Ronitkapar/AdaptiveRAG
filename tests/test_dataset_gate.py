"""
tests.test_dataset_gate
-----------------------
Offline tests for `scripts/validate_phase7_dataset.py`.

These never touch the gitignored canonical corpus: each test builds a two-document
chunk corpus of a few hundred characters in `tmp_path`, so the grounding checks are
exercised against real files while the suite stays deterministic and offline. The
point of the module is that it *rejects* bad labels, so most tests are negative:
each asserts a specific check fires on a specific, minimal defect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.validate_phase7_dataset import (
    CORPUS_ABSENT_MESSAGE,
    main,
    run_gate,
)

DOC_A = "doc_a"
DOC_B = "doc_b"

# Real-ish chunk text. Kept long enough that a legitimate quote clears the
# 20-character minimum without being the whole chunk.
TEXT_A = (
    "## 1 Intro\n\n"
    "The retrieval corpus uses a two stage pipeline. The first stage builds an index "
    "over the passage collection and the second stage reranks the candidates. "
    "Saturation of the term weight is what\nkeeps repeated occurrences bounded."
)
TEXT_B = (
    "## 2 Body\n\n"
    "A separate document describes the relevance feedback loop and the notion of a "
    "probability of relevance for each query document pair. The estimator is described "
    "over the following paragraphs of discussion."
)

CHUNK_A = f"{DOC_A}::structure_aware_v1::c00001"
CHUNK_B = f"{DOC_B}::structure_aware_v1::c00001"

QUOTE_A = "The first stage builds an index over the passage collection"
QUOTE_B = "the notion of a probability of relevance for each query document pair"


# --- fixtures -----------------------------------------------------------------


def _write_chunks(chunks_dir: Path) -> Path:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "chunk_id": CHUNK_A,
            "document_id": DOC_A,
            "text": TEXT_A,
            "metadata": {"section_path": ["1 Intro"]},
        },
        {
            "chunk_id": CHUNK_B,
            "document_id": DOC_B,
            "text": TEXT_B,
            "metadata": {"section_path": ["2 Body", "2.1 Relevance"]},
        },
    ]
    path = chunks_dir / f"{DOC_A}.chunks.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return chunks_dir


def _write_manifest(path: Path) -> Path:
    path.write_text(
        json.dumps([{"document_id": DOC_A}, {"document_id": DOC_B}]),
        encoding="utf-8",
    )
    return path


def _example(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": "p1",
        "query": "How is the index built in the first stage?",
        "reference_answer": "The first stage builds an index over the passage collection.",
        "relevant_documents": [DOC_A],
        "relevant_sections": [
            {"document_id": DOC_A, "section_path_prefix": ["1 Intro"]}
        ],
        "relevant_chunks": [],
        "category": "factual",
        "requires_multi_hop": False,
        "notes": "",
        "dataset_version": "test_ds",
        "split": "calibration",
    }
    record.update(overrides)
    return record


def _ledger(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": "p1",
        "split": "calibration",
        "relevant_documents": [DOC_A],
        "evidence": [
            {"document_id": DOC_A, "chunk_id": CHUNK_A, "quote": QUOTE_A}
        ],
        "answer_terms": ["passage collection"],
        "curator_notes": "The index build is described in the intro of doc_a.",
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny two-document chunk corpus plus a matching manifest."""
    return _write_chunks(tmp_path / "chunks"), _write_manifest(tmp_path / "papers.json")


def _run(
    tmp_path: Path,
    corpus: tuple[Path, Path],
    examples: list[dict[str, Any]],
    ledgers: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the gate over explicitly written dataset/ledger files."""
    chunks_dir, manifest = corpus
    dataset = _write_jsonl(tmp_path / "ds.jsonl", examples)
    ledger = _write_jsonl(tmp_path / "lg.jsonl", ledgers)
    kwargs.setdefault("min_per_category", 1)
    return run_gate(
        dataset, ledger, chunks_dir=chunks_dir, manifest_path=manifest, **kwargs
    )


def _pair(**overrides: Any) -> list[dict[str, Any], list[dict[str, Any]]]:
    """One calibration example plus a test example, both about doc_a.

    Two records are the minimum that satisfies S4 (both splits non-empty).
    """
    second_example = _example(
        example_id="p2",
        query="What does the second stage do?",
        split="test",
    )
    second_ledger = _ledger(example_id="p2", split="test")
    return [overrides.pop("example", _example()), second_example], [
        overrides.pop("ledger", _ledger()),
        second_ledger,
    ]


def _checks(report: dict[str, Any]) -> set[str]:
    return {failure["check"] for failure in report["failures"]}


def _reasons(report: dict[str, Any]) -> str:
    return " | ".join(failure["reason"] for failure in report["failures"])


# --- positive baseline --------------------------------------------------------


def test_valid_dataset_and_ledger_passes(tmp_path: Path, corpus: tuple[Path, Path]):
    """The happy path: two grounded records, both splits populated, gate open."""
    examples, ledgers = _pair()
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["summary"]["gate_open"] is True
    assert report["grounding_verified"] is True
    assert report["summary"]["records_pass"] == 2
    assert report["summary"]["grounded"] == 2
    assert report["summary"]["grounding_failures"] == 0
    assert report["failures"] == []
    assert report["counts"]["by_split"] == {"calibration": 1, "test": 1}


def test_quote_may_span_newlines(tmp_path: Path, corpus: tuple[Path, Path]):
    """Comparison is whitespace-normalised, so a quote copied across a newline passes."""
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_A,
                "chunk_id": CHUNK_A,
                "quote": "the term weight is what\nkeeps repeated occurrences",
            }
        ],
        answer_terms=["repeated occurrences"],
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["summary"]["gate_open"] is True
    assert report["summary"]["grounded"] == 2


# --- G4: fabricated quote -----------------------------------------------------


def test_tampered_quote_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """A quote that is not in the chunk text is a fabricated label and must fail."""
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_A,
                "chunk_id": CHUNK_A,
                "quote": "The first stage builds an inverted index over the corpus",
            }
        ]
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["summary"]["gate_open"] is False
    assert "G4" in _checks(report)
    assert "not a verbatim substring" in _reasons(report)
    # Only p1 is tampered; p2 must still pass, so this is not a blanket failure.
    assert report["summary"]["failed_example_ids"] == ["p1"]
    assert report["summary"]["records_pass"] == 1


def test_quote_below_minimum_length_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """A 12-character quote is not evidence; it clears nothing on its own."""
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_A,
                "chunk_id": CHUNK_A,
                "quote": "two stage pipeline",
            }
        ]
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G4" in _checks(report)
    assert "below the 20-char minimum" in _reasons(report)


# --- G5: ungrounded answer ----------------------------------------------------


def test_answer_term_absent_from_quotes_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """An answer term that appears in no verified quote is invented content."""
    ledger = _ledger(answer_terms=["passage collection", "reciprocal rank fusion"])
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G5" in _checks(report)
    assert "reciprocal rank fusion" in _reasons(report)


def test_answer_term_match_is_case_insensitive(tmp_path: Path, corpus: tuple[Path, Path]):
    """Answer terms are matched case-insensitively, so casing cannot fail a label."""
    ledger = _ledger(answer_terms=["PASSAGE COLLECTION"])
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["summary"]["gate_open"] is True


# --- G3: bad evidence pointers ------------------------------------------------


def test_chunk_id_not_on_disk_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """An invented chunk_id must fail: nothing on disk backs the quote."""
    ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_A,
                "chunk_id": f"{DOC_A}::structure_aware_v1::c09999",
                "quote": QUOTE_A,
            }
        ]
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G3" in _checks(report)
    assert "not found in corpus" in _reasons(report)


def test_evidence_document_not_in_relevant_documents_fails(
    tmp_path: Path, corpus: tuple[Path, Path]
):
    """Quoting a chunk from a document the record does not label is incoherent."""
    ledger = _ledger(
        relevant_documents=[DOC_A, DOC_B],
        evidence=[{"document_id": DOC_B, "chunk_id": CHUNK_B, "quote": QUOTE_B}],
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G3" in _checks(report)
    assert "is not in relevant_documents" in _reasons(report)


def test_evidence_document_mismatch_against_chunk_fails(
    tmp_path: Path, corpus: tuple[Path, Path]
):
    """The evidence's document_id must match the document the chunk actually is."""
    ledger = _ledger(
        relevant_documents=[DOC_A, DOC_B],
        evidence=[{"document_id": DOC_B, "chunk_id": CHUNK_A, "quote": QUOTE_A}],
    )
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G3" in _checks(report)
    assert "evidence claims" in _reasons(report)


# --- G6: dead section label ---------------------------------------------------


def test_section_prefix_matching_no_chunk_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """A section label pointing at nothing would zero out derived chunk relevance."""
    example = _example(
        relevant_sections=[
            {"document_id": DOC_A, "section_path_prefix": ["9 Nonexistent Section"]}
        ]
    )
    examples, ledgers = _pair(example=example)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G6" in _checks(report)
    assert "matches no chunk" in _reasons(report)


def test_deeper_section_prefix_also_matches(tmp_path: Path, corpus: tuple[Path, Path]):
    """Prefix matching is per element, so ['2 Body', '2.1 Relevance'] matches."""
    example = _example(
        relevant_documents=[DOC_B],
        relevant_sections=[
            {"document_id": DOC_B, "section_path_prefix": ["2 Body", "2.1 Relevance"]}
        ],
    )
    ledger = _ledger(
        relevant_documents=[DOC_B],
        evidence=[{"document_id": DOC_B, "chunk_id": CHUNK_B, "quote": QUOTE_B}],
        answer_terms=["probability of relevance"],
    )
    examples, ledgers = _pair(example=example, ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["summary"]["gate_open"] is True


# --- G1/G2: ledger correspondence ---------------------------------------------


def test_ledger_split_disagreement_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """The ledger's asserted split must equal the record's split."""
    ledger = _ledger(split="test")
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G2" in _checks(report)
    assert "ledger split" in _reasons(report)


def test_ledger_relevant_documents_disagreement_fails(
    tmp_path: Path, corpus: tuple[Path, Path]
):
    ledger = _ledger(relevant_documents=[DOC_B])
    examples, ledgers = _pair(ledger=ledger)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G2" in _checks(report)
    assert "ledger relevant_documents" in _reasons(report)


def test_record_without_ledger_record_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """A dataset record with no provenance is unprovenanced and must fail."""
    examples, ledgers = _pair()
    examples.append(_example(example_id="p3"))
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G1" in _checks(report)
    assert "no ledger record" in _reasons(report)


def test_orphan_ledger_record_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """A ledger record with no benchmark record is rot, and must fail."""
    examples, ledgers = _pair()
    ledgers.append(_ledger(example_id="p9"))
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "G1" in _checks(report)
    assert "orphan provenance" in _reasons(report)


# --- S3/S4: splits -------------------------------------------------------------


def test_duplicate_example_id_across_splits_fails(
    tmp_path: Path, corpus: tuple[Path, Path]
):
    """The same id in calibration and test is exactly the leakage S3 exists to stop."""
    examples, ledgers = _pair()
    examples.append(_example(example_id="p1", split="test"))
    ledgers.append(_ledger(example_id="p1", split="test"))
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "S3" in _checks(report)
    assert "must be disjoint" in _reasons(report)
    assert report["summary"]["gate_open"] is False


def test_empty_test_split_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """An empty test split means there is no untouched final set."""
    examples = [_example(), _example(example_id="p2")]
    ledgers = [_ledger(), _ledger(example_id="p2")]
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["summary"]["gate_open"] is False
    assert "S4" in _checks(report)
    assert "'test'" in _reasons(report)
    assert report["counts"]["by_split"] == {"calibration": 2, "test": 0}


# --- S5: category minimums ----------------------------------------------------


def test_too_small_category_cell_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """A category below the documented minimum is reported by name, not silently kept."""
    examples, ledgers = _pair()
    report = _run(tmp_path, corpus, examples, ledgers, min_per_category=3)

    assert "S5" in _checks(report)
    assert report["summary"]["thin_categories"] == {"factual": 2}
    assert "below the required minimum of 3" in _reasons(report)
    assert report["summary"]["grounded"] == 2  # grounding still clean; S5 is separate


def test_relaxed_category_threshold_is_recorded(tmp_path: Path, corpus: tuple[Path, Path]):
    """A run below the Phase 7 target must say so in the artifact."""
    examples, ledgers = _pair()
    report = _run(tmp_path, corpus, examples, ledgers, min_per_category=3)

    assert report["parameters"]["min_category_relaxed"] is True
    assert report["parameters"]["target_min_per_category"] == 5


def test_target_satisfied_is_not_flagged_as_relaxed(tmp_path: Path, corpus: tuple[Path, Path]):
    examples, ledgers = _pair()
    report = _run(tmp_path, corpus, examples, ledgers, min_per_category=2,
                  target_min_per_category=2)

    assert report["parameters"]["min_category_relaxed"] is False


# --- S1: structural -----------------------------------------------------------


def test_unknown_document_id_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    """A document id absent from papers.json is rejected before grounding."""
    example = _example(relevant_documents=["doc_does_not_exist"])
    examples, ledgers = _pair(example=example)
    report = _run(tmp_path, corpus, examples, ledgers)

    assert "S2" in _checks(report)
    assert "unknown relevant document" in _reasons(report)


def test_invalid_category_fails(tmp_path: Path, corpus: tuple[Path, Path]):
    example = _example(category="vibes")
    examples, ledgers = _pair(example=example)
    report = _run(tmp_path, corpus, examples, ledgers)

    # pydantic's own Literal rejects it first; either way the record does not
    # reach grounding and the gate closes.
    assert "S1" in _checks(report)
    assert "category" in _reasons(report)
    assert report["summary"]["gate_open"] is False


def test_unparseable_line_fails_without_hiding_the_rest(tmp_path: Path, corpus: tuple[Path, Path]):
    """One malformed line becomes a failure record, not an exception that hides the rest."""
    dataset = _write_jsonl(tmp_path / "ds.jsonl", [_example()])
    with open(dataset, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"example_id": "broken"}) + "\n")
    ledger = _write_jsonl(tmp_path / "lg.jsonl", [_ledger()])
    chunks_dir, manifest = corpus

    report = run_gate(
        dataset,
        ledger,
        chunks_dir=chunks_dir,
        manifest_path=manifest,
        min_per_category=1,
    )

    assert "S1" in _checks(report)
    assert "does not validate as EvaluationExample" in _reasons(report)
    # The good record is still audited and reported.
    assert [r["example_id"] for r in report["records"]] == ["p1"]
    assert report["summary"]["records_pass"] == 1


# --- absent corpus ------------------------------------------------------------


def test_absent_corpus_exits_non_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The corpus is gitignored; its absence must fail loudly, not skip grounding."""
    chunks_dir = tmp_path / "no_chunks_here"
    chunks_dir.mkdir()
    manifest = _write_manifest(tmp_path / "papers.json")
    dataset = _write_jsonl(tmp_path / "ds.jsonl", [_example(), _example(example_id="p2", split="test")])
    ledger = _write_jsonl(
        tmp_path / "lg.jsonl", [_ledger(), _ledger(example_id="p2", split="test")]
    )
    artifact = tmp_path / "gate.json"

    exit_code = main([
        "--dataset", str(dataset),
        "--ledger", str(ledger),
        "--chunks-dir", str(chunks_dir),
        "--manifest", str(manifest),
        "--out", str(artifact),
        "--json-only",
    ])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "cannot verify grounding" in stderr
    assert "download_corpus.py" in stderr
    report = json.loads(artifact.read_text(encoding="utf-8"))
    assert report["grounding_verified"] is False
    assert report["summary"]["grounded"] is None
    assert report["summary"]["records_unverified"] == 2


def test_structural_only_records_grounding_not_verified(
    tmp_path: Path, corpus: tuple[Path, Path]
):
    """--structural-only is the only way through, and it says so in the artifact."""
    chunks_dir, manifest = corpus
    dataset = _write_jsonl(
        tmp_path / "ds.jsonl", [_example(), _example(example_id="p2", split="test")]
    )
    ledger = _write_jsonl(
        tmp_path / "lg.jsonl", [_ledger(), _ledger(example_id="p2", split="test")]
    )

    report = run_gate(
        dataset,
        ledger,
        chunks_dir=chunks_dir,
        manifest_path=manifest,
        structural_only=True,
        min_per_category=1,
    )

    assert report["grounding_verified"] is False
    assert report["summary"]["grounded"] is None
    assert report["summary"]["records_unverified"] == 2
    assert report["summary"]["records_pass"] == 0
    assert {r["status"] for r in report["records"]} == {"UNVERIFIED"}


def test_structural_only_does_not_hide_structural_failures(
    tmp_path: Path, corpus: tuple[Path, Path]
):
    """Skipping grounding must not also skip S4."""
    chunks_dir, manifest = corpus
    dataset = _write_jsonl(tmp_path / "ds.jsonl", [_example(), _example(example_id="p2")])
    ledger = _write_jsonl(
        tmp_path / "lg.jsonl", [_ledger(), _ledger(example_id="p2")]
    )

    report = run_gate(
        dataset,
        ledger,
        chunks_dir=chunks_dir,
        manifest_path=manifest,
        structural_only=True,
        min_per_category=1,
    )

    assert report["summary"]["gate_open"] is False
    assert "S4" in _checks(report)


def test_absent_corpus_message_names_the_rebuild_chain() -> None:
    """The absent-corpus message must name the regeneration path, not just complain."""
    message = CORPUS_ABSENT_MESSAGE.format(chunks_dir="x")
    for token in ("download_corpus.py", "ingest_corpus.py", "build_chunks.py"):
        assert token in message


# --- non-vacuity --------------------------------------------------------------


def test_gate_is_not_vacuous(tmp_path: Path, corpus: tuple[Path, Path]):
    """Two runs differing only in one quote character must differ in verdict.

    Without this, a gate that ignored its inputs and returned open would satisfy
    every other test in this file equally well.
    """
    good_examples, good_ledgers = _pair()
    bad_ledger = _ledger(
        evidence=[
            {
                "document_id": DOC_A,
                "chunk_id": CHUNK_A,
                "quote": QUOTE_A.replace("passage collection", "passage collection!"),
            }
        ]
    )
    bad_examples, bad_ledgers = _pair(ledger=bad_ledger)

    (tmp_path / "good").mkdir()
    (tmp_path / "bad").mkdir()
    good = _run(tmp_path / "good", corpus, good_examples, good_ledgers)
    bad = _run(tmp_path / "bad", corpus, bad_examples, bad_ledgers)

    assert good["summary"]["gate_open"] is True
    assert bad["summary"]["gate_open"] is False
    assert good["summary"]["grounded"] == 2
    assert bad["summary"]["grounded"] == 1


def test_artifact_records_provenance(tmp_path: Path, corpus: tuple[Path, Path]):
    """The artifact must carry enough provenance to be cited later."""
    examples, ledgers = _pair()
    report = _run(tmp_path, corpus, examples, ledgers)

    assert report["gate_version"] == "phase7_dataset_gate_v1"
    assert report["timestamp"]
    assert report["git_commit"]
    assert len(report["dataset"]["sha256"]) == 16
    assert len(report["ledger"]["sha256"]) == 16
    assert report["dataset"]["dataset_version"] == "test_ds"
    assert report["environment"]["python"]
