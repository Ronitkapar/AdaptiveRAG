#!/usr/bin/env python
"""
scripts.validate_phase7_dataset
-------------------------------
Phase 7.0c -- the *dataset* verification gate.

`scripts/validate_environment.py` answers "can each benchmark arm actually run
against the real corpus?". This script answers the prior question: "are the
relevance labels themselves trustworthy?". Without that second answer the
environment gate is measuring retrieval over an unverified benchmark, and every
Phase 7 conclusion -- including "adaptive routing beats fixed strategies" --
inherits whatever was wrong with the labels.

The threat is specific. Phase 7 needs ~100 queries, hand-written relevance
labels, and verbatim quotes justifying them. Under that much manual authoring the
failure mode is not a typo: it is a curator writing a plausible question, a
plausible answer, and a quote that sounds like the paper but appears in no chunk
of it. Nothing downstream could detect that, because nothing downstream ever
re-reads the source. So the gate re-reads the source, mechanically, for every
label.

=============================================================================
THE LEDGER FORMAT
=============================================================================

Evidence lives in a *separate* JSONL file, one record per query, not in the
benchmark itself. Two reasons:

  * The benchmark is a metric input. `EvaluationExample` is `extra="forbid"`, and
    curation provenance is not a metric. Mixing them would either widen the
    schema (changing what every consumer parses) or force every evidence blob
    through `ReferenceInfo`, which the evaluator copies into every trace.
  * A 1:1 companion file can be enforced. Provenance smuggled into the
    benchmark can be dropped from it without anyone noticing; a ledger record
    with no benchmark record is a gate failure, so it cannot quietly rot.

Record shape (one JSON object per line):

    {
      "example_id":        "p7_001",
      "split":             "calibration",
      "relevant_documents":["rag_lewis_2020"],
      "evidence": [
        {"document_id": "rag_lewis_2020",
         "chunk_id":    "rag_lewis_2020::structure_aware_v1::c00005",
         "quote":       "<verbatim substring of that chunk's text>"}
      ],
      "answer_terms": ["RAG-Sequence", "latent"],
      "curator_notes": "why this document is relevant"
    }

Field rules:

  * `example_id` / `split` / `relevant_documents` are *assertions about the
    benchmark record*. The gate compares them for exact equality (order matters
    for documents, because the order is part of the label). A disagreement is a
    failure: it means the curator changed the answer and did not change the
    justification, or vice versa.
  * `evidence` -- at least one item. Each names a real chunk and quotes it.
  * `answer_terms` -- the content terms of `reference_answer`. Every one must
    occur in the concatenated quotes. This is what stops a fluent, invented
    answer: an answer cannot be grounded in source text it does not share.
  * `curator_notes` -- free text, not checked, but required to be non-empty so a
    record cannot be waved through without a stated reason.

=============================================================================
THE CHECKS
=============================================================================

Structural (S)
  S1  every line parses as `EvaluationExample`; `example_id` unique;
      `relevant_documents` non-empty; `category` in VALID_CATEGORIES;
      `split` in VALID_SPLITS.
  S2  every labelled `document_id` exists in the papers manifest
      (`evaluation.dataset.validate_against_corpus`).
  S3  `validate_split_separation` -- no `example_id` in both splits.
  S4  both splits non-empty. An empty test split is a hard failure: it means
      there is no untouched final set, and every "final" number would be scored
      on data already used to pick thresholds.
  S5  per-category and per-split counts reported; a category cell below
      `--min-per-category` (default 5) fails. E7's per-category breakdown is
      noise below that, and reporting it anyway is how a routing win gets
      claimed from three queries.

Grounding (G)
  G1  ledger <-> dataset is 1:1 in both directions.
  G2  ledger `relevant_documents` and `split` equal the dataset's exactly.
  G3  each evidence `chunk_id` exists on disk; the chunk's own `document_id`
      equals the evidence `document_id`; that document is in
      `relevant_documents`.
  G4  **the quote is a verbatim substring of that chunk's `text`**, and is at
      least `--min-quote-chars` (default 20) characters. Comparison is on
      *whitespace-normalised* text: both the chunk text and the quote are passed
      through `" ".join(s.split())`, which collapses every run of whitespace
      (including the newlines and the irregular inter-word spacing of PDF text
      extraction) to a single space and strips leading/trailing space. Nothing
      else is altered -- no case folding, no punctuation stripping, no ligature
      or hyphen repair, because each of those would let a mangled or paraphrased
      quote through. So `"RAG-Token"` matches, `"rag token"` does not.
  G5  every `answer_terms` entry occurs in the concatenated quotes,
      case-insensitively, on the same whitespace-normalised text.
  G6  every `relevant_sections[i].section_path_prefix` is a list-prefix of the
      `section_path` of at least one chunk in a relevant document. A section
      label pointing at nothing is worse than no label: `relevant_chunk_ids`
      intersects retrieved chunks against it, so a dead prefix silently
      produces an empty relevance set and a chunk-level `Precision@k` of 0.
  G7  **the quote is contiguous inside a single column of its source PDF**
      (opt-in, `--check-column-contiguity`; off by default because it needs the
      raw PDFs). G4 is necessary but not sufficient on a two-column page: the
      chunker reads both columns in one pass, so a span that jumps from the
      bottom of the left column to the top of the right one is still a verbatim
      substring of the flattened chunk text, and being long it clears the
      20-character floor. A spliced quote joins two claims the paper never made
      together, which makes a `reference_answer` look supported when it is not --
      a wrong relevance label that no downstream consumer could detect. G7
      re-reads `data/raw/<document_id>.pdf`, decomposes each provenance page
      into its columns, and requires the quote to appear in one of them.
      `column_check` in the artifact is `enabled`, `disabled`, or
      `indeterminate`; only a quote that is contiguous in the *flattened* page
      and in no single column is `contaminated`. A page that cannot be split
      confidently, and a quote that cannot be located at all, are
      `indeterminate`: this check may only accuse when it is certain, because a
      false positive would reject honest evidence. See `page_column_runs`.

=============================================================================
WHY AN ABSENT CORPUS IS A FAILURE, NOT A SKIP
=============================================================================

`data/processed/chunks/` is gitignored. It is regenerable
(`scripts/download_corpus.py` -> `ingest_corpus.py` -> `build_chunks.py`) but it
is routinely absent on a fresh clone, and the test suite skips corpus-dependent
tests when it is. A gate that skipped grounding in that state would report
`gate_open: true` on a dataset whose quotes had never been checked -- and the
artifact would say nothing about the difference, so the next step would treat an
unverified dataset as verified. Here the absence is instead:

  * exit 1, with the rebuild command in the message; and
  * overridable only by an explicit `--structural-only`, which records
    `grounding_verified: false` in the artifact so no downstream consumer can
    mistake it for a verified dataset.

=============================================================================
USAGE
=============================================================================

    python scripts/validate_phase7_dataset.py \\
        --dataset data/evaluation/phase7_eval_pilot.jsonl \\
        --ledger  data/evaluation/phase7_eval_pilot.ledger.jsonl

    python scripts/validate_phase7_dataset.py --structural-only   # unverified
    python scripts/validate_phase7_dataset.py --min-per-category 3  # pilot scale
    python scripts/validate_phase7_dataset.py --check-column-contiguity  # G7

Exit status is 0 only when every enabled check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import (  # noqa: E402
    CHUNKS_DIR,
    EVALUATION_DIR,
    PAPERS_MANIFEST_PATH,
    RAW_DATA_DIR,
    REPO_ROOT,
)
from adaptive_rag.evaluation.dataset import (  # noqa: E402
    VALID_CATEGORIES,
    VALID_SPLITS,
    validate_against_corpus,
    validate_split_separation,
)
from adaptive_rag.errors import AdaptiveRAGError  # noqa: E402
from adaptive_rag.schemas.experiment import EvaluationExample  # noqa: E402

GATE_VERSION = "phase7_dataset_gate_v1"

DEFAULT_ARTIFACT_PATH = REPO_ROOT / "experiments" / "phase7" / "dataset_gate.json"

# Phase 7's per-category analysis target. Below this, a per-category mean is
# dominated by which two or three queries happened to land in the cell, and
# reporting it is how "adaptive wins on fine-grained queries" gets claimed from
# noise. Small curation pilots cannot reach this (12 queries cannot fill 6
# categories 5 deep), so the threshold is a flag -- but it is recorded in the
# artifact next to the numbers, and a run below target is flagged as relaxed.
TARGET_MIN_PER_CATEGORY = 5
DEFAULT_MIN_PER_CATEGORY = TARGET_MIN_PER_CATEGORY

# A quote shorter than this is not evidence of anything: it is a phrase that
# could plausibly appear in the paper for unrelated reasons, so it cannot
# distinguish "the curator read this chunk" from "the curator guessed".
DEFAULT_MIN_QUOTE_CHARS = 20

CORPUS_ABSENT_MESSAGE = (
    "corpus absent at {chunks_dir} -- cannot verify grounding; rebuild with "
    "scripts/download_corpus.py -> scripts/ingest_corpus.py -> "
    "scripts/build_chunks.py, or re-run with --structural-only to record an "
    "unverified dataset"
)

_CHUNK_GLOB = "*.chunks.jsonl"


# --- ledger contract ----------------------------------------------------------


class EvidenceItem(BaseModel):
    """One quoted chunk justifying a relevance label."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    quote: str


class LedgerRecord(BaseModel):
    """Curation provenance for exactly one benchmark record."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    split: str
    relevant_documents: list[str]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    answer_terms: list[str] = Field(default_factory=list)
    curator_notes: str = ""


# --- helpers ------------------------------------------------------------------


def normalise(text: str) -> str:
    """Whitespace-normalise text for substring comparison.

    Collapses every whitespace run to a single space and strips the ends.
    Nothing else changes -- see the module docstring, check G4, for why case
    folding and punctuation repair are deliberately *not* done here.
    """
    return " ".join(text.split())


def corpus_present(chunks_dir: Path) -> bool:
    return bool(list(chunks_dir.glob(_CHUNK_GLOB)))


def load_chunks(chunks_dir: Path) -> dict[str, dict[str, Any]]:
    """Index every canonical chunk by id -> {document_id, section_path, text}."""
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(chunks_dir.glob(_CHUNK_GLOB)):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                index[record["chunk_id"]] = {
                    "document_id": record["document_id"],
                    "section_path": list(record["metadata"]["section_path"]),
                    # Page provenance drives the opt-in G7 column-contiguity
                    # check. Read defensively: a chunk written before provenance
                    # was recorded must not break the checks that do not need it.
                    "pages": [
                        int(p) for p in (record.get("provenance") or {}).get("pages", [])
                    ],
                    "text": record["text"],
                }
    return index


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows.append({"line_no": line_no, "payload": json.loads(line)})
    return rows


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return "unknown"


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _snapshot() -> dict[str, Any]:
    """Record what the dataset was validated against (same shape as the 7.0 gate)."""
    from importlib.metadata import distributions

    packages: dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name in {"numpy", "pydantic"}:
            packages[name] = dist.version or "unknown"

    return {
        "python": sys.version.split()[0],
        "platform": __import__("platform").platform(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": packages,
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --- S1/S2: per-record structure ----------------------------------------------


def _check_structure(
    raw_rows: list[dict[str, Any]],
    manifest_ids: set[str],
    manifest_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse and structurally validate each dataset line.

    Returns (examples by id, failures). A bad line becomes a failure record
    rather than an exception, so one malformed row cannot hide the audit of the
    other ninety-nine.
    """
    by_id: dict[str, Any] = {}
    seen_split: dict[str, str] = {}
    failures: list[dict[str, Any]] = []

    for row in raw_rows:
        line_no, payload = row["line_no"], row["payload"]

        def fail(reason: str, example_id: str | None = None, check: str = "S1") -> None:
            failures.append(
                {
                    "example_id": example_id,
                    "line_no": line_no,
                    "check": check,
                    "reason": reason,
                }
            )

        try:
            example = EvaluationExample.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            fail(f"line does not validate as EvaluationExample: {exc}")
            continue

        if not example.example_id:
            fail("empty example_id")
            continue
        if example.example_id in by_id:
            # A repeat id inside one split double-counts a query; the same id in
            # both splits is calibration/test leakage. The second is attributed to
            # S3 rather than S1 so the report names which rule was broken, and the
            # record is dropped either way.
            previous_split = seen_split[example.example_id]
            cross_split = previous_split != example.split
            fail(
                f"duplicate example_id {example.example_id} in '{previous_split}' and "
                f"'{example.split}' splits; calibration and test must be disjoint"
                if cross_split
                else f"duplicate example_id {example.example_id} in the "
                f"'{example.split}' split",
                example_id=example.example_id,
                check="S3" if cross_split else "S1",
            )
            continue
        if not example.relevant_documents:
            fail("relevant_documents is empty", example_id=example.example_id)
            continue
        if example.category not in VALID_CATEGORIES:
            fail(
                f"invalid category '{example.category}'; expected one of "
                f"{sorted(VALID_CATEGORIES)}",
                example_id=example.example_id,
            )
            continue
        if example.split not in VALID_SPLITS:
            fail(
                f"invalid split '{example.split}'; expected one of {list(VALID_SPLITS)}",
                example_id=example.example_id,
            )
            continue

        unknown_docs = [d for d in example.relevant_documents if d not in manifest_ids]
        if unknown_docs:
            fail(
                f"unknown relevant document(s) {unknown_docs}; not in {manifest_name}",
                example_id=example.example_id,
                check="S2",
            )
            continue

        unknown_sections = [
            ref.document_id
            for ref in example.relevant_sections
            if ref.document_id not in manifest_ids
        ]
        if unknown_sections:
            fail(
                f"unknown relevant_sections document(s) {unknown_sections}",
                example_id=example.example_id,
                check="S2",
            )
            continue

        by_id[example.example_id] = example
        seen_split[example.example_id] = example.split

    return by_id, failures


# --- G2-G6: grounding ---------------------------------------------------------


def check_grounding(
    example: EvaluationExample,
    ledger: LedgerRecord,
    chunks: dict[str, dict[str, Any]],
    min_quote_chars: int = DEFAULT_MIN_QUOTE_CHARS,
) -> list[str]:
    """Grounding checks G2-G6 for one matched dataset/ledger pair.

    Returns a list of human-readable problems; empty means fully grounded.
    """
    problems: list[str] = []

    # G2 -- the ledger's assertions about the benchmark record must hold exactly.
    if ledger.relevant_documents != example.relevant_documents:
        problems.append(
            f"G2: ledger relevant_documents {ledger.relevant_documents} != dataset "
            f"{example.relevant_documents}"
        )
    if ledger.split != example.split:
        problems.append(
            f"G2: ledger split '{ledger.split}' != dataset split '{example.split}'"
        )
    if not ledger.evidence:
        problems.append("G3: ledger has no evidence items")
    if not ledger.answer_terms:
        problems.append("G5: ledger has no answer_terms")
    if not ledger.curator_notes.strip():
        problems.append("curator_notes is empty")

    # G3/G4 -- every evidence item points at a real chunk and quotes it verbatim.
    quote_text: list[str] = []
    for position, item in enumerate(ledger.evidence):
        chunk = chunks.get(item.chunk_id)
        if chunk is None:
            problems.append(
                f"G3: evidence[{position}] chunk_id '{item.chunk_id}' not found in corpus"
            )
            continue
        if chunk["document_id"] != item.document_id:
            problems.append(
                f"G3: evidence[{position}] chunk '{item.chunk_id}' has document_id "
                f"'{chunk['document_id']}' but evidence claims '{item.document_id}'"
            )
            continue
        if item.document_id not in example.relevant_documents:
            problems.append(
                f"G3: evidence[{position}] document '{item.document_id}' is not in "
                f"relevant_documents {example.relevant_documents}"
            )
            continue

        normalised_quote = normalise(item.quote)
        if len(normalised_quote) < min_quote_chars:
            problems.append(
                f"G4: evidence[{position}] quote is {len(normalised_quote)} chars, "
                f"below the {min_quote_chars}-char minimum"
            )
            continue
        if normalised_quote not in normalise(chunk["text"]):
            problems.append(
                f"G4: evidence[{position}] quote is not a verbatim substring of chunk "
                f"{item.chunk_id} -- fabricated or mis-copied evidence"
            )
            continue
        quote_text.append(normalised_quote)

    # G5 -- the reference answer must share vocabulary with the verified source.
    haystack = normalise(" ".join(quote_text)).lower()
    for term in ledger.answer_terms:
        if normalise(term).lower() not in haystack:
            problems.append(
                f"G5: answer_term '{term}' does not appear in the verified evidence "
                f"quotes -- reference_answer is not grounded in the cited source"
            )

    # G6 -- section labels must point at real chunks in a relevant document.
    relevant_ids = set(example.relevant_documents)
    for position, ref in enumerate(example.relevant_sections):
        if ref.document_id not in relevant_ids:
            problems.append(
                f"G6: relevant_sections[{position}] document '{ref.document_id}' is not "
                f"in relevant_documents"
            )
            continue
        matched = any(
            chunk["document_id"] == ref.document_id
            and chunk["section_path"][: len(ref.section_path_prefix)]
            == ref.section_path_prefix
            for chunk in chunks.values()
        )
        if not matched:
            problems.append(
                f"G6: relevant_sections[{position}] prefix "
                f"{ref.section_path_prefix} matches no chunk in {ref.document_id}"
            )

    return problems


# --- G7: column contiguity ------------------------------------------------------


# The three run-level states recorded in the artifact. `disabled` means the
# check did not run; it must never be readable as "passed".
COLUMN_CHECK_DISABLED = "disabled"
COLUMN_CHECK_ENABLED = "enabled"
COLUMN_CHECK_INDETERMINATE = "indeterminate"

# Per-quote verdicts.
COLUMN_VERIFIED = "verified"
COLUMN_CONTAMINATED = "contaminated"
COLUMN_INDETERMINATE = "indeterminate"

RAW_ABSENT_MESSAGE = (
    "raw corpus absent at {raw_dir} -- cannot run the column-contiguity check "
    "(G7); rebuild with scripts/download_corpus.py, or re-run without "
    "--check-column-contiguity so the artifact records the check as not run"
)

# --- column-splitting geometry -------------------------------------------------
#
# Every threshold below exists to make the check *conservative*. The failure it
# guards against is a false positive: rejecting an honest quote because a
# geometry heuristic mis-split a page that was never two-column. Under-flagging a
# splice is a missed catch; over-flagging one destroys good evidence. Where the
# two conflict, this file takes the miss.
_FULL_WIDTH_FRAC = 0.55
# Justified body text in a column ends at the same x for almost every line, so
# line widths are sharply bimodal: column-width lines, and anything narrower
# (headings, figures, last lines of paragraphs) or wider (tables spanning the
# page). Keeping only lines at >= 85% of the modal narrow width is what removes
# figures and table rules from the geometry estimate.
_COLUMN_WIDTH_FRAC = 0.85
# A gutter narrower than this is not a gutter. Two-column papers set it at
# roughly 12-24pt at 10pt body type; single-column text has none at all.
_MIN_GUTTER_PT = 3.0
_MIN_COLUMN_WIDTH_FRAC = 0.10
_MIN_LINES_PER_COLUMN = 4
# A "column" holding a tenth of its partner's lines is a figure or a table
# fragment, not a column.
_MIN_COLUMN_LINE_RATIO = 0.25
# Layout analysis needs enough lines to have a mode at all.
_MIN_LINES_FOR_LAYOUT = 8
# A character that covers more than half the gutter band is text inside the
# gutter, which means the page is not cleanly two-column (a rule, a figure
# axis label, a wide table). Such a page is `indeterminate`, never split.
_MIN_GUTTER_OVERLAP_FRAC = 0.5
# Crop tolerance around the gutter midpoint, in points. Word-final hyphens and
# footnote marks sit a hair past the nominal column edge; dropping them would
# break an honest quote in half.
_CROP_TOLERANCE_PT = 2.0
_QUOTE_PREVIEW_CHARS = 160


def _cluster_chars_into_lines(chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group characters into visual lines by vertical position.

    Mirrors `adaptive_rag.ingestion.normalizer._cluster_chars_into_lines`
    (3.0pt vertical tolerance, sorted by `(top, x0)`) on purpose: the whole point
    is that the reconstructed page text reads like the chunk text, so a quote
    that G4 accepted is also found here. Two columns sharing a baseline merge
    into one line, which is exactly the flattening the chunker performs.
    """
    ordered = sorted(chars, key=lambda c: (round(float(c["top"]), 1), float(c["x0"])))
    if not ordered:
        return []
    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_top = float(ordered[0]["top"])
    for char in ordered:
        if abs(float(char["top"]) - current_top) <= 3.0:
            current.append(char)
        else:
            lines.append(current)
            current = [char]
            current_top = float(char["top"])
    if current:
        lines.append(current)
    return lines


def _line_text(line: list[dict[str, Any]]) -> str:
    """Join one line's characters, inserting a space where horizontal gaps occur.

    Same rule as the ingestor's `_line_text` (2.5pt gap threshold).
    """
    if not line:
        return ""
    parts: list[str] = []
    previous_x1 = float(line[0]["x0"])
    for char in line:
        if float(char["x0"]) - previous_x1 > 2.5:
            parts.append(" ")
        parts.append(str(char.get("text", "")))
        previous_x1 = float(char["x1"])
    return "".join(parts).strip()


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile; no statistics import for one formula."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _gutter_candidate(
    lines: list[list[dict[str, Any]]], lo: float, hi: float
) -> dict[str, Any] | None:
    """Estimate the gutter inside the x-range `[lo, hi]`, or None if there is none.

    The estimate is deliberately indirect. Column-width lines are split into a
    left group and a right group by the midpoint of the range; each group's
    median inner edge is the column boundary. Medians rather than maxima,
    because one figure or table rule crossing the range would otherwise move the
    boundary; column-width lines only, because a short line never reaches an
    edge. A real gutter shows up as a positive gap between those medians even
    when a minority of lines bridge it -- which is why a fixed midpoint rule
    alone is not enough, and why three columns are caught separately by
    `_any_gutter`.
    """
    width = hi - lo
    if width <= 0:
        return None
    inside = [
        [c for c in line if lo <= float(c["x0"]) and float(c["x1"]) <= hi]
        for line in lines
    ]
    inside = [line for line in inside if line]
    spans = [
        (min(float(c["x0"]) for c in line), max(float(c["x1"]) for c in line))
        for line in inside
    ]
    narrow = [(a, b) for a, b in spans if (b - a) < _FULL_WIDTH_FRAC * width]
    if len(narrow) < _MIN_LINES_FOR_LAYOUT:
        return None
    modal = _quantile([b - a for a, b in narrow], 0.5)
    column_width = [(a, b) for a, b in narrow if (b - a) >= _COLUMN_WIDTH_FRAC * modal]
    if len(column_width) < _MIN_LINES_FOR_LAYOUT:
        return None
    midpoint = (lo + hi) / 2.0
    left = [s for s in column_width if (s[0] + s[1]) / 2 < midpoint]
    right = [s for s in column_width if (s[0] + s[1]) / 2 >= midpoint]
    if min(len(left), len(right)) < _MIN_LINES_PER_COLUMN:
        return None
    if min(len(left), len(right)) / max(len(left), len(right)) < _MIN_COLUMN_LINE_RATIO:
        return None
    left_edge = _quantile([b for _, b in left], 0.5)
    right_edge = _quantile([a for a, _ in right], 0.5)
    if right_edge - left_edge < _MIN_GUTTER_PT:
        return None
    return {
        "left_edge": left_edge,
        "right_edge": right_edge,
        "left_extent": (min(a for a, _ in left), max(b for _, b in left)),
        "right_extent": (min(a for a, _ in right), max(b for _, b in right)),
    }


def _spans_gutter(
    line: list[dict[str, Any]], midpoint: float, min_gap: float
) -> bool:
    """True when one visual line carries text across the gutter with no clean gap.

    Two columns set on a shared baseline merge into a single char cluster, and
    that cluster is *not* a full-width line -- treating it as one would rebuild
    the flattened page text inside the column runs and defeat the check. Only a
    line whose left and right parts are closer than `min_gap` (a wide table row,
    a full-width heading) is a genuine spanning element.
    """
    left_max = max(
        (float(c["x1"]) for c in line if float(c["x1"]) <= midpoint + _CROP_TOLERANCE_PT),
        default=None,
    )
    right_min = min(
        (float(c["x0"]) for c in line if float(c["x0"]) >= midpoint - _CROP_TOLERANCE_PT),
        default=None,
    )
    if left_max is None or right_min is None:
        return False
    return (right_min - left_max) < min_gap


def page_column_runs(page: Any) -> dict[str, Any]:
    """Decompose one PDF page into reading-order runs, or classify its layout.

    Returns `{"layout", "flat", "columns", "info"}` where `layout` is:

      * `single` -- no column structure; `columns` is the whole page, so any
        quote in it is contiguous by construction and nothing can be flagged;
      * `two_column` -- a clean gutter was found; `columns` holds the left
        column, the right column, and the text of lines that span the gutter;
      * `indeterminate` -- the page has some column-like structure but cannot be
        split confidently (text inside the gutter, or an internal gutter meaning
        three or more columns). Callers must not flag quotes here.

    `flat` is the whole page in the chunker's reading order, used only to tell
    "not contiguous in any column" apart from "not found at all".
    """
    chars = [c for c in (page.chars or []) if str(c.get("text", "")) != ""]
    lines = _cluster_chars_into_lines(chars)
    flat = normalise(" ".join(_line_text(line) for line in lines))
    width = float(page.width)
    whole = {"layout": "single", "flat": flat, "columns": [flat], "info": {}}

    if not lines:
        return {"layout": "indeterminate", "flat": flat, "columns": [flat], "info": {}}

    gutter = _gutter_candidate(lines, 0.0, width)
    if gutter is None:
        return whole
    left_edge = float(gutter["left_edge"])
    right_edge = float(gutter["right_edge"])
    left_margin = min(float(c["x0"]) for line in lines for c in line)
    right_margin = max(float(c["x1"]) for line in lines for c in line)
    left_width = left_edge - left_margin
    right_width = right_margin - right_edge
    info = {
        "gutter": (round(left_edge, 1), round(right_edge, 1)),
        "left_width": round(left_width, 1),
        "right_width": round(right_width, 1),
    }
    if min(left_width, right_width) < _MIN_COLUMN_WIDTH_FRAC * width:
        return whole

    # Three or more columns would read as "two" columns plus a seam inside one
    # of them, and the seam would be found by the midpoint rule. Splitting a
    # column down that seam would reject honest quotes, so a page that has one
    # is indeterminate rather than split. The probe re-runs the full estimate on
    # each column's own extent, so an ordinary whitespace gap inside a figure or
    # a table does not count -- it is not a column.
    nested = [
        extent
        for extent in (gutter["left_extent"], gutter["right_extent"])
        if _gutter_candidate(lines, extent[0], extent[1]) is not None
    ]
    info["nested_gutters"] = len(nested)
    if nested:
        return {"layout": "indeterminate", "flat": flat, "columns": [flat], "info": info}

    band = right_edge - left_edge
    intruders = [
        c
        for c in chars
        if min(float(c["x1"]), right_edge) - max(float(c["x0"]), left_edge)
        > _MIN_GUTTER_OVERLAP_FRAC * band
    ]
    info["gutter_intruders"] = len(intruders)
    if intruders:
        return {"layout": "indeterminate", "flat": flat, "columns": [flat], "info": info}

    midpoint = (left_edge + right_edge) / 2.0
    runs = []
    for keep in (
        lambda c: float(c["x1"]) <= midpoint + _CROP_TOLERANCE_PT,
        lambda c: float(c["x0"]) >= midpoint - _CROP_TOLERANCE_PT,
    ):
        selected = [[c for c in line if keep(c)] for line in lines]
        runs.append(normalise(" ".join(_line_text(line) for line in selected if line)))
    spanning = [line for line in lines if _spans_gutter(line, midpoint, band)]
    runs.append(normalise(" ".join(_line_text(line) for line in spanning)))
    return {"layout": "two_column", "flat": flat, "columns": runs, "info": info}


class ColumnChecker:
    """Locates each quote inside a single column of its source PDF.

    Lazily opens `data/raw/<document_id>.pdf`, extracts only the pages named in
    a chunk's `provenance.pages`, and caches the page decomposition. Every quote
    verdict is one of `verified`, `contaminated`, or `indeterminate`; only
    `contaminated` is ever treated as a failure.
    """

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.layouts: dict[str, int] = {}
        self._pages: dict[tuple[str, int], dict[str, Any]] = {}

    def raw_corpus_present(self) -> bool:
        return self.raw_dir.is_dir() and any(self.raw_dir.glob("*.pdf"))

    def _page_runs(self, document_id: str, page_no: int) -> dict[str, Any] | None:
        key = (document_id, page_no)
        if key in self._pages:
            return self._pages[key]
        path = self.raw_dir / f"{document_id}.pdf"
        if not path.is_file():
            return None
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                if page_no < 1 or page_no > len(pdf.pages):
                    return None
                runs = page_column_runs(pdf.pages[page_no - 1])
        except Exception:  # noqa: BLE001 -- unreadable source is indeterminate
            return None
        layout = str(runs["layout"])
        self.layouts[layout] = self.layouts.get(layout, 0) + 1
        self._pages[key] = runs
        return runs

    def verdict(self, quote: str, chunk: dict[str, Any]) -> tuple[str, str]:
        """Classify one quote against the columns of its chunk's source pages."""
        normalised_quote = normalise(quote)
        if not normalised_quote:
            return COLUMN_INDETERMINATE, "quote is empty after whitespace normalisation"
        pages = sorted({int(p) for p in chunk.get("pages", []) if int(p) > 0})
        if not pages:
            return COLUMN_INDETERMINATE, "chunk carries no page provenance"

        flat_hits: list[int] = []
        unconfident: list[int] = []
        for page_no in pages:
            runs = self._page_runs(chunk["document_id"], page_no)
            if runs is None:
                return (
                    COLUMN_INDETERMINATE,
                    f"page {page_no} of {chunk['document_id']}.pdf could not be read",
                )
            # An unsplit page is not a one-column page: its single run is the
            # whole flattened page, so a quote found there proves nothing about
            # columns. It can neither confirm nor accuse.
            if runs["layout"] == "indeterminate":
                unconfident.append(page_no)
                continue
            if any(normalised_quote in run for run in runs["columns"]):
                return (
                    COLUMN_VERIFIED,
                    f"contiguous inside one column of page {page_no} "
                    f"({runs['layout']} layout)",
                )
            if normalised_quote in runs["flat"]:
                flat_hits.append(page_no)

        if flat_hits and not unconfident:
            return (
                COLUMN_CONTAMINATED,
                f"contiguous only in the flattened text of page(s) {flat_hits}, "
                f"never inside a single column",
            )
        if unconfident:
            return (
                COLUMN_INDETERMINATE,
                f"page layout could not be resolved confidently on page(s) {unconfident}",
            )
        return (
            COLUMN_INDETERMINATE,
            "quote is not contiguous in the reconstructed text of any provenance "
            "page (chunk-only text such as tables, equations, or headings is not "
            "reproduced by the page reconstruction)",
        )

def _column_check_record(
    example_id: str,
    ledger: LedgerRecord,
    chunks: dict[str, dict[str, Any]],
    checker: ColumnChecker,
    failures: list[dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, Any]:
    """Run G7 over one record's quotes and fold the outcome into the report.

    A quote is `contaminated` only when it is verbatim-continuous in the
    flattened text of a provenance page and in no single column of any of them.
    Anything the column analysis cannot settle -- an unreadable page, a layout
    it will not split, a chunk whose text (tables, equations, headings) the page
    reconstruction does not reproduce -- is `indeterminate` and never fails the
    gate. The quote preview is truncated so a human can act on the finding
    without opening the ledger.
    """
    verdicts: list[dict[str, Any]] = []
    record_status = COLUMN_VERIFIED
    for position, item in enumerate(ledger.evidence):
        chunk = chunks.get(item.chunk_id)
        counts["quotes"] += 1
        if chunk is None or chunk["document_id"] != item.document_id:
            verdict, detail = COLUMN_INDETERMINATE, (
                "chunk is not on disk, so its source pages cannot be read"
            )
        else:
            verdict, detail = checker.verdict(item.quote, chunk)
        counts[verdict] += 1
        verdicts.append(
            {
                "index": position,
                "chunk_id": item.chunk_id,
                "status": verdict,
                "detail": detail,
            }
        )
        if verdict != COLUMN_CONTAMINATED:
            continue
        record_status = COLUMN_CONTAMINATED
        preview = normalise(item.quote)
        if len(preview) > _QUOTE_PREVIEW_CHARS:
            preview = preview[:_QUOTE_PREVIEW_CHARS] + "..."
        failures.append(
            {
                "example_id": example_id,
                "line_no": None,
                "check": "G7",
                "reason": (
                    f"G7: evidence[{position}] quote for chunk {item.chunk_id} is "
                    f"{detail} -- the quote splices unrelated text across a column "
                    f"break. quote: '{preview}'"
                ),
            }
        )
    if record_status != COLUMN_CONTAMINATED and not any(
        v["status"] == COLUMN_VERIFIED for v in verdicts
    ):
        # Nothing was confirmed and nothing was accused: this record's quotes
        # could not be resolved, which is not the same as passing.
        record_status = COLUMN_INDETERMINATE
    return {"status": record_status, "evidence": verdicts}


# --- the gate -----------------------------------------------------------------


def run_gate(
    dataset_path: Path,
    ledger_path: Path,
    *,
    chunks_dir: Path = CHUNKS_DIR,
    manifest_path: Path = PAPERS_MANIFEST_PATH,
    structural_only: bool = False,
    min_per_category: int = DEFAULT_MIN_PER_CATEGORY,
    target_min_per_category: int = TARGET_MIN_PER_CATEGORY,
    min_quote_chars: int = DEFAULT_MIN_QUOTE_CHARS,
    check_column_contiguity: bool = False,
    raw_dir: Path = RAW_DATA_DIR,
    allow_test_only: bool = False,
) -> dict[str, Any]:
    """Validate a benchmark dataset against its curation ledger and the corpus.

    `allow_test_only` (pre-registered for Phase 15, docs/phases/phase-15.md
    §15.17) permits a dataset whose calibration split is empty and whose
    test split is non-empty -- the shape of a confirmation benchmark that
    fits nothing. It is default-off; without it S4 behaves exactly as
    before. When active and the shape holds, the artifact records
    `single_split_test_only: true` so the accommodation is visible.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"ledger not found: {ledger_path}")

    corpus_available = corpus_present(chunks_dir)
    structural_only = structural_only or not corpus_available
    grounding_verified = not structural_only

    # G7 is opt-in because it needs the raw PDFs, which are not part of the
    # chunk corpus the other grounding checks read. Disabled is recorded as
    # `disabled`, never as a pass.
    column_checker = ColumnChecker(raw_dir)
    column_enabled = check_column_contiguity and grounding_verified
    column_status = COLUMN_CHECK_DISABLED
    column_counts = {"quotes": 0, COLUMN_VERIFIED: 0, COLUMN_CONTAMINATED: 0,
                     COLUMN_INDETERMINATE: 0}
    if column_enabled and not column_checker.raw_corpus_present():
        column_enabled = False
        column_status = COLUMN_CHECK_INDETERMINATE

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_ids = {p["document_id"] for p in manifest}

    failures: list[dict[str, Any]] = []
    if check_column_contiguity and not column_checker.raw_corpus_present():
        failures.append(
            {
                "example_id": None,
                "line_no": None,
                "check": "G7",
                "reason": RAW_ABSENT_MESSAGE.format(raw_dir=raw_dir),
            }
        )
    raw_rows = load_jsonl(dataset_path)
    by_id, structural_failures = _check_structure(
        raw_rows, manifest_ids, manifest_path.name
    )
    failures.extend(structural_failures)
    examples = list(by_id.values())

    # S2/S3 -- cross-check against the dataset module's own rules. The per-record
    # pass above applies the same rules earlier and with per-record attribution;
    # these calls are the independent confirmation that the shared rule set
    # agrees, so the two cannot silently diverge.
    if examples:
        try:
            validate_against_corpus(examples, manifest_path=manifest_path)
        except AdaptiveRAGError as exc:
            failures.append(
                {
                    "example_id": None,
                    "line_no": None,
                    "check": "S2",
                    "reason": f"validate_against_corpus rejected the dataset: {exc}",
                }
            )
    if examples:
        try:
            validate_split_separation(examples)
        except AdaptiveRAGError as exc:
            failures.append(
                {
                    "example_id": None,
                    "line_no": None,
                    "check": "S3",
                    "reason": f"validate_split_separation rejected the dataset: {exc}",
                }
            )

    # S4 -- both splits non-empty.
    split_counts = {split: 0 for split in VALID_SPLITS}
    category_counts: dict[str, int] = {}
    category_split_counts: dict[str, dict[str, int]] = {}
    for example in examples:
        split_counts[example.split] = split_counts.get(example.split, 0) + 1
        category_counts[example.category] = category_counts.get(example.category, 0) + 1
        cell = category_split_counts.setdefault(
            example.category, {s: 0 for s in VALID_SPLITS}
        )
        cell[example.split] += 1

    empty_splits = [s for s in VALID_SPLITS if split_counts.get(s, 0) == 0]
    test_only_shape = empty_splits == ["calibration"]
    if empty_splits and not (allow_test_only and test_only_shape):
        failures.append(
            {
                "example_id": None,
                "line_no": None,
                "check": "S4",
                "reason": (
                    f"split(s) {empty_splits} are empty; a benchmark with no untouched "
                    f"test split cannot report final numbers -- every 'test' metric "
                    f"would be scored on queries already used to choose thresholds"
                ),
            }
        )

    # S5 -- per-category minimum.
    thin_categories = {
        category: count
        for category, count in sorted(category_counts.items())
        if count < min_per_category
    }
    for category, count in thin_categories.items():
        failures.append(
            {
                "example_id": None,
                "line_no": None,
                "check": "S5",
                "reason": (
                    f"category '{category}' has {count} record(s), below the required "
                    f"minimum of {min_per_category}; per-category analysis at this cell "
                    f"size is noise"
                ),
            }
        )

    # --- G1: ledger 1:1, both directions ---
    ledger_rows = load_jsonl(ledger_path)
    ledger_by_id: dict[str, LedgerRecord] = {}
    for row in ledger_rows:
        line_no, payload = row["line_no"], row["payload"]
        try:
            record = LedgerRecord.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            failures.append(
                {
                    "example_id": payload.get("example_id"),
                    "line_no": line_no,
                    "check": "G1",
                    "reason": f"ledger line does not validate as LedgerRecord: {exc}",
                }
            )
            continue
        if record.example_id in ledger_by_id:
            failures.append(
                {
                    "example_id": record.example_id,
                    "line_no": line_no,
                    "check": "G1",
                    "reason": f"duplicate ledger record for {record.example_id}",
                }
            )
            continue
        ledger_by_id[record.example_id] = record

    for example_id in sorted(set(by_id) - set(ledger_by_id)):
        failures.append(
            {
                "example_id": example_id,
                "line_no": None,
                "check": "G1",
                "reason": "dataset record has no ledger record -- relevance is unprovenanced",
            }
        )
    for example_id in sorted(set(ledger_by_id) - set(by_id)):
        failures.append(
            {
                "example_id": example_id,
                "line_no": None,
                "check": "G1",
                "reason": "ledger record has no dataset record -- orphan provenance",
            }
        )

    # --- G2-G6 per record ---
    chunks = load_chunks(chunks_dir) if grounding_verified else {}
    record_results: list[dict[str, Any]] = []
    for example_id in sorted(by_id):
        example = by_id[example_id]
        ledger = ledger_by_id.get(example_id)
        entry: dict[str, Any] = {
            "example_id": example_id,
            "category": example.category,
            "split": example.split,
            "relevant_documents": example.relevant_documents,
            "evidence_items": len(ledger.evidence) if ledger else 0,
            "answer_terms": list(ledger.answer_terms) if ledger else [],
            "status": "PASS",
            "failures": [],
            "column_check": {"status": "not_run", "evidence": []},
        }
        if ledger is None:
            entry["status"] = "FAIL"
            entry["failures"] = ["G1: no ledger record"]
            entry["column_check"] = {"status": "not_run", "evidence": []}
        elif not grounding_verified:
            entry["status"] = "UNVERIFIED"
            entry["column_check"] = {"status": "not_run", "evidence": []}
        else:
            problems = check_grounding(example, ledger, chunks, min_quote_chars)
            entry["failures"] = problems
            for problem in problems:
                failures.append(
                    {
                        "example_id": example_id,
                        "line_no": None,
                        "check": problem.split(":", 1)[0],
                        "reason": problem,
                    }
                )
            entry["status"] = "FAIL" if problems else "PASS"
            if not column_enabled:
                entry["column_check"]["status"] = (
                    "disabled" if not check_column_contiguity else "not_run"
                )
            if column_enabled:
                entry["column_check"] = _column_check_record(
                    example_id, ledger, chunks, column_checker, failures, column_counts
                )
                if entry["column_check"]["status"] == COLUMN_CONTAMINATED:
                    entry["status"] = "FAIL"
        record_results.append(entry)

    total_records = len(examples)
    if column_enabled:
        column_status = (
            COLUMN_CHECK_ENABLED
            if column_counts[COLUMN_VERIFIED] + column_counts[COLUMN_CONTAMINATED] > 0
            else COLUMN_CHECK_INDETERMINATE
        )
    grounded = sum(1 for r in record_results if r["status"] == "PASS" and grounding_verified)
    grounding_failures = [
        f for f in failures if f["check"].startswith("G")
    ]
    gate_open = not failures

    return {
        "gate_version": GATE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": _snapshot(),
        "grounding_verified": grounding_verified,
        "corpus": {
            "chunks_dir": _rel(chunks_dir),
            "present": corpus_available,
            "n_chunks": len(chunks),
        },
        # `column_check` is the run-level state and is one of
        # enabled | disabled | indeterminate. `disabled` means the check did not
        # run; it must never be read as a pass. `indeterminate` means it could
        # not be run at all (no raw corpus). `column_check_counts` is per-quote
        # and always present, so a reader can see the shape of the result even
        # when the check was off.
        "column_check": column_status,
        "column_check_counts": column_counts,
        "column_check_detail": {
            "raw_dir": _rel(raw_dir),
            "raw_corpus_present": column_checker.raw_corpus_present(),
            "page_layouts": dict(sorted(column_checker.layouts.items())),
        },
        "dataset": {
            "path": _rel(dataset_path),
            "sha256": _file_sha256(dataset_path),
            "n_records": total_records,
            "dataset_version": examples[0].dataset_version if examples else None,
        },
        "ledger": {
            "path": _rel(ledger_path),
            "sha256": _file_sha256(ledger_path),
            "n_records": len(ledger_rows),
        },
        "parameters": {
            "structural_only": structural_only,
            "allow_test_only": allow_test_only,
            "single_split_test_only": bool(allow_test_only and test_only_shape),
            "min_per_category": min_per_category,
            "target_min_per_category": target_min_per_category,
            "min_category_relaxed": min_per_category < target_min_per_category,
            "min_quote_chars": min_quote_chars,
        },
        "counts": {
            "by_split": split_counts,
            "by_category": dict(sorted(category_counts.items())),
            "by_category_split": {
                category: dict(sorted(cells.items()))
                for category, cells in sorted(category_split_counts.items())
            },
            "categories_absent": sorted(set(VALID_CATEGORIES) - set(category_counts)),
        },
        "records": record_results,
        "summary": {
            "records_total": total_records,
            "records_pass": sum(1 for r in record_results if r["status"] == "PASS"),
            "records_fail": sum(1 for r in record_results if r["status"] == "FAIL"),
            "records_unverified": sum(
                1 for r in record_results if r["status"] == "UNVERIFIED"
            ),
            "grounded": grounded if grounding_verified else None,
            "grounding_failures": len(grounding_failures),
            "column_contaminated": column_counts[COLUMN_CONTAMINATED],
            "column_indeterminate": column_counts[COLUMN_INDETERMINATE],
            "column_verified": column_counts[COLUMN_VERIFIED],
            "structural_failures": len(failures) - len(grounding_failures),
            "failed_example_ids": [r["example_id"] for r in record_results if r["status"] == "FAIL"],
            "thin_categories": thin_categories,
            "gate_open": gate_open,
        },
        "failures": failures,
    }


# --- reporting ----------------------------------------------------------------


def print_report(report: dict[str, Any]) -> None:
    params = report["parameters"]
    relaxed = (
        f", RELAXED from {params['target_min_per_category']}"
        if params["min_category_relaxed"]
        else ""
    )
    print(f"Phase 7 dataset gate -- {report['gate_version']}")
    print(f"  dataset  {report['dataset']['path']} "
          f"({report['dataset']['n_records']} records, "
          f"sha256={report['dataset']['sha256']})")
    print(f"  ledger   {report['ledger']['path']} "
          f"({report['ledger']['n_records']} records)")
    print(f"  commit   {report['git_commit']}")
    corpus = report["corpus"]
    print(f"  corpus   {corpus['chunks_dir']} "
          f"present={corpus['present']} chunks={corpus['n_chunks']}")
    column = report["column_check"]
    column_counts = report["column_check_counts"]
    print(f"  column_check={column} "
          f"(quotes={column_counts['quotes']} "
          f"verified={column_counts['verified']} "
          f"contaminated={column_counts['contaminated']} "
          f"indeterminate={column_counts['indeterminate']})")
    print(f"  grounding_verified={report['grounding_verified']} "
          f"structural_only={params['structural_only']} "
          f"min_per_category={params['min_per_category']}{relaxed} "
          f"min_quote_chars={params['min_quote_chars']}")
    print()

    counts = report["counts"]
    print("  splits     " + "  ".join(f"{k}={v}" for k, v in counts["by_split"].items()))
    print("  categories")
    thin = report["summary"]["thin_categories"]
    for category, count in counts["by_category"].items():
        cells = counts["by_category_split"].get(category, {})
        flag = "   <-- BELOW MIN" if category in thin else ""
        print(f"    {category:<14} n={count:<3} calibration={cells.get('calibration', 0)}"
              f" test={cells.get('test', 0)}{flag}")
    print()

    for record in report["records"]:
        print(f"  [{record['status']:>10}] {record['example_id']:<10} "
              f"{record['category']:<13} {record['split']:<12} "
              f"docs={','.join(record['relevant_documents'])} "
              f"evidence={record['evidence_items']}")
        for problem in record["failures"]:
            print(f"               - {problem}")
        column_entry = record.get("column_check")
        if column_entry and column_entry["status"] != "not_run":
            print(f"               column_check={column_entry['status']}")

    summary = report["summary"]
    print()
    if report["failures"]:
        print("  failures")
        for failure in report["failures"]:
            print(f"    - [{failure['check']}] {failure['example_id'] or '-'}: "
                  f"{failure['reason']}")
        print()
    print(f"  records pass={summary['records_pass']}/{summary['records_total']} "
          f"fail={summary['records_fail']} "
          f"unverified={summary['records_unverified']}")
    if summary["grounded"] is not None:
        print(f"  grounded {summary['grounded']}/{summary['records_total']} "
              f"(grounding failures={summary['grounding_failures']})")
    print(f"  GATE {'OPEN' if summary['gate_open'] else 'CLOSED'}"
          + ("" if report["grounding_verified"]
             else "  [UNVERIFIED: structural checks only]"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 7 evaluation dataset against its curation ledger.",
    )
    parser.add_argument("--dataset", type=Path,
                        default=EVALUATION_DIR / "phase7_eval_pilot.jsonl")
    parser.add_argument("--ledger", type=Path,
                        default=EVALUATION_DIR / "phase7_eval_pilot.ledger.jsonl")
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--manifest", type=Path, default=PAPERS_MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Skip grounding checks. Records grounding_verified=false.",
    )
    parser.add_argument("--min-per-category", type=int, default=DEFAULT_MIN_PER_CATEGORY)
    parser.add_argument("--target-min-per-category", type=int,
                        default=TARGET_MIN_PER_CATEGORY)
    parser.add_argument("--min-quote-chars", type=int, default=DEFAULT_MIN_QUOTE_CHARS)
    parser.add_argument(
        "--check-column-contiguity",
        action="store_true",
        help=(
            "Run G7: require every quote to be contiguous inside a single column of "
            "its source PDF, so a quote spliced across a column break is rejected. "
            "Needs the raw corpus at data/raw/. Off by default; when off the "
            "artifact records column_check=disabled."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--allow-test-only",
        action="store_true",
        help=(
            "Permit a test-only dataset (empty calibration split, non-empty "
            "test split): the shape of a confirmation benchmark that fits "
            "nothing. Pre-registered for Phase 15 (docs/phases/phase-15.md "
            "§15.17); recorded in the artifact as single_split_test_only. "
            "Off by default; S4 is otherwise unchanged."
        ),
    )
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    report = run_gate(
        args.dataset,
        args.ledger,
        chunks_dir=args.chunks_dir,
        manifest_path=args.manifest,
        structural_only=args.structural_only,
        min_per_category=args.min_per_category,
        target_min_per_category=args.target_min_per_category,
        min_quote_chars=args.min_quote_chars,
        check_column_contiguity=args.check_column_contiguity,
        raw_dir=args.raw_dir,
        allow_test_only=args.allow_test_only,
    )

    if args.check_column_contiguity and not report["column_check_detail"]["raw_corpus_present"]:
        print(RAW_ABSENT_MESSAGE.format(raw_dir=args.raw_dir), file=sys.stderr)

    if not report["corpus"]["present"]:
        message = CORPUS_ABSENT_MESSAGE.format(chunks_dir=args.chunks_dir)
        if not args.structural_only:
            print(message, file=sys.stderr)
        else:
            print(f"{message}\n  (continuing with --structural-only: this dataset is "
                  f"NOT grounding-verified)", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")

    if not args.json_only:
        print_report(report)
        print()

    return 0 if report["summary"]["gate_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
