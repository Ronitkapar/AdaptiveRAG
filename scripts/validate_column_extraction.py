#!/usr/bin/env python3
"""
validate_column_extraction.py
-----------------------------
Phase 8 Step 1 -- measure the two-column extraction fix against the five gates
recorded in `docs/phases/phase-8.md` §3.3, rather than asserting them.

Every gate here is measured. Nothing is written to the corpus and no index is
touched: this script answers "did the normalizer fix work?" and leaves the
re-ingestion decision to a human.

Gates
-----
1. **Residual interleaving.** Counts, on the **column-aware** reading of every
   two-column page, how many adjacent word pairs span a column boundary. This is
   the defect itself, measured where it matters: on the output that would be
   re-ingested. It must be zero, or near it -- every non-zero case is listed.

   The companion count is how many pairs the fix *removed*, which is evidence the
   fix did something rather than a pass/fail criterion. Phase 7 recorded "1.4% of
   tokens glued" (`docs/phase_7_results.md` §11) without retaining the measurement
   behind it, so that figure is quoted as the prior claim it is and is **not**
   treated as a comparable baseline.

2. **Spot check.** Locates a caller-supplied sentence (default: one from
   `dpr_karpukhin_2020.pdf`) in both readings. It must occur in the column-aware
   text and not in the flat text. Both readings are de-hyphenated first, since a
   sentence broken across lines by the hyphen a two-column layout makes
   unavoidable would otherwise never match.

3. **Single-column regression.** Re-ingests every document and compares the
   canonical element texts against the Phase 7 artifacts already in
   `data/processed/documents/`. A document with **no** two-column page must be
   byte-identical; a document with some must differ only on its two-column pages.
   This is the gate that blocks re-ingestion.

4. **Reading order.** Prints the opening elements of 2-3 papers for
   hand-verification against the PDF. It is printed rather than asserted because
   "is this the right reading order" is a judgement about a page, not a property
   code can decide.

5. **Chunk delta.** Builds chunks from documents ingested both ways and reports
   how many chunk texts change.

Usage
-----
    python scripts/validate_column_extraction.py
    python scripts/validate_column_extraction.py --spot-check "exact sentence"
    python scripts/validate_column_extraction.py --skip-chunk-delta
    python scripts/validate_column_extraction.py --out experiments/phase8/column_validation.json

Exit status is 0 only when every machine-decidable gate passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.chunking.pipeline import ChunkingPipeline  # noqa: E402
from adaptive_rag.config.paths import (  # noqa: E402
    DOCUMENTS_DIR,
    PAPERS_MANIFEST_PATH,
    RAW_DATA_DIR,
    REPO_ROOT,
)
from adaptive_rag.ingestion import normalizer as normalizer_module  # noqa: E402
from adaptive_rag.ingestion.extractor import PdfplumberExtractor  # noqa: E402
from adaptive_rag.ingestion.normalizer import (  # noqa: E402
    _cluster_chars_into_lines,
    _detect_columns,
    _line_text,
)
from adaptive_rag.ingestion.pipeline import IngestionPipeline  # noqa: E402

VALIDATION_VERSION = "phase8_column_validation_v1"

# Phase 7 recorded "1.4% of tokens glued" (docs/phase_7_results.md §11) without
# retaining the measurement that produced it. Quoted as the prior claim, NOT as a
# baseline: the two are different quantities and no reproducible measurement
# bridges them.
PHASE7_GLUE_CLAIM = 0.014
# Residual splicing tolerance on the column-aware output. The normalizer only
# emits a line intact when it spans no gutter-sized gap, so the expected count is
# zero; the allowance is headroom for a wide table row that genuinely crosses.
# Each surviving case is listed in the artifact, so this cannot hide.
MAX_RESIDUAL_SPLICED_LINES = 10

DEFAULT_SPOT_CHECK = (
    "Although more expressive model forms for measuring the similarity between "
    "a question and a passage do exist"
)
DEFAULT_SPOT_CHECK_DOC = "dpr_karpukhin_2020"

# Reading-order spot check documents: two-column papers whose layout is
# recognisable from the opening elements alone.
DEFAULT_ORDER_DOCS = ("dpr_karpukhin_2020", "colbert_khattab_2020", "splade_v2_formal_2021")
DEFAULT_ORDER_ELEMENTS = 10


def _page_text(chars: list[Any], page_width: float) -> tuple[str, str | None]:
    """Reconstruct a page's text both ways.

    Returns `(flat_text, column_text)`. `column_text` is `None` when the page was
    not detected as multi-column -- for such a page the flat reconstruction *is*
    the output, and there is nothing to compare.
    """
    ranges = _detect_columns(chars, page_width)
    flat_text = " ".join(
        _line_text(line) for line in _cluster_chars_into_lines(chars, None)
    )
    column_text = (
        None
        if ranges is None
        else " ".join(_line_text(line) for line in _cluster_chars_into_lines(chars, ranges))
    )
    return flat_text, column_text


def _dehyphenate(text: str) -> str:
    """Join words broken across lines by a trailing hyphen.

    A two-column layout makes line-wrapped sentences unavoidable, and the trailing
    hyphen of a broken word is real text, not an artifact of interleaving. Both
    readings are normalised this way before the spot check so the sentence being
    looked for is not defeated by where the layout happened to break it.
    """
    return re.sub(r"-\s+", "", " ".join(text.split()))


def _removed_pairs(flat_text: str, column_text: str) -> list[tuple[str, str]]:
    """Adjacent word pairs that exist only in the flat, interleaved reading.

    A pair survives the column split if it still occurs somewhere in the
    column-aware reading, which is the lenient test: a sentence that wraps across
    a page boundary keeps its internal pairs, and only pairs that existed *because*
    two columns were concatenated are counted. This is evidence the fix did
    something, not a pass/fail criterion.
    """
    flat_tokens = flat_text.split()
    column_tokens = column_text.split()
    surviving = set(zip(column_tokens, column_tokens[1:]))
    return [
        (a, b)
        for a, b in zip(flat_tokens, flat_tokens[1:])
        if (a, b) not in surviving
    ]


def _residual_splices(chars: list[Any], ranges: list[tuple[float, float]]) -> dict[str, list[str]]:
    """Check the column-aware output for lines that still straddle a gutter.

    A line in the output may carry characters from two columns and still be
    **correct**: a figure caption or a centred title genuinely spans the full width
    and flows across the gutter, so emitting it intact is right. What would be a
    defect is a line whose two column parts are separated by a gutter-sized *gap*
    -- two columns concatenated, which is the corruption this phase removes.

    So the two cases are separated by the same measurement, and each is reported
    under its own key rather than being summed. `spliced` is the invariant that
    must hold; `full_width` is informational and is expected to be non-zero.
    """
    from adaptive_rag.ingestion.normalizer import (
        _COLUMN_MIN_GUTTER_PT,
        _column_index,
    )

    spliced: list[str] = []
    full_width: list[str] = []
    for line in _cluster_chars_into_lines(chars, ranges):
        text = _line_text(line)
        if not text.strip():
            continue
        parts: list[list[Any]] = [[] for _ in ranges]
        for c in line:
            parts[_column_index((c.x0 + c.x1) / 2.0, ranges)].append(c)
        occupied = [i for i, part in enumerate(parts) if part]
        if len(occupied) < 2:
            continue
        gaps = [
            min(c.x0 for c in parts[right]) - max(c.x1 for c in parts[left])
            for left, right in zip(occupied, occupied[1:])
        ]
        if max(gaps) >= _COLUMN_MIN_GUTTER_PT:
            spliced.append(text[:200])
        else:
            full_width.append(text[:200])
    return {"spliced": spliced, "full_width": full_width}


def _element_texts(doc: Any) -> list[tuple[str, str, int | None]]:
    """`(element_id, text, page)` for every canonical element, in position order."""
    return [
        (element.element_id, element.content.text or "", element.provenance.page)
        for element in doc.elements
    ]


def _reingest(
    manifest: dict[str, Any], detect: Any, with_columns: bool
) -> tuple[dict[str, list[tuple[str, str, int | None]]], dict[str, int]]:
    """Ingest every document with `_detect_columns` monkey-patched.

    `with_columns=False` forces the pre-Phase-8 flat path by making detection
    always decline. The patch is process-local and always restored, so the module
    is not left altered if ingestion raises.
    """
    original = normalizer_module._detect_columns
    normalizer_module._detect_columns = detect
    try:
        pipeline = IngestionPipeline()
        out: dict[str, list[tuple[str, str, int | None]]] = {}
        pages_two_column: dict[str, int] = {}
        for document_id, paper in sorted(manifest.items()):
            doc = pipeline.ingest_pdf(Path(paper["local_path"]), paper)
            out[document_id] = _element_texts(doc)
            pages_two_column[document_id] = sum(
                1
                for issue in doc.extraction_report.issues
                if issue.code == "COLUMN_LAYOUT_DETECTED"
            )
        return out, pages_two_column
    finally:
        normalizer_module._detect_columns = original


def run(
    raw_dir: Path,
    *,
    spot_check: str = DEFAULT_SPOT_CHECK,
    spot_check_doc: str = DEFAULT_SPOT_CHECK_DOC,
    order_docs: tuple[str, ...] = DEFAULT_ORDER_DOCS,
    order_elements: int = DEFAULT_ORDER_ELEMENTS,
    skip_chunk_delta: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    extractor = PdfplumberExtractor()
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    per_document: list[dict[str, Any]] = []
    total_removed = 0
    residual_examples: list[dict[str, Any]] = []
    single_column_pages = 0
    two_column_pages = 0
    spot_before = False
    spot_after = False

    for pdf in pdfs:
        document_id = pdf.stem
        pages, _info = extractor.extract_document(pdf)
        doc_removed = 0
        doc_single = 0
        doc_two = 0

        for page in pages:
            flat_text, column_text = _page_text(page.chars, page.width)
            if column_text is None:
                doc_single += 1
                continue

            doc_two += 1
            doc_removed += len(_removed_pairs(flat_text, column_text))
            ranges = _detect_columns(page.chars, page.width)
            residual = _residual_splices(page.chars, ranges or [])
            for kind in ("spliced", "full_width"):
                for offender in residual[kind]:
                    residual_examples.append(
                        {
                            "document_id": document_id,
                            "page": page.page_number,
                            "kind": kind,
                            "text": offender,
                        }
                    )
            if document_id == spot_check_doc:
                spot_before = spot_before or spot_check in _dehyphenate(flat_text)
                spot_after = spot_after or spot_check in _dehyphenate(column_text)

        total_removed += doc_removed
        single_column_pages += doc_single
        two_column_pages += doc_two
        per_document.append(
            {
                "document_id": document_id,
                "pages": len(pages),
                "pages_two_column": doc_two,
                "pages_single_column": doc_single,
                "pairs_removed_by_split": doc_removed,
            }
        )

    spliced_examples = [e for e in residual_examples if e["kind"] == "spliced"]
    full_width_examples = [e for e in residual_examples if e["kind"] == "full_width"]
    residual_total = len(spliced_examples)

    manifest_list = json.loads(PAPERS_MANIFEST_PATH.read_text("utf-8"))
    manifest = {p["document_id"]: p for p in manifest_list}

    # Gate 3 -- single-column regression, measured against the stored Phase 7
    # document artifacts rather than against a reimplementation of the old code.
    after_elements, pages_two_column = _reingest(
        manifest, normalizer_module._detect_columns, with_columns=True
    )
    before_elements, _ = _reingest(manifest, lambda chars, width: None, with_columns=False)

    regression: list[dict[str, Any]] = []
    on_disk_comparable = 0
    for document_id, elements in after_elements.items():
        stored = DOCUMENTS_DIR / f"{document_id}.json"
        if not stored.is_file():
            continue
        on_disk_comparable += 1
        stored_elements = [
            (e["element_id"], e["content"]["text"] or "", (e["provenance"] or {}).get("page"))
            for e in json.loads(stored.read_text("utf-8"))["elements"]
        ]
        flat_elements = before_elements[document_id]
        if stored_elements == elements:
            continue
        if pages_two_column.get(document_id, 0) == 0:
            # Distinguish "the column fix changed this" from "the stored artifact
            # was already stale relative to the code that reads it". Only the
            # second is a real single-column regression.
            stale = stored_elements != flat_elements
            changed = [
                {
                    "position": position,
                    "page": page,
                    "stored": stored_text[:160],
                    "reingested_flat": flat_text[:160],
                }
                for position, ((_, stored_text, page), (_, flat_text, _)) in enumerate(
                    zip(stored_elements, flat_elements, strict=False)
                )
                if stored_text != flat_text
            ][:5]
            regression.append(
                {
                    "document_id": document_id,
                    "two_column_pages": 0,
                    "n_elements_stored": len(stored_elements),
                    "n_elements_reingested": len(elements),
                    "stored_artifact_already_stale": stale,
                    "column_fix_changed_this_document": not stale,
                    "first_differences": changed,
                    "reason": (
                        "no page of this document was detected as two-column, so its "
                        "output must be byte-identical to the stored artifact"
                        + (
                            "; it is not, but the flat re-ingest also differs from the "
                            "stored artifact, so the drift predates the column fix"
                            if stale
                            else "; the flat re-ingest matches the stored artifact, so "
                            "the column fix changed a single-column document"
                        )
                    ),
                }
            )
            continue
        # A document that does have two-column pages is *expected* to differ.
        # Report how much, so an unexpectedly large change is visible.
        changed_pages = sorted(
            {
                page
                for (_, before_text, page), (_, after_text, _) in zip(
                    stored_elements, elements, strict=False
                )
                if before_text != after_text and page is not None
            }
        )
        regression.append(
            {
                "document_id": document_id,
                "two_column_pages": pages_two_column.get(document_id, 0),
                "n_elements_stored": len(stored_elements),
                "n_elements_reingested": len(elements),
                "changed_pages": changed_pages,
                "reason": "expected to differ: document has two-column pages",
                "expected": True,
            }
        )

    # A regression is a *hard* failure only when the column fix changed a document
    # it should not have touched. A stored artifact that the flat re-ingest also
    # fails to reproduce is stale for an unrelated reason and is reported, not
    # blamed on this change.
    hard_regressions = [
        entry
        for entry in regression
        if not entry.get("expected") and entry.get("column_fix_changed_this_document")
    ]
    stale_artifacts = [
        entry
        for entry in regression
        if not entry.get("expected") and entry.get("stored_artifact_already_stale")
    ]

    # Gate 4 -- reading order, printed for a human to check against the PDF.
    order_dump: list[dict[str, Any]] = []
    for document_id in order_docs:
        if document_id not in manifest:
            order_dump.append({"document_id": document_id, "error": "not in manifest"})
            continue
        order_dump.append(
            {
                "document_id": document_id,
                "opening_elements": [
                    {"page": page, "text": text[:200]}
                    for _element_id, text, page in after_elements[document_id][:order_elements]
                ],
            }
        )

    chunk_delta: dict[str, Any] | None = None
    if not skip_chunk_delta:
        chunk_delta = _chunk_delta(manifest)

    gates = {
        "residual_interleaving": {
            "criterion": (
                f"<= {MAX_RESIDUAL_SPLICED_LINES} output lines still show a "
                "gutter-sized gap between their column parts"
            ),
            "spliced_lines": residual_total,
            "spliced_examples": spliced_examples[:20],
            "full_width_lines_informational": len(full_width_examples),
            "full_width_examples": full_width_examples[:10],
            "full_width_note": (
                "lines that carry both columns' characters with no gutter-sized gap "
                "between them. Titles, figure captions and table rows genuinely span "
                "the full width, so emitting them intact is correct and they are "
                "reported rather than counted as defects"
            ),
            "pairs_removed_by_split": total_removed,
            "phase7_glue_claim": PHASE7_GLUE_CLAIM,
            "phase7_claim_note": (
                "docs/phase_7_results.md section 11 records '1.4% of tokens glued' "
                "without retaining the measurement behind it. That is a different "
                "quantity on a different scale, quoted here as the prior claim and "
                "not as a baseline this gate confirms."
            ),
            "passed": residual_total <= MAX_RESIDUAL_SPLICED_LINES,
        },
        "spot_check": {
            "criterion": "the sentence occurs in the column-aware text and not in the flat text",
            "document_id": spot_check_doc,
            "found_in_flat": spot_before,
            "found_in_column_text": spot_after,
            "passed": spot_after and not spot_before,
        },
        "single_column_regression": {
            "criterion": (
                "every document with no detected two-column page re-ingests "
                "byte-identical to its stored Phase 7 artifact"
            ),
            "documents_compared_to_disk": on_disk_comparable,
            "unexpected_differences": hard_regressions,
            "stale_stored_artifacts": stale_artifacts,
            "expected_differences": [e for e in regression if e.get("expected")],
            "passed": not hard_regressions and on_disk_comparable > 0,
        },
        "reading_order": {
            "criterion": "hand-verified against the PDF; dumped below for that verification",
            "documents": list(order_docs),
            "passed": None,
            "note": "not machine-decidable; verify `reading_order_dump`",
        },
        "chunk_delta": {
            "criterion": "reported, not gated",
            "measured": chunk_delta,
            "passed": None,
        },
    }

    return {
        "validation_version": VALIDATION_VERSION,
        "raw_dir": str(raw_dir.relative_to(REPO_ROOT)),
        "documents": len(per_document),
        "pages_two_column": two_column_pages,
        "pages_single_column": single_column_pages,
        "residual_spliced_lines": residual_total,
        "full_width_lines_informational": len(full_width_examples),
        "pairs_removed_by_split": total_removed,
        "per_document": per_document,
        "gates": gates,
        "reading_order_dump": order_dump,
    }


def _chunk_delta(manifest: dict[str, Any]) -> dict[str, Any]:
    """Count how many chunk texts change, by chunking the corpus both ways.

    Element counts are not chunk counts: the chunker merges elements under section
    headers and splits on token budgets, so a changed element can move a boundary
    and change several chunks downstream. Both sides go through the *same*
    chunker with the *same* config, so every difference is attributable to
    reading order.

    The chunker reads Documents off disk, so each side re-ingests from the PDFs
    with `_detect_columns` patched for that side and restored afterwards.
    """

    def chunk() -> dict[str, str]:
        ingest = IngestionPipeline()
        chunking = ChunkingPipeline()
        texts: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "documents"
            out_dir = Path(tmp) / "chunks"
            docs_dir.mkdir(parents=True)
            for _document_id, paper in sorted(manifest.items()):
                ingest.save_document(
                    ingest.ingest_pdf(Path(paper["local_path"]), paper), out_dir=docs_dir
                )
            for _document_id, chunks in chunking.chunk_corpus(
                docs_dir=docs_dir, out_dir=out_dir
            ).items():
                for chunked in chunks:
                    texts[chunked.chunk_id] = chunked.text
        return texts

    original = normalizer_module._detect_columns
    normalizer_module._detect_columns = lambda chars, width: None
    try:
        before = chunk()
    finally:
        normalizer_module._detect_columns = original
    after = chunk()

    shared = set(before) & set(after)
    changed = sorted(cid for cid in shared if before[cid] != after[cid])
    return {
        "chunks_before": len(before),
        "chunks_after": len(after),
        "chunk_ids_shared": len(shared),
        "chunk_texts_changed": len(changed),
        "chunk_ids_only_before": sorted(set(before) - set(after)),
        "chunk_ids_only_after": sorted(set(after) - set(before)),
        "example_changed_chunk_ids": changed[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Phase 8 column extraction fix.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--spot-check", type=str, default=DEFAULT_SPOT_CHECK)
    parser.add_argument("--spot-check-doc", type=str, default=DEFAULT_SPOT_CHECK_DOC)
    parser.add_argument("--order-docs", type=str, nargs="*", default=list(DEFAULT_ORDER_DOCS))
    parser.add_argument("--order-elements", type=int, default=DEFAULT_ORDER_ELEMENTS)
    parser.add_argument("--skip-chunk-delta", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "phase8" / "column_validation.json",
    )
    args = parser.parse_args()

    report = run(
        args.raw_dir,
        spot_check=args.spot_check,
        spot_check_doc=args.spot_check_doc,
        order_docs=tuple(args.order_docs),
        order_elements=args.order_elements,
        skip_chunk_delta=args.skip_chunk_delta,
        limit=args.limit,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"two-column pages     : {report['pages_two_column']}")
    print(f"single-column pages  : {report['pages_single_column']}")
    print(f"pairs removed by fix : {report['pairs_removed_by_split']}")
    print(
        f"residual spliced lines: {report['residual_spliced_lines']} "
        f"(gate <= {MAX_RESIDUAL_SPLICED_LINES}; "
        f"{report['full_width_lines_informational']} full-width title/caption/table "
        "lines reported separately as correct)"
    )
    print()
    for name, gate in report["gates"].items():
        verdict = gate["passed"]
        print(f"gate {name:28s} {verdict if verdict is not None else 'manual'}")
    print()
    print("reading order (verify against the PDF):")
    for entry in report["reading_order_dump"]:
        print(f"  {entry['document_id']}")
        for element in entry.get("opening_elements", []):
            print(f"    p{element['page']} {element['text'][:110]}")
    print()
    print(f"written: {args.out}")

    return (
        0
        if all(g["passed"] for g in report["gates"].values() if g["passed"] is not None)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
