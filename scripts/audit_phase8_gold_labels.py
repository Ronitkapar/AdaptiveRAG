#!/usr/bin/env python3
"""
audit_phase8_gold_labels.py
----------------------------
Phase 8 Step 4 -- audit the frozen benchmark's gold labels against the clean corpus.

The finding this script exists to act on: **the gold section labels are themselves
partly corrupt.** They were derived by reading the corpus as the pre-Phase-8
extractor produced it, so a label like

    ('3.4 Offline Indexing: Computing & Storing computation)andreduceacrossqueryterms...')

carries a real heading followed by two columns of body prose spliced onto it. Those
labels happen to match the *garbled* section paths, and stop matching once reading
order is fixed.

This matters asymmetrically, and the asymmetry decides what Phase 8 may claim:

* `recall_at_5`, `mrr` and `hit_at_5` key on `relevant_documents`, which is
  document-level and untouched by both re-chunking and label repair. The headline
  before/after comparison rests on these and is sound.
* `ndcg_at_5` and `precision_at_k` derive chunk relevance by matching
  `metadata.section_path` against the section labels
  (`evaluation/dataset.py:158`). They are therefore only as good as the labels,
  and **are not comparable across arms until the labels are repaired**.

Rather than silently carrying that, the script quantifies it and proposes repairs,
leaving the decision to a human:

* `suspicious` -- labels carrying the splice signature (too long, too many words,
  or fused-case glue). Reported per document.
* `proposed` -- the clean section path each suspicious label most likely meant,
  recovered by matching the label's leading clean text against the after-arm paths.
  A proposal is `high` confidence only when the label's own leading segment matches
  a real path prefix exactly.

Nothing here rewrites the dataset. The frozen benchmark is the benchmark, and a
label repaired by a similarity heuristic has to be read by a person before it is
allowed to change a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import (  # noqa: E402
    EVALUATION_DIR,
    PHASE8_AFTER_CHUNKS_DIR,
    PHASE8_BEFORE_CHUNKS_DIR,
    REPO_ROOT,
)
from adaptive_rag.schemas import Chunk  # noqa: E402

AUDIT_VERSION = "phase8_gold_label_audit_v1"

# A real heading is short and title-like. A spliced one carries body prose.
MAX_HEADING_WORDS = 8
MAX_HEADING_CHARS = 60
# Two or more over-long words in one heading is the signature of fused prose; a
# legitimate technical heading has the odd long token, not a run of them.
LONG_WORD_CHARS = 14
MAX_LONG_WORDS = 1
# A section title is words, so it has spaces. The corruption interleaving two
# columns produces is a space-free run of body prose -- `runtimeefficiency`,
# `fordocumentranking` -- and that is what this threshold catches.
#
# The value is taken from the data rather than guessed. Across the 140 section
# references in the frozen benchmark, the longest single token in a label that
# resolves against the clean corpus is **18** characters, and the shortest token
# that only ever appears in a spliced label is **26**. 20 sits in that gap, and
# flags 20 of the 50 damaged labels with no false positives among the 90 valid
# ones. Raise it and real splices slip through; lower it and legitimate
# identifiers and URLs start being called corruption.
MAX_PLAUSIBLE_WORD_CHARS = 20

# Worst-last, so a path's confidence is the confidence of its least certain link.
CONFIDENCE_ORDER = ("high", "medium", "low", "none")


def load_chunks(chunks_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
    return chunks


def section_paths(chunks: list[Chunk]) -> dict[str, set[tuple[str, ...]]]:
    """Every section path present in a corpus, keyed by document."""
    paths: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for chunk in chunks:
        paths[chunk.metadata.document_id].add(tuple(chunk.metadata.section_path))
    return paths


def is_corrupt(label: tuple[str, ...], resolves: bool) -> bool:
    """True when a section label is demonstrably damaged, not merely long.

    Length and word count do not diagnose anything: several legitimate section
    titles in this corpus are long strings -- a postal address picked up as a
    heading, say -- and an earlier version of this check flagged 57 labels of which
    most were perfectly valid. Two things are actual evidence instead:

    * the label contains a space-free run too long to be a word, which is what
      splicing two columns of prose together leaves behind; or
    * the label does not resolve against the clean corpus at all.

    A label that is merely wordy, and resolves, is left alone.
    """
    if any(
        len(token) > MAX_PLAUSIBLE_WORD_CHARS
        for part in label
        for token in part.split()
    ):
        return True
    return not resolves


def _recover_component(
    component: str, candidates: set[str]
) -> tuple[str | None, str]:
    """Recover the heading a spliced component was built from.

    A splice *appends* body prose to a heading that was already complete, so the
    heading is a literal prefix of the component. Matching on that is far stronger
    than trimming to a plausible length: it either finds the exact heading or
    admits it did not, instead of quietly returning a plausible-looking fragment
    that is not the real one.

    The longest candidate that prefixes the component wins, so `3.4 Offline
    Indexing` is preferred over `3` when both would match.
    """
    prefixed = [c for c in candidates if component.startswith(c) and c]
    if prefixed:
        return max(prefixed, key=len), "high"

    best: str | None = None
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, component[:MAX_HEADING_CHARS], candidate).ratio()
        if score > best_score:
            best, best_score = candidate, score
    if best is None:
        return None, "none"
    return best, "low" if best_score < 0.8 else "medium"


def propose(
    label: tuple[str, ...], candidates: set[tuple[str, ...]]
) -> tuple[tuple[str, ...] | None, str]:
    """Best clean section path for a spliced label, and how much to trust it.

    Each label component is recovered independently and the confidence of the
    weakest component wins, because a path is only as trustworthy as its least
    certain link: a perfect parent heading followed by a guessed child is still a
    guess.
    """
    if not candidates:
        return None, "none"

    flat = {part for path in candidates for part in path}
    recovered: list[str] = []
    worst = "high"
    for component in label:
        match, confidence = _recover_component(component, flat)
        if match is None:
            return None, "none"
        recovered.append(match)
        if CONFIDENCE_ORDER.index(confidence) > CONFIDENCE_ORDER.index(worst):
            worst = confidence

    # Only offer the reconstruction if it is a path the corpus actually has.
    proposal = tuple(recovered)
    if proposal in candidates:
        return proposal, worst
    return None, "none" if worst == "none" else worst


@dataclass
class LabelRef:
    example_id: str
    document_id: str
    prefix: tuple[str, ...]

    def resolves(self, paths: dict[str, set[tuple[str, ...]]]) -> bool:
        return any(
            path[: len(self.prefix)] == self.prefix
            for path in paths.get(self.document_id, set())
        )


@dataclass
class Report:
    total_refs: int = 0
    suspicious: list[dict[str, Any]] = field(default_factory=list)
    confidence: Counter = field(default_factory=Counter)


def audit(dataset: Path, before_dir: Path, after_dir: Path) -> dict[str, Any]:
    examples = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    before = section_paths(load_chunks(before_dir))
    after = section_paths(load_chunks(after_dir))

    refs: list[LabelRef] = [
        LabelRef(example["example_id"], ref["document_id"], tuple(ref["section_path_prefix"]))
        for example in examples
        for ref in (example.get("relevant_sections") or [])
    ]

    entries: list[dict[str, Any]] = []
    for ref in refs:
        record: dict[str, Any] = {
            "example_id": ref.example_id,
            "document_id": ref.document_id,
            "section_path_prefix": list(ref.prefix),
            "resolves_before": ref.resolves(before),
            "resolves_after": ref.resolves(after),
        }
        record["corrupt"] = is_corrupt(ref.prefix, record["resolves_after"])
        if record["corrupt"]:
            candidate, confidence = propose(ref.prefix, after.get(ref.document_id, set()))
            record["proposed_prefix"] = list(candidate) if candidate else None
            record["confidence"] = confidence
        entries.append(record)

    suspicious = [e for e in entries if e["corrupt"]]
    confidence = Counter(e.get("confidence", "n/a") for e in suspicious)
    regressed = [e for e in entries if e["resolves_before"] and not e["resolves_after"]]
    repaired = [e for e in entries if not e["resolves_before"] and e["resolves_after"]]
    broken_both = [
        e for e in entries if not e["resolves_before"] and not e["resolves_after"]
    ]

    return {
        "audit_version": AUDIT_VERSION,
        "dataset": str(dataset.relative_to(REPO_ROOT)),
        "arms": {
            "before": str(before_dir.relative_to(REPO_ROOT)),
            "after": str(after_dir.relative_to(REPO_ROOT)),
        },
        "examples": len(examples),
        "section_refs_total": len(refs),
        "labels_corrupt": len(suspicious),
        "labels_corrupt_fraction": (len(suspicious) / len(refs)) if refs else 0.0,
        "corrupt_by_document": Counter(
            e["document_id"] for e in suspicious
        ).most_common(),
        "proposal_confidence": dict(confidence),
        "resolution": {
            "resolved_before": sum(1 for e in entries if e["resolves_before"]),
            "resolved_after": sum(1 for e in entries if e["resolves_after"]),
            "regressed_before_to_after": len(regressed),
            "newly_resolved_after": len(repaired),
            "unresolved_in_both": len(broken_both),
        },
        "metric_impact": {
            "document_level_metrics_unaffected": ["recall_at_5", "mrr", "hit_at_5"],
            "section_dependent_metrics": ["ndcg_at_5", "precision_at_k"],
            "note": (
                "recall_at_5 / mrr / hit_at_5 key on relevant_documents only "
                "(evaluation/retrieval.py:134,151) and are valid on both arms. "
                "ndcg_at_5 and precision_at_k derive chunk relevance by matching "
                "section_path against these labels (evaluation/dataset.py:158), so "
                "they are NOT comparable across arms until the labels are repaired. "
                "Per docs/phases/phase-8.md section 6, they are reported as not "
                "measured on the new corpus rather than as measured on labels known "
                "to be corrupt."
            ),
        },
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVALUATION_DIR / "phase7_eval_v1.jsonl",
        help="Frozen benchmark to audit. Never rewritten.",
    )
    parser.add_argument("--before-dir", type=Path, default=PHASE8_BEFORE_CHUNKS_DIR)
    parser.add_argument("--after-dir", type=Path, default=PHASE8_AFTER_CHUNKS_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "phase8" / "gold_label_audit.json",
    )
    args = parser.parse_args()

    report = audit(args.dataset, args.before_dir, args.after_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"section refs                : {report['section_refs_total']}")
    print(f"corrupt (spliced) labels    : {report['labels_corrupt']} "
          f"({report['labels_corrupt_fraction']:.0%})")
    print(f"proposal confidence         : {report['proposal_confidence']}")
    print()
    resolution = report["resolution"]
    print(f"resolve before / after      : {resolution['resolved_before']} / "
          f"{resolution['resolved_after']} of {report['section_refs_total']}")
    print(f"regressed by the column fix : {resolution['regressed_before_to_after']}")
    print(f"unresolved in both arms     : {resolution['unresolved_in_both']}")
    print()
    print("metric impact:")
    print(f"  unaffected : {', '.join(report['metric_impact']['document_level_metrics_unaffected'])}")
    print(f"  blocked    : {', '.join(report['metric_impact']['section_dependent_metrics'])}")
    print()
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())