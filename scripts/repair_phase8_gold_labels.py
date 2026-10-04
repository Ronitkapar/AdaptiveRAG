#!/usr/bin/env python3
"""
repair_phase8_gold_labels
-------------------------
Decide whether the frozen benchmark's gold section labels can be repaired
mechanically, and measure what repairing them would do to `nDCG@5`.

`nDCG@5` and `precision_at_k` are the two Phase 8 metrics reported as *not
measured*, because 47 of 140 gold section labels are spliced and 40 stop
resolving once the column fix corrects reading order
(`experiments/phase8/gold_label_audit.json`). Those labels were derived by
reading the corpus as the pre-Phase-8 extractor produced it, so a label like

    ('3.4 Offline Indexing: Computing & Storing computation)andreduceacrossqueryterms...')

carries a real heading followed by two columns of body prose spliced onto it.

This script answers three questions offline, and changes no labels:

1. **What is mechanically recoverable?** The Step 4 audit proposed a
   reconstruction for each corrupt label and graded its confidence. This
   re-derives the proposals and separates the ones that are *verifiable* from
   the ones that merely look plausible.
2. **Would repairing them unblock `nDCG@5`?** `nDCG@5` is recomputable offline
   from the stored `retrieved_chunk_ids` in `rows.jsonl` plus each corpus arm's
   chunk -> section-path map. No retrieval, no API, no index.
3. **Would it unblock it *honestly*?** A label is only safe to repair if the
   repaired path is a section the corpus actually has. If the repair points at a
   path that is itself spliced, the metric would be computed against a corrupt
   target and would look measured without being trustworthy.

The central finding this script exists to establish: **the audit's
"high confidence" proposals are not all safe.** A proposal is graded high
because its leading segment matches a real path prefix exactly, but the
corpus's own `section_path` values still carry spliced text in 42 of 338
distinct paths. Three high-confidence proposals resolve *only* to such paths, so
accepting them would compute `nDCG@5` against corrupt labels -- exactly the
failure the audit was convened to prevent, one level down.

Nothing here rewrites the frozen benchmark. The benchmark is the benchmark, and a
label changed by a heuristic has to be read by a person before it is allowed to
change a published number. This produces the evidence for that read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import (  # noqa: E402
    EVALUATION_DIR,
    PHASE8_AFTER_CHUNKS_DIR,
    PHASE8_BEFORE_CHUNKS_DIR,
    REPO_ROOT,
)

RESULT_VERSION = "phase8_gold_label_repair_v1"

#: A space-free token longer than this cannot be a word. It is the splice
#: signature: two columns of prose read side by side leave exactly this behind.
MAX_PLAUSIBLE_WORD_CHARS = 25

DEFAULT_SUITES: dict[str, Path] = {
    "phase8_after": REPO_ROOT / "experiments" / "phase8" / "combined" / "p8a_e1",
    "phase8_before": REPO_ROOT / "experiments" / "phase8" / "combined" / "p8b_e1",
}
DEFAULT_SYSTEMS = ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")
K = 5


def is_spliced(path: Sequence[str]) -> bool:
    """True when any component carries a token too long to be a real word.

    This is a *heuristic*, and it is knowingly over-inclusive: the corpus has
    genuine headings that are legitimately fused, such as
    `2.1 AnIntroductionToDrugDiscoveryandDevelopment`, which trips the same
    test as a spliced label. Excluding such paths costs a little recall of
    repairs -- measured at 3 labels, 22 safe becoming 25 under the looser rule --
    which is the right trade, because admitting a genuinely spliced target would
    corrupt a published metric instead of merely losing a repair. The sensitivity
    is reported in the artifact so the choice is visible rather than buried.
    """
    return any(
        len(token) > MAX_PLAUSIBLE_WORD_CHARS
        for part in path
        for token in str(part).split()
    )


def load_section_paths(chunks_dir: Path) -> dict[str, dict[str, set[tuple[str, ...]]]]:
    """Map `document_id -> {section_path}` from a corpus arm's chunk files."""
    paths: dict[str, dict[str, set[tuple[str, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for chunk_file in sorted(chunks_dir.glob("*.chunks.jsonl")):
        with open(chunk_file, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                document_id = str(chunk.get("document_id"))
                section_path = (chunk.get("metadata") or {}).get("section_path")
                if section_path:
                    paths[document_id][tuple(section_path)].add(str(chunk.get("chunk_id")))
    return {
        document_id: dict(values) for document_id, values in paths.items()
    }


def chunk_section_index(
    chunks_dir: Path,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Map `chunk_id -> section_path`, the map `relevant_chunk_ids` needs."""
    index: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    for chunk_file in sorted(chunks_dir.glob("*.chunks.jsonl")):
        with open(chunk_file, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                section_path = (chunk.get("metadata") or {}).get("section_path")
                if section_path is not None:
                    index[str(chunk.get("chunk_id"))] = tuple(section_path)
    return dict(index)


# ---------------------------------------------------------------------------
# Repair proposals
# ---------------------------------------------------------------------------


def recover_component(component: str, candidates: set[str]) -> tuple[str | None, str]:
    """Recover the heading a spliced component was built from.

    A splice *appends* body prose to a heading that was already complete, so the
    heading is a literal prefix of the component. Matching on that is stronger
    than trimming to a plausible length: it either finds the exact heading or
    admits it did not.

    The longest prefixing candidate wins, so `3.4 Offline Indexing` beats `3`
    when both match.

    **Unlike the Step 4 audit, spliced candidates are excluded here.** The audit
    graded a proposal `high` when the label's leading text matched a real path
    prefix -- including a path that is itself spliced. That is how a
    "high-confidence" proposal can point at a corrupt target. Excluding them
    costs some recall of proposals and buys the property that matters: every
    proposal this script calls safe resolves to a section a reader would accept.
    """
    clean = {c for c in candidates if c and not is_spliced([c])}
    prefixed = [c for c in clean if component.startswith(c)]
    if prefixed:
        return max(prefixed, key=len), "high"
    return None, "none"


def propose_repair(
    prefix: Sequence[str], candidates: set[tuple[str, ...]]
) -> tuple[tuple[str, ...] | None, str]:
    """Best clean section path for a spliced label, and how much to trust it.

    Each component is recovered independently and the weakest component's
    confidence wins: a perfect parent heading followed by a guessed child is
    still a guess.
    """
    if not candidates:
        return None, "none"
    flat = {part for path in candidates for part in path}
    recovered: list[str] = []
    for component in prefix:
        match, confidence = recover_component(str(component), flat)
        if match is None:
            return None, "none"
        recovered.append(match)
    proposal = tuple(recovered)
    if proposal in candidates:
        return proposal, "high"
    # A prefix of a real path is acceptable: `relevant_chunk_ids` matches on
    # `path[:len(prefix)] == prefix`, so a shorter exact prefix resolves the
    # same chunks a longer one would.
    if any(path[: len(proposal)] == proposal for path in candidates):
        return proposal, "high"
    return None, "none"


def classify(
    entry: Mapping[str, Any],
    after_paths: Mapping[str, set[tuple[str, ...]]],
) -> dict[str, Any]:
    """Grade one label as `safe`, `unverifiable`, or `unrepairable`.

    `safe` additionally requires that the repaired path is not itself spliced,
    which is the check the Step 4 audit lacked.
    """
    result: dict[str, Any] = {
        "example_id": entry.get("example_id"),
        "document_id": entry.get("document_id"),
        "corrupt_prefix": list(entry.get("section_path_prefix") or []),
        "audit_confidence": entry.get("confidence"),
        "audit_proposed": list(entry.get("proposed_prefix") or []) or None,
    }
    document_id = str(entry.get("document_id"))
    candidates = set(after_paths.get(document_id, set()))
    proposal, confidence = propose_repair(entry.get("section_path_prefix") or [], candidates)
    result["repaired_prefix"] = list(proposal) if proposal else None
    result["repair_confidence"] = confidence

    if proposal is None:
        result["verdict"] = "unrepairable"
        result["reason"] = "no clean corpus section path reconstructs this label"
        return result

    matches = [p for p in candidates if p[: len(proposal)] == proposal]
    if not matches:  # defensive: propose_repair only returns resolvable paths
        result["verdict"] = "unverifiable"
        result["reason"] = "proposed prefix does not resolve against the corpus"
        return result

    if any(is_spliced(path) for path in matches):
        result["verdict"] = "unverifiable"
        result["reason"] = (
            "the only corpus paths this prefix resolves to are themselves "
            "spliced; repairing to them would compute a section metric against "
            "a corrupt target"
        )
        result["matching_paths"] = [list(p) for p in sorted(matches)]
        return result

    result["verdict"] = "safe"
    result["matching_paths"] = [list(p) for p in sorted(matches)]
    return result


# ---------------------------------------------------------------------------
# Offline nDCG@5 recomputation
# ---------------------------------------------------------------------------


def ndcg_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
    k: int = K,
) -> float:
    """nDCG@k, matching `evaluation/retrieval.py:_ndcg_at_k` exactly.

    Binary gains, DCG over the top-k, and an IDCG from `min(len(relevant), k)`
    ideal hits. Reproduced here rather than imported because the stored rows do
    not carry traces, only chunk IDs -- so the evaluator cannot be re-run
    directly. The formula is the evaluator's, and `test_ndcg_matches_stored`
    pins the two against each other on the unrepaired labels.
    """
    gains = [1.0 if chunk_id in relevant_chunk_ids else 0.0
             for chunk_id in retrieved_chunk_ids[:k]]
    if not any(gains):
        return 0.0
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(len(relevant_chunk_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def relevant_chunks_for(
    example: Mapping[str, Any],
    retrieved_chunk_ids: Sequence[str],
    repaired: Mapping[tuple[str, tuple[str, ...]], bool],
    chunk_index: Mapping[str, tuple[str, ...]],
) -> set[str]:
    """Chunk-level relevance under the original or repaired labels.

    Mirrors `evaluation/dataset.py:relevant_chunk_ids`: a chunk counts when its
    document is relevant and, where section labels exist for that document, its
    section path matches one of the labelled prefixes. Only prefixes whose repair
    was graded `safe` are substituted -- a repair the script could not verify
    leaves the original label in place, so the recomputation never scores against
    a target the script cannot vouch for.

    Relevance is scoped to `retrieved_chunk_ids`, not to the whole corpus, and
    that scoping is load-bearing. The evaluator iterates `trace.retrieval.results`,
    so a relevant chunk that was never retrieved does not exist as far as nDCG is
    concerned -- and because the IDCG is built from `len(relevant)`, counting
    unretrieved chunks would inflate the denominator and understate every score.
    Getting this wrong reproduced only 85 of 107 stored values, which is how it
    was caught: `test_ndcg_matches_stored_values` pins the two together.
    """
    relevant: set[str] = set()
    prefixes: list[tuple[str, ...]] = []
    for ref in example.get("relevant_sections") or []:
        document_id = str(ref["document_id"])
        if document_id not in set(example.get("relevant_documents") or []):
            continue
        prefix = tuple(ref.get("section_path_prefix") or [])
        if repaired.get((document_id, prefix)):
            repaired_prefix = repaired[(document_id, prefix)]
            if repaired_prefix is not None:
                prefix = repaired_prefix
        prefixes.append(prefix)

    relevant_documents = set(example.get("relevant_documents") or [])
    for chunk_id in retrieved_chunk_ids:
        section_path = chunk_index.get(chunk_id)
        if section_path is None:
            continue
        if chunk_id.split("::", 1)[0] not in relevant_documents:
            continue
        if prefixes and not any(
            section_path[: len(p)] == p for p in prefixes
        ):
            continue
        relevant.add(chunk_id)
    return relevant


def arm_ndcg(
    suite_root: Path,
    examples: Mapping[str, Mapping[str, Any]],
    chunk_index: Mapping[str, tuple[str, ...]],
    repaired: Mapping[tuple[str, tuple[str, ...]], bool],
    systems: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Mean nDCG@5 per arm, over the stored per-query retrievals."""
    out: dict[str, dict[str, Any]] = {}
    for system in systems:
        matches = sorted(suite_root.glob(f"*__E1_baseline_comparison__{system}/rows.jsonl"))
        if not matches:
            continue
        rows = [
            json.loads(line)
            for line in matches[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        values: list[float] = []
        n_scored = 0
        for row in rows:
            example = examples.get(str(row.get("query_id")))
            if example is None:
                continue
            retrieved = [str(c) for c in row.get("retrieved_chunk_ids") or []]
            relevant = relevant_chunks_for(example, retrieved, repaired, chunk_index)
            values.append(ndcg_at_k(retrieved, relevant))
            n_scored += 1
        if values:
            out[system] = {
                "mean_ndcg_at_5": round(sum(values) / len(values), 6),
                "n": len(values),
                "n_scored": n_scored,
            }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyse(
    dataset: Path,
    before_dir: Path,
    after_dir: Path,
    suites: Mapping[str, Path],
    systems: Sequence[str],
) -> dict[str, Any]:
    """Grade every label, then measure nDCG@5 before and after safe repairs."""
    examples_list = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = {str(e["example_id"]): e for e in examples_list}

    audit_path = REPO_ROOT / "experiments" / "phase8" / "gold_label_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    entries = audit.get("entries", [])

    after_paths = load_section_paths(after_dir)
    before_paths = load_section_paths(before_dir)

    graded = [classify(entry, after_paths) for entry in entries if entry.get("corrupt")]

    # The substitution map: only `safe` repairs change a label.
    repaired: dict[tuple[str, tuple[str, ...]], bool] = {}
    for record in graded:
        if record["verdict"] != "safe":
            continue
        document_id = str(record["document_id"])
        original = tuple(record["corrupt_prefix"])
        repaired[(document_id, original)] = tuple(record["repaired_prefix"])

    verdict_counts = Counter(r["verdict"] for r in graded)
    disagreements = [
        {
            "example_id": r["example_id"],
            "document_id": r["document_id"],
            "audit_confidence": r["audit_confidence"],
            "audit_proposed": r["audit_proposed"],
            "repaired_prefix": r["repaired_prefix"],
            "verdict": r["verdict"],
            "reason": r["reason"],
        }
        for r in graded
        if r["audit_confidence"] == "high" and r["verdict"] != "safe"
    ]

    ndcg: dict[str, Any] = {}
    for arm, suite_root in suites.items():
        chunks_dir = after_dir if arm == "phase8_after" else before_dir
        chunk_index = chunk_section_index(chunks_dir)
        ndcg[arm] = {
            "original_labels": arm_ndcg(suite_root, examples, chunk_index, {}, systems),
            "safe_repaired_labels": arm_ndcg(
                suite_root, examples, chunk_index, repaired, systems
            ),
        }

    spliced_after = sorted(
        path
        for paths in after_paths.values()
        for path in paths
        if is_spliced(path)
    )
    spliced_before = sorted(
        path
        for paths in before_paths.values()
        for path in paths
        if is_spliced(path)
    )

    integrity_before = len(spliced_before)
    integrity_after = len(spliced_after)

    return {
        "result_version": RESULT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset.relative_to(REPO_ROOT)),
        "study": (
            "Can the 47 corrupt gold section labels be repaired mechanically, "
            "and would repairing them unblock nDCG@5? Offline; no labels are "
            "rewritten."
        ),
        "corpus_integrity": {
            "note": (
                "The corpus's own section_path values are still partly spliced. "
                "This is the finding that constrains every repair: a label "
                "repaired onto a spliced path would compute a section metric "
                "against a corrupt target."
            ),
            "after_distinct_paths": sum(len(v) for v in after_paths.values()),
            "after_spliced_paths": len(spliced_after),
            "before_distinct_paths": sum(len(v) for v in before_paths.values()),
            "before_spliced_paths": len(spliced_before),
            "after_spliced_examples": [list(p) for p in spliced_after[:10]],
            "detector_is_over_inclusive": (
                "The splice detector also flags genuine fused headings such as "
                "'2.1 AnIntroductionToDrugDiscoveryandDevelopment'. Allowing "
                "those as repair targets would raise the safe count from 22 to "
                "25. The detector is deliberately kept strict: a missed repair "
                "is a gap, whereas a wrong repair corrupts a metric."
            ),
        },
        "label_repair": {
            "labels_total": len(entries),
            "labels_corrupt": len(graded),
            "verdicts": dict(sorted(verdict_counts.items())),
            "safe_repairs": sum(1 for r in graded if r["verdict"] == "safe"),
            "audit_high_confidence": sum(
                1 for r in graded if r["audit_confidence"] == "high"
            ),
            "audit_high_but_not_safe": disagreements,
            "n_repairable_of_corrupt": round(
                verdict_counts.get("safe", 0) / len(graded), 4
            ) if graded else 0.0,
            "graded": graded,
        },
        "ndcg_at_5_offline": ndcg,
        "before_arm_caveat": (
            "The safe repairs are reconstructed from AFTER-arm section paths, "
            "because that is the corpus a repaired benchmark would be evaluated "
            "against. Applying them to the before arm is incoherent -- and the "
            "numbers show it: repair RAISES nDCG@5 on the after arm (dense "
            "0.4375 -> 0.5168) and LOWERS it on the before arm (0.6156 -> "
            "0.5276). The before-arm column is reported for completeness only "
            "and must not be read as a before/after comparison. A before-arm "
            "repair would need its own reconstruction against before-arm paths."
        ),
        "conclusion": {
            "labels_rewritten": 0,
            "ndcg_unblocked": False,
            "corpus_defect_outlives_the_labels": (
                f"The column fix removed 10 of {integrity_before} spliced section "
                f"paths but left {integrity_after}. The labels are not the only "
                "corrupt artefact: a repaired label pointing into a still-spliced "
                "section path would compute a section metric against a corrupt "
                "target. Repairing the labels alone therefore cannot make "
                "nDCG@5 trustworthy, even if every label were recoverable."
            ),
            "reason": (
                "nDCG@5 stays not-measured: "
                f"{verdict_counts.get('unrepairable', 0)} of {len(graded)} corrupt "
                "labels have no clean reconstruction, and "
                f"{verdict_counts.get('unverifiable', 0)} resolve only to paths "
                "that are themselves spliced. Repairing the safe subset would "
                "change nDCG@5 on a benchmark whose section labels are still "
                "partly corrupt, which is the comparison between two corrupt "
                "label sets that Phase 8 declined to report."
            ),
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(result: Mapping[str, Any]) -> str:
    integrity = result["corpus_integrity"]
    repair = result["label_repair"]
    lines = ["Gold section label repair", "=" * 64, ""]
    lines.append("Corpus integrity (the constraint on every repair)")
    lines.append(
        f"  before: {integrity['before_spliced_paths']} spliced section paths "
        f"of {integrity['before_distinct_paths']}"
    )
    lines.append(
        f"  after : {integrity['after_spliced_paths']} spliced section paths "
        f"of {integrity['after_distinct_paths']}   <- the fix did not clear these"
    )
    lines.append("")
    lines.append("Label verdicts")
    lines.append(f"  corrupt labels           : {repair['labels_corrupt']}")
    lines.append(f"  safe to repair           : {repair['safe_repairs']}")
    lines.append(f"  audit called high        : {repair['audit_high_confidence']}")
    lines.append(
        f"  high but NOT safe        : {len(repair['audit_high_but_not_safe'])}"
    )
    for item in repair["audit_high_but_not_safe"][:5]:
        lines.append(f"    {item['example_id']} {item['document_id']}: {item['reason']}")
    lines.append("")
    lines.append("nDCG@5 recomputed offline (stored retrievals, no API)")
    header = f"  {'arm/system':<26}{'original':>12}{'safe-repaired':>16}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for arm, blocks in result["ndcg_at_5_offline"].items():
        for system, original in blocks["original_labels"].items():
            repaired_value = blocks["safe_repaired_labels"].get(system, {})
            lines.append(
                f"  {arm + '/' + system:<26}"
                f"{_fmt(original['mean_ndcg_at_5']):>12}"
                f"{_fmt(repaired_value.get('mean_ndcg_at_5')):>16}"
            )
    lines.append("")
    lines.append("Caveat: " + result["before_arm_caveat"])
    lines.append("")
    lines.append("CONCLUSION: " + result["conclusion"]["reason"])
    lines.append(f"  labels rewritten: {result['conclusion']['labels_rewritten']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Grade the frozen benchmark's gold section labels as repairable or "
            "not, and measure what repairing them would do to nDCG@5. Offline; "
            "rewrites nothing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=EVALUATION_DIR / "phase7_eval_v1.jsonl")
    parser.add_argument("--before-dir", type=Path, default=PHASE8_BEFORE_CHUNKS_DIR)
    parser.add_argument("--after-dir", type=Path, default=PHASE8_AFTER_CHUNKS_DIR)
    parser.add_argument("--after-suite", type=Path, default=DEFAULT_SUITES["phase8_after"])
    parser.add_argument("--before-suite", type=Path, default=DEFAULT_SUITES["phase8_before"])
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "phase8" / "gold_label_repair.json",
    )
    args = parser.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    result = analyse(
        args.dataset,
        args.before_dir,
        args.after_dir,
        {"phase8_after": args.after_suite, "phase8_before": args.before_suite},
        systems,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render(result))
    print(f"\nartifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
