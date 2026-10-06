"""
tests.test_phase8_gold_label_repair
-------------------------------------
The gold-label repair driver (`scripts/repair_phase8_gold_labels.py`).

Two behaviours are pinned here, both because getting them wrong is quiet.

*The nDCG reproduction.* The script recomputes nDCG@5 offline from the stored
`retrieved_chunk_ids` rather than re-running the evaluator, so its arithmetic has
to be shown to agree with the evaluator's. An earlier version scoped relevance to
the whole corpus instead of to the retrieved set; because the IDCG is built from
`len(relevant)`, that inflated the denominator and reproduced only 85 of 107
stored values. Nothing about the output looked wrong. `test_ndcg_matches_stored`
is the check that catches it.

*The spliced-candidate exclusion.* The Step 4 audit graded a repair `high` when
the label's leading text matched a real path prefix, including prefixes that are
themselves spliced. That admits repairs pointing at corrupt targets. This script
excludes them, so `test_repair_refuses_a_spliced_candidate` pins the difference.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repair_phase8_gold_labels", REPO_ROOT / "scripts" / "repair_phase8_gold_labels.py"
)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


# ---------------------------------------------------------------------------
# The splice detector
# ---------------------------------------------------------------------------


def test_is_spliced_flags_a_long_space_free_token():
    assert repair.is_spliced(["3.4 Offline Indexing: Computing & Storing computation)andreduceacrossqueryterms"])
    assert repair.is_spliced(["2 BM25)thatrelyoninvertedindexes.WhileBOWmodelsremain"])


def test_is_spliced_accepts_an_ordinary_heading():
    """Word count does not diagnose splicing; only an unbroken run does.

    Several genuine headings in this corpus are long strings, and an earlier
    length-based check flagged 57 labels of which most were valid.
    """
    assert not repair.is_spliced(["4 EXPERIMENTAL EVALUATION", "4.1 Methodology"])
    assert not repair.is_spliced(["3.4 Offline Indexing: Computing & Storing"])


def test_is_spliced_is_known_to_over_flag_fused_headings():
    """Documented limitation, pinned so a change to it is deliberate.

    The corpus contains genuinely fused headings that trip the same test as a
    spliced label. The detector stays strict on purpose -- a missed repair is a
    gap, a wrong repair corrupts a metric -- and the measured cost is 3 labels
    (22 safe, or 25 under the looser rule).
    """
    fused_but_real = "2.1 AnIntroductionToDrugDiscoveryandDevelopment"
    assert repair.is_spliced([fused_but_real])


# ---------------------------------------------------------------------------
# Repair proposals
# ---------------------------------------------------------------------------


def test_repair_recovers_a_heading_that_is_a_literal_prefix_of_the_splice():
    label = ("1. RECIPROCAL RANK FUSION shown in table 1, indicated that k = 60 was near-optimal,",)
    candidates = {("1. RECIPROCAL RANK FUSION",), ("2. DISCUSSION",)}
    proposal, confidence = repair.propose_repair(label, candidates)
    assert proposal == ("1. RECIPROCAL RANK FUSION",)
    assert confidence == "high"


def test_repair_refuses_a_spliced_candidate():
    """The Step 4 audit would have accepted this; a corrupt target must not pass."""
    label = ("3.2 AbstractiveQuestionAnswering",)
    candidates = {("3.2 AbstractiveQuestionAnswering",)}  # itself spliced
    proposal, _ = repair.propose_repair(label, candidates)
    assert proposal is None


def test_repair_accepts_a_shorter_exact_prefix():
    """`relevant_chunk_ids` matches on `path[:len(prefix)]`, so a shorter
    exact prefix resolves the same chunks and is a valid repair."""
    label = ("3.4 Offline Indexing: Computing & Storing computation)andreduceacrossqueryterms",)
    candidates = {("3.4 Offline Indexing: Computing & Storing", "3.4.1 Inverted Index")}
    proposal, confidence = repair.propose_repair(label, candidates)
    assert proposal == ("3.4 Offline Indexing: Computing & Storing",)
    assert confidence == "high"


def test_repair_gives_up_rather_than_guessing():
    label = ("0 LLMs,haveemergedasapromisingapproach",)
    proposal, confidence = repair.propose_repair(label, set())
    assert proposal is None
    assert confidence == "none"


def test_classify_marks_a_clean_reconstruction_safe():
    entry = {
        "example_id": "q1",
        "document_id": "doc1",
        "section_path_prefix": ["1. RECIPROCAL RANK FUSION shown in table 1,"],
        "confidence": "high",
        "proposed_prefix": ["1. RECIPROCAL RANK FUSION"],
    }
    record = repair.classify(entry, {"doc1": {("1. RECIPROCAL RANK FUSION",)}})
    assert record["verdict"] == "safe"
    assert record["repaired_prefix"] == ["1. RECIPROCAL RANK FUSION"]


def test_classify_marks_an_unreconstructable_label_unrepairable():
    entry = {
        "example_id": "q2",
        "document_id": "doc1",
        "section_path_prefix": ["1 KEYWORDS", "3.1 SPLADE"],
        "confidence": "high",
        "proposed_prefix": None,
    }
    record = repair.classify(entry, {"doc1": {("1 KEYWORDS",)}})
    assert record["verdict"] == "unrepairable"
    assert record["repaired_prefix"] is None


# ---------------------------------------------------------------------------
# nDCG@5 arithmetic
# ---------------------------------------------------------------------------


def test_ndcg_is_one_when_the_only_hit_is_first():
    assert repair.ndcg_at_k(["a", "b", "c"], {"a"}) == 1.0


def test_ndcg_is_zero_when_nothing_retrieved_is_relevant():
    assert repair.ndcg_at_k(["a", "b"], {"z"}) == 0.0


def test_ndcg_matches_the_evaluator_formula():
    """DCG over binary gains, IDCG from `min(len(relevant), k)`."""
    retrieved = ["a", "b", "c", "d", "e", "f"]
    relevant = {"b", "d"}
    gains = [1.0 if c in relevant else 0.0 for c in retrieved[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), 5)))
    assert repair.ndcg_at_k(retrieved, relevant) == pytest.approx(dcg / idcg)


def test_relevance_is_scoped_to_retrieved_chunks():
    """An unretrieved relevant chunk must not enter the IDCG.

    This is the bug the stored-value test caught: computing relevance over the
    whole corpus inflates `len(relevant)`, and every score comes out too low.
    """
    example = {
        "relevant_documents": ["doc1"],
        "relevant_sections": [
            {"document_id": "doc1", "section_path_prefix": ["3 Experiments"]}
        ],
    }
    chunk_index = {
        "doc1::structure_aware_v1::c1": ("3 Experiments", "3.1 Setup"),
        "doc1::structure_aware_v1::c2": ("3 Experiments", "3.2 Results"),
        "doc1::structure_aware_v1::c3": ("2 Methods",),
    }
    retrieved = ["doc1::structure_aware_v1::c1"]
    relevant = repair.relevant_chunks_for(example, retrieved, {}, chunk_index)
    # Only c1 was retrieved and it matches; c2 and c3 must not be counted.
    assert relevant == {"doc1::structure_aware_v1::c1"}


def test_relevance_falls_back_to_document_level_without_section_labels():
    example = {
        "relevant_documents": ["doc1"],
        "relevant_sections": [],
    }
    chunk_index = {
        "doc1::structure_aware_v1::c1": ("9 Appendix",),
        "other::structure_aware_v1::c2": ("1 Intro",),
    }
    relevant = repair.relevant_chunks_for(
        example, ["doc1::structure_aware_v1::c1", "other::structure_aware_v1::c2"], {}, chunk_index
    )
    assert relevant == {"doc1::structure_aware_v1::c1"}


def test_an_unverified_repair_leaves_the_original_label_in_place():
    """Only `safe` repairs are substituted; anything else keeps the old label."""
    example = {
        "relevant_documents": ["doc1"],
        "relevant_sections": [
            {"document_id": "doc1", "section_path_prefix": ["SPLICED LABEL"]}
        ],
    }
    chunk_index = {"doc1::structure_aware_v1::c1": ("3 Experiments",)}
    repaired = {("doc1", ("SPLICED LABEL",)): None}
    relevant = repair.relevant_chunks_for(
        example, ["doc1::structure_aware_v1::c1"], repaired, chunk_index
    )
    # The original (spliced) label does not match, so nothing is relevant.
    assert relevant == set()


# ---------------------------------------------------------------------------
# Against the real, stored values
# ---------------------------------------------------------------------------

_SUITES = {
    "after": "experiments/phase8/combined/p8a_e1",
    "before": "experiments/phase8/combined/p8b_e1",
}
_DATASET = REPO_ROOT / "data" / "evaluation" / "phase7_eval_v1.jsonl"


def _load_examples() -> dict[str, Any]:
    return {
        json.loads(line)["example_id"]: json.loads(line)
        for line in _DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


@pytest.mark.parametrize("arm", sorted(_SUITES))
def test_ndcg_matches_stored_values(arm: str):
    """The offline recomputation must equal what the evaluator recorded.

    Run against the real artifacts on both corpus arms, unrepaired. This is the
    only check that would have caught the relevance-scoping bug, because it
    compares against 107 independently-produced numbers rather than against
    the script's own arithmetic.
    """
    suite = REPO_ROOT / _SUITES[arm]
    rows_paths = sorted(suite.glob("*__E1_baseline_comparison__dense/rows.jsonl"))
    if not rows_paths:
        pytest.skip("Phase 8 E1 artifacts are not present")
    rows_path = rows_paths[0]

    chunk_index = repair.chunk_section_index(
        REPO_ROOT / "data" / f"processed_phase8_{arm}" / "chunks"
    )
    examples = _load_examples()
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    mismatches = []
    for row in rows:
        retrieved = [str(c) for c in row.get("retrieved_chunk_ids") or []]
        relevant = repair.relevant_chunks_for(
            examples[row["query_id"]], retrieved, {}, chunk_index
        )
        mine = repair.ndcg_at_k(retrieved, relevant)
        if abs(mine - float(row["ndcg_at_5"])) > 1e-9:
            mismatches.append((row["query_id"], mine, row["ndcg_at_5"]))

    assert not mismatches, (
        f"{len(mismatches)} of {len(rows)} stored ndcg_at_5 values did not "
        f"reproduce on the {arm} corpus; first: {mismatches[:3]}"
    )
