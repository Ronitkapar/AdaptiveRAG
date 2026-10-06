"""
test_gold_label_audit.py
------------------------
Phase 8 Step 4 -- the gold-label audit in `scripts/audit_phase8_gold_labels.py`.

The audit's job is to decide whether a section label is *demonstrably* damaged,
and to refuse to guess when it is not. Both halves matter: over-flagging turns a
valid label into a spurious repair, and under-flagging leaves a spliced label in
place where it silently corrupts nDCG@5. The tests below pin the distinction, using
the real splice signatures found in this corpus rather than invented ones.
"""

from __future__ import annotations

import pytest

from scripts.audit_phase8_gold_labels import (
    MAX_PLAUSIBLE_WORD_CHARS,
    is_corrupt,
    propose,
    section_paths,
)


class _Meta:
    def __init__(self, document_id, section_path):
        self.document_id = document_id
        self.section_path = section_path


class _Chunk:
    def __init__(self, document_id, section_path):
        self.metadata = _Meta(document_id, section_path)


CLEAN_PATHS = {
    "paper": {
        ("1 Introduction",),
        ("2 Methods", "2.4 Training"),
        ("2 Methods", "2.5 Decoding"),
        ("3 Experiments",),
        ("1. RECIPROCAL RANK FUSION",),
    }
}


# --- corruption detection ----------------------------------------------------


def test_a_spliced_label_is_corrupt_because_it_resolves_nowhere():
    """The primary evidence: the label matches nothing in the clean corpus."""
    spliced = ("1. RECIPROCAL RANK FUSION shown in table 1, indicated that k = 60 was near-optimal,",)
    assert is_corrupt(spliced, resolves=False)


def test_a_space_free_prose_run_is_corrupt_even_when_it_happens_to_resolve():
    """Glue is corruption on its face, whatever the section tree says.

    This corpus's splices are all lowercase -- `runtimeefficiency`,
    `fordocumentranking` -- so a camelCase detector finds nothing here. What they
    have in common is no space at all, which no heading contains. A label carrying
    one cannot be trusted even if it happens to match a path.
    """
    glued = ("Modelruntimeefficiencyandrobustness",)
    token = glued[0].split()[0]
    assert len(token) > MAX_PLAUSIBLE_WORD_CHARS
    assert is_corrupt(glued, resolves=True)


def test_a_long_but_valid_label_is_not_corrupt():
    """Length alone diagnoses nothing.

    Several real section titles in this corpus are long strings -- a postal address
    picked up as a heading, for one. An earlier version of this check used length
    and word count as evidence and flagged 57 labels of which 5 were perfectly
    valid; the count heuristic had to go.
    """
    long_but_real = ("2 Yahoo! Research, Av. Diagonal 177, Barcelona 08028, Spain",)
    # Long overall, but every token is a plausible word -- which is the distinction
    # the length heuristic was missing.
    assert len(long_but_real[0]) > 55
    assert max(len(t) for t in long_but_real[0].split()) <= MAX_PLAUSIBLE_WORD_CHARS
    assert not is_corrupt(long_but_real, resolves=True)


def test_a_clean_label_resolving_in_the_clean_corpus_is_not_corrupt():
    assert not is_corrupt(("2 Methods", "2.4 Training"), resolves=True)


# --- repair proposals --------------------------------------------------------


def test_a_splice_is_repaired_by_taking_the_heading_it_contains():
    """The reliable signal: a splice appends prose to a heading already complete.

    So the real heading is a literal prefix of the label, and finding it is
    recovery rather than a guess.
    """
    label = ("1. RECIPROCAL RANK FUSION shown in table 1, indicated that k = 60 was near-optimal,",)
    repaired, confidence = propose(label, CLEAN_PATHS["paper"])
    assert repaired == ("1. RECIPROCAL RANK FUSION",)
    assert confidence == "high"


def test_the_longest_matching_heading_wins():
    """`2.4 Training` must not be recovered as `2` when both would prefix-match."""
    label = ("2 Methods", "2.4 Training on the retriever")
    repaired, confidence = propose(label, CLEAN_PATHS["paper"])
    assert repaired == ("2 Methods", "2.4 Training")
    assert confidence == "high"


def test_a_label_whose_components_are_all_real_is_left_alone():
    repaired, confidence = propose(("2 Methods", "2.4 Training"), CLEAN_PATHS["paper"])
    assert repaired == ("2 Methods", "2.4 Training")
    assert confidence == "high"


def test_a_reconstruction_the_corpus_does_not_contain_is_withheld():
    """Offering a path that exists nowhere is worse than offering nothing.

    A proposal that does not resolve would be applied to the benchmark and then
    match no chunk, which reads as "nothing relevant was retrieved" rather than as
    "this label is broken".
    """
    label = ("9 Nonexistent Section", "9.9 Also Nonexistent")
    repaired, _confidence = propose(label, CLEAN_PATHS["paper"])
    assert repaired is None


def test_an_empty_candidate_set_yields_no_proposal():
    assert propose(("1 Introduction",), set()) == (None, "none")


def test_proposal_confidence_is_the_weakest_link():
    """A perfect parent followed by a guessed child is still a guess.

    Otherwise a path could be reported as `high` on the strength of its first
    component and be trusted for all of them.
    """
    # '2 Methods' is an exact prefix match; '2.5 Nothing Like It' is not a prefix
    # of anything and can only be matched by similarity, so the path is a guess.
    label = ("2 Methods", "2.5 Nothing Like This Heading At All")
    _repaired, confidence = propose(label, CLEAN_PATHS["paper"])
    assert confidence in ("medium", "low")


# --- section path collection -------------------------------------------------


def test_section_paths_are_collected_per_document():
    chunks = [
        _Chunk("a", ["1 Intro"]),
        _Chunk("a", ["2 Body"]),
        _Chunk("b", ["1 Only"]),
    ]
    paths = section_paths(chunks)
    assert paths["a"] == {("1 Intro",), ("2 Body",)}
    assert paths["b"] == {("1 Only",)}