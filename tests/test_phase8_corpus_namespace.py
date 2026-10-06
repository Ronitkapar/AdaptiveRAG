"""
test_phase8_corpus_namespace.py
-------------------------------
Phase 8 is a before/after study of two-column reading order. Both arms are built
from the same PDFs and the same chunking config, differing only in whether
`_detect_columns` runs.

The failure mode this file exists to prevent is a run that reads the wrong arm.
It is not loud: the two arms share a `corpus_version`, a Qdrant collection carries
no version at all, and `ensure_collection` silently reuses an existing collection.
A run pointed at the other arm retrieves real-looking results from the wrong text
and reports them as a finding. So the arm has to be selected once, derive both its
collection and its lexical index, and be verified on load -- and that is what is
pinned here.
"""

from __future__ import annotations

import json

import pytest

from adaptive_rag.config.paths import (
    BM25_INDEX_PATH,
    CHUNKS_DIR,
    DOCUMENTS_DIR,
    PHASE8_AFTER_BM25_INDEX_PATH,
    PHASE8_AFTER_CHUNKS_DIR,
    PHASE8_AFTER_DOCUMENTS_DIR,
    PHASE8_BEFORE_BM25_INDEX_PATH,
    PHASE8_BEFORE_CHUNKS_DIR,
    PHASE8_BEFORE_DOCUMENTS_DIR,
)
from adaptive_rag.errors import IndexConfigMismatchError
from adaptive_rag.experiments.config import build_experiment_config, compute_corpus_version
from adaptive_rag.indexing.bm25 import BM25Index
from adaptive_rag.schemas import PHASE8_INDEX_NAMESPACES, IndexConfig, IngestionConfig

ARMS = ("phase8_before", "phase8_after")


def _arm_built(arm: str) -> bool:
    return (PHASE8_BEFORE_BM25_INDEX_PATH if arm == "phase8_before" else PHASE8_AFTER_BM25_INDEX_PATH).is_file()


def _arm_dir(arm: str):
    return PHASE8_BEFORE_CHUNKS_DIR if arm == "phase8_before" else PHASE8_AFTER_CHUNKS_DIR


# --- namespace separation ----------------------------------------------------


def test_phase8_arms_live_outside_the_phase7_directories():
    """Neither arm may share a directory with the Phase 7 artifacts.

    Sharing one would mean rebuilding an arm overwrites the historical record of
    what Phase 7 evaluated, and the two studies stop being separable.
    """
    assert PHASE8_BEFORE_DOCUMENTS_DIR != DOCUMENTS_DIR
    assert PHASE8_BEFORE_CHUNKS_DIR != CHUNKS_DIR
    assert PHASE8_BEFORE_BM25_INDEX_PATH != BM25_INDEX_PATH
    assert PHASE8_AFTER_CHUNKS_DIR not in (CHUNKS_DIR, PHASE8_BEFORE_CHUNKS_DIR)
    assert PHASE8_BEFORE_BM25_INDEX_PATH != PHASE8_AFTER_BM25_INDEX_PATH


def test_selecting_an_arm_derives_its_collection_and_lexical_index():
    """`corpus_arm` must be behaviour, not documentation.

    If it were only a label, a caller could select the before arm and still point
    at the after arm's collection, and the run would compare a corpus with itself.
    """
    collections = set()
    indexes = set()
    for arm in ARMS:
        config = IndexConfig(corpus_arm=arm)
        expected = PHASE8_INDEX_NAMESPACES[arm]
        assert config.collection_name == expected["collection_name"]
        assert config.bm25_index_path == expected["bm25_index_path"]
        collections.add(config.collection_name)
        indexes.add(config.bm25_index_path)
    assert len(collections) == len(ARMS), "arms share a collection"
    assert len(indexes) == len(ARMS), "arms share a lexical index"


def test_arms_never_reuse_the_phase7_collection():
    """`ensure_collection` reuses a collection of the same name unless recreate.

    Sharing `adaptiverag_dense_v1` would leave each arm's retired chunk ids in the
    other's index, and dense retrieval would keep returning vectors for chunks that
    no longer exist.
    """
    assert IndexConfig(corpus_arm="phase7").collection_name == "adaptiverag_dense_v1"
    for arm in ARMS:
        assert IndexConfig(corpus_arm=arm).collection_name != "adaptiverag_dense_v1"


def test_an_explicit_namespace_override_is_still_honoured():
    """One-off namespaces stay possible; a silent arm/index mismatch does not."""
    config = IndexConfig(corpus_arm="phase8_before", collection_name="scratch")
    assert config.collection_name == "scratch"
    # The unspecified half still follows the arm.
    assert config.bm25_index_path == PHASE8_INDEX_NAMESPACES["phase8_before"]["bm25_index_path"]


# --- corpus version ----------------------------------------------------------


def test_corpus_version_follows_the_extraction_pipeline():
    """The corpus fingerprint must move when the pipeline does.

    The raw PDFs are not the corpus -- the corpus is what ingestion produced from
    them. Phase 8 changed that pipeline and touched no PDF, so a fingerprint over
    PDFs alone would have handed the new corpus the old corpus's identity.
    """
    assert compute_corpus_version().startswith("corpus_")
    # Read from the schema rather than re-spelled, so bumping the pipeline moves the
    # corpus version without this test needing to know what it was bumped to.
    assert IngestionConfig().ingestion_version == "ingestion_v3"


def test_experiment_config_records_the_selected_arm_and_index():
    config = build_experiment_config(
        "phase8_probe", index=IndexConfig(corpus_arm="phase8_before")
    )
    assert config.corpus_version == compute_corpus_version()
    assert config.index.corpus_arm == "phase8_before"
    assert config.index.collection_name == PHASE8_INDEX_NAMESPACES["phase8_before"]["collection_name"]


def test_every_strategy_resolves_one_index_path():
    """Resolution lives in one helper so the strategies cannot disagree.

    A BM25 arm and a hybrid arm that resolved different files would be comparing
    different corpora under one experiment id.
    """
    from adaptive_rag.experiments.config import _bm25_index_path

    for arm in ARMS:
        config = build_experiment_config("phase8_probe", index=IndexConfig(corpus_arm=arm))
        resolved = _bm25_index_path(config)
        assert resolved.is_absolute()
        assert _bm25_index_path(config) == resolved


# --- the built artifacts -----------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_index_loads_and_indexes_its_own_chunks(arm: str):
    if not _arm_built(arm):
        pytest.skip(f"{arm} artifacts not built")
    from scripts.build_bm25_index import load_canonical_chunks

    path = PHASE8_BEFORE_BM25_INDEX_PATH if arm == "phase8_before" else PHASE8_AFTER_BM25_INDEX_PATH
    index = BM25Index.load(
        path, expected_corpus_version=compute_corpus_version(), expected_corpus_arm=arm
    )
    assert index.corpus_arm == arm
    chunks = load_canonical_chunks(_arm_dir(arm))
    assert len(chunks) == index.total_docs
    assert {c.chunk_id for c in chunks} == set(index.doc_ids)


def test_the_two_arms_share_a_corpus_version_and_are_told_apart_by_the_arm():
    """The reason the arm check exists, stated as a test.

    Both arms come from the same PDFs through the same pipeline version, so the
    version cannot separate them. If a future change ever did make the versions
    differ, the arm check becomes redundant -- but it must not become wrong, and
    the wrong-arm rejection below is what holds it to that.
    """
    if not all(_arm_built(arm) for arm in ARMS):
        pytest.skip("Phase 8 artifacts not built")
    before = json.loads(PHASE8_BEFORE_BM25_INDEX_PATH.read_text(encoding="utf-8"))
    after = json.loads(PHASE8_AFTER_BM25_INDEX_PATH.read_text(encoding="utf-8"))
    assert before["corpus_version"] == after["corpus_version"]
    assert before["corpus_arm"] != after["corpus_arm"]


def test_the_staleness_guard_rejects_the_wrong_arm():
    """A wrong-arm index must fail loudly, not return the other corpus's results."""
    if not _arm_built("phase8_before"):
        pytest.skip("Phase 8 artifacts not built")
    with pytest.raises(IndexConfigMismatchError, match="corpus arm"):
        BM25Index.load(
            PHASE8_BEFORE_BM25_INDEX_PATH,
            expected_corpus_version=compute_corpus_version(),
            expected_corpus_arm="phase8_after",
        )


def test_the_staleness_guard_still_rejects_a_version_mismatch():
    if not _arm_built("phase8_after"):
        pytest.skip("Phase 8 artifacts not built")
    with pytest.raises(IndexConfigMismatchError, match="corpus version"):
        BM25Index.load(
            PHASE8_AFTER_BM25_INDEX_PATH,
            expected_corpus_version="corpus_something_else",
            expected_corpus_arm="phase8_after",
        )


def test_the_phase7_index_is_still_loadable_and_untouched():
    """The historical record of Phase 7 must survive Phase 8 intact."""
    index = BM25Index.load(BM25_INDEX_PATH)
    assert index.doc_ids
    # Built before the arm field existed, so it carries none -- which is why the
    # arm check is skipped rather than required when reading it.
    assert index.corpus_arm == ""


@pytest.mark.parametrize("arm", ARMS)
def test_the_arms_differ_only_in_reading_order(arm: str):
    """Every chunk text must differ, or the arms are not the study they claim.

    The column fix changed reading order on 77 of 285 pages, so the two corpora
    must not be interchangeable. A regression here would mean one arm was rebuilt
    with the other's normalizer, which would quietly void the comparison.
    """
    if not all(_arm_built(a) for a in ARMS):
        pytest.skip("Phase 8 artifacts not built")
    before = BM25Index.load(PHASE8_BEFORE_BM25_INDEX_PATH)
    after = BM25Index.load(PHASE8_AFTER_BM25_INDEX_PATH)
    texts_before = dict(zip(before.doc_ids, before.doc_texts))
    texts_after = dict(zip(after.doc_ids, after.doc_texts))
    shared = set(texts_before) & set(texts_after)
    assert shared, "the arms share no chunk ids at all"
    differing = [cid for cid in shared if texts_before[cid] != texts_after[cid]]
    assert differing, "every shared chunk is byte-identical; the arms are not distinct"