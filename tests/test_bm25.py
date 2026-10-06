"""
test_bm25.py
------------
Unit and integration tests for BM25Index and BM25Retriever.
Verifies tokenization, scoring monotonicity, index serialization,
protocol compliance, rank assignment, empty query handling, and provenance preservation.
"""

from pathlib import Path
import pytest

from adaptive_rag.errors import IndexConfigMismatchError, IndexUnavailableError, InvalidQueryError
from adaptive_rag.indexing.bm25 import BM25Index, tokenize_text
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.retrieval.bm25 import BM25Retriever
from adaptive_rag.schemas import (
    BM25RetrievalConfig,
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingMetadata,
    RetrievalConfig,
)


def _make_test_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    ordinal: int = 0,
    page: int = 1,
    sha256: str = "hash123",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=doc_id,
            doc_title=f"Doc {doc_id}",
            section_path=["1. Section"],
            headings=["Section"],
            element_ids=[f"e{ordinal}"],
            element_types=["paragraph"],
            page_start=page,
            page_end=page,
            token_count=len(text.split()),
            char_count=len(text),
        ),
        provenance=ChunkProvenance(document_id=doc_id, pages=[page], source_sha256=sha256),
        chunking_metadata=ChunkingMetadata(
            chunking_version="structure_aware_v1",
            config_hash="cfg",
            ordinal=ordinal,
        ),
    )


@pytest.fixture
def sample_chunks():
    return [
        _make_test_chunk("c1", "doc1", "The Okapi BM25 retrieval model is a bag-of-words retrieval function.", 0),
        _make_test_chunk("c2", "doc1", "Dense retrieval uses neural representations and inner product search.", 1),
        _make_test_chunk("c3", "doc2", "BM25 balances term frequency and document length normalization.", 2),
        _make_test_chunk("c4", "doc2", "Information retrieval evaluation commonly measures Recall and MRR.", 3),
    ]


def test_tokenize_text_normalization():
    assert tokenize_text("BM25 & Okapi: 2009-model!") == ["bm25", "okapi", "2009", "model"]
    assert tokenize_text("CAFÉ & naïve résumé") == ["cafe", "naive", "resume"]
    assert tokenize_text("") == []


def test_bm25_index_build_and_search(sample_chunks):
    index = BM25Index(k1=1.2, b=0.75, corpus_version="test_corp")
    n_docs = index.build_from_chunks(sample_chunks, corpus_version="test_corp")
    assert n_docs == 4
    assert index.total_docs == 4
    assert index.avgdl > 0

    results = index.search("okapi bm25", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id in ("c1", "c3")
    assert results[0].score >= results[1].score
    assert all(r.score > 0 for r in results)


def test_bm25_retriever_satisfies_retriever_protocol(sample_chunks):
    index = BM25Index()
    index.build_from_chunks(sample_chunks, corpus_version="test_corp")
    retriever = BM25Retriever(index=index, corpus_version="test_corp")

    assert isinstance(retriever, Retriever)
    assert retriever.method == "bm25"


def test_bm25_retriever_rank_and_provenance(sample_chunks):
    index = BM25Index()
    index.build_from_chunks(sample_chunks, corpus_version="test_corp")
    retriever = BM25Retriever(index=index, corpus_version="test_corp")

    resp = retriever.retrieve("neural representations dense", top_k=3)
    assert resp.status == "ok"
    assert resp.retrieval_method == "bm25"
    assert len(resp.results) > 0
    assert resp.results[0].chunk_id == "c2"
    assert resp.results[0].rank == 1
    assert resp.results[0].provenance.source_sha256 == "hash123"
    assert resp.results[0].metadata.document_id == "doc1"
    assert resp.retrieval_metadata.retriever_version == "bm25_v1"
    assert resp.retrieval_metadata.top_k == 3
    assert resp.retrieval_metadata.latency_ms > 0
    assert resp.retrieval_metadata.k1 == 1.2
    assert resp.retrieval_metadata.b == 0.75


def test_empty_query_raises_invalid_query_error(sample_chunks):
    index = BM25Index()
    index.build_from_chunks(sample_chunks)
    retriever = BM25Retriever(index=index)

    with pytest.raises(InvalidQueryError):
        retriever.retrieve("   ")


def test_zero_match_query_returns_no_results(sample_chunks):
    index = BM25Index()
    index.build_from_chunks(sample_chunks)
    retriever = BM25Retriever(index=index)

    resp = retriever.retrieve("xylophone quantum astrophysics", top_k=3)
    assert resp.status == "no_results"
    assert len(resp.results) == 0


def test_bm25_index_persistence_roundtrip(tmp_path: Path, sample_chunks):
    index_path = tmp_path / "bm25_index.json"
    index = BM25Index(k1=1.5, b=0.8, corpus_version="corp_v1")
    index.build_from_chunks(sample_chunks, corpus_version="corp_v1")
    index.save(index_path)

    assert index_path.is_file()

    loaded = BM25Index.load(index_path, expected_corpus_version="corp_v1")
    assert loaded.total_docs == 4
    assert loaded.k1 == 1.5
    assert loaded.b == 0.8
    assert loaded.corpus_version == "corp_v1"

    # Mismatching corpus version raises error
    with pytest.raises(IndexConfigMismatchError):
        BM25Index.load(index_path, expected_corpus_version="corp_v2")

    # Missing file raises IndexUnavailableError
    with pytest.raises(IndexUnavailableError):
        BM25Index.load(tmp_path / "non_existent.json")


def test_bm25_config_validation():
    # Setting retrieval_method='bm25' auto-aligns retriever_version to 'bm25_v1'
    cfg = RetrievalConfig(retrieval_method="bm25")
    assert cfg.retriever_version == "bm25_v1"
    assert cfg.retrieval_method == "bm25"

    bm25_cfg = BM25RetrievalConfig(k1=1.4, b=0.7)
    assert bm25_cfg.retriever_version == "bm25_v1"
    assert bm25_cfg.k1 == 1.4
    assert bm25_cfg.b == 0.7


def test_term_frequency_monotonicity():
    """Higher term frequency must yield a higher score, all else equal."""
    chunks = [
        _make_test_chunk("low", "doc1", "retrieval appears once here"),
        _make_test_chunk("high", "doc2", "retrieval appears retrieval appears retrieval appears"),
    ]
    index = BM25Index()
    index.build_from_chunks(chunks)
    results = {r.chunk_id: r.score for r in index.search("retrieval", top_k=2)}
    assert results["high"] > results["low"]


def test_document_length_normalization_b():
    """With b > 0 a shorter matching doc outranks an identical-term longer doc;
    with b = 0 length normalization is disabled so term frequency dominates."""
    short_terms = "bm25"
    long_terms = "bm25 " + "padding word " * 30
    chunks = [
        _make_test_chunk("short", "doc1", short_terms),
        _make_test_chunk("long", "doc2", long_terms),
    ]
    norm = BM25Index(b=0.75)
    norm.build_from_chunks(chunks)
    res_norm = {r.chunk_id: r.score for r in norm.search("bm25", top_k=2)}
    assert res_norm["short"] > res_norm["long"]

    flat = BM25Index(b=0.0)
    flat.build_from_chunks(chunks)
    res_flat = {r.chunk_id: r.score for r in flat.search("bm25", top_k=2)}
    # Without length normalization both docs have tf=1 and equal length weighting
    assert res_flat["short"] == pytest.approx(res_flat["long"])


def test_filters_are_applied_via_payload_matching(sample_chunks):
    index = BM25Index()
    index.build_from_chunks(sample_chunks)
    retriever = BM25Retriever(index=index)

    resp = retriever.retrieve("retrieval", top_k=10, filters={"document_id": "doc2"})
    assert resp.status == "ok"
    assert all(r.metadata.document_id == "doc2" for r in resp.results)


def test_end_to_end_build_save_load_retrieve(tmp_path: Path, sample_chunks):
    """Integration: build index from chunks, persist, reload, retrieve via BM25Retriever."""
    index_path = tmp_path / "e2e_index.json"
    index = BM25Index(corpus_version="e2e_corp")
    index.build_from_chunks(sample_chunks, corpus_version="e2e_corp")
    index.save(index_path)

    loaded = BM25Index.load(index_path, expected_corpus_version="e2e_corp")
    retriever = BM25Retriever(index=loaded, corpus_version="e2e_corp")
    assert retriever.method == "bm25"

    resp = retriever.retrieve("term frequency document length", top_k=3)
    assert resp.status == "ok"
    assert resp.retrieval_method == "bm25"
    assert resp.results[0].chunk_id == "c3"
    assert resp.results[0].provenance.source_sha256 == "hash123"
    assert resp.query == "term frequency document length"
    assert resp.retrieval_metadata.corpus_version == "e2e_corp"
    assert resp.retrieval_metadata.index_id == "adaptiverag_bm25_v1"


def test_build_experiment_config_stamps_consistent_versions():
    from adaptive_rag.experiments import build_experiment_config

    config = build_experiment_config(
        name="bm25_baseline_v1",
        retrieval=RetrievalConfig(retrieval_method="bm25"),
        corpus_version="corpus_test",
    )
    assert config.retrieval.retrieval_method == "bm25"
    assert config.retrieval.retriever_version == "bm25_v1"
    assert config.component_versions["retrieval"] == "bm25_v1"

    dense_config = build_experiment_config(name="dense_baseline_v1", corpus_version="corpus_test")
    assert dense_config.retrieval.retriever_version == "dense_v1"
    assert dense_config.component_versions["retrieval"] == "dense_v1"


def _canonical_corpus_files():
    from adaptive_rag.config.paths import BM25_INDEX_PATH

    return BM25_INDEX_PATH.is_file()


@pytest.mark.skipif(
    not _canonical_corpus_files(),
    reason="canonical BM25 index not present (gitignored artifact)",
)
def test_regression_lexical_keywords_on_canonical_corpus():
    """Known lexical keywords must surface the expected documents at top ranks.

    Asserted against the **built index**, not against a freshly-indexed copy of
    `data/processed/chunks`. Those two have drifted apart -- the chunk files on
    disk are a later rebuild that chunking the current documents does not
    reproduce, so they no longer describe the corpus that was indexed -- and only
    the index is what BM25 retrieval actually reads. Building a throwaway index
    from the chunk files would have tested an artifact nothing serves while
    looking like it tested the real one.

    Rebuild the index with `scripts/build_bm25_index.py` after any re-chunking;
    `tests/test_phase8_corpus_namespace.py` covers that the index and the corpus
    version agree.
    """
    from adaptive_rag.config.paths import BM25_INDEX_PATH

    index = BM25Index.load(BM25_INDEX_PATH)
    assert index.doc_ids, "canonical BM25 index is empty"

    for keyword, expected_doc in (
        ("Robertson", "bm25_robertson_2009"),
        ("ColBERT", "colbert_khattab_2020"),
        ("BM25", "bm25_robertson_2009"),
    ):
        results = index.search(keyword, top_k=3)
        assert results, f"no results for {keyword}"
        top_docs = [r.metadata.document_id for r in results[:2]]
        assert expected_doc in top_docs, f"{keyword}: expected {expected_doc} in top-2, got {top_docs}"
