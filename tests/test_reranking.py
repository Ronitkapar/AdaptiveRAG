"""
tests.test_reranking
--------------------
Phase 5 tests for second-stage cross-encoder scoring.
Everything here is offline and deterministic: `FakeReranker` replaces the ONNX
backend, and the real retrievers run over in-memory indexes. The single live
test (the real ONNX session) is integration-marked and deselected by default.
"""

from pathlib import Path

import json
import subprocess
import sys
import pytest
from qdrant_client import QdrantClient

from adaptive_rag.errors import InvalidQueryError, RerankingError
from adaptive_rag.evaluation import EfficiencyEvaluator, RetrievalEvaluator, load_evaluation_dataset
from adaptive_rag.experiments import build_experiment_config
from adaptive_rag.experiments.runner import ExperimentRunner
from adaptive_rag.indexing.bm25 import BM25Index
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.reranking.base import Reranker
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.retrieval.bm25 import BM25Retriever
from adaptive_rag.retrieval.dense import DenseRetriever
from adaptive_rag.retrieval.hybrid import HybridRetriever
from adaptive_rag.retrieval.reranked import RerankedRetriever
from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingMetadata,
    IndexConfig,
    RetrievalConfig,
    RetrievalResult,
)
from tests.fakes import FakeEmbeddingModel, FakeReranker, StubRetriever


# --- helpers ----------------------------------------------------------------


def _result(chunk_id: str, rank: int, score: float = 1.0, text: str | None = None) -> RetrievalResult:
    body = text if text is not None else f"text for {chunk_id}"
    return RetrievalResult(
        chunk_id=chunk_id,
        text=body,
        score=score,
        rank=rank,
        metadata=ChunkMetadata(
            document_id=chunk_id.split("::")[0],
            doc_title=chunk_id.split("::")[0],
            section_path=["1. Section"],
            headings=["Section"],
            element_ids=[f"e-{chunk_id}"],
            element_types=["paragraph"],
            page_start=1,
            page_end=2,
            token_count=len(body.split()),
            char_count=len(body),
        ),
        provenance=ChunkProvenance(
            document_id=chunk_id.split("::")[0], pages=[1, 2], source_sha256="sha123"
        ),
    )


def _ranked(chunk_ids: list[str]) -> list[RetrievalResult]:
    return [
        _result(cid, rank, score=1.0 / rank)
        for rank, cid in enumerate(chunk_ids, start=1)
    ]


def _reranked(base, reranker, **config_kwargs) -> RerankedRetriever:
    method = f"{base.method}_rerank"
    config = RetrievalConfig(retrieval_method=method, **config_kwargs)
    return RerankedRetriever(
        base_retriever=base,
        reranker=reranker,
        config=config,
        corpus_version="test_corp",
    )


# --- 1-3: scoring and ordering ---------------------------------------------


def test_rerank_score_drives_ordering():
    """A reranker that inverts the candidate order produces the inverted output."""
    base = StubRetriever("dense", _ranked(["a", "b", "c", "d"]))
    reranker = FakeReranker(scores=[0.1, 0.9, 0.5, 0.7])
    retriever = _reranked(base, reranker, top_k=4, rerank_candidate_k=4)

    resp = retriever.retrieve("query", top_k=4)

    assert [r.chunk_id for r in resp.results] == ["b", "d", "c", "a"]


def test_score_fields_carry_both_signals():
    base = StubRetriever("dense", _ranked(["a", "b", "c"]))
    base_by_id = {r.chunk_id: r for r in base.results}
    reranker = FakeReranker(scores=[0.1, 0.9, 0.5])
    retriever = _reranked(base, reranker, top_k=3, rerank_candidate_k=3)

    resp = retriever.retrieve("query", top_k=3)

    for result in resp.results:
        original = base_by_id[result.chunk_id]
        assert result.score == pytest.approx(
            {"a": 0.1, "b": 0.9, "c": 0.5}[result.chunk_id]
        )
        assert result.retrieval_score == pytest.approx(original.score)
        assert result.retrieval_rank == original.rank


def test_original_rank_is_preserved_in_retrieval_rank():
    base = StubRetriever("bm25", _ranked(["a", "b", "c", "d", "e"]))
    reranker = FakeReranker(scores=[5.0, 4.0, 3.0, 2.0, 1.0])
    retriever = _reranked(base, reranker, top_k=3, rerank_candidate_k=5)

    resp = retriever.retrieve("query", top_k=3)

    assert [r.rank for r in resp.results] == [1, 2, 3]
    assert [r.retrieval_rank for r in resp.results] == [1, 2, 3]
    # Rank is renumbered after the reorder, so a real inversion must show up.
    inverted = FakeReranker(scores=[1.0, 2.0, 3.0, 4.0, 5.0])
    resp2 = _reranked(base, inverted, top_k=5, rerank_candidate_k=5).retrieve("q", top_k=5)
    assert [r.chunk_id for r in resp2.results] == ["e", "d", "c", "b", "a"]
    assert [r.rank for r in resp2.results] == [1, 2, 3, 4, 5]
    assert [r.retrieval_rank for r in resp2.results] == [5, 4, 3, 2, 1]


# --- 4-5: top_k behaviour ---------------------------------------------------


def test_top_k_truncates_and_records_counts():
    base = StubRetriever("dense", _ranked([f"c{i}" for i in range(20)]))
    reranker = FakeReranker(scores=[float(i) for i in range(20)])
    retriever = _reranked(base, reranker, top_k=5, rerank_candidate_k=20)

    resp = retriever.retrieve("query", top_k=5)

    assert len(resp.results) == 5
    assert [r.rank for r in resp.results] == [1, 2, 3, 4, 5]
    meta = resp.retrieval_metadata
    assert meta.candidate_count == 20
    assert meta.result_count == 5
    assert meta.rerank_top_k == 5
    assert meta.rerank_candidate_k == 20
    assert meta.rerank_latency_ms is not None and meta.rerank_latency_ms >= 0.0


def test_top_k_greater_than_candidate_count_returns_all():
    base = StubRetriever("dense", _ranked(["a", "b", "c"]))
    reranker = FakeReranker(scores=[0.3, 0.2, 0.1])
    retriever = _reranked(base, reranker, top_k=10, rerank_candidate_k=10)

    resp = retriever.retrieve("query", top_k=10)

    assert [r.chunk_id for r in resp.results] == ["a", "b", "c"]
    assert len(resp.results) == 3
    assert resp.retrieval_metadata.result_count == 3


# --- 6-7: candidate depth ---------------------------------------------------


@pytest.mark.parametrize("candidate_k", [12, 20, 40])
def test_base_is_queried_at_rerank_depth(candidate_k: int):
    """The base retriever must be asked for rerank_candidate_k, never top_k."""
    base = StubRetriever("dense", _ranked([f"c{i}" for i in range(candidate_k)]))
    reranker = FakeReranker(scores=[1.0] * candidate_k)
    retriever = _reranked(
        base, reranker, top_k=10, rerank_candidate_k=candidate_k
    )

    retriever.retrieve("query", top_k=10)

    assert base.call_count == 1
    assert base.calls[0]["top_k"] == candidate_k
    assert base.calls[0]["top_k"] != 10
    assert len(retriever.retrieve("query", top_k=10).results) == 10


def test_hybrid_depth_couples_to_rerank_candidate_k():
    dense = StubRetriever("dense", _ranked([f"d{i}" for i in range(40)]))
    lexical = StubRetriever("bm25", _ranked([f"b{i}" for i in range(40)]))
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=lexical,
        config=RetrievalConfig(retrieval_method="hybrid", top_k=10, candidate_k=40),
        corpus_version="test_corp",
    )
    reranker = FakeReranker(scores=[1.0] * 40)
    retriever = _reranked(hybrid, reranker, top_k=10, rerank_candidate_k=40)

    resp = retriever.retrieve("query", top_k=10)

    assert dense.calls[0]["top_k"] == 40
    assert lexical.calls[0]["top_k"] == 40
    assert hybrid.method == "hybrid"
    assert resp.retrieval_method == "hybrid_rerank"


# --- 8: empty candidates ----------------------------------------------------


def test_empty_candidates_skip_the_model_entirely():
    base = StubRetriever("dense", [])
    reranker = FakeReranker(scores=[1.0])
    retriever = _reranked(base, reranker, top_k=5, rerank_candidate_k=20)

    resp = retriever.retrieve("query", top_k=5)

    assert resp.status == "no_results"
    assert resp.results == []
    assert resp.retrieval_metadata.rerank_latency_ms == 0.0
    assert reranker.call_count == 0
    assert base.call_count == 1


# --- 9-10: determinism ------------------------------------------------------


def test_equal_scores_fall_back_to_retrieval_rank_then_chunk_id():
    ids = ["z", "a", "m"]
    base = StubRetriever("dense", _ranked(ids))
    reranker = FakeReranker(scores=[0.5, 0.5, 0.5])
    retriever = _reranked(base, reranker, top_k=3, rerank_candidate_k=3)

    resp = retriever.retrieve("query", top_k=3)

    # All scores tie, so the pre-rerank order decides; chunk_id only breaks a tie
    # that retrieval_rank cannot, which the next case forces.
    assert [r.chunk_id for r in resp.results] == ["z", "a", "m"]
    assert [r.score for r in resp.results] == [0.5, 0.5, 0.5]

    duplicate_rank = [
        _result("b", 1), _result("a", 1), _result("c", 2),
    ]
    base2 = StubRetriever("dense", duplicate_rank)
    tied = _reranked(
        base2, FakeReranker(scores=[0.5, 0.5, 0.5]), top_k=3, rerank_candidate_k=3
    )
    resp2 = tied.retrieve("query", top_k=3)
    assert [r.chunk_id for r in resp2.results] == ["a", "b", "c"]


def test_repeated_runs_are_byte_identical():
    base = StubRetriever("dense", _ranked([f"c{i}" for i in range(12)]))
    reranker = FakeReranker(scores=[float(i % 4) for i in range(12)])
    retriever = _reranked(base, reranker, top_k=6, rerank_candidate_k=12)

    first = retriever.retrieve("query", top_k=6)
    second = retriever.retrieve("query", top_k=6)

    assert [r.chunk_id for r in first.results] == [r.chunk_id for r in second.results]
    assert [r.score for r in first.results] == [r.score for r in second.results]


def test_batch_size_invariance():
    """Batching cannot change the outcome: scoring is per-pair with no interaction."""
    ids = [f"c{i}" for i in range(17)]

    def run(batch_size: int):
        base = StubRetriever("dense", _ranked(ids))
        reranker = FakeReranker(
            score_fn=lambda p: float(len(p) % 7), version=f"fake_b{batch_size}"
        )
        # batch_size lives on the backend; the protocol itself takes one call.
        reranker.batch_size = batch_size
        retriever = _reranked(base, reranker, top_k=17, rerank_candidate_k=17)
        return retriever.retrieve("query", top_k=17)

    one = run(1)
    many = run(64)

    assert [r.chunk_id for r in one.results] == [r.chunk_id for r in many.results]
    assert [r.score for r in one.results] == [r.score for r in many.results]


# --- 11-13: passthrough fidelity --------------------------------------------


def test_chunk_metadata_is_preserved_verbatim():
    base = StubRetriever("dense", _ranked(["doc1::c1", "doc2::c2"]))
    original = {r.chunk_id: r for r in base.results}
    retriever = _reranked(
        base, FakeReranker(scores=[0.1, 0.9]), top_k=2, rerank_candidate_k=2
    )

    resp = retriever.retrieve("query", top_k=2)

    for result in resp.results:
        assert result.metadata == original[result.chunk_id].metadata
        assert result.metadata.document_id
        assert result.metadata.doc_title
        assert result.metadata.section_path
        assert result.metadata.headings
        assert result.metadata.element_ids
        assert result.metadata.page_start is not None
        assert result.metadata.page_end is not None
        assert result.metadata.token_count >= 0


def test_provenance_and_text_are_untouched():
    base = StubRetriever("dense", _ranked(["doc1::c1", "doc2::c2"]))
    original = {r.chunk_id: r for r in base.results}
    retriever = _reranked(
        base, FakeReranker(scores=[0.1, 0.9]), top_k=2, rerank_candidate_k=2
    )

    resp = retriever.retrieve("query", top_k=2)

    for result in resp.results:
        source = original[result.chunk_id]
        assert result.text == source.text
        assert result.provenance == source.provenance
        assert result.provenance.source_sha256 == "sha123"
        assert result.provenance.pages == [1, 2]


# --- 14: latency accounting -------------------------------------------------


def test_latency_is_the_sum_of_its_two_stages():
    base = StubRetriever("dense", _ranked(["a", "b", "c"]), latency_ms=12.0)
    retriever = _reranked(
        base, FakeReranker(scores=[0.1, 0.2, 0.3]), top_k=3, rerank_candidate_k=3
    )

    meta = retriever.retrieve("query", top_k=3).retrieval_metadata

    assert meta.latency_ms == pytest.approx(
        meta.candidate_generation_latency_ms + meta.rerank_latency_ms
    )
    assert meta.candidate_generation_latency_ms == pytest.approx(12.0)
    assert meta.latency_ms >= 12.0
    assert meta.search_latency_ms == pytest.approx(0.75)


# --- 15-18: failure semantics ----------------------------------------------


def test_reranker_failure_propagates_with_original_type():
    base = StubRetriever("dense", _ranked(["a", "b"]))
    reranker = FakeReranker(raises=RerankingError("model exploded"))
    retriever = _reranked(base, reranker, top_k=2, rerank_candidate_k=2)

    with pytest.raises(RerankingError) as excinfo:
        retriever.retrieve("query", top_k=2)

    assert str(excinfo.value) == "model exploded"


def test_reranker_failure_is_recorded_as_retrieval_failed(tmp_path: Path):
    dataset = load_evaluation_dataset()[:1]
    base = StubRetriever("dense", _ranked(["a", "b"]))
    retriever = _reranked(
        base, FakeReranker(raises=RerankingError("model exploded")),
        top_k=2, rerank_candidate_k=2,
    )
    config = build_experiment_config(
        name="dense_rerank_v1",
        retrieval=retriever.config,
        corpus_version="corpus_test",
    )
    runner = ExperimentRunner(
        retriever=retriever,
        generator=None,
        evaluators=[RetrievalEvaluator(), EfficiencyEvaluator()],
        output_root=tmp_path / "experiments",
    )
    summary = runner.run(config, dataset, run_id="rerank_failure")

    assert summary["manifest"]["status_counts"]["retrieval_failed"] == 1
    assert summary["manifest"]["error_types"] == ["RerankingError"]
    trace = json.loads(
        (tmp_path / "experiments" / "rerank_failure" / "traces.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert trace["retrieval"] is None
    assert trace["error"]["stage"] == "retrieval"
    assert trace["error"]["error_type"] == "RerankingError"


def test_fallback_is_off_by_default():
    base = StubRetriever("dense", _ranked(["a", "b"]))
    retriever = _reranked(
        base,
        FakeReranker(raises=RerankingError("boom")),
        top_k=2,
        rerank_candidate_k=2,
    )
    assert retriever.config.rerank_fallback is False
    with pytest.raises(RerankingError):
        retriever.retrieve("query", top_k=2)


def test_fallback_is_opt_in_and_observable():
    base = StubRetriever("dense", _ranked(["a", "b", "c"]))
    base_by_id = {r.chunk_id: r for r in base.results}
    retriever = _reranked(
        base,
        FakeReranker(raises=RerankingError("boom")),
        top_k=2,
        rerank_candidate_k=3,
        rerank_fallback=True,
    )

    resp = retriever.retrieve("query", top_k=2)

    assert len(resp.results) == 2
    assert [r.chunk_id for r in resp.results] == ["a", "b"]
    for result in resp.results:
        assert result.score == pytest.approx(base_by_id[result.chunk_id].score)
        assert result.retrieval_score == pytest.approx(base_by_id[result.chunk_id].score)
        assert result.retrieval_rank == result.rank
    assert resp.retrieval_metadata.rerank_fallback is True


def test_empty_query_never_reaches_base_or_reranker():
    base = StubRetriever("dense", _ranked(["a"]))
    reranker = FakeReranker(scores=[1.0])
    retriever = _reranked(base, reranker, top_k=1, rerank_candidate_k=5)

    with pytest.raises(InvalidQueryError):
        retriever.retrieve("   ")

    assert base.call_count == 0
    assert reranker.call_count == 0


# --- 19-21: protocol, score contract, configuration -------------------------


def test_reranked_retriever_conforms_to_protocols():
    base = StubRetriever("dense", _ranked(["a"]))
    reranker = FakeReranker(scores=[1.0])
    retriever = _reranked(base, reranker, top_k=1, rerank_candidate_k=1)

    assert isinstance(retriever, Retriever)
    assert isinstance(reranker, Reranker)
    assert retriever.method == "dense_rerank"


@pytest.mark.parametrize(
    "base_method,expected",
    [
        ("dense", "dense_rerank"),
        ("bm25", "bm25_rerank"),
        ("hybrid", "hybrid_rerank"),
    ],
)
def test_method_names_the_whole_pipeline(base_method: str, expected: str):
    base = StubRetriever(base_method, _ranked(["a"]))
    retriever = _reranked(base, FakeReranker(scores=[1.0]), rerank_candidate_k=1)

    assert retriever.method == expected
    assert retriever.retrieve("q").retrieval_method == expected


def test_score_count_mismatch_is_a_typed_error_not_a_truncation():
    base = StubRetriever("dense", _ranked(["a", "b", "c"]))
    # Two scores for three candidates: silently zipping would drop a chunk.
    retriever = _reranked(
        base, FakeReranker(scores=[0.1, 0.2]), top_k=3, rerank_candidate_k=3
    )

    with pytest.raises(RerankingError):
        retriever.retrieve("query", top_k=3)


def test_config_aligns_version_and_validates_depth():
    assert RetrievalConfig(retrieval_method="dense_rerank").retriever_version == (
        "dense_rerank_v1"
    )
    assert RetrievalConfig(retrieval_method="bm25_rerank").retriever_version == (
        "bm25_rerank_v1"
    )
    assert RetrievalConfig(retrieval_method="hybrid_rerank").retriever_version == (
        "hybrid_rerank_v1"
    )

    with pytest.raises(ValueError):
        RetrievalConfig(
            retrieval_method="dense_rerank",
            rerank_enabled=True,
            top_k=10,
            rerank_candidate_k=5,
        )
    with pytest.raises(ValueError):
        RetrievalConfig(
            retrieval_method="hybrid_rerank",
            rerank_enabled=True,
            top_k=10,
            candidate_k=10,
            rerank_candidate_k=20,
        )


def test_config_hash_covers_rerank_settings():
    from adaptive_rag.config.hashing import compute_config_hash

    def digest(**overrides):
        return compute_config_hash(
            RetrievalConfig(
                retrieval_method="dense_rerank", rerank_enabled=True, **overrides
            ).model_dump(mode="json")
        )

    baseline = digest()
    assert digest(rerank_model_id="other/model") != baseline
    assert digest(rerank_candidate_k=40) != baseline
    assert digest(rerank_device="cpu") != baseline


# --- 22-24: offline integration over real retrievers ------------------------


def _chunk(chunk_id: str, doc_id: str, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=doc_id,
            doc_title=doc_id,
            section_path=["1. Section"],
            headings=["Section"],
            element_ids=[f"e{ordinal}"],
            element_types=["paragraph"],
            page_start=1,
            page_end=1,
            token_count=len(text.split()),
            char_count=len(text),
        ),
        provenance=ChunkProvenance(document_id=doc_id, pages=[1], source_sha256="sha123"),
        chunking_metadata=ChunkingMetadata(
            chunking_version="structure_aware_v1", config_hash="cfg", ordinal=ordinal
        ),
    )


CHUNK_TEXTS = [
    "Okapi BM25 is a probabilistic ranking function based on term saturation.",
    "Dense retrieval encodes queries and passages into a shared vector space.",
    "Reciprocal rank fusion merges several ranked lists without score calibration.",
    "Reranking applies a cross-encoder to the candidate list produced by retrieval.",
]


@pytest.fixture
def corpus_chunks() -> list[Chunk]:
    return [
        _chunk(f"c{i}", f"doc{i}", text, i) for i, text in enumerate(CHUNK_TEXTS)
    ]


@pytest.fixture
def dense_retriever(corpus_chunks):
    model = FakeEmbeddingModel(dimension=8)
    store = QdrantVectorStore(
        config=IndexConfig(collection_name="rerank_col"), client=QdrantClient(location=":memory:")
    )
    store.ensure_collection(dim=8, distance="cosine")
    store.upsert(corpus_chunks, [model.embed_query(c.text) for c in corpus_chunks])
    return DenseRetriever(
        embedding_model=model,
        vector_store=store,
        config=RetrievalConfig(top_k=4),
        corpus_version="test_corp",
    )


@pytest.fixture
def bm25_retriever(corpus_chunks):
    index = BM25Index(corpus_version="test_corp")
    index.build_from_chunks(corpus_chunks, corpus_version="test_corp")
    return BM25Retriever(index=index, corpus_version="test_corp")


def _run_through_runner(tmp_path: Path, retriever, run_id: str) -> dict:
    dataset = load_evaluation_dataset()[:3]
    config = build_experiment_config(
        name="dense_rerank_v1",
        retrieval=retriever.config,
        corpus_version="corpus_test",
    )
    runner = ExperimentRunner(
        retriever=retriever,
        generator=None,
        evaluators=[RetrievalEvaluator(), EfficiencyEvaluator()],
        output_root=tmp_path / "experiments",
    )
    summary = runner.run(config, dataset, run_id=run_id)
    metrics = {
        m["name"]: m["value"] for m in summary["metrics"]["efficiency"]["metrics"]
    }
    traces = [
        json.loads(line)
        for line in (
            tmp_path / "experiments" / run_id / "traces.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"summary": summary, "metrics": metrics, "traces": traces}


def test_dense_to_rerank_through_runner(tmp_path: Path, dense_retriever):
    retriever = _reranked(
        dense_retriever,
        FakeReranker(score_fn=lambda p: float(len(p) % 5)),
        top_k=4,
        rerank_candidate_k=4,
    )
    outcome = _run_through_runner(tmp_path, retriever, "dense_rerank")

    assert outcome["summary"]["manifest"]["retrieval_method"] == "dense_rerank"
    assert outcome["summary"]["trace_count"] == 3
    assert outcome["metrics"]["rerank_latency_ms_mean"] is not None
    assert outcome["metrics"]["rerank_candidate_count_mean"] == pytest.approx(4.0)
    assert outcome["metrics"]["retrieval_latency_ms_mean"] is not None
    assert (tmp_path / "experiments" / "dense_rerank" / "metrics_retrieval.json").is_file()
    for trace in outcome["traces"]:
        assert trace["rerank_latency_ms"] is not None
        assert trace["candidate_generation_latency_ms"] is not None
        assert trace["rerank_fallback"] is False


def test_bm25_to_rerank_through_runner(tmp_path: Path, bm25_retriever):
    retriever = _reranked(
        bm25_retriever,
        FakeReranker(score_fn=lambda p: float(p.count("a"))),
        top_k=4,
        rerank_candidate_k=4,
    )
    outcome = _run_through_runner(tmp_path, retriever, "bm25_rerank")

    assert outcome["summary"]["manifest"]["retrieval_method"] == "bm25_rerank"
    assert outcome["metrics"]["rerank_latency_ms_mean"] is not None
    assert outcome["traces"][0]["retrieval"]["retrieval_method"] == "bm25_rerank"


def test_hybrid_to_rerank_preserves_fusion_diagnostics(
    tmp_path: Path, dense_retriever, bm25_retriever
):
    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        config=RetrievalConfig(
            retrieval_method="hybrid", top_k=4, candidate_k=4
        ),
        corpus_version="test_corp",
    )
    retriever = _reranked(
        hybrid,
        FakeReranker(score_fn=lambda p: float(len(p) % 3)),
        top_k=4,
        rerank_candidate_k=4,
    )
    outcome = _run_through_runner(tmp_path, retriever, "hybrid_rerank")

    assert outcome["summary"]["manifest"]["retrieval_method"] == "hybrid_rerank"
    meta = outcome["traces"][0]["retrieval"]["retrieval_metadata"]
    # The wrapper carries the fused base metadata through untouched.
    assert meta["fusion_method"] == "rrf"
    assert meta["rrf_k"] == 60
    assert meta["dense_candidate_count"] is not None
    assert meta["bm25_candidate_count"] is not None
    assert meta["fusion_latency_ms"] is not None
    assert meta["rerank_latency_ms"] is not None
    assert retriever.base_retriever is hybrid
    assert bm25_retriever.index.total_docs == 4


# --- 25: canonical corpus regression ----------------------------------------


def _canonical_corpus_files():
    from adaptive_rag.config.paths import CHUNKS_DIR

    return sorted(CHUNKS_DIR.glob("*.chunks.jsonl"))


@pytest.mark.skipif(
    not _canonical_corpus_files(),
    reason="canonical chunk corpus not present (gitignored artifacts)",
)
def test_regression_reranking_preserves_retrieval_correctness():
    """A reranker reorders; it must never remove a document the first stage found."""
    from adaptive_rag.config.paths import CHUNKS_DIR
    from adaptive_rag.experiments.config import compute_corpus_version
    from scripts.build_bm25_index import load_canonical_chunks

    chunks = load_canonical_chunks(CHUNKS_DIR)
    corpus_ver = compute_corpus_version()
    index = BM25Index(corpus_version=corpus_ver)
    index.build_from_chunks(chunks, corpus_version=corpus_ver)
    lexical = BM25Retriever(index=index, corpus_version=corpus_ver)

    # Score every canonical chunk by whether it shares a token with the query:
    # a real relevance signal, computed without a model.
    def lexical_overlap(query: str) -> float:
        terms = set(query.lower().split())

        def score(text: str) -> float:
            tokens = set(text.lower().split())
            return float(len(terms & tokens))

        return score

    for keyword, expected_doc in (
        ("Robertson", "bm25_robertson_2009"),
        ("ColBERT", "colbert_khattab_2020"),
        ("BM25", "bm25_robertson_2009"),
    ):
        base = lexical.retrieve(keyword, top_k=20)
        base_ids = {r.chunk_id for r in base.results}
        assert expected_doc in {r.metadata.document_id for r in base.results}

        retriever = _reranked(
            lexical,
            FakeReranker(score_fn=lexical_overlap(keyword)),
            top_k=10,
            rerank_candidate_k=20,
        )
        resp = retriever.retrieve(keyword, top_k=10)
        reranked_ids = {r.chunk_id for r in resp.results}

        # A reranker may reorder the pool but must never add or drop a chunk,
        # and the document the first stage found must survive the reorder.
        assert len(resp.results) == 10
        assert reranked_ids <= base_ids
        assert expected_doc in {r.metadata.document_id for r in resp.results}
        assert [r.rank for r in resp.results] == list(range(1, 11))
        # The lexical overlap signal must actually have driven the order.
        overlap_scores = [r.score for r in resp.results]
        assert overlap_scores == sorted(overlap_scores, reverse=True)


# --- 26-28: regression with reranking disabled -----------------------------


def test_instantiate_components_returns_plain_retrievers_when_reranking_is_off(
    monkeypatch, corpus_chunks
):
    """The single construction site must not wrap when the second stage is off."""
    from adaptive_rag.experiments import config as exp_config
    from adaptive_rag.retrieval.bm25 import BM25Retriever as _BM25
    from adaptive_rag.retrieval.dense import DenseRetriever as _Dense
    from adaptive_rag.retrieval.hybrid import HybridRetriever as _Hybrid

    index = BM25Index(corpus_version="corpus_test")
    index.build_from_chunks(corpus_chunks, corpus_version="corpus_test")

    class _Store:
        def __init__(self, config=None, client=None):
            self.collection_name = config.collection_name if config else "col"

        def count(self):
            return len(corpus_chunks)

        def close(self):
            pass

    class _Index:
        @staticmethod
        def load(expected_corpus_version=None):
            return index

    monkeypatch.setattr(exp_config, "BM25Index", _Index)
    monkeypatch.setattr(exp_config, "QdrantVectorStore", _Store)
    monkeypatch.setattr(
        exp_config, "AICreditsEmbeddingModel", lambda config=None: FakeEmbeddingModel()
    )

    cases = [
        ("dense", _Dense),
        ("bm25", _BM25),
        ("hybrid", _Hybrid),
    ]
    for method, expected_type in cases:
        config = exp_config.build_experiment_config(
            name=f"{method}_baseline_v1",
            retrieval=RetrievalConfig(retrieval_method=method),
            corpus_version="corpus_test",
        )
        components = exp_config.instantiate_components(config)
        retriever = components[2]
        assert type(retriever) is expected_type, method
        assert not isinstance(retriever, RerankedRetriever), method
        assert config.component_versions.get("reranker") is None


def test_component_versions_record_the_reranker_only_when_enabled():
    from adaptive_rag.experiments import config as exp_config

    plain = exp_config.build_experiment_config(
        name="dense_baseline_v1",
        retrieval=RetrievalConfig(retrieval_method="dense"),
        corpus_version="corpus_test",
    )
    scored = exp_config.build_experiment_config(
        name="dense_rerank_v1",
        retrieval=RetrievalConfig(
            retrieval_method="dense_rerank", rerank_enabled=True
        ),
        corpus_version="corpus_test",
    )

    assert "reranker" not in plain.component_versions
    assert scored.component_versions["reranker"] == "onnx_cross_encoder_v1"
    assert scored.component_versions["retrieval"] == "dense_rerank_v1"


def test_rerank_disabled_config_is_unchanged_by_new_fields():
    dense = RetrievalConfig(retrieval_method="dense")
    lexical = RetrievalConfig(retrieval_method="bm25")
    fused = RetrievalConfig(retrieval_method="hybrid")

    for config in (dense, lexical, fused):
        assert config.rerank_enabled is False
        assert config.rerank_candidate_k == 20
    assert dense.retriever_version == "dense_v1"
    assert lexical.retriever_version == "bm25_v1"
    assert fused.retriever_version == "hybrid_v1"


@pytest.mark.parametrize("run_name", ["bm25_baseline_v1", "20260929T175539Z-hybrid_baseline_v1"])
def test_existing_run_artifacts_still_validate(run_name: str):
    """Phase 4 artifacts on disk must parse against the extended schemas."""
    from adaptive_rag.config.paths import EXPERIMENTS_DIR
    from adaptive_rag.schemas import ExperimentTrace

    run_dir = EXPERIMENTS_DIR / run_name
    if not (run_dir / "traces.jsonl").is_file():
        pytest.skip(f"{run_name} not present on this machine")

    traces_path = run_dir / "traces.jsonl"
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trace = ExperimentTrace.model_validate_json(line)
                assert trace.candidate_generation_latency_ms is None
                assert trace.rerank_latency_ms is None


def test_efficiency_output_is_unchanged_when_no_trace_carries_rerank_fields(
    tmp_path: Path,
):
    """A non-reranked run's efficiency report must not gain new metric names."""
    dataset = load_evaluation_dataset()[:2]
    base = StubRetriever("dense", _ranked(["a", "b"]))
    config = build_experiment_config(
        name="dense_baseline_v1",
        retrieval=RetrievalConfig(retrieval_method="dense"),
        corpus_version="corpus_test",
    )
    summary = ExperimentRunner(
        retriever=base,
        generator=None,
        evaluators=[EfficiencyEvaluator()],
        output_root=tmp_path / "experiments",
    ).run(config, dataset, run_id="plain")

    names = {m["name"] for m in summary["metrics"]["efficiency"]["metrics"]}
    assert "retrieval_latency_ms_mean" in names
    assert not any(name.startswith("rerank_") for name in names)
    assert "candidate_generation_latency_ms_mean" not in names
    assert "rerank_fallback_count" not in names


def test_efficiency_reports_the_rerank_cost_split(tmp_path: Path, dense_retriever):
    retriever = _reranked(
        dense_retriever,
        FakeReranker(score_fn=lambda p: float(len(p) % 5)),
        top_k=4,
        rerank_candidate_k=4,
    )
    outcome = _run_through_runner(tmp_path, retriever, "split")
    metrics = outcome["metrics"]

    assert metrics["rerank_latency_ms_mean"] is not None
    assert metrics["rerank_latency_ms_p50"] is not None
    assert metrics["rerank_latency_ms_p95"] is not None
    assert metrics["candidate_generation_latency_ms_mean"] is not None
    assert metrics["rerank_candidate_count_mean"] == pytest.approx(4.0)
    assert metrics["rerank_result_count_mean"] == pytest.approx(4.0)
    assert metrics["rerank_fallback_count"] == pytest.approx(0.0)


# --- Comparison CLI ---------------------------------------------------------


def _write_fake_run(
    root: Path,
    name: str,
    method: str,
    recall: float,
    *,
    candidate_k: int | None = None,
    trace_count: int = 20,
    corpus_version: str = "corpus_test",
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "retrieval": {
                    "metrics": [
                        {"name": "recall_at_k", "k": 5, "value": recall, "n": trace_count},
                        {"name": "mrr", "k": None, "value": recall / 2, "n": trace_count},
                    ]
                },
                "efficiency": {
                    "metrics": [
                        {
                            "name": "retrieval_latency_ms_mean",
                            "k": None,
                            "value": 1.0,
                            "n": trace_count,
                        },
                        {
                            "name": "rerank_latency_ms_mean",
                            "k": None,
                            "value": 0.5,
                            "n": trace_count,
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_version": corpus_version,
                "trace_count": trace_count,
                "retrieval_method": method,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps({"retrieval": {"rerank_candidate_k": candidate_k}}), encoding="utf-8"
    )
    return run_dir


def _run_comparison(experiments_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_reranking.py"
    return subprocess.run(
        [sys.executable, str(script), *extra],
        capture_output=True,
        text=True,
        cwd=script.parent.parent,
    )


def test_comparison_reports_base_versus_reranked(tmp_path: Path):
    base = _write_fake_run(tmp_path, "dense_baseline_v1", "dense", 0.60)
    rerank = _write_fake_run(tmp_path, "dense_rerank_v1", "dense_rerank", 0.80)

    proc = _run_comparison(
        tmp_path, "--dense-run", str(base), "--dense-rerank-run", str(rerank)
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dense: base vs reranked" in proc.stdout
    assert "corpus: base=corpus_test rerank=corpus_test MATCH" in proc.stdout
    assert "method: base=dense rerank=dense_rerank OK" in proc.stdout
    assert "recall_at_k@k=5" in proc.stdout
    assert "rerank_latency_ms_mean" in proc.stdout


def test_comparison_rejects_a_method_mismatch(tmp_path: Path):
    base = _write_fake_run(tmp_path, "dense_baseline_v1", "dense", 0.6)
    bad = _write_fake_run(tmp_path, "dense_rerank_v1", "dense", 0.8)

    proc = _run_comparison(
        tmp_path, "--dense-run", str(base), "--dense-rerank-run", str(bad)
    )

    assert proc.returncode == 2
    assert "method: base=dense rerank=dense UNEXPECTED" in proc.stdout


def test_comparison_rejects_a_corpus_mismatch(tmp_path: Path):
    base = _write_fake_run(tmp_path, "dense_baseline_v1", "dense", 0.6)
    rerank = _write_fake_run(
        tmp_path, "dense_rerank_v1", "dense_rerank", 0.8, corpus_version="corpus_other"
    )

    proc = _run_comparison(
        tmp_path, "--dense-run", str(base), "--dense-rerank-run", str(rerank)
    )

    assert proc.returncode == 2
    assert "MISMATCH" in proc.stdout


def test_comparison_rejects_a_missing_baseline(tmp_path: Path):
    rerank = _write_fake_run(tmp_path, "dense_rerank_v1", "dense_rerank", 0.8)

    proc = _run_comparison(tmp_path, "--dense-rerank-run", str(rerank))

    assert proc.returncode == 1


def test_comparison_never_auto_discovers_rerank_runs(tmp_path: Path):
    """A rerank run on disk must not appear without an explicit flag."""
    _write_fake_run(tmp_path, "dense_baseline_v1", "dense", 0.6)
    _write_fake_run(tmp_path, "dense_rerank_v1", "dense_rerank", 0.8)

    proc = _run_comparison(tmp_path)

    assert proc.returncode == 1
    assert "No reranked run supplied" in proc.stdout + proc.stderr


def test_comparison_depth_ablation_table(tmp_path: Path):
    base = _write_fake_run(tmp_path, "hybrid_baseline_v1", "hybrid", 0.7)
    k10 = _write_fake_run(
        tmp_path, "hybrid_rerank_k10_v1", "hybrid_rerank", 0.75, candidate_k=10
    )
    k40 = _write_fake_run(
        tmp_path, "hybrid_rerank_k40_v1", "hybrid_rerank", 0.85, candidate_k=40
    )

    proc = _run_comparison(
        tmp_path,
        "--hybrid-run",
        str(base),
        "--hybrid-rerank-run",
        str(k40),
        "--ablation-run",
        str(k10),
        "--ablation-run",
        str(k40),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Depth ablation (final top_k held constant):" in proc.stdout
    assert "k=10" in proc.stdout and "k=40" in proc.stdout
    assert "rerank_candidate_count_mean" in proc.stdout


def test_comparison_leaves_the_phase_four_script_frozen():
    """compare_retrievers.py must stay untouched by the Phase 5 tooling."""
    compare_retrievers = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_retrievers.py"
    )
    text = compare_retrievers.read_text(encoding="utf-8")
    assert "rerank" not in text.lower()
    assert f"{'metric':<22}{'dense':>12}{'bm25':>12}{'delta':>12}" not in text
    # The Phase 4 two-run table is produced by an f-string with no spaces in the
    # source, so assert the literal format fields instead.
    assert "{'metric':<22}{'dense':>12}{'bm25':>12}{'delta':>12}" in text


# --- 35: live ONNX backend ---------------------------------------------------


@pytest.mark.integration
def test_onnx_cross_encoder_backend_runs(tmp_path: Path):
    """Download the pinned cross-encoder, load the session, and score a batch."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    from adaptive_rag.reranking import OnnxCrossEncoderReranker

    passages = [
        "BM25 is a bag-of-words ranking function.",
        "Cross-encoders score a query and passage jointly.",
        "Dense retrieval embeds text into a vector space.",
        "Reciprocal rank fusion merges ranked lists.",
    ]

    def build(batch_size: int) -> OnnxCrossEncoderReranker:
        return OnnxCrossEncoderReranker(
            model_revision="main",
            device="cpu",
            batch_size=batch_size,
            model_dir=tmp_path,
        )

    scorer = build(4)
    assert not scorer.is_loaded, "loading must be deferred to first use"
    scores = scorer.score("what does BM25 do?", passages)

    assert len(scores) == len(passages)
    assert all(s == s for s in scores), "no NaN logits"
    assert scorer.is_loaded
    assert scorer.device == "cpu"
    assert scorer.model_id == "Xenova/ms-marco-MiniLM-L-6-v2"

    # Batching must not perturb the scores.
    one_at_a_time = build(1).score("what does BM25 do?", passages)
    assert one_at_a_time == pytest.approx(scores, abs=1e-5)

    # The BM25 passage must outrank the unrelated fusion passage.
    # NOTE: this model's head is a single raw regression logit whose scale is
    # entirely negative for non-matches, so the comparison is between two scored
    # passages. Asserting against 0.0 (or against scores[0] * 0) would compare a
    # score with itself plus a zero baseline and never actually rank anything.
    bm25_score = scores[passages.index("BM25 is a bag-of-words ranking function.")]
    fusion_score = scores[passages.index("Reciprocal rank fusion merges ranked lists.")]
    assert bm25_score > fusion_score


@pytest.mark.integration
def test_onnx_backend_rejects_a_missing_artifact(tmp_path: Path):
    from adaptive_rag.errors import RerankerModelError
    from adaptive_rag.reranking import OnnxCrossEncoderReranker

    scorer = OnnxCrossEncoderReranker(
        model_id="definitely/not-a-real-model", model_dir=tmp_path / "empty"
    )
    with pytest.raises(RerankerModelError):
        scorer.score("query", ["passage"])