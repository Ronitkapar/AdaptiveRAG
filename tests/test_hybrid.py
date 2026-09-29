"""
tests.test_hybrid
-----------------
Phase 4 tests for Reciprocal Rank Fusion and the HybridRetriever composition.
All offline and deterministic: no credentials, no network, no live indexes.
"""

from pathlib import Path

import json
import pytest

from adaptive_rag.errors import ConfigurationError, IndexUnavailableError, InvalidQueryError
from adaptive_rag.evaluation import EfficiencyEvaluator, RetrievalEvaluator, load_evaluation_dataset
from adaptive_rag.experiments import build_experiment_config
from adaptive_rag.experiments.runner import ExperimentRunner
from adaptive_rag.indexing.bm25 import BM25Index
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.retrieval.bm25 import BM25Retriever
from adaptive_rag.retrieval.dense import DenseRetriever
from adaptive_rag.retrieval.fusion import reciprocal_rank_fusion
from adaptive_rag.retrieval.hybrid import HybridRetriever
from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingMetadata,
    HybridRetrievalConfig,
    IndexConfig,
    RetrievalConfig,
    RetrievalResult,
)
from tests.fakes import FakeEmbeddingModel, StubRetriever

RRF_K = 60


def _result(
    chunk_id: str,
    rank: int,
    score: float = 1.0,
    text: str | None = None,
    doc_id: str | None = None,
    sha: str = "hash123",
    section_path: list[str] | None = None,
) -> RetrievalResult:
    document_id = doc_id or chunk_id.split("::")[0]
    body = text if text is not None else f"text for {chunk_id}"
    return RetrievalResult(
        chunk_id=chunk_id,
        text=body,
        score=score,
        rank=rank,
        metadata=ChunkMetadata(
            document_id=document_id,
            doc_title=document_id,
            section_path=section_path or ["1. Section"],
            headings=["Section"],
            element_ids=[f"e-{chunk_id}"],
            element_types=["paragraph"],
            token_count=len(body.split()),
            char_count=len(body),
        ),
        provenance=ChunkProvenance(document_id=document_id, pages=[1], source_sha256=sha),
    )


def _ranked(chunk_ids: list[str], **kwargs) -> list[RetrievalResult]:
    return [_result(cid, rank, **kwargs) for rank, cid in enumerate(chunk_ids, start=1)]


def _hybrid(
    dense: StubRetriever,
    bm25: StubRetriever,
    **config_kwargs,
) -> HybridRetriever:
    config = RetrievalConfig(retrieval_method="hybrid", **config_kwargs)
    return HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        config=config,
        corpus_version="test_corp",
    )


def _stubs(dense_ids: list[str], bm25_ids: list[str], **kwargs):
    return (
        StubRetriever("dense", _ranked(dense_ids, **kwargs)),
        StubRetriever("bm25", _ranked(bm25_ids, **kwargs)),
    )


# --- Pure RRF function ------------------------------------------------------


def test_rrf_scores_and_order():
    """Exact RRF values; order is derived from them, not assumed."""
    dense = _ranked(["A", "B", "C"])
    bm25 = _ranked(["B", "C", "D"])

    fused = reciprocal_rank_fusion([dense, bm25], rrf_k=RRF_K)
    scores = {r.chunk_id: r.score for r in fused}

    assert scores == {
        "A": pytest.approx(1 / 61),
        "B": pytest.approx(1 / 62 + 1 / 61),
        "C": pytest.approx(1 / 63 + 1 / 62),
        "D": pytest.approx(1 / 63),
    }
    assert [r.chunk_id for r in fused] == ["B", "C", "A", "D"]


def test_rrf_duplicate_appears_once_with_summed_contribution():
    """A chunk at dense rank 1 and lexical rank 4 contributes 1/61 + 1/64 once."""
    fused = reciprocal_rank_fusion(
        [_ranked(["Y", "A", "B", "C"]), _ranked(["D", "E", "F", "Y"])], rrf_k=RRF_K
    )
    scores = {r.chunk_id: r.score for r in fused}

    assert len(fused) == 7
    assert scores["Y"] == pytest.approx(1 / 61 + 1 / 64)


def test_rrf_dedupes_by_chunk_id_not_text():
    """Identical text under two distinct chunk_ids must survive as two results."""
    a = _result("chunk_a", 1, text="identical body")
    b = _result("chunk_b", 2, text="identical body")

    fused = reciprocal_rank_fusion([[a], [b]], rrf_k=RRF_K)

    assert [r.chunk_id for r in fused] == ["chunk_a", "chunk_b"]


def test_rrf_single_source_chunks_get_one_term():
    fused = reciprocal_rank_fusion(
        [_ranked(["only_dense"]), _ranked(["only_bm25"])], rrf_k=RRF_K
    )
    scores = {r.chunk_id: r.score for r in fused}

    assert scores == {
        "only_dense": pytest.approx(1 / 61),
        "only_bm25": pytest.approx(1 / 61),
    }


def test_rrf_unequal_lengths_no_padding():
    dense_ids = [f"d{i}" for i in range(20)]
    bm25_ids = [f"b{i}" for i in range(7)]

    fused = reciprocal_rank_fusion(
        [_ranked(dense_ids), _ranked(bm25_ids)], rrf_k=RRF_K
    )
    scores = {r.chunk_id: r.score for r in fused}

    assert len(fused) == 27
    for i, cid in enumerate(dense_ids):
        assert scores[cid] == pytest.approx(1 / (RRF_K + i + 1))
    for i, cid in enumerate(bm25_ids):
        assert scores[cid] == pytest.approx(1 / (RRF_K + i + 1))


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], rrf_k=RRF_K) == []
    only_bm25 = reciprocal_rank_fusion(
        [[], _ranked(["A", "B"])], rrf_k=RRF_K
    )
    assert [r.chunk_id for r in only_bm25] == ["A", "B"]
    only_dense = reciprocal_rank_fusion(
        [_ranked(["A", "B"]), []], rrf_k=RRF_K
    )
    assert [r.chunk_id for r in only_dense] == ["A", "B"]


def test_rrf_rejects_negative_rrf_k():
    with pytest.raises(ConfigurationError):
        reciprocal_rank_fusion([_ranked(["A"])], rrf_k=-1)


def test_rrf_tie_break_is_deterministic():
    """Exact ties resolve by chunk_id, never by input order."""
    fused = reciprocal_rank_fusion(
        [_ranked(["b_chunk"]), _ranked(["a_chunk"])], rrf_k=RRF_K
    )
    assert [r.chunk_id for r in fused] == ["a_chunk", "b_chunk"]

    reversed_order = reciprocal_rank_fusion(
        [_ranked(["a_chunk"]), _ranked(["b_chunk"])], rrf_k=RRF_K
    )
    assert [r.chunk_id for r in reversed_order] == ["a_chunk", "b_chunk"]


# --- HybridRetriever -------------------------------------------------------


def test_hybrid_satisfies_retriever_protocol():
    dense, bm25 = _stubs(["A"], ["B"])
    retriever = _hybrid(dense, bm25)

    assert isinstance(retriever, Retriever)
    assert retriever.method == "hybrid"


def test_hybrid_rejects_empty_query_before_calling_constituents():
    dense, bm25 = _stubs(["A"], ["B"])
    retriever = _hybrid(dense, bm25)

    with pytest.raises(InvalidQueryError):
        retriever.retrieve("   ")

    assert dense.call_count == 0
    assert bm25.call_count == 0


def test_hybrid_propagates_constituent_failure_unchanged():
    boom = IndexUnavailableError("bm25 index exploded")
    dense = StubRetriever("dense", _ranked(["A"]))
    bm25 = StubRetriever("bm25", raises=boom)
    retriever = _hybrid(dense, bm25)

    with pytest.raises(IndexUnavailableError) as excinfo:
        retriever.retrieve("query", top_k=5)

    assert excinfo.value is boom
    assert str(excinfo.value) == "bm25 index exploded"


def test_hybrid_returns_no_results_when_both_branches_empty():
    dense, bm25 = _stubs([], [])
    retriever = _hybrid(dense, bm25)

    resp = retriever.retrieve("query", top_k=5)

    assert resp.retrieval_method == "hybrid"
    assert resp.status == "no_results"
    assert resp.results == []
    assert resp.retrieval_metadata.latency_ms > 0
    assert resp.retrieval_metadata.dense_candidate_count == 0
    assert resp.retrieval_metadata.bm25_candidate_count == 0


def test_hybrid_empty_branch_survives_as_single_source():
    dense, bm25 = _stubs([], ["A", "B"])
    resp = _hybrid(dense, bm25).retrieve("query", top_k=5)

    assert resp.status == "ok"
    assert [r.chunk_id for r in resp.results] == ["A", "B"]
    assert resp.results[0].score == pytest.approx(1 / 61)


def test_hybrid_passes_candidate_k_down_and_truncates_to_top_k():
    dense_ids = [f"d{i}" for i in range(20)]
    bm25_ids = [f"b{i}" for i in range(20)]
    dense, bm25 = _stubs(dense_ids, bm25_ids)
    retriever = _hybrid(dense, bm25, top_k=5, candidate_k=20)

    resp = retriever.retrieve("query", top_k=5)

    assert dense.call_count == 1 and bm25.call_count == 1
    assert dense.calls[0]["top_k"] == 20
    assert bm25.calls[0]["top_k"] == 20
    assert len(resp.results) == 5
    assert [r.rank for r in resp.results] == [1, 2, 3, 4, 5]
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_forwards_filters_to_both_branches():
    dense, bm25 = _stubs(["A"], ["B"])
    retriever = _hybrid(dense, bm25)

    retriever.retrieve("query", top_k=5, filters={"document_id": "doc1"})

    assert dense.calls[0]["filters"] == {"document_id": "doc1"}
    assert bm25.calls[0]["filters"] == {"document_id": "doc1"}
    assert dense.calls[0]["top_k"] == 20 and bm25.calls[0]["top_k"] == 20


def test_hybrid_preserves_metadata_and_prefers_dense_payload():
    shared = "shared::c0"
    dense = StubRetriever(
        "dense",
        [
            _result(shared, 1, text="dense body", sha="dense_sha"),
            _result("dense_only", 2, text="dense only", sha="dense_only_sha"),
        ],
    )
    bm25 = StubRetriever(
        "bm25",
        [
            _result(shared, 1, text="bm25 body", sha="bm25_sha"),
            _result("bm25_only", 2, text="bm25 only", sha="bm25_only_sha"),
        ],
    )

    resp = _hybrid(dense, bm25).retrieve("query", top_k=10)
    by_id = {r.chunk_id: r for r in resp.results}

    assert by_id[shared].text == "dense body"
    assert by_id[shared].provenance.source_sha256 == "dense_sha"
    assert by_id[shared].metadata.section_path == ["1. Section"]
    assert by_id["bm25_only"].text == "bm25 only"
    assert by_id["bm25_only"].provenance.source_sha256 == "bm25_only_sha"
    assert by_id["dense_only"].provenance.source_sha256 == "dense_only_sha"


def test_hybrid_metadata_reports_fusion_diagnostics():
    dense, bm25 = _stubs(["A", "B"], ["B", "C"])
    retriever = _hybrid(dense, bm25, rrf_k=RRF_K, candidate_k=20, top_k=5)

    resp = retriever.retrieve("query", top_k=5)
    meta = resp.retrieval_metadata

    assert meta.retriever_version == "hybrid_v1"
    assert meta.fusion_method == "rrf"
    assert meta.rrf_k == RRF_K
    assert meta.candidate_k == 20
    assert meta.top_k == 5
    assert meta.dense_candidate_count == 2
    assert meta.bm25_candidate_count == 2
    assert resp.results[0].chunk_id == "B"
    assert resp.results[0].score == pytest.approx(1 / 61 + 1 / 62)
    # Fused scores are RRF values, never a constituent's native score.
    assert all(r.score < 0.05 for r in resp.results)


def test_hybrid_latency_accounts_for_both_branches_and_fusion():
    dense = StubRetriever("dense", _ranked(["A"]), latency_ms=1.5, search_latency_ms=0.75)
    bm25 = StubRetriever("bm25", _ranked(["B"]), latency_ms=2.0, search_latency_ms=0.25)
    resp = _hybrid(dense, bm25).retrieve("query", top_k=5)
    meta = resp.retrieval_metadata

    assert meta.dense_latency_ms == 1.5
    assert meta.bm25_latency_ms == 2.0
    assert meta.search_latency_ms == pytest.approx(1.0)
    assert meta.latency_ms > 0
    assert meta.latency_ms >= meta.fusion_latency_ms
    # The reported total is measured end-to-end here, not copied from a branch.
    assert meta.latency_ms != meta.dense_latency_ms
    assert meta.latency_ms != meta.bm25_latency_ms


def test_hybrid_is_deterministic_across_runs():
    dense_ids = [f"d{i}" for i in range(8)]
    bm25_ids = [f"b{i}" for i in range(8)]
    retriever = _hybrid(*_stubs(dense_ids, bm25_ids))

    first = retriever.retrieve("query", top_k=10)
    second = retriever.retrieve("query", top_k=10)

    assert [r.chunk_id for r in first.results] == [r.chunk_id for r in second.results]
    assert [r.score for r in first.results] == [r.score for r in second.results]


# --- Configuration ---------------------------------------------------------


def test_hybrid_config_validation():
    cfg = RetrievalConfig(retrieval_method="hybrid")
    assert cfg.retriever_version == "hybrid_v1"
    assert cfg.rrf_k == 60
    assert cfg.candidate_k == 20
    assert cfg.top_k == 10

    dedicated = HybridRetrievalConfig()
    assert dedicated.retriever_version == "hybrid_v1"
    assert dedicated.retrieval_method == "hybrid"
    assert dedicated.rrf_k == 60
    assert dedicated.candidate_k == 20

    with pytest.raises(ValueError):
        RetrievalConfig(retrieval_method="hybrid", top_k=20, candidate_k=10)


def test_build_experiment_config_stamps_hybrid_versions():
    config = build_experiment_config(
        name="hybrid_baseline_v1",
        retrieval=RetrievalConfig(retrieval_method="hybrid"),
        corpus_version="corpus_test",
    )
    assert config.retrieval.retrieval_method == "hybrid"
    assert config.retrieval.retriever_version == "hybrid_v1"
    assert config.component_versions["retrieval"] == "hybrid_v1"


def test_hybrid_params_are_covered_by_config_hash():
    a = build_experiment_config(
        name="hybrid",
        retrieval=RetrievalConfig(retrieval_method="hybrid"),
        corpus_version="corpus_test",
    )
    b = build_experiment_config(
        name="hybrid",
        retrieval=RetrievalConfig(retrieval_method="hybrid", rrf_k=10, candidate_k=30),
        corpus_version="corpus_test",
    )
    assert a.config_hash != b.config_hash


# --- Integration -----------------------------------------------------------


def _make_test_chunk(chunk_id: str, doc_id: str, text: str, ordinal: int = 0) -> Chunk:
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
            token_count=len(text.split()),
            char_count=len(text),
        ),
        provenance=ChunkProvenance(document_id=doc_id, pages=[1], source_sha256="sha123"),
        chunking_metadata=ChunkingMetadata(
            chunking_version="structure_aware_v1",
            config_hash="cfg",
            ordinal=ordinal,
        ),
    )


def _build_hybrid_components(corpus_version: str = "corpus_test"):
    """Real dense + real BM25 over the same chunks, both fully offline."""
    from qdrant_client import QdrantClient

    dataset = load_evaluation_dataset()
    model = FakeEmbeddingModel(dimension=16)
    store = QdrantVectorStore(
        config=IndexConfig(collection_name="hybrid_test"),
        client=QdrantClient(location=":memory:"),
    )
    store.ensure_collection(dim=16)

    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    ordinal = 0
    for example in dataset:
        for doc_id in example.relevant_documents:
            text = f"{example.query} {example.reference_answer} {doc_id}"
            chunk_id = f"{doc_id}::hybrid_v1::c{ordinal:05d}"
            ordinal += 1
            chunks.append(_make_test_chunk(chunk_id, doc_id, text, ordinal))
            vectors.append(model.embed_query(text))

    store.upsert(chunks, vectors)

    index = BM25Index(corpus_version=corpus_version)
    index.build_from_chunks(chunks, corpus_version=corpus_version)

    retrieval = RetrievalConfig(
        retrieval_method="hybrid", top_k=10, candidate_k=20, rrf_k=RRF_K
    )
    dense = DenseRetriever(
        embedding_model=model,
        vector_store=store,
        config=retrieval,
        corpus_version=corpus_version,
    )
    bm25 = BM25Retriever(index=index, config=retrieval, corpus_version=corpus_version)
    return HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        config=retrieval,
        corpus_version=corpus_version,
    ), dataset


def test_hybrid_end_to_end_through_experiment_runner(tmp_path: Path):
    retriever, dataset = _build_hybrid_components()
    config = build_experiment_config(
        name="hybrid_baseline_v1",
        retrieval=RetrievalConfig(retrieval_method="hybrid", top_k=10),
        corpus_version="corpus_test",
    )

    runner = ExperimentRunner(
        retriever=retriever,
        generator=None,
        evaluators=[RetrievalEvaluator(), EfficiencyEvaluator()],
        output_root=tmp_path / "experiments",
    )
    summary = runner.run(config, dataset, run_id="hybrid_offline")

    run_dir = tmp_path / "experiments" / "hybrid_offline"
    assert summary["trace_count"] == len(dataset)
    assert (run_dir / "traces.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert summary["manifest"]["retrieval_method"] == "hybrid"
    assert summary["manifest"]["status_counts"].get("retrieval_failed", 0) == 0

    recall5 = next(
        m["value"]
        for m in summary["metrics"]["retrieval"]["metrics"]
        if m["name"] == "recall_at_k" and m["k"] == 5
    )
    assert recall5 is not None and recall5 > 0
    assert "retrieval" in summary["metrics"]
    assert "efficiency" in summary["metrics"]


def test_hybrid_response_shape_from_real_components():
    retriever, dataset = _build_hybrid_components()
    example = dataset[0]

    resp = retriever.retrieve(example.query, top_k=10)
    meta = resp.retrieval_metadata

    assert resp.retrieval_method == "hybrid"
    assert resp.status == "ok"
    assert len(resp.results) == 10
    assert [r.rank for r in resp.results] == list(range(1, 11))
    assert len({r.chunk_id for r in resp.results}) == 10
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.metadata.document_id for r in resp.results)
    assert all(r.provenance.source_sha256 == "sha123" for r in resp.results)
    assert meta.fusion_method == "rrf"
    assert meta.candidate_k == 20
    assert meta.dense_candidate_count == 20
    assert meta.bm25_candidate_count > 0
    # End-to-end cost covers both constituent calls plus fusion.
    assert meta.latency_ms >= meta.dense_latency_ms + meta.bm25_latency_ms
    assert meta.search_latency_ms >= 0.0
    assert meta.fusion_latency_ms >= 0.0


def test_runner_records_constituent_failure_without_fallback(tmp_path: Path):
    dataset = load_evaluation_dataset()[:1]
    failing = StubRetriever("dense", raises=IndexUnavailableError("vector store down"))
    healthy = StubRetriever("bm25", _ranked(["anything"]))
    retriever = _hybrid(failing, healthy)

    config = build_experiment_config(
        name="hybrid_baseline_v1",
        retrieval=RetrievalConfig(retrieval_method="hybrid"),
        corpus_version="corpus_test",
    )
    runner = ExperimentRunner(
        retriever=retriever,
        generator=None,
        evaluators=[RetrievalEvaluator(), EfficiencyEvaluator()],
        output_root=tmp_path / "experiments",
    )
    summary = runner.run(config, dataset, run_id="hybrid_failure")

    assert summary["manifest"]["status_counts"]["retrieval_failed"] == 1
    assert summary["manifest"]["error_types"] == ["IndexUnavailableError"]

    trace_path = tmp_path / "experiments" / "hybrid_failure" / "traces.jsonl"
    trace = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert trace["retrieval"] is None
    assert trace["error"]["error_type"] == "IndexUnavailableError"
    assert "vector store down" in trace["error"]["message"]


# --- Canonical corpus regression -------------------------------------------


def _canonical_corpus_files():
    from adaptive_rag.config.paths import CHUNKS_DIR

    return sorted(CHUNKS_DIR.glob("*.chunks.jsonl"))


@pytest.mark.skipif(
    not _canonical_corpus_files(),
    reason="canonical chunk corpus not present (gitignored artifacts)",
)
def test_regression_hybrid_on_canonical_corpus():
    """Known keywords surface the expected documents after fusion.

    The dense branch is a deterministic stub ordered by chunk size (a retrieval
    signal unrelated to the query terms); the lexical branch is the real
    BM25Retriever over the canonical corpus. The fused list must beat both
    branches rather than reproducing either one.
    """
    from adaptive_rag.config.paths import CHUNKS_DIR
    from adaptive_rag.experiments.config import compute_corpus_version
    from scripts.build_bm25_index import load_canonical_chunks

    chunks = load_canonical_chunks(CHUNKS_DIR)
    assert len(chunks) > 0

    corpus_ver = compute_corpus_version()
    index = BM25Index(corpus_version=corpus_ver)
    index.build_from_chunks(chunks, corpus_version=corpus_ver)
    lexical = BM25Retriever(index=index, corpus_version=corpus_ver)
    retriever_config = RetrievalConfig(retrieval_method="hybrid", top_k=10, candidate_k=20)

    size_ranked = _ranked([c.chunk_id for c in sorted(chunks, key=lambda c: -len(c.text))[:20]])
    dense = StubRetriever("dense", size_ranked)
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=lexical,
        config=retriever_config,
        corpus_version=corpus_ver,
    )

    for keyword, expected_doc in (
        ("Robertson", "bm25_robertson_2009"),
        ("ColBERT", "colbert_khattab_2020"),
        ("BM25", "bm25_robertson_2009"),
    ):
        resp = hybrid.retrieve(keyword, top_k=5)
        top_docs = [r.metadata.document_id for r in resp.results]
        assert expected_doc in top_docs, f"{keyword}: expected {expected_doc}, got {top_docs}"
        assert len(resp.results) == len({r.chunk_id for r in resp.results})

    resp = hybrid.retrieve("Robertson", top_k=10)
    fused_ids = [r.chunk_id for r in resp.results]
    lexical_ids = [r.chunk_id for r in lexical.retrieve("Robertson", top_k=20).results]
    size_ids = [r.chunk_id for r in size_ranked]
    # Fusion is a third ordering, and it is a superset of neither branch alone:
    # it keeps lexical-only hits and also keeps hits only the other branch found.
    assert fused_ids != lexical_ids[: len(fused_ids)]
    assert set(lexical_ids) - set(fused_ids)
    assert set(fused_ids) - set(lexical_ids)
    assert set(fused_ids) != set(size_ids[: len(fused_ids)])


# --- Comparison CLI --------------------------------------------------------


def _write_fake_run(root: Path, name: str, method: str, recall: float) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "retrieval": {
                    "metrics": [
                        {"name": "recall_at_k", "k": 5, "value": recall, "n": 20},
                    ]
                },
                "efficiency": {
                    "metrics": [
                        {"name": "retrieval_latency_ms_mean", "k": None, "value": 1.0, "n": 20},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_version": "corpus_test",
                "trace_count": 20,
                "retrieval_method": method,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _run_comparison(experiments_dir: Path, *extra: str) -> str:
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_retrievers.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--experiments-dir", str(experiments_dir), *extra],
        capture_output=True,
        text=True,
        cwd=script.parent.parent,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_comparison_hybrid_column_is_opt_in(tmp_path: Path):
    """A hybrid run present on disk must not change the default two-run table."""
    _write_fake_run(tmp_path, "dense_baseline_v1", "dense", 0.9)
    _write_fake_run(tmp_path, "bm25_baseline_v1", "bm25", 0.7)
    _write_fake_run(tmp_path, "hybrid_baseline_v1", "hybrid", 0.8)

    default_output = _run_comparison(tmp_path)
    assert "method:         hybrid=" not in default_output
    assert f"{'metric':<22}{'dense':>12}{'bm25':>12}{'delta':>12}" in default_output
    assert "Δhyb-dense" not in default_output

    with_hybrid = _run_comparison(
        tmp_path, "--hybrid-run", str(tmp_path / "hybrid_baseline_v1")
    )
    assert "method:         hybrid=hybrid OK" in with_hybrid
    assert "Δhyb-dense" in with_hybrid
    assert f"{'metric':<22}{'dense':>12}{'bm25':>12}{'hybrid':>12}" in with_hybrid


def test_comparison_rejects_wrong_method_for_hybrid_run(tmp_path: Path):
    _write_fake_run(tmp_path, "dense_baseline_v1", "dense", 0.9)
    _write_fake_run(tmp_path, "bm25_baseline_v1", "bm25", 0.7)
    bad = _write_fake_run(tmp_path, "hybrid_baseline_v1", "dense", 0.8)

    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_retrievers.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--experiments-dir",
            str(tmp_path),
            "--hybrid-run",
            str(bad),
        ],
        capture_output=True,
        text=True,
        cwd=script.parent.parent,
    )
    assert proc.returncode == 2
    assert "method:         hybrid=dense UNEXPECTED" in proc.stdout
