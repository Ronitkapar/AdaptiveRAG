"""
tests.test_experiment_runner
----------------------------
Offline end-to-end validation of ExperimentRunner with fake components.
"""

from pathlib import Path

import pytest

from adaptive_rag.evaluation import (
    EfficiencyEvaluator,
    GenerationEvaluator,
    RetrievalEvaluator,
    load_evaluation_dataset,
)
from adaptive_rag.experiments import build_experiment_config
from adaptive_rag.experiments.runner import ExperimentRunner
from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.retrieval.dense import DenseRetriever
from adaptive_rag.schemas import IndexConfig, RetrievalConfig
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingMetadata,
)
from tests.fakes import FakeEmbeddingModel, FakeGenerator


def _build_stub_retriever() -> DenseRetriever:
    """Fake-backed retriever preloaded with chunks covering every eval document."""
    from adaptive_rag.evaluation import load_evaluation_dataset  # local import for fixture clarity

    dataset = load_evaluation_dataset()
    model = FakeEmbeddingModel(dimension=16)

    from qdrant_client import QdrantClient

    store = QdrantVectorStore(
        config=IndexConfig(collection_name="experiment_runner_test"),
        client=QdrantClient(location=":memory:"),
    )
    store.ensure_collection(dim=16)

    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    ordinal = 0
    for example in dataset:
        for doc_id in example.relevant_documents:
            text = f"Reference evidence for {doc_id}: {example.reference_answer[:120]}"
            chunk_id = f"{doc_id}::structure_aware_v1::c{ordinal:05d}"
            ordinal += 1
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=text,
                    metadata=ChunkMetadata(
                        document_id=doc_id,
                        doc_title=doc_id,
                        section_path=["1. Test"],
                        headings=["Test"],
                        element_ids=[f"e{ordinal}"],
                        element_types=["paragraph"],
                        token_count=20,
                        char_count=len(text),
                    ),
                    provenance=ChunkProvenance(
                        document_id=doc_id, pages=[1], source_sha256="stub"
                    ),
                    chunking_metadata=ChunkingMetadata(
                        chunking_version="structure_aware_v1",
                        config_hash="stub",
                        ordinal=ordinal,
                    ),
                )
            )
            vectors.append(model.embed_query(text))

    store.upsert(chunks, vectors)
    return DenseRetriever(
        embedding_model=model, vector_store=store, config=RetrievalConfig(top_k=5)
    )


def test_runner_end_to_end_offline(tmp_path: Path):
    dataset = load_evaluation_dataset()
    config = build_experiment_config(corpus_version="corpus_test")
    config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"enable_llm_judge": False})}
    )
    retriever = _build_stub_retriever()

    runner = ExperimentRunner(
        retriever=retriever,
        generator=FakeGenerator(),
        context_builder=ContextBuilder(),
        evaluators=[
            RetrievalEvaluator(),
            GenerationEvaluator(),
            EfficiencyEvaluator(),
        ],
        output_root=tmp_path / "experiments",
    )

    summary = runner.run(config, dataset, run_id="offline_smoke")

    assert summary["trace_count"] == len(dataset)
    assert summary["manifest"]["status_counts"]["ok"] == len(dataset)
    assert (tmp_path / "experiments" / "offline_smoke" / "traces.jsonl").is_file()
    assert (tmp_path / "experiments" / "offline_smoke" / "metrics.json").is_file()
    assert (tmp_path / "experiments" / "offline_smoke" / "report.md").is_file()

    retrieval = summary["metrics"]["retrieval"]
    recall5 = next(
        m["value"] for m in retrieval["metrics"] if m["name"] == "recall_at_k" and m["k"] == 5
    )
    assert recall5 is not None and recall5 > 0

    generation = summary["metrics"]["generation"]
    assert "judge" in generation["aggregates"]


def test_runner_retrieval_only_records_ok_traces_and_a_total_clock(tmp_path: Path):
    """The Phase 7 protocol at the runner seam: no generator, no generation clock.

    Pins three things at once -- retrieval-only yields `status="ok"` rather than
    `generation_failed`, `generation` stays `None` with no generation latency, and
    `total_latency_ms` is populated instead of left unset.
    """
    from adaptive_rag.schemas import ExperimentTrace

    dataset = load_evaluation_dataset()[:3]
    config = build_experiment_config(corpus_version="corpus_test")
    retriever = _build_stub_retriever()

    runner = ExperimentRunner(
        retriever=retriever,
        evaluators=[RetrievalEvaluator(), EfficiencyEvaluator()],
        output_root=tmp_path / "experiments",
        retrieval_only=True,
    )
    assert runner.generator is None

    summary = runner.run(config, dataset, run_id="retrieval_only_smoke")

    assert summary["trace_count"] == 3
    assert summary["manifest"]["status_counts"] == {
        "ok": 3,
        "empty": 0,
        "retrieval_failed": 0,
        "generation_failed": 0,
    }
    assert summary["manifest"]["error_types"] == []

    traces = [
        ExperimentTrace.model_validate_json(line)
        for line in (tmp_path / "experiments" / "retrieval_only_smoke" / "traces.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(traces) == 3
    for trace in traces:
        assert trace.status == "ok"
        assert trace.retrieval is not None
        assert trace.generation is None
        assert trace.generation_latency_ms is None
        assert trace.total_latency_ms == trace.retrieval_latency_ms
        assert trace.total_latency_ms and trace.total_latency_ms > 0


def test_runner_default_protocol_still_generates(tmp_path: Path):
    """`retrieval_only=False` is the default, so no existing call path changed."""
    dataset = load_evaluation_dataset()[:2]
    config = build_experiment_config(corpus_version="corpus_test")
    generator = FakeGenerator()

    runner = ExperimentRunner(
        retriever=_build_stub_retriever(),
        generator=generator,
        evaluators=[RetrievalEvaluator()],
        output_root=tmp_path / "experiments",
    )

    summary = runner.run(config, dataset, run_id="generated_smoke")

    assert generator.call_count == 2
    assert summary["manifest"]["status_counts"]["ok"] == 2
