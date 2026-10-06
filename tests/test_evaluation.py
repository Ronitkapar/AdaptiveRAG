"""
tests.test_evaluation
---------------------
Unit tests for retrieval, generation, and efficiency evaluators.
All metrics derive from raw traces; no live providers are touched.
"""

import pytest

from adaptive_rag.evaluation import (
    EfficiencyEvaluator,
    GenerationEvaluator,
    RetrievalEvaluator,
    load_evaluation_dataset,
    rouge_l,
    token_f1,
)
from adaptive_rag.experiments import build_experiment_config
from adaptive_rag.schemas import (
    ChunkMetadata,
    ChunkProvenance,
    GenerationResult,
    TokenUsage,
)


def _retrieval_result(chunk_id: str, doc_id: str, rank: int, score: float):
    from adaptive_rag.schemas import RetrievalResult

    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"Evidence passage about {doc_id}",
        score=score,
        rank=rank,
        metadata=ChunkMetadata(
            document_id=doc_id,
            doc_title=doc_id,
            section_path=["1. Test"],
            headings=["Test"],
            element_ids=["e1"],
            element_types=["paragraph"],
            token_count=10,
            char_count=20,
        ),
        provenance=ChunkProvenance(document_id=doc_id, pages=[1], source_sha256="s"),
    )


def test_load_dataset_validates_against_corpus():
    dataset = load_evaluation_dataset()
    assert len(dataset) == 20
    assert all(ex.category for ex in dataset)
    assert all(ex.relevant_documents for ex in dataset)


def test_token_f1_and_rouge_l_reference_values():
    assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert token_f1("apple orange", "banana grape") == 0.0
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l("", "nonempty") == 0.0


def test_retrieval_evaluator_known_case():
    from adaptive_rag.schemas import ExperimentTrace, ReferenceInfo, RetrievalMetadata, RetrievalResponse

    example = load_evaluation_dataset()[0]
    rel_doc = example.relevant_documents[0]

    results = [
        _retrieval_result("c-other", "some_other_doc", rank=1, score=0.9),
        _retrieval_result("c-hit", rel_doc, rank=2, score=0.8),
    ]
    response = RetrievalResponse(
        query=example.query,
        results=results,
        retrieval_method="dense",
        status="ok",
        retrieval_metadata=RetrievalMetadata(
            top_k=2,
            retriever_version="dense_v1",
            embedding_model_id="stub",
            embedding_dim=8,
            index_id="idx",
            collection="col",
            corpus_version="corpus_test",
            latency_ms=1.0,
            query_embedding_latency_ms=0.5,
            search_latency_ms=0.5,
        ),
    )
    trace = ExperimentTrace(
        trace_id="t:eval_001",
        experiment_id="t",
        example_id=example.example_id,
        query=example.query,
        category=example.category,
        retrieval=response,
        reference=ReferenceInfo(
            reference_answer=example.reference_answer,
            relevant_documents=example.relevant_documents,
        ),
        status="ok",
        config_hash="cfg",
        corpus_version="corpus_test",
    )

    report = RetrievalEvaluator().evaluate([trace], [example])
    by_key = {(m.name, m.k): m.value for m in report.metrics}
    assert by_key[("recall_at_k", 1)] == 0.0
    assert by_key[("recall_at_k", 3)] == pytest.approx(1.0)
    assert by_key[("mrr", None)] == pytest.approx(0.5)


def test_generation_and_efficiency_evaluators_offline():
    from adaptive_rag.schemas import ExperimentTrace, ReferenceInfo

    example = load_evaluation_dataset()[0]
    generation = GenerationResult(
        answer=f"Dense retrieval facts about {example.relevant_documents[0]} [Source 1]",
        source_chunk_ids=["c-hit"],
        model="stub",
        usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
        latency_ms=42.0,
        finish_reason="stop",
        status="ok",
        generator_version="stub_v1",
        prompt_version="stub",
    )
    trace = ExperimentTrace(
        trace_id="t:eval_001",
        experiment_id="t",
        example_id=example.example_id,
        query=example.query,
        category=example.category,
        generation=generation,
        reference=ReferenceInfo(
            reference_answer=example.reference_answer,
            relevant_documents=example.relevant_documents,
        ),
        status="ok",
        generation_latency_ms=42.0,
        total_latency_ms=50.0,
        usage=generation.usage,
        estimated_cost_usd=0.0001,
        config_hash="cfg",
        corpus_version="corpus_test",
    )

    config = build_experiment_config(corpus_version="corpus_test")
    config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"enable_llm_judge": False})}
    )

    generation_report = GenerationEvaluator().evaluate([trace], [example], config.evaluation)
    gen_metrics = {m.name: m.value for m in generation_report.metrics}
    assert gen_metrics["answer_count"] == 1.0
    assert gen_metrics["citation_coverage"] == 1.0
    assert generation_report.aggregates["judge"]["enabled"] is False

    efficiency_report = EfficiencyEvaluator().evaluate([trace], [example])
    eff = {m.name: m.value for m in efficiency_report.metrics}
    assert eff["total_latency_ms_mean"] == pytest.approx(50.0)
    assert eff["total_tokens_total"] == 120.0
