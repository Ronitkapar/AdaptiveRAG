"""Service-layer tests: BM25-only, offline, deterministic.

Provider-backed arms (dense/hybrid/rerank) are deliberately *not*
exercised here: they cost real embedding API calls. The suite stays
offline by construction.
"""

import pytest

from adaptive_rag.schemas import RetrievalResponse
from frontend.ui import service


@pytest.fixture(autouse=True)
def _isolated_cache():
    service.clear_cache()
    yield
    service.clear_cache()


def test_bm25_retrieve_returns_real_response():
    response = service.retrieve("bm25", "What is RAG?", top_k=3)
    assert isinstance(response, RetrievalResponse)
    assert response.retrieval_method == "bm25"
    assert response.status == "ok"
    assert 1 <= len(response.results) <= 3
    first = response.results[0]
    assert first.rank == 1
    assert first.text.strip()
    assert first.metadata.document_id
    assert first.chunk_id.startswith(first.metadata.document_id)


def test_retriever_is_cached_by_config():
    first = service.build_retriever("bm25")
    second = service.build_retriever("bm25")
    assert first is second


def test_sequential_bm25_only_comparison():
    records = service.retrieve_sequential(["bm25"], "What is RAG?", top_k=2)
    assert len(records) == 1
    assert records[0]["strategy"] == "bm25"
    assert records[0]["error"] is None
    assert records[0]["response"].status == "ok"


def test_bm25_only_adaptive_runs_offline():
    routing = service.adaptive_routing(["bm25"])
    response = service.retrieve("adaptive", "What is RAG?", top_k=3, routing=routing)
    assert response.retrieval_method == "adaptive"
    assert response.retrieval_metadata.routing is not None
    trace = response.retrieval_metadata.routing
    assert trace.initial_strategy == "bm25"
    assert trace.final_strategy == "bm25"
