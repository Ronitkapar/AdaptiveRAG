"""
tests.test_routing
------------------
Phase 6 tests for adaptive routing.

Everything here is offline and deterministic. The real `QueryFeatureAnalyzer`,
`RuleBasedRouter`, `SufficiencyChecker`, and `EscalationPolicy` are exercised
directly; retrieval is stubbed with `StubRetriever` so routing behaviour is tested
independently of any index. One canonical-corpus test exercises the full
BM25-only adaptive path over a real persisted index and skips when it is absent.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from adaptive_rag.errors import ConfigurationError, InvalidQueryError
from adaptive_rag.evaluation import (
    EfficiencyEvaluator,
    RetrievalEvaluator,
    RoutingEvaluator,
)
from adaptive_rag.experiments import build_experiment_config
from adaptive_rag.experiments.runner import ExperimentRunner
from adaptive_rag.retrieval.adaptive import AdaptiveRetriever
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.routing import (
    EscalationPolicy,
    QueryAnalyzer,
    QueryFeatureAnalyzer,
    Router,
    RuleBasedRouter,
    SufficiencyChecker,
)
from adaptive_rag.schemas import (
    ChunkMetadata,
    ChunkProvenance,
    RetrievalConfig,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
    RoutingConfig,
)
from tests.fakes import FakeAnalyzer, FakeRouter, StubRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_STRATEGIES = ("bm25", "dense", "hybrid", "hybrid_rerank")


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


def _ranked(chunk_ids: list[str], text: str | None = None) -> list[RetrievalResult]:
    return [
        _result(cid, rank, score=1.0 / rank, text=text)
        for rank, cid in enumerate(chunk_ids, start=1)
    ]


def _bm25_only_config(**overrides) -> RoutingConfig:
    """Two-rung BM25 → dense configuration, valid for escalation tests."""
    fields = {
        "available_strategies": ["bm25", "dense"],
        "escalation_ladder": ["bm25", "dense"],
    }
    fields.update(overrides)
    return RoutingConfig(**fields)


def _adaptive(
    retrievers: dict,
    *,
    routing: RoutingConfig | None = None,
    router=None,
    analyzer=None,
) -> AdaptiveRetriever:
    cfg = routing or _bm25_only_config()
    return AdaptiveRetriever(
        retrievers=retrievers,
        router=router or RuleBasedRouter(cfg),
        analyzer=analyzer or QueryFeatureAnalyzer(),
        sufficiency_checker=SufficiencyChecker(cfg),
        escalation_policy=EscalationPolicy(cfg),
        retrieval_config=RetrievalConfig(retrieval_method="adaptive"),
        routing_config=cfg,
        corpus_version="test_corp",
    )


# --- 1: query analyzer ------------------------------------------------------


def test_analyzer_is_deterministic():
    """Identical queries must produce byte-identical features."""
    a = QueryFeatureAnalyzer()
    q = "Why does masked language modeling help latent-variable retrieval?"
    assert a.analyze(q).model_dump() == a.analyze(q).model_dump()


def test_analyzer_satisfies_protocol():
    assert isinstance(QueryFeatureAnalyzer(), QueryAnalyzer)


def test_analyzer_short_simple_query():
    f = QueryFeatureAnalyzer().analyze("BM25")
    assert f.query_length_words == 1
    assert f.content_term_count == 1
    assert f.lexical_density == 1.0
    assert f.complexity_score < 0.2


def test_analyzer_technical_query_records_terminology():
    f = QueryFeatureAnalyzer().analyze("How is reciprocal rank fusion used with BM25?")
    assert f.technical_term_count >= 2
    assert f.technical_ratio > 0.0


def test_analyzer_entity_indicators():
    """Quoted spans, acronyms, and identifiers are all counted."""
    quoted = QueryFeatureAnalyzer().analyze('What does "latent variable" mean?')
    acronym = QueryFeatureAnalyzer().analyze("What is RAG in DPR?")
    identifier = QueryFeatureAnalyzer().analyze("What is the k1 parameter?")
    assert quoted.entity_indicator_count >= 1
    assert acronym.entity_indicator_count >= 1
    assert identifier.entity_indicator_count >= 1
    assert QueryFeatureAnalyzer().analyze("what is this about").entity_indicator_count == 0


def test_analyzer_multi_concept_query():
    single = QueryFeatureAnalyzer().analyze("Describe the retriever.")
    multi = QueryFeatureAnalyzer().analyze(
        "Compare DPR and Contriever and REALM architectures"
    )
    assert single.multi_concept is False
    assert multi.multi_concept is True
    assert multi.concept_count > single.concept_count


def test_analyzer_comparison_query():
    f = QueryFeatureAnalyzer().analyze("Compare BM25 and dense retrieval")
    assert f.question_type == "compare"
    assert f.comparison_indicator_count >= 1


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is BM25?", "what"),
        ("Why does REALM work?", "why"),
        ("How does the retriever train?", "how"),
        ("Which paper introduces BM25?", "which"),
        ("Who wrote the RAG paper?", "who"),
        ("Define reciprocal rank fusion", "define"),
        ("Robertson and Zaragoza 2009", "other"),
    ],
)
def test_analyzer_question_types(query: str, expected: str):
    assert QueryFeatureAnalyzer().analyze(query).question_type == expected


def test_analyzer_complexity_is_bounded():
    """Complexity stays inside [0, 1] for both trivial and maximal queries."""
    a = QueryFeatureAnalyzer()
    assert 0.0 <= a.analyze("BM25").complexity_score <= 1.0
    monster = (
        "Compare why BM25 versus dense retrieval and hybrid RRF reranking differ "
        "in latency across ColBERT, SPLADE, DPR and REALM benchmarks"
    )
    assert 0.0 <= a.analyze(monster).complexity_score <= 1.0


def test_analyzer_complexity_rises_with_difficulty():
    a = QueryFeatureAnalyzer()
    simple = a.analyze("Define BM25").complexity_score
    complex_q = a.analyze(
        "Compare why BM25 versus dense retrieval and hybrid reranking differ in latency"
    ).complexity_score
    assert complex_q > simple


def test_router_satisfies_protocol_and_returns_valid_decision():
    r = RuleBasedRouter()
    assert isinstance(r, Router)
    d = r.route(QueryFeatureAnalyzer().analyze("What is BM25?"), top_k=10)
    assert d.strategy in ALL_STRATEGIES
    assert d.final_top_k == 10
    assert d.router_kind == "rule_based"
    assert d.candidate_k > 0


def test_router_records_evidence_for_every_available_strategy():
    d = RuleBasedRouter().route(QueryFeatureAnalyzer().analyze("What is BM25?"), top_k=5)
    assert {e.strategy for e in d.evidence} == set(ALL_STRATEGIES)
    assert all(isinstance(e.score, float) for e in d.evidence)


def test_router_exact_terminology_query_selects_cheap_lexical_strategy():
    """A definitional lookup is BM25's home turf and must not cost 600 ms."""
    d = RuleBasedRouter().route(
        QueryFeatureAnalyzer().analyze("Define reciprocal rank fusion"), top_k=10
    )
    assert d.strategy == "bm25"
    assert d.reranking_required is False


def test_router_explanatory_query_prefers_semantic_strategy():
    d = RuleBasedRouter().route(
        QueryFeatureAnalyzer().analyze(
            "Why does masked language modeling help latent-variable retrieval?"
        ),
        top_k=10,
    )
    assert d.strategy in ("dense", "hybrid")


def test_router_evidence_affects_scores():
    """Disabling a group the winner relies on must change the evidence."""
    a = QueryFeatureAnalyzer()
    r = RuleBasedRouter()
    query = "Define reciprocal rank fusion"
    full = r.route(a.analyze(query), top_k=10)
    ablated = RuleBasedRouter(
        RoutingConfig(enabled_feature_groups=["complexity", "multi_concept"])
    ).route(a.analyze(query), top_k=10)

    full_bm25 = next(e for e in full.evidence if e.strategy == "bm25")
    ablated_bm25 = next(e for e in ablated.evidence if e.strategy == "bm25")
    assert set(full_bm25.contributions) != set(ablated_bm25.contributions)
    assert ablated.feature_groups_disabled


def test_router_configuration_changes_behaviour():
    """Cost weight must actually move decisions, not merely be recorded."""
    a = QueryFeatureAnalyzer().analyze("Compare DPR and Contriever architectures")
    unpenalized = [x["strategy"] for x in RuleBasedRouter(
        RoutingConfig(cost_weight=0.0)).route(a, top_k=10).metadata["score_ranking"]]
    penalized = [x["strategy"] for x in RuleBasedRouter(
        RoutingConfig(cost_weight=5.0)).route(a, top_k=10).metadata["score_ranking"]]
    assert unpenalized != penalized
    assert penalized.index("hybrid_rerank") > unpenalized.index("hybrid_rerank")


def test_router_never_selects_an_unavailable_strategy():
    cfg = RoutingConfig(
        available_strategies=["bm25"],
        escalation_ladder=["bm25"],
        # A one-rung ladder has nowhere to escalate to, so the step bound must
        # drop with it; the config validator enforces exactly this coupling.
        max_escalation_steps=0,
        cost_weight=0.0,
    )
    d = RuleBasedRouter(cfg).route(
        QueryFeatureAnalyzer().analyze(
            "Compare why dense retrieval and hybrid reranking differ across ColBERT and SPLADE"
        ),
        top_k=10,
    )
    assert d.strategy == "bm25"
    assert {e.strategy for e in d.evidence} == {"bm25"}


def test_router_is_deterministic():
    a = QueryFeatureAnalyzer()
    r = RuleBasedRouter()
    f = a.analyze("Compare BM25 and dense retrieval across three corpora")
    assert r.route(f, top_k=10).model_dump() == r.route(f, top_k=10).model_dump()


def test_router_confidence_is_bounded_and_explained():
    d = RuleBasedRouter().route(
        QueryFeatureAnalyzer().analyze("Define reciprocal rank fusion"), top_k=10
    )
    assert 0.0 <= d.confidence <= 1.0
    assert d.metadata["score_ranking"][0]["strategy"] == d.strategy


def test_router_confidence_is_zero_without_a_runner_up():
    """A single candidate means no discriminating evidence exists."""
    cfg = RoutingConfig(
        available_strategies=["bm25"],
        escalation_ladder=["bm25"],
        max_escalation_steps=0,
    )
    d = RuleBasedRouter(cfg).route(
        QueryFeatureAnalyzer().analyze("Compare A and B"), top_k=5
    )
    assert d.confidence == 0.0


def test_router_reranking_flag_matches_selected_strategy():
    """`reranking_required` is a claim about the decision, so it must be true."""
    r = RuleBasedRouter(
        RoutingConfig(
            available_strategies=["bm25", "hybrid_rerank"],
            escalation_ladder=["bm25", "hybrid_rerank"],
        )
    )
    for query in ["What is BM25?", "Compare DPR and ColBERT and REALM and SPLADE across k1 b"]:
        d = r.route(QueryFeatureAnalyzer().analyze(query), top_k=10)
        assert d.reranking_required == (d.strategy == "hybrid_rerank")


def test_router_rejects_empty_available_strategies():
    """An empty candidate set is a configuration error, not a silent default."""
    cfg = RoutingConfig.model_construct(available_strategies=[], escalation_ladder=[])
    with pytest.raises(ConfigurationError):
        RuleBasedRouter(cfg).route(QueryFeatureAnalyzer().analyze("BM25"), top_k=5)


@pytest.mark.parametrize("bad", ["", "   ", "\t\n", "!"])
def test_analyzer_rejects_unusable_query(bad: str):
    with pytest.raises(InvalidQueryError):
        QueryFeatureAnalyzer().analyze(bad)


# --- 3: retrieval sufficiency -----------------------------------------------


def _response(results, method: str = "bm25"):
    return RetrievalResponse(
        query="q",
        results=list(results),
        retrieval_method=method,
        status="ok" if results else "no_results",
        retrieval_metadata=RetrievalMetadata(
            top_k=10,
            retriever_version=f"{method}_v1",
            index_id="idx",
            corpus_version="test_corp",
            latency_ms=1.0,
            search_latency_ms=0.5,
        ),
    )


def test_sufficiency_accepts_grounded_evidence():
    """Query terms present in the returned text must read as sufficient."""
    cfg = _bm25_only_config()
    features = QueryFeatureAnalyzer().analyze("reciprocal rank fusion")
    response = _response(_ranked(["a", "b", "c", "d"], text="reciprocal rank fusion formula"))
    d = SufficiencyChecker(cfg).check(features, response, "bm25")
    assert d.sufficient is True
    assert d.coverage == 1.0
    assert d.result_count == 4


def test_sufficiency_rejects_ungrounded_evidence():
    """Query terms absent from everything returned must read as insufficient."""
    cfg = _bm25_only_config()
    features = QueryFeatureAnalyzer().analyze("reciprocal rank fusion")
    response = _response(_ranked(["a", "b", "c", "d"], text="unrelated catalysis prose"))
    d = SufficiencyChecker(cfg).check(features, response, "bm25")
    assert d.sufficient is False
    assert d.coverage == 0.0
    assert d.top1_coverage == 0.0


def test_sufficiency_rejects_empty_result_set():
    cfg = _bm25_only_config()
    features = QueryFeatureAnalyzer().analyze("reciprocal rank fusion")
    d = SufficiencyChecker(cfg).check(features, _response([]), "bm25")
    assert d.sufficient is False
    assert d.result_count == 0
    assert d.coverage == 0.0


def test_sufficiency_threshold_is_configurable():
    """Raising the threshold must be able to flip a decision, or it is inert.

    Partial grounding is required here: with perfect evidence every signal is 1.0
    and no threshold could change the verdict, which would make the assertion
    vacuous rather than a real test of configurability.
    """
    cfg = _bm25_only_config()
    features = QueryFeatureAnalyzer().analyze("reciprocal rank fusion tokenizer")
    # Only one of the two query terms appears in the returned text.
    response = _response(_ranked(["a", "b", "c", "d"], text="reciprocal rank fusion formula"))
    partial = SufficiencyChecker(cfg).check(features, response, "bm25")
    assert partial.coverage == pytest.approx(0.75)  # 3 of 4 content terms found

    lenient = SufficiencyChecker(
        _bm25_only_config(sufficiency_threshold=0.5)
    ).check(features, response, "bm25")
    strict = SufficiencyChecker(
        _bm25_only_config(sufficiency_threshold=0.99)
    ).check(features, response, "bm25")
    assert lenient.sufficient is True
    assert strict.sufficient is False


def test_sufficiency_is_deterministic():
    cfg = _bm25_only_config()
    features = QueryFeatureAnalyzer().analyze("reciprocal rank fusion")
    response = _response(_ranked(["a", "b", "c"], text="reciprocal rank fusion"))
    checker = SufficiencyChecker(cfg)
    assert checker.check(features, response, "bm25").model_dump() == (
        checker.check(features, response, "bm25").model_dump()
    )


def test_sufficiency_records_every_signal():
    cfg = _bm25_only_config()
    features = QueryFeatureAnalyzer().analyze("reciprocal rank fusion")
    response = _response(_ranked(["a", "b", "c"], text="reciprocal rank fusion"))
    d = SufficiencyChecker(cfg).check(features, response, "bm25")
    assert {"result_count", "lexical_coverage", "top1_coverage"} == {s.name for s in d.signals}
    assert d.reason


def test_sufficiency_does_not_read_ground_truth():
    """The checker must not import dataset or label machinery (guard-backed)."""
    source = (REPO_ROOT / "src" / "adaptive_rag" / "routing" / "sufficiency.py").read_text(
        encoding="utf-8"
    )
    for token in ("EvaluationExample", "relevant_documents", "dataset"):
        assert token not in source


# --- 4: escalation policy ---------------------------------------------------


def test_escalation_follows_the_ladder():
    policy = EscalationPolicy(RoutingConfig())
    assert policy.next_strategy("bm25") == "dense"
    assert policy.next_strategy("dense") == "hybrid"
    assert policy.next_strategy("hybrid") == "hybrid_rerank"


def test_escalation_stops_at_the_top_rung():
    """The strongest strategy has nowhere to go; re-running it is not escalation."""
    policy = EscalationPolicy(RoutingConfig())
    assert policy.next_strategy("hybrid_rerank") is None
    assert policy.is_terminal("hybrid_rerank") is True


def test_escalation_cannot_cycle():
    """A total order makes the forbidden BM25→…→BM25 path unreachable."""
    policy = EscalationPolicy(RoutingConfig())
    seen = [policy.next_strategy("bm25")]
    while seen[-1] is not None:
        seen.append(policy.next_strategy(seen[-1]))
    assert len(seen) == len(set(seen)), "escalation revisited a strategy"
    assert seen[-1] is None


def test_escalation_respects_max_steps():
    assert EscalationPolicy(RoutingConfig(max_escalation_steps=1)).allows_escalation(0) is True
    assert EscalationPolicy(RoutingConfig(max_escalation_steps=1)).allows_escalation(1) is False
    zero = EscalationPolicy(
        RoutingConfig(available_strategies=["bm25"], escalation_ladder=["bm25"],
                      max_escalation_steps=0)
    )
    assert zero.allows_escalation(0) is False


def test_escalation_rejects_a_ladder_with_unavailable_rungs():
    cfg = RoutingConfig.model_construct(
        available_strategies=["bm25"], escalation_ladder=["bm25", "dense", "hybrid"]
    )
    with pytest.raises(ConfigurationError):
        EscalationPolicy(cfg)


def test_escalation_rejects_a_cyclic_ladder():
    cfg = RoutingConfig.model_construct(
        available_strategies=["bm25", "dense"], escalation_ladder=["bm25", "dense", "bm25"]
    )
    with pytest.raises(ConfigurationError):
        EscalationPolicy(cfg)


def test_escalation_of_a_strategy_off_the_ladder_is_none():
    policy = EscalationPolicy(RoutingConfig())
    assert policy.next_strategy("nonexistent") is None


# --- 5: adaptive retriever --------------------------------------------------

GROUNDED = "reciprocal rank fusion formula and bm25 scoring"
UNGROUNDED = "unrelated catalysis prose with no shared vocabulary"


def _two_stage(text_for_bm25: str, text_for_dense: str = GROUNDED):
    bm25 = StubRetriever("bm25", _ranked([f"b{i}" for i in range(4)], text=text_for_bm25), latency_ms=1.7)
    dense = StubRetriever("dense", _ranked([f"d{i}" for i in range(4)], text=text_for_dense), latency_ms=600.0)
    return {"bm25": bm25, "dense": dense}, bm25, dense


def test_adaptive_satisfies_retriever_protocol():
    retrievers, _, _ = _two_stage(GROUNDED)
    a = _adaptive(retrievers)
    assert isinstance(a, Retriever)
    assert a.method == "adaptive"


def test_adaptive_returns_without_escalating_when_sufficient():
    retrievers, bm25, dense = _two_stage(GROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    meta = resp.retrieval_metadata
    assert resp.retrieval_method == "adaptive"
    assert meta.adaptive_initial_strategy == "bm25"
    assert meta.adaptive_final_strategy == "bm25"
    assert meta.adaptive_stage_count == 1
    assert meta.adaptive_escalated is False
    assert meta.adaptive_sufficient is True
    # The expensive branch must not have run at all.
    assert dense.call_count == 0
    assert bm25.call_count == 1


def test_adaptive_escalates_when_evidence_is_insufficient():
    retrievers, bm25, dense = _two_stage(UNGROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    meta = resp.retrieval_metadata
    assert meta.adaptive_sufficient is False
    assert meta.adaptive_escalated is True
    assert meta.adaptive_initial_strategy == "bm25"
    assert meta.adaptive_final_strategy == "dense"
    assert meta.adaptive_stage_count == 2
    assert bm25.call_count == 1 and dense.call_count == 1


def test_adaptive_final_results_come_from_the_escalated_strategy():
    """Escalation must replace the results, not merely annotate them."""
    retrievers, _, _ = _two_stage(UNGROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    assert [r.chunk_id for r in resp.results] == ["d0", "d1", "d2", "d3"]
    assert [r.rank for r in resp.results] == [1, 2, 3, 4]


def test_adaptive_trace_records_both_stages():
    retrievers, _, _ = _two_stage(UNGROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    trace = resp.retrieval_metadata.routing
    assert trace is not None
    assert trace.initial_strategy == "bm25"
    assert trace.final_strategy == "dense"
    assert trace.stage_count == 2
    assert len(trace.stage_latencies_ms) == 2
    assert trace.initial_result_count == 4
    assert trace.sufficiency is not None and trace.sufficiency.sufficient is False
    assert trace.escalation.escalated is True
    assert trace.decision.strategy == "bm25"
    assert trace.decision.confidence >= 0.0
    assert trace.features.content_terms


def test_adaptive_never_escalates_when_disabled():
    retrievers, _, dense = _two_stage(UNGROUNDED)
    a = _adaptive(retrievers, routing=_bm25_only_config(escalation_enabled=False))
    resp = a.retrieve("reciprocal rank fusion", top_k=4)
    assert resp.retrieval_metadata.adaptive_escalated is False
    assert dense.call_count == 0


def test_adaptive_escalates_at_most_once():
    """One bounded escalation: even a fully ungrounded first stage stops there."""
    retrievers, bm25, dense = _two_stage(UNGROUNDED, text_for_dense=UNGROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    meta = resp.retrieval_metadata
    assert meta.adaptive_stage_count == 2
    assert bm25.call_count == 1 and dense.call_count == 1
    assert meta.adaptive_final_strategy == "dense"


def test_adaptive_does_not_escalate_past_the_top_rung():
    retrievers, _, _ = _two_stage(UNGROUNDED)
    # A ladder whose top rung is bm25 cannot escalate at all.
    cfg = RoutingConfig(
        available_strategies=["bm25"], escalation_ladder=["bm25"], max_escalation_steps=0
    )
    a = _adaptive(retrievers, routing=cfg)
    resp = a.retrieve("reciprocal rank fusion", top_k=4)
    assert resp.retrieval_metadata.adaptive_escalated is False
    assert resp.retrieval_metadata.adaptive_final_strategy == "bm25"


def test_adaptive_skips_sufficiency_when_disabled():
    retrievers, _, dense = _two_stage(UNGROUNDED)
    a = _adaptive(retrievers, routing=_bm25_only_config(sufficiency_enabled=False))
    resp = a.retrieve("reciprocal rank fusion", top_k=4)
    assert resp.retrieval_metadata.routing.sufficiency is None
    assert resp.retrieval_metadata.adaptive_sufficient is None
    assert dense.call_count == 0


def test_adaptive_latency_accounts_for_every_stage():
    """latency_ms must be the true end-to-end cost, not just the last stage."""
    retrievers, _, _ = _two_stage(UNGROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    meta = resp.retrieval_metadata
    expected = meta.adaptive_routing_latency_ms + sum(meta.routing.stage_latencies_ms)
    assert meta.latency_ms == pytest.approx(expected)


def test_adaptive_preserves_results_and_provenance():
    retrievers, _, _ = _two_stage(GROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    for r in resp.results:
        assert r.text
        assert r.metadata.document_id
        assert r.provenance.document_id
        assert r.provenance.source_sha256 == "sha123"


def test_adaptive_does_not_forward_score_threshold():
    """A threshold must not silently shrink the pool before sufficiency sees it."""
    retrievers, bm25, _ = _two_stage(GROUNDED)
    cfg = RetrievalConfig(retrieval_method="adaptive", score_threshold=0.9)
    a = _adaptive(retrievers)
    a.config = cfg
    resp = a.retrieve("reciprocal rank fusion", top_k=4)
    assert bm25.calls[0]["top_k"] == 4
    assert resp.retrieval_metadata.score_threshold == 0.9


def test_adaptive_forwards_filters_to_stages():
    retrievers, bm25, _ = _two_stage(GROUNDED)
    _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4, filters={"doc": "x"})
    assert bm25.calls[0]["filters"] == {"doc": "x"}


@pytest.mark.parametrize("bad", ["", "   "])
def test_adaptive_rejects_empty_query_without_calling_a_retriever(bad: str):
    retrievers, bm25, dense = _two_stage(GROUNDED)
    with pytest.raises(InvalidQueryError):
        _adaptive(retrievers).retrieve(bad, top_k=4)
    assert bm25.call_count == 0 and dense.call_count == 0


def test_adaptive_propagates_strategy_failure():
    """No silent fallback: a raising strategy must surface as retrieval_failed."""
    boom = RuntimeError("index offline")
    retrievers = {
        "bm25": StubRetriever("bm25", _ranked(["a", "b", "c", "d"]), raises=boom),
        "dense": StubRetriever("dense", _ranked(["d0"], text=GROUNDED)),
    }
    with pytest.raises(RuntimeError, match="index offline"):
        _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)


def test_adaptive_uses_the_router_before_any_retrieval():
    retrievers, bm25, _ = _two_stage(GROUNDED)
    router = FakeRouter(decision=RuleBasedRouter(_bm25_only_config()).route(
        QueryFeatureAnalyzer().analyze("reciprocal rank fusion"), top_k=4))
    _adaptive(retrievers, router=router).retrieve("reciprocal rank fusion", top_k=4)
    assert router.call_count == 1
    assert router.calls[0]["top_k"] == 4
    assert router.calls[0]["features"].content_terms


def test_adaptive_is_deterministic():
    retrievers, _, _ = _two_stage(UNGROUNDED)
    a = _adaptive(retrievers)
    first = a.retrieve("reciprocal rank fusion", top_k=4)
    second = a.retrieve("reciprocal rank fusion", top_k=4)
    assert [r.chunk_id for r in first.results] == [r.chunk_id for r in second.results]
    assert first.retrieval_metadata.adaptive_final_strategy == (
        second.retrieval_metadata.adaptive_final_strategy
    )


# --- 6: configuration -------------------------------------------------------


def test_config_aligns_adaptive_method_and_version():
    """The retrieval_method <-> retriever_version invariant must still hold."""
    cfg = RetrievalConfig(retrieval_method="adaptive")
    assert cfg.retriever_version == "adaptive_v1"


def test_config_rejects_an_escalation_rung_that_cannot_run():
    """A ladder may only name strategies the router can actually select."""
    with pytest.raises(Exception):
        RoutingConfig(available_strategies=["bm25"], escalation_ladder=["bm25", "dense"])


def test_config_rejects_a_duplicate_ladder_entry():
    with pytest.raises(Exception):
        RoutingConfig(
            available_strategies=["bm25", "dense"],
            escalation_ladder=["bm25", "dense", "bm25"],
        )


def test_config_rejects_an_unknown_strategy():
    with pytest.raises(Exception):
        RoutingConfig(
            available_strategies=["bm25", "telepathy"],
            escalation_ladder=["bm25", "telepathy"],
        )


def test_config_rejects_out_of_range_steps():
    with pytest.raises(Exception):
        RoutingConfig(max_escalation_steps=9)


def test_config_rejects_out_of_range_thresholds():
    with pytest.raises(Exception):
        RoutingConfig(sufficiency_threshold=1.5)
    with pytest.raises(Exception):
        RoutingConfig(cost_weight=-1.0)


def test_config_hash_covers_routing_settings():
    """A routing change must change the hash, or a run is not reproducible."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    base = build_experiment_config(
        name="adaptive_v1",
        corpus_version="corpus_test",
        retrieval=RetrievalConfig(retrieval_method="adaptive"),
        routing=RoutingConfig(available_strategies=["bm25"], escalation_ladder=["bm25"],
                              max_escalation_steps=0),
    )
    changed = build_experiment_config(
        name="adaptive_v1",
        corpus_version="corpus_test",
        retrieval=RetrievalConfig(retrieval_method="adaptive"),
        routing=RoutingConfig(
            available_strategies=["bm25"], escalation_ladder=["bm25"],
            max_escalation_steps=0, cost_weight=0.9,
        ),
    )
    assert base.config_hash != changed.config_hash


def test_component_versions_record_routing_only_when_adaptive():
    adaptive = build_experiment_config(
        name="adaptive_v1", corpus_version="corpus_test",
        retrieval=RetrievalConfig(retrieval_method="adaptive"),
    )
    fixed = build_experiment_config(name="bm25_baseline_v1", corpus_version="corpus_test")
    assert adaptive.component_versions.get("routing") == "rule_based_v1"
    assert "routing" not in fixed.component_versions


# --- 7: trace and evaluator -------------------------------------------------


def test_fixed_strategy_trace_has_no_routing():
    """Phase 2-5 traces must remain valid and carry no routing payload."""
    from adaptive_rag.schemas import ExperimentTrace, ReferenceInfo

    trace = ExperimentTrace(
        trace_id="t:1", experiment_id="e", example_id="1", query="q", category="factual",
        reference=ReferenceInfo(reference_answer="a", relevant_documents=["d"]),
        status="ok", config_hash="h", corpus_version="c",
    )
    assert trace.routing is None
    assert trace.model_dump(mode="json")["routing"] is None


def test_adaptive_trace_round_trips_through_json():
    retrievers, _, _ = _two_stage(UNGROUNDED)
    resp = _adaptive(retrievers).retrieve("reciprocal rank fusion", top_k=4)
    payload = resp.retrieval_metadata.model_dump(mode="json")
    from adaptive_rag.schemas import RetrievalMetadata

    assert RetrievalMetadata.model_validate(payload).routing is not None


def test_routing_evaluator_is_silent_without_routing():
    """A fixed-strategy run must emit no routing metrics at all."""
    from adaptive_rag.schemas import ExperimentTrace, ReferenceInfo

    trace = ExperimentTrace(
        trace_id="t:1", experiment_id="e", example_id="1", query="q", category="factual",
        reference=ReferenceInfo(reference_answer="a", relevant_documents=["d"]),
        status="ok", config_hash="h", corpus_version="c",
    )
    report = RoutingEvaluator().evaluate([trace])
    assert report.metrics == []
    assert report.aggregates["routed_traces"] == 0


def test_routing_evaluator_reports_escalation_and_distribution():
    from adaptive_rag.schemas import ExperimentTrace, ReferenceInfo

    routed = _adaptive(*_two_stage(UNGROUNDED)[:1]).retrieve("reciprocal rank fusion", top_k=4)
    settled = _adaptive(*_two_stage(GROUNDED)[:1]).retrieve("reciprocal rank fusion", top_k=4)
    traces = [
        ExperimentTrace(
            trace_id=f"t:{i}", experiment_id="e", example_id=str(i), query="q",
            category="factual",
            reference=ReferenceInfo(reference_answer="a", relevant_documents=["d"]),
            status="ok", config_hash="h", corpus_version="c",
            retrieval=resp, routing=resp.retrieval_metadata.routing,
            retrieval_latency_ms=resp.retrieval_metadata.latency_ms,
        )
        for i, resp in enumerate([routed, settled])
    ]
    report = RoutingEvaluator().evaluate(traces)
    values = {m.name: m.value for m in report.metrics}
    assert values["escalation_rate"] == 0.5
    assert values["retrieval_stages_mean"] == 1.5
    dist = report.aggregates["final_strategy_distribution"]
    assert dist["dense"]["count"] == 1
    assert dist["bm25"]["count"] == 1
    # Every known strategy appears, so a zero is an observation, not an omission.
    assert dist["hybrid_rerank"] == {"count": 0, "share": 0.0}


def test_routing_evaluator_reports_zero_rerank_rate_by_default():
    from adaptive_rag.schemas import ExperimentTrace, ReferenceInfo

    resp = _adaptive(*_two_stage(GROUNDED)[:1]).retrieve("reciprocal rank fusion", top_k=4)
    trace = ExperimentTrace(
        trace_id="t:1", experiment_id="e", example_id="1", query="q", category="factual",
        reference=ReferenceInfo(reference_answer="a", relevant_documents=["d"]),
        status="ok", config_hash="h", corpus_version="c",
        retrieval=resp, routing=resp.retrieval_metadata.routing,
    )
    values = {m.name: m.value for m in RoutingEvaluator().evaluate([trace]).metrics}
    assert values["rerank_rate"] == 0.0


# --- 8: end-to-end through the experiment runner ---------------------------


def test_adaptive_runs_end_to_end_through_the_runner(tmp_path):
    from adaptive_rag.schemas import EvaluationExample

    retrievers, _, _ = _two_stage(GROUNDED)
    config = build_experiment_config(
        name="adaptive_v1", corpus_version="corpus_test",
        retrieval=RetrievalConfig(retrieval_method="adaptive"),
        routing=RoutingConfig(
            available_strategies=["bm25", "dense"], escalation_ladder=["bm25", "dense"]
        ),
    )
    dataset = [
        EvaluationExample(
            example_id="e1", query="reciprocal rank fusion",
            reference_answer="a", relevant_documents=["b0"], category="factual",
        )
    ]
    runner = ExperimentRunner(
        retriever=_adaptive(retrievers, routing=config.routing),
        generator=None,
        evaluators=[RetrievalEvaluator(), EfficiencyEvaluator(), RoutingEvaluator()],
        output_root=tmp_path,
    )
    summary = runner.run(config, dataset)
    run_dir = tmp_path / summary["experiment_id"]
    assert (run_dir / "traces.jsonl").is_file()
    assert (run_dir / "metrics_routing.json").is_file()

    trace_line = (run_dir / "traces.jsonl").read_text().strip().splitlines()[0]
    trace = json.loads(trace_line)
    assert trace["routing"] is not None
    assert trace["routing"]["initial_strategy"] == "bm25"
    assert trace["retrieval"]["retrieval_method"] == "adaptive"
    routing_metrics = json.loads((run_dir / "metrics_routing.json").read_text())
    assert routing_metrics["aggregates"]["routed_traces"] == 1


def test_adaptive_failure_becomes_retrieval_failed_in_the_runner(tmp_path):
    from adaptive_rag.schemas import EvaluationExample

    retrievers = {
        "bm25": StubRetriever("bm25", _ranked(["a"], text=GROUNDED), raises=RuntimeError("boom")),
        "dense": StubRetriever("dense", _ranked(["d"], text=GROUNDED)),
    }
    config = build_experiment_config(
        name="adaptive_v1", corpus_version="corpus_test",
        retrieval=RetrievalConfig(retrieval_method="adaptive"),
    )
    dataset = [
        EvaluationExample(
            example_id="e1", query="reciprocal rank fusion",
            reference_answer="a", relevant_documents=["b0"], category="factual",
        )
    ]
    summary = ExperimentRunner(
        retriever=_adaptive(retrievers), generator=None,
        evaluators=[RetrievalEvaluator(), EfficiencyEvaluator(), RoutingEvaluator()],
        output_root=tmp_path,
    ).run(config, dataset)
    trace = json.loads(
        (tmp_path / summary["experiment_id"] / "traces.jsonl").read_text().strip().splitlines()[0]
    )
    assert trace["status"] == "retrieval_failed"
    assert trace["error"]["error_type"] == "RuntimeError"
    assert trace["retrieval"] is None


# --- 9: CLI -----------------------------------------------------------------


def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"), *cli_args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )


def test_cli_accepts_the_adaptive_retriever():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "adaptive" in result.stdout


def test_cli_rejects_rerank_with_adaptive():
    """Reranking is a routing decision, so the combination must be an error."""
    result = _run_cli("--retriever", "adaptive", "--rerank", "--no-judge", "--no-generation")
    assert result.returncode != 0
    assert "--rerank cannot be combined" in result.stderr


# --- 10: canonical-corpus regression ---------------------------------------


def test_adaptive_over_the_real_persisted_bm25_index():
    """Full runtime path over the canonical corpus, auto-skipped when absent.

    This is the end-to-end check that routing does not merely work against stubs:
    a real BM25 index, a real query, real latency, and a real trace.
    """
    from adaptive_rag.config.paths import BM25_INDEX_PATH
    from adaptive_rag.indexing.bm25 import BM25Index
    from adaptive_rag.retrieval.bm25 import BM25Retriever

    if not BM25_INDEX_PATH.is_file():
        pytest.skip("BM25 index not built; run scripts/build_bm25_index.py")

    index = BM25Index.load()
    if index.total_docs == 0:
        pytest.skip("BM25 index is empty")

    cfg = RoutingConfig(
        available_strategies=["bm25"], escalation_ladder=["bm25"], max_escalation_steps=0
    )
    retriever = BM25Retriever(
        index=index, config=RetrievalConfig(retrieval_method="bm25"),
        corpus_version=index.corpus_version,
    )
    adaptive = AdaptiveRetriever(
        retrievers={"bm25": retriever},
        router=RuleBasedRouter(cfg),
        analyzer=QueryFeatureAnalyzer(),
        sufficiency_checker=SufficiencyChecker(cfg),
        escalation_policy=EscalationPolicy(cfg),
        retrieval_config=RetrievalConfig(retrieval_method="adaptive"),
        routing_config=cfg,
        corpus_version=index.corpus_version,
    )

    resp = adaptive.retrieve("What is BM25?", top_k=10)
    meta = resp.retrieval_metadata

    assert meta.retriever_version == "adaptive_v1"
    assert meta.corpus_version == index.corpus_version
    assert resp.retrieval_method == "adaptive"
    assert meta.routing is not None
    assert meta.adaptive_initial_strategy == "bm25"
    assert meta.adaptive_stage_count == 1
    assert meta.latency_ms > 0.0
    assert 0 <= len(resp.results) <= 10
    # The results must be the real BM25 results, not a re-ranked variant.
    baseline = retriever.retrieve("What is BM25?", top_k=10)
    assert [r.chunk_id for r in resp.results] == [r.chunk_id for r in baseline.results]


def test_adaptive_real_index_surfaces_the_expected_document():
    """Sanity check that the adaptive path is not silently returning noise."""
    from adaptive_rag.config.paths import BM25_INDEX_PATH
    from adaptive_rag.indexing.bm25 import BM25Index
    from adaptive_rag.retrieval.bm25 import BM25Retriever

    if not BM25_INDEX_PATH.is_file():
        pytest.skip("BM25 index not built")
    index = BM25Index.load()
    if index.total_docs == 0:
        pytest.skip("BM25 index is empty")

    cfg = RoutingConfig(
        available_strategies=["bm25"], escalation_ladder=["bm25"], max_escalation_steps=0
    )
    adaptive = AdaptiveRetriever(
        retrievers={
            "bm25": BM25Retriever(
                index=index, config=RetrievalConfig(retrieval_method="bm25"),
                corpus_version=index.corpus_version,
            )
        },
        router=RuleBasedRouter(cfg),
        analyzer=QueryFeatureAnalyzer(),
        sufficiency_checker=SufficiencyChecker(cfg),
        escalation_policy=EscalationPolicy(cfg),
        retrieval_config=RetrievalConfig(retrieval_method="adaptive"),
        routing_config=cfg,
        corpus_version=index.corpus_version,
    )
    ids = {r.chunk_id for r in adaptive.retrieve("Robertson BM25 scoring", top_k=10).results}
    assert any("bm25" in chunk_id for chunk_id in ids)
