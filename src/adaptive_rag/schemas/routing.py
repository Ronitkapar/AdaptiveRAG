"""
schemas.routing
---------------
Data contracts for Phase 6 adaptive routing: query features, per-strategy
evidence, the routing decision, the retrieval-sufficiency judgement, and the
escalation outcome.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# The four pipelines the adaptive router may select. `hybrid_rerank` is the Phase 5
# second-stage pipeline; the other three are the Phase 2-4 first stages. This is the
# single definition; schemas/config.py imports it rather than redeclaring it.
STRATEGY_ORDER: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")

StrategyName = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]

QuestionType = Literal[
    "what",
    "how",
    "why",
    "which",
    "who",
    "when",
    "where",
    "compare",
    "define",
    "other",
]


class QueryFeatures(BaseModel):
    """Cheap, deterministic query characteristics produced by the query analyzer.

    Every field is derived from the raw query string alone: no index, no corpus,
    no model, and no network. The feature set is deliberately explicit so a later
    learned router can consume exactly the same signals as the rule-based router.
    """

    model_config = ConfigDict(extra="forbid")

    query_length_words: int
    query_length_chars: int
    content_terms: list[str] = Field(default_factory=list)
    content_term_count: int
    lexical_density: float
    entity_indicator_count: int
    entity_ratio: float
    technical_term_count: int
    technical_ratio: float
    semantic_indicator_count: int
    semantic_ratio: float
    question_type: QuestionType
    concept_count: int
    multi_concept: bool
    comparison_indicator_count: int
    complexity_score: float
    analyzer_version: str


class StrategyEvidence(BaseModel):
    """Per-strategy routing evidence and the contributions that produced it.

    `contributions` maps a signal group to its weighted contribution, so any score
    can be explained term by term without re-running the router.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: StrategyName
    score: float
    cost_penalty: float
    contributions: dict[str, float] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """A router's complete, structured decision about what should happen.

    The router decides *what* runs; the selected retriever decides *how*. This
    contract is what makes a decision reproducible, loggable, and comparable
    across rule-based and learned routers.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: StrategyName
    # Support for the selected strategy relative to the runner-up. This is a
    # routing-strength signal, NOT a calibrated probability, and it deliberately
    # does not participate in the sufficiency decision.
    confidence: float
    evidence: list[StrategyEvidence] = Field(default_factory=list)
    candidate_k: int
    final_top_k: int
    reranking_required: bool
    router_kind: Literal["rule_based"] = "rule_based"
    router_version: str
    analyzer_version: str
    feature_groups_used: list[str] = Field(default_factory=list)
    feature_groups_disabled: list[str] = Field(default_factory=list)
    cost_weight: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SufficiencySignal(BaseModel):
    """One observable signal contributing to a sufficiency judgement."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    passed: bool
    weight: float = 1.0


class SufficiencyDecision(BaseModel):
    """Whether retrieved evidence justifies continuing or escalating.

    Computed at query time from the query and the retrieved text alone. It never
    reads dataset labels, which would make the check unavailable at query time.
    """

    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    score: float
    threshold: float
    signals: list[SufficiencySignal] = Field(default_factory=list)
    reason: str
    checker_version: str
    result_count: int
    coverage: float | None = None
    top1_coverage: float | None = None


class EscalationDecision(BaseModel):
    """Whether the pipeline escalated, and to which stronger strategy."""

    model_config = ConfigDict(extra="forbid")

    escalated: bool
    from_strategy: StrategyName
    to_strategy: StrategyName | None = None
    reason: str
    step_index: int
    max_steps: int
    policy_version: str


class RoutingTrace(BaseModel):
    """The complete Phase 6 execution trace for a single query.

    Carried on `RetrievalMetadata.routing`, so the routing story travels with the
    response itself and reaches the experiment trace without the retriever keeping
    mutable per-query state.

    `initial_chunk_ids` / `initial_document_ids` are the Phase 7 measurement hook
    for the *discarded* pre-escalation evidence: the first stage's ranked results,
    in rank order. They are populated **only when the query actually escalated**,
    because a query that did not escalate kept its initial results as the final
    ones (`adaptive.py` returns `final = initial`), so its "quality before" is
    already the persisted `retrieval.results` and re-recording it would only
    duplicate bytes. `None` therefore means "not recorded because nothing was
    discarded", and stays distinguishable from `[]` ("recorded, and the first
    stage returned nothing"). This mirrors the Phase 5 reranking fields on
    `ExperimentTrace`, which are likewise absent on runs that never reranked.

    Scores are deliberately not captured: retrieval quality (recall, precision,
    hit, nDCG, MRR) is computed from ranked `chunk_id` and `document_id` alone, and
    the score scales of BM25, cosine similarity, and a cross-encoder are not
    comparable across stages, so a persisted score would invite an invalid
    "score improved" comparison.

    Every field is optional so traces persisted before this extension keep
    validating unchanged under `extra="forbid"`.
    """

    model_config = ConfigDict(extra="forbid")

    features: QueryFeatures
    decision: RoutingDecision
    initial_strategy: StrategyName
    initial_latency_ms: float
    initial_result_count: int
    # Discarded pre-escalation evidence, rank-ordered; None unless escalation ran.
    initial_chunk_ids: list[str] | None = None
    initial_document_ids: list[str] | None = None
    sufficiency: SufficiencyDecision | None = None
    escalation: EscalationDecision | None = None
    final_strategy: StrategyName
    stage_count: int
    stage_latencies_ms: list[float] = Field(default_factory=list)
    routing_latency_ms: float