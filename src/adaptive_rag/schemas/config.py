"""
schemas.config
--------------
Pydantic configuration models for all pipeline stages.
Leaf models with no downstream dependencies to avoid circular imports.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_rag.schemas.routing import STRATEGY_ORDER, StrategyName


class IngestionConfig(BaseModel):
    """Configuration for PDF extraction and document ingestion."""

    model_config = ConfigDict(extra="forbid")

    extractor_backend: Literal["pdfplumber"] = "pdfplumber"
    ingestion_version: str = "ingestion_v3"
    min_page_chars_warning: int = 100
    extract_tables: bool = True
    extract_figures: bool = True
    extract_equations: bool = True


class ChunkingConfig(BaseModel):
    """Configuration for structure-aware document chunking."""

    model_config = ConfigDict(extra="forbid")

    chunking_version: str = "structure_aware_v1"
    target_tokens: int = 500
    max_tokens: int = 800
    min_tokens: int = 150
    overlap_tokens: int = 80
    include_section_header: bool = True
    include_reference_sections: bool = True
    merge_undersized_sections: bool = True


class EmbeddingConfig(BaseModel):
    """Configuration for text embeddings."""

    model_config = ConfigDict(extra="forbid")

    embedding_pipeline_version: str = "embedding_v1"
    provider: Literal["aicredits"] = "aicredits"
    model_id: str = "text-embedding-3-large"
    dimensions: int | None = None
    normalize: bool = True
    batch_size: int = 50
    max_retries: int = 4


class IndexConfig(BaseModel):
    """Configuration for vector store indexing."""

    model_config = ConfigDict(extra="forbid")

    index_version: str = "index_v2"
    # Which Phase 8 corpus arm this experiment reads. The arms differ only in
    # whether two-column reading order was handled correctly, so naming the arm is
    # the whole of what distinguishes one Phase 8 run from the other -- and getting
    # it wrong would silently compare a corpus with itself.
    #
    # "phase7" is the historical namespace: the `adaptiverag_dense_v1` collection
    # and `storage/bm25/bm25_index.json` that Phase 7 actually evaluated (713
    # chunks). It is kept loadable as the record of that study, and is NOT a valid
    # Phase 8 arm -- see `docs/phases/phase-8.md` section 4.
    corpus_arm: Literal["phase7", "phase8_before", "phase8_after"] = "phase8_after"
    # Distinct collections, never a rebuild. `ensure_collection` reuses an existing
    # collection of the same name unless `recreate=True`, so sharing a name across
    # arms would leave each arm's retired chunk ids in the other's index and every
    # dense retrieval would return vectors for chunks that no longer exist.
    collection_name: str = "adaptiverag_dense_v2_after"
    # Likewise for the lexical index: one file per arm, and pointing this at the
    # wrong one is caught by the corpus-version guard on load.
    bm25_index_path: str = "storage/bm25/bm25_index_phase8_after.json"
    distance: Literal["cosine", "dot", "euclidean"] = "cosine"
    batch_size: int = 128

    @model_validator(mode="after")
    def _namespace_follows_arm(self) -> "IndexConfig":
        """Derive the collection and lexical index from the arm.

        Without this, `corpus_arm` would be documentation rather than behaviour: a
        caller selecting an arm could still point at another arm's collection and
        lexical index, and the run would quietly compare the wrong corpus with
        itself. Explicit values are honoured so a one-off namespace stays possible,
        but a silent arm/index mismatch is not.

        The corpus-version guard on the lexical index catches half of this mistake
        on its own; nothing catches a wrong *collection*, because a collection
        carries no corpus version.
        """
        expected = PHASE8_INDEX_NAMESPACES[self.corpus_arm]
        updates = {}
        if self.collection_name == IndexConfig.model_fields["collection_name"].default:
            updates["collection_name"] = expected["collection_name"]
        if self.bm25_index_path == IndexConfig.model_fields["bm25_index_path"].default:
            updates["bm25_index_path"] = expected["bm25_index_path"]
        for field, value in updates.items():
            setattr(self, field, value)
        return self


# Every Phase 8 corpus arm's index namespace. Kept beside `IndexConfig` so the
# mapping from arm to storage is stated once: a run that selects an arm must select
# its collection and its lexical index together, and two hand-maintained lists would
# eventually disagree.
PHASE8_INDEX_NAMESPACES: dict[str, dict[str, str]] = {
    "phase7": {
        "collection_name": "adaptiverag_dense_v1",
        "bm25_index_path": "storage/bm25/bm25_index.json",
    },
    "phase8_before": {
        "collection_name": "adaptiverag_dense_v2_before",
        "bm25_index_path": "storage/bm25/bm25_index_phase8_before.json",
    },
    "phase8_after": {
        "collection_name": "adaptiverag_dense_v2_after",
        "bm25_index_path": "storage/bm25/bm25_index_phase8_after.json",
    },
}


class DenseRetrievalConfig(BaseModel):
    """Configuration specifically for dense vector retrieval."""

    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "dense_v1"
    retrieval_method: Literal["dense"] = "dense"
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None


class BM25RetrievalConfig(BaseModel):
    """Configuration specifically for BM25 lexical retrieval."""

    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "bm25_v1"
    retrieval_method: Literal["bm25"] = "bm25"
    k1: float = 1.2
    b: float = 0.75
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None


class HybridRetrievalConfig(BaseModel):
    """Configuration specifically for hybrid dense + BM25 fusion retrieval."""

    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "hybrid_v1"
    retrieval_method: Literal["hybrid"] = "hybrid"
    rrf_k: int = 60
    candidate_k: int = 20
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None


class RerankerConfig(BaseModel):
    """Configuration specifically for second-stage cross-encoder reranking."""

    model_config = ConfigDict(extra="forbid")

    reranker_version: str = "onnx_cross_encoder_v1"
    model_id: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_revision: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 16
    max_length: int = 512
    candidate_k: int = 20
    top_k: int = 10
    fallback_to_retrieval: bool = False


# Fixed pipeline order lives in schemas/routing.py and is imported above; it is used
# here to break score ties deterministically and to order strategy distributions in
# reports, never to infer capability.

# The six query-characteristic groups the rule-based router scores. Each is a
# normalized [0, 1] signal derived from QueryFeatures, and each can be switched
# off independently to measure how much it actually contributes.
FEATURE_GROUPS: tuple[str, ...] = (
    "lexical",
    "semantic",
    "entity",
    "complexity",
    "question_type",
    "multi_concept",
)


class RetrievalConfig(BaseModel):
    """Configuration for retrieval (dense, bm25, hybrid, a reranked variant, or adaptive)."""

    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "dense_v1"
    retrieval_method: Literal[
        "dense",
        "bm25",
        "hybrid",
        "dense_rerank",
        "bm25_rerank",
        "hybrid_rerank",
        "adaptive",
    ] = "dense"
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None
    # BM25-specific parameters
    k1: float = 1.2
    b: float = 0.75
    # Hybrid / RRF-specific parameters
    rrf_k: int = 60
    candidate_k: int = 20
    # Second-stage cross-encoder parameters (ignored unless rerank_enabled)
    rerank_enabled: bool = False
    rerank_model_id: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_model_revision: str | None = None
    rerank_device: Literal["auto", "cpu", "cuda"] = "auto"
    rerank_batch_size: int = 16
    rerank_max_length: int = 512
    rerank_candidate_k: int = 20
    rerank_fallback: bool = False

    @model_validator(mode="after")
    def validate_method_and_version(self) -> "RetrievalConfig":
        default_version = {
            "dense": "dense_v1",
            "bm25": "bm25_v1",
            "hybrid": "hybrid_v1",
            "dense_rerank": "dense_rerank_v1",
            "bm25_rerank": "bm25_rerank_v1",
            "hybrid_rerank": "hybrid_rerank_v1",
            "adaptive": "adaptive_v1",
        }
        if self.retriever_version in default_version.values():
            # Auto-align a default strategy version to the active retrieval method
            object.__setattr__(
                self, "retriever_version", default_version[self.retrieval_method]
            )
        if self.retrieval_method == "hybrid" and self.candidate_k < self.top_k:
            raise ValueError("hybrid candidate_k must be >= top_k")
        if self.rerank_enabled:
            if self.rerank_candidate_k < self.top_k:
                raise ValueError("rerank_candidate_k must be >= top_k")
            if self.retrieval_method == "hybrid_rerank" and (
                self.candidate_k < self.rerank_candidate_k
            ):
                raise ValueError(
                    "hybrid_rerank candidate_k must be >= rerank_candidate_k"
                )
        return self


class RoutingConfig(BaseModel):
    """Configuration for Phase 6 adaptive routing.

    Holds the rule-based router's weighted evidence table, the measured retrieval
    cost table, the retrieval-sufficiency thresholds, and the bounded escalation
    ladder. Everything that changes a routing outcome lives here, so it is
    captured by `config_hash` and a run is reproducible from its recorded
    `config.json` alone.

    The cost table is seeded from retrieval latencies actually measured in Phase 5
    (docs/phases/phase-5.md §8, retrieval-only, 20-example benchmark). It is a
    default to be re-measured, not a theoretical constant: Phase 5 already recorded
    a documented cost estimate that was wrong by two orders of magnitude.
    """

    model_config = ConfigDict(extra="forbid")

    router_version: str = "rule_based_v1"
    analyzer_version: str = "analyzer_v1"
    sufficiency_version: str = "sufficiency_v1"
    escalation_version: str = "escalation_v1"

    # Strategies the router may select. Restricting this is also how a BM25-only
    # adaptive run stays fully offline: the dense branch is then never built.
    available_strategies: list[StrategyName] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid", "hybrid_rerank"]
    )

    # Weighted evidence table: strategy -> signal group -> weight. Weights may be
    # negative, which is how a signal arguing *against* a strategy (semantic
    # phrasing for BM25, say) is expressed.
    rule_weights: dict[str, dict[str, float]] = Field(default_factory=lambda: {
        "bm25": {
            "lexical": 1.0,
            "entity": 1.5,
            "question_type": 0.8,
            "semantic": -0.6,
            "complexity": -0.8,
            "multi_concept": -1.0,
        },
        "dense": {
            "lexical": -0.3,
            "entity": 0.2,
            "question_type": 0.6,
            "semantic": 1.5,
            "complexity": 0.3,
            "multi_concept": 0.2,
        },
        "hybrid": {
            "lexical": 0.5,
            "entity": 0.3,
            "question_type": 0.5,
            "semantic": 1.0,
            "complexity": 1.0,
            "multi_concept": 1.5,
        },
        "hybrid_rerank": {
            "lexical": 0.0,
            "entity": 0.0,
            "question_type": 0.5,
            "semantic": 0.3,
            "complexity": 1.5,
            "multi_concept": 1.0,
        },
    })

    # Signal groups consulted when scoring. Removing groups is the Phase 6
    # signal-ablation lever; it is configuration, not a code change.
    enabled_feature_groups: list[str] = Field(
        default_factory=lambda: list(FEATURE_GROUPS)
    )

    # Empirically measured retrieval latency per strategy, in milliseconds.
    #
    # Re-measured in Phase 7 under the protocol in `evaluation/measurement.py`:
    # medians over 5 repetitions x 20 queries (n=100 per arm), 5 warm-up queries
    # discarded, arm order and query order rotated between repetitions. Provenance
    # is recorded in `experiments/phase7/strategy_cost_ms.json`.
    #
    # These replaced a Phase 5 seed taken from single-pass per-query means with no
    # warm-up, which encoded a dense/hybrid gap narrower than this hardware's
    # own run-to-run spread.
    #
    # Read `stage_median_ms` in that artifact before trusting these as pure
    # retrieval cost: for dense, ~426 ms of the ~452 ms median is the live
    # `text-embedding-3-large` call and only ~24 ms is the vector search. The
    # router consumes the ratios between these numbers, so a provider outage or
    # a faster provider changes the table's meaning.
    strategy_cost_ms: dict[str, float] = Field(default_factory=lambda: {
        "bm25": 2.25,
        "dense": 451.78,
        "hybrid": 455.97,
        "hybrid_rerank": 3854.41,
    })

    # Quality-vs-cost trade-off knob. 0.0 is pure evidence; higher values subtract
    # `cost_weight * cost(strategy) / max_cost` from a strategy's score, so the
    # router only spends more when the evidence supports spending more.
    cost_weight: float = 0.25

    candidate_k: int = 20

    # Retrieval-sufficiency check (query time, label free).
    sufficiency_enabled: bool = True
    sufficiency_threshold: float = 0.5
    min_results: int = 3
    coverage_threshold: float = 0.5
    top1_coverage_threshold: float = 0.3
    # Off by default: BM25 magnitudes, cosine similarity, RRF scores, and cross-
    # encoder logits are not comparable, so one floor would be an invented number.
    # Kept configurable because a per-strategy floor is legitimate once measured.
    score_floor: dict[str, float] | None = None

    # Bounded escalation.
    escalation_enabled: bool = True
    escalation_ladder: list[StrategyName] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid", "hybrid_rerank"]
    )
    max_escalation_steps: int = 1

    @model_validator(mode="after")
    def validate_routing_policy(self) -> "RoutingConfig":
        named = (
            set(self.rule_weights)
            | set(self.strategy_cost_ms)
            | set(self.escalation_ladder)
            | set(self.available_strategies)
        )
        unknown = named - set(STRATEGY_ORDER)
        if unknown:
            raise ValueError(f"unknown retrieval strategy in routing config: {sorted(unknown)}")

        if not self.available_strategies:
            raise ValueError("available_strategies must not be empty")
        if len(set(self.available_strategies)) != len(self.available_strategies):
            raise ValueError("available_strategies must not contain duplicates")

        if len(set(self.escalation_ladder)) != len(self.escalation_ladder):
            raise ValueError("escalation_ladder must not contain duplicates")
        if not set(self.escalation_ladder) <= set(self.available_strategies):
            raise ValueError(
                "escalation_ladder must be a subset of available_strategies: a rung "
                "that cannot run must never appear in the ladder"
            )
        if not 0 <= self.max_escalation_steps <= len(self.escalation_ladder) - 1:
            raise ValueError(
                "max_escalation_steps must be between 0 and len(escalation_ladder) - 1"
            )

        unknown_groups = set(self.enabled_feature_groups) - set(FEATURE_GROUPS)
        if unknown_groups:
            raise ValueError(f"unknown feature group(s): {sorted(unknown_groups)}")
        if not self.enabled_feature_groups:
            raise ValueError("enabled_feature_groups must not be empty")

        if not 0.0 <= self.sufficiency_threshold <= 1.0:
            raise ValueError("sufficiency_threshold must be within [0, 1]")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be within [0, 1]")
        if not 0.0 <= self.top1_coverage_threshold <= 1.0:
            raise ValueError("top1_coverage_threshold must be within [0, 1]")
        if self.min_results < 1:
            raise ValueError("min_results must be >= 1")
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")
        if self.cost_weight < 0.0:
            raise ValueError("cost_weight must be >= 0")
        return self


class ContextConfig(BaseModel):
    """Configuration for retrieved context assembly."""

    model_config = ConfigDict(extra="forbid")

    context_template_version: str = "context_v1"
    max_context_tokens: int = 6000
    max_chunks: int = 5


class GenerationConfig(BaseModel):
    """Configuration for Groq answer generation."""

    model_config = ConfigDict(extra="forbid")

    generator_version: str = "groq_v1"
    model: str = "openai/gpt-oss-120b"
    prompt_version: str = "generation_prompt_v1"
    temperature: float = 0.0
    max_completion_tokens: int = 1024
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"


class EvaluationConfig(BaseModel):
    """Configuration for retrieval, generation, and efficiency evaluation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_version: str = "eval_v1"
    k_grid: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    judge_model: str = "openai/gpt-oss-20b"
    judge_prompt_version: str = "judge_prompt_v1"
    enable_llm_judge: bool = True
    use_judge_cache: bool = True
