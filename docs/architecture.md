# AdaptiveRAG — System Architecture

## 1. Purpose

AdaptiveRAG is a retrieval research/engineering system designed to study
whether retrieval strategies can be selected according to query
characteristics while balancing:

* retrieval quality
* answer quality
* latency
* token usage
* estimated inference cost

The system is developed progressively so that each retrieval strategy can be
measured independently before adaptive routing is introduced.

---

# 2. Retrieval Strategy Progression

The eventual architecture supports:

```text
Dense
   ↓
BM25
   ↓
Hybrid
   ↓
Reranking
   ↓
Adaptive Routing
```

The current implementation has established the Dense-RAG baseline (Phase 2),
the BM25 lexical baseline (Phase 3), the hybrid rank-fusion strategy (Phase 4),
and second-stage cross-encoder reranking (Phase 5). All four operate over the
same canonical chunk corpus and share the same retrieval/evaluation contracts.

Future retrieval strategies should reuse the existing retrieval and
evaluation contracts wherever possible.

---

# 3a. Phase 3 BM25 Branch

Phase 3 adds an independent lexical branch alongside the dense pipeline:

```text
Structured Chunks (canonical corpus)
        ↓
BM25Index (inverted index, JSON at storage/bm25/)
        ↓
BM25Retriever (native Okapi BM25 scores)
        ↓
Retrieved Context (common RetrievalResponse)
        ↓
Context Builder / Evaluation
```

Guarantees:

* Corpus parity: BM25 reads the same canonical chunks as dense, without
  modifying them or assuming a fixed count.
* Zero coupling: dense and BM25 never import or fall back to each other.
* Common protocol: both implement `Retriever` and emit `RetrievalResponse`
  with `retrieval_method ∈ {"dense", "bm25"}`, so the evaluation pipeline
  stays strategy-agnostic.
* Scores are native BM25 values (unnormalized); provenance and metadata are
  preserved end-to-end.

---

# 3b. Phase 4 Hybrid Branch

Phase 4 adds a composition layer above the two existing retrievers. It owns no
index, no embedding model, and no tokenizer:

```text
        Canonical Chunk Corpus
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 DenseRetriever       BM25Retriever
 (Qdrant, cosine)     (BM25Index, JSON)
       │                   │
       └─────────┬─────────┘
                 ▼
      reciprocal_rank_fusion(ranked_lists, rrf_k=60)
                 ▼
        truncate → top_k, renumber ranks 1..n
                 ▼
     RetrievalResponse(retrieval_method="hybrid")
                 ▼
      ExperimentRunner + evaluators (unchanged)
```

Guarantees:

* **Composition, not reimplementation.** `hybrid.py` orchestrates; it contains
  no tokenization, embedding, index access, or alternative fusion logic, and no
  `try`/`except` (architecture-guard enforced on the AST).
* **Rank-based fusion.** `score(chunk) = Σ_lists 1 / (rrf_k + rank)`. Cosine
  similarity and BM25 magnitudes are never mixed, and no score normalization is
  introduced.
* **Deterministic output.** Deduplication is by `chunk_id`; ordering is
  `(-rrf_score, best_single_source_rank, chunk_id)`, so nothing depends on
  input or dict iteration order.
* **Candidate depth.** Both branches are queried at `candidate_k` (default 20)
  and truncation to `top_k` happens only after fusion.
* **Fail closed.** Constituent exceptions propagate unchanged, so a broken
  branch yields `status="retrieval_failed"` with the original error type —
  never a silent dense-only or BM25-only answer.
* **No third index.** Hybrid reuses the existing Qdrant collection and the
  persisted BM25 index; there is no `storage/hybrid/` and no build script.
  Because the dense branch embeds every query, a hybrid run requires
  `AICREDITS_API_KEY` even with `--no-judge --no-generation`.

`retrieval_method ∈ {"dense", "bm25", "hybrid"}` across configuration, traces,
and manifests, with `retriever_version` aligned (`hybrid ⇔ hybrid_v1`).

---

# 3c. Phase 5 Reranking Branch

Phase 5 adds a second stage above any first-stage retriever. It reorders what
that retriever already returned; it never decides *whether* to run, and it owns
no index, no embedding model, and no corpus handle:

```text
   DenseRetriever | BM25Retriever | HybridRetriever   (all unchanged)
                          │
                          │  RetrievalResponse, ≤ N candidates
                          ▼
                 ┌─────────────────────┐
                 │   RerankedRetriever  │   candidate_generation_latency_ms
                 │                     │   ← reused from the base response
                 │   reranker.score(query, [c.text ...])   ──► rerank_latency_ms
                 │   sort by (-rerank_score, retrieval_rank, chunk_id)
                 │   truncate → top_k, renumber ranks 1..n
                 └─────────────────────┘
                          │  RetrievalResponse(retrieval_method="<base>_rerank")
                          ▼
             ExperimentRunner + evaluators (unchanged)
```

`OnnxCrossEncoderReranker` sits behind a `Reranker` protocol and runs on ONNX
Runtime over a pre-exported ms-marco cross-encoder. No torch, no transformers,
and `sentence-transformers` stays banned.

Guarantees:

* **Composition, not reimplementation.** `reranked.py` holds only a
  `base_retriever` and a `reranker`. It contains no `qdrant`, `idf`,
  `tokenize`, `embedding_model`, or `vector_store`, imports none of the concrete
  retrievers, and architecture-guards enforce all of it. `dense.py`, `bm25.py`,
  `hybrid.py`, and `fusion.py` are untouched and independently guard-checked.
* **Score separation.** `score` is the rerank score; the first-stage signal
  survives as `retrieval_score` and the pre-rerank position as
  `retrieval_rank`. Rerank scores are never averaged, normalized, or blended
  with cosine / BM25 / RRF values, and they are ranking signals rather than
  calibrated probabilities.
* **Deterministic output.** Ordering is `(-rerank_score, retrieval_rank,
  chunk_id)`, so equal scores never depend on input order. Scoring is per-pair
  with no cross-pair interaction, so output is batch-size invariant.
* **Candidate depth.** The base is queried at `rerank_candidate_k` (default 20)
  and truncation to `top_k` happens only after scoring. For
  `hybrid_rerank`, `--rerank-candidate-k` also raises the fusion `candidate_k`,
  so one flag controls depth end-to-end.
* **Fail closed by default.** A reranker exception propagates unchanged and
  yields `status="retrieval_failed"` with the original error type. The single
  `try`/`except` in the module is gated on `config.rerank_fallback`, and an AST
  guard asserts the handler both re-raises and references that flag. When the
  opt-in fallback is taken, `rerank_fallback=True` surfaces in every trace, in
  `rerank_fallback_count`, and in the manifest.
* **No model on an empty pool.** An empty candidate set returns
  `status="no_results"` with `rerank_latency_ms == 0.0` and no model
  invocation; that is distinct from a reranker failure.
* **No threshold forwarding.** `score_threshold` is not passed to the base
  retriever — mirroring the Phase 4 §3b decision — because applying one there
  would silently shrink the candidate pool before the ranker sees it. The
  configured value is echoed into metadata for traceability only.
* **Additive cost accounting.** `latency_ms` stays the full end-to-end total
  (the Phase 4 hybrid rule) so existing efficiency metrics keep their meaning;
  the candidate-generation / rerank split is added alongside it and its metrics
  are emitted only for reranked runs.

`retrieval_method ∈ {"dense", "bm25", "hybrid", "dense_rerank", "bm25_rerank",
"hybrid_rerank"}` across configuration, traces, and manifests, with
`retriever_version` aligned (`dense_rerank ⇔ dense_rerank_v1`, and likewise for
bm25 and hybrid). Every `rerank_*` setting lives inside `RetrievalConfig` and is
covered by `config_hash`; `component_versions["reranker"]` appears only when
reranking is enabled.

---

# 3. Phase 2 Pipeline

The locked Phase 2 pipeline is:

```text
14 Research Papers
        ↓
PDF Ingestion
        ↓
Hierarchical Documents
        ↓
Structure-Aware Chunking
        ↓
Structured Chunks
        ↓
Embedding Model
        ↓
Embedding Cache
        ↓
Qdrant
        ↓
Dense Retriever
        ↓
Retrieved Context
        ↓
Context Builder
        ↓
Groq Generator
        ↓
Answer + Sources
        ↓
Experiment Trace
        ↓
Evaluation
```

---

# 4. Ingestion Layer

The ingestion layer converts PDFs into structured, provenance-preserving
Documents.

Responsibilities:

* text extraction
* heading extraction
* table preservation where reliably extractable
* equation preservation where reliably extractable
* figure/caption representation
* page information
* logical section structure
* source order
* provenance
* extraction failure reporting

Ingestion does not perform:

* chunking
* embedding
* indexing
* retrieval
* generation
* routing

---

# 5. Document Model

Documents preserve both physical and logical structure.

Conceptually:

```text
Document
├── document_id
├── metadata
├── pages[]
├── sections[]
└── provenance
```

Sections contain ordered elements and may contain subsections.

---

# 6. Element Model

Supported element types include:

```text
Paragraph
Heading
Figure
Table
Equation
Caption
List
```

Elements have deterministic identity and source order.

Provenance includes at least:

```text
document_id
page
```

Bounding boxes are included only when reliably available.

---

# 7. Chunking Layer

Phase 2 uses structure-aware chunking.

The conceptual flow is:

```text
Document
   ↓
Sections
   ↓
Subsections
   ↓
Elements
   ↓
Coherent Element Groups
   ↓
Chunks
```

Chunking preserves:

* source order
* section path
* headings
* element IDs
* page range
* provenance

Chunks are derived artifacts; the Document remains canonical.

---

# 8. Chunk Contract

A Chunk conceptually contains:

```text
chunk_id
document_id
text
section_path
headings
element_ids
page_start
page_end
provenance
chunking_metadata
```

Embeddings are deliberately not stored in the canonical Chunk model.

The frozen Phase 2 corpus contains:

```text
713 chunks
Mean: approximately 408 tokens
Median: approximately 447 tokens
106 atomic chunks
Maximum: approximately 891 tokens
```

The maximum chunk is intentionally an unsplit table.

---

# 9. Embedding Layer

Phase 2 uses:

```text
AICredits
    ↓
text-embedding-3-large
```

The embedding implementation is hidden behind an embedding interface.

The embedding cache is content/configuration aware so incompatible embeddings
are not accidentally reused.

---

# 10. Vector Index

Phase 2 uses:

```text
Qdrant
```

locally.

The vector store is hidden behind a vector-store abstraction.

Index metadata records information such as:

* corpus version
* chunking version/configuration
* embedding model/configuration
* vector dimension
* similarity configuration
* index version

---

# 11. Dense Retrieval

The DenseRetriever:

```text
Query
 ↓
Query Embedding
 ↓
Qdrant Search
 ↓
Ranked Retrieval Results
```

The Phase 2 baseline uses a fixed `top_k`.

Retrieval results preserve:

* chunk ID
* text
* score
* rank
* metadata
* provenance

Dense retrieval does not:

* rerank
* generate
* perform BM25
* perform hybrid retrieval or rank fusion
* perform adaptive routing

---

# 12. Generation Layer

Generation is independent from embedding.

```text
Generator
    ↓
Groq API
    ↓
Selected Generation Model
```

The generator receives a structured GenerationRequest rather than accessing
the corpus directly.

---

# 13. Context Builder

The Context Builder sits between retrieval and generation:

```text
RetrievalResponse
        ↓
Context Builder
        ↓
GenerationRequest
```

It:

* preserves retrieval order
* formats retrieved chunks
* includes source metadata
* respects context/token limits
* prepares generator input

---

# 14. Generation Result

Generation produces a structured result containing information such as:

```text
answer
source/chunk IDs
model
token usage
latency
metadata
```

Prompt configuration is versioned.

---

# 3c. Phase 6 Adaptive Routing

Phase 6 composes a decision layer *above* the fixed strategies. It owns no index,
no embedding model, no tokenizer, and no ranking logic:

```text
                         Query
                            │
                            ▼
                  ┌───────────────────────┐
                  │  QueryFeatureAnalyzer  │  deterministic, no model
                  │  → QueryFeatures       │
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │    RuleBasedRouter     │  weighted evidence
                  │  → RoutingDecision     │  (confidence + evidence)
                  └───────────┬───────────┘
                              ▼
              initial retrieval (existing Phase 2–5 Retriever)
                              │  RetrievalResponse
                              ▼
                  ┌───────────────────────┐
                  │   SufficiencyChecker   │  label-free runtime evidence
                  └───────────┬───────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
          sufficient                 insufficient
                │                           │
                │                  ┌────────▼─────────┐
                │                  │ EscalationPolicy │  ≤ 1 step,
                │                  └────────┬─────────┘  strictly stronger
                │                           ▼
                │              stronger strategy retrieval
                └─────────────┬─────────────┘
                              ▼
                   RetrievalResponse
                   retrieval_method="adaptive"
                   latency_ms = routing + Σ executed stages
                   routing = RoutingTrace
                              │
                              ▼
       ExperimentRunner → RoutingEvaluator (+ existing evaluators, unchanged)
```

Guarantees:

* **Composition, not reimplementation.** `adaptive.py` imports no concrete
  retriever and contains no `try`/`except`; `routing/` may not import `.indexing`,
  `.ingestion`, `.chunking`, or `.embeddings`. Enforced by architecture guards.
* **Common protocol.** `AdaptiveRetriever` conforms to `Retriever` and emits a
  `RetrievalResponse` with `retrieval_method="adaptive"` (paired with
  `retriever_version="adaptive_v1"`), so the evaluation pipeline stays
  strategy-agnostic.
* **Preserved provenance.** The winning stage's results, ranks, scores,
  metadata, and Phase 5 rerank diagnostics pass through untouched; only routing
  diagnostics are layered on top.
* **Honest cost.** `latency_ms` is analysis + routing + every stage that actually
  ran, so `EfficiencyEvaluator` prices an adaptive run with no special case.
* **Bounded.** The escalation ladder is a total order, so
  `BM25 → Dense → Hybrid → BM25` is impossible by construction, not merely
  untested. `max_escalation_steps` defaults to 1.
* **Runtime-only decisions.** Sufficiency is judged from the query and the
  retrieved text; it never reads benchmark labels or evaluation metrics.
* **Optional metrics.** `RoutingEvaluator` emits nothing when no trace carries
  routing, so Phase 2–5 runs keep byte-identical artifacts.

---

# 4a. Phase 6 Routing Configuration

`RoutingConfig` holds everything that can change a routing outcome, so it is
covered by `config_hash` and a run is reproducible from its recorded
`config.json`:

| Field | Purpose |
| --- | --- |
| `available_strategies` | which strategies the router may select |
| `rule_weights` | strategy → signal group → weight (may be negative) |
| `enabled_feature_groups` | which of the six signal groups are scored |
| `strategy_cost_ms` | measured per-strategy latency (Phase 5 §8) |
| `cost_weight` | quality-vs-cost trade-off knob (0.0 = pure evidence) |
| `sufficiency_*` | check enablement, threshold, min results, coverage floors |
| `score_floor` | optional per-strategy floor; `None` by default |
| `escalation_enabled` / `escalation_ladder` / `max_escalation_steps` | bounded escalation |

Restricting `available_strategies` also restricts which indexes are constructed,
which is how a BM25-only adaptive run stays fully offline.

---

# 15. Evaluation Architecture

Evaluation is an independent observer.

```text
System
   ↓
Raw Experiment Trace
   ↓
Evaluation
   ↓
Metrics
```

Evaluation is not embedded inside retrieval or generation.

It is intended to remain reusable for all future retrieval strategies.

---

# 16. Evaluation Dimensions

### Retrieval

Potential/current metrics include:

* Recall@k
* Precision@k
* Hit@k
* MRR
* nDCG

### Generation

Metrics include:

* lexical F1
* ROUGE-L
* cached model-based judging where configured
* correctness/relevance/groundedness-oriented evaluation

### Efficiency

Measure:

* retrieval latency
* generation latency
* total latency
* input tokens
* output tokens
* total tokens
* estimated cost

---

# 17. Experiment Architecture

The ExperimentRunner is retrieval-strategy agnostic.

Conceptually:

```text
Evaluation Dataset
        ↓
ExperimentRunner
        ↓
Retriever
        ↓
Context Builder
        ↓
Generator
        ↓
Raw Traces
        ↓
Evaluation
        ↓
Metrics + Artifacts
```

The same architecture should support future retrieval strategies.

---

# 18. Raw Experiment Traces

Per-query traces retain information needed for analysis and debugging:

```text
query
retrieval method
retrieved chunks
scores
ranks
reference/relevance information
answer
source IDs
latency
token usage
estimated cost
errors
configuration/version
```

On reranked runs the trace additionally carries the second-stage split
(`candidate_generation_latency_ms`, `rerank_latency_ms`,
`rerank_candidate_count`, `rerank_result_count`, `rerank_fallback`). All five
default to `None`, so traces recorded before Phase 5 still validate against the
schema unchanged.

On adaptive runs the trace additionally carries `routing`, a `RoutingTrace`
holding the query features, the routing decision with its per-strategy evidence
and confidence, the initial strategy, its latency and result count, the
sufficiency decision with every signal, the escalation decision with its reason,
the final strategy, the per-stage latencies, and the routing overhead. It
defaults to `None`, so all traces recorded before Phase 6 still validate.

Aggregate metrics should be traceable back to raw results.

---

# 19. Provider Boundaries

Provider-specific implementations are isolated.

```text
Embedding
    ↓
AICredits
    ↓
text-embedding-3-large
```

and:

```text
Generation
    ↓
Groq
    ↓
Generation Model
```

The rest of the application should use the project interfaces rather than
provider-specific APIs directly.

---

# 20. Runtime vs Experiment Path

Runtime path:

```text
User Query
    ↓
Retriever
    ↓
Context Builder
    ↓
Generator
    ↓
Answer + Sources
```

Experiment path:

```text
Evaluation Dataset
    ↓
Retriever
    ↓
Generator
    ↓
Evaluation
    ↓
Metrics
    ↓
Traces + Artifacts
```

These paths share core components but have different responsibilities.

---

# 21. Architectural Principle

The project follows this principle:

> Build fixed retrieval systems first, evaluate them independently, and only
> then build the adaptive system that chooses between them.

This prevents the adaptive system from being evaluated against poorly defined
or incomparable retrieval baselines.
