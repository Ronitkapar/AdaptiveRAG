# AdaptiveRAG — Architecture Decisions

This file contains the current accepted architectural decisions.

It is not a conversation history.

When a decision changes, update the existing decision rather than continuously
appending historical discussion.

---

## ADR-001 — Progressive Retrieval Strategy Development

Status: ACCEPTED

Retrieval strategies will be implemented progressively:

```text
Dense
→ BM25
→ Hybrid
→ Reranking
→ Adaptive Routing
```

Fixed retrieval strategies must be established before adaptive routing.

---

## ADR-002 — Phase 2 Is a Fixed Dense-RAG Baseline

Status: ACCEPTED

Phase 2 implements only a fixed Dense-RAG pipeline.

It does not implement:

* BM25
* Hybrid retrieval
* Reranking
* Adaptive routing
* Query classification
* Dynamic strategy selection

The purpose is to establish a reproducible baseline for later comparison.

---

## ADR-003 — Structure-Aware Documents and Chunks

Status: ACCEPTED

PDFs are converted into hierarchical Documents containing structured elements
and provenance.

Chunking is structure-aware and preserves:

* document identity
* section/heading information
* source element IDs
* page information
* provenance

The canonical Chunk does not contain embeddings.

---

## ADR-004 — Provider Isolation

Status: ACCEPTED

Embedding and generation providers are separate concerns and are hidden behind
their own interfaces.

Phase 2:

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
Groq API
    ↓
Generation model
```

The rest of the application must not depend directly on provider-specific
implementation details.

---

## ADR-005 — Qdrant as Phase 2 Vector Store

Status: ACCEPTED

Phase 2 uses local Qdrant for dense vector storage.

The application interacts with a vector-store abstraction rather than
coupling retrieval logic directly to Qdrant.

---

## ADR-006 — Fixed Dense Retrieval

Status: ACCEPTED

The Phase 2 DenseRetriever uses a fixed `top_k`.

Phase 2 does not dynamically choose:

* retrieval strategy
* retrieval depth
* query routing

Retrieved scores and provenance are preserved for later analysis.

---

## ADR-007 — Retrieval Strategy Abstraction

Status: ACCEPTED

Retrieval strategies should be independently replaceable behind a common
retrieval abstraction.

The same general retrieval contract should eventually support:

```text
DenseRetriever
BM25Retriever
HybridRetriever
RerankedRetriever
AdaptiveRetriever
```

---

## ADR-008 — Evaluation Is Strategy-Agnostic

Status: ACCEPTED

Evaluation is independent from individual retrieval implementations.

The same evaluation infrastructure should eventually evaluate:

* Dense
* BM25
* Hybrid
* Reranked
* Adaptive

Evaluation should not need to be rewritten for each retrieval strategy.

---

## ADR-009 — Independent Evaluation Dataset

Status: ACCEPTED

The evaluation dataset is independent of the retrieval implementation.

Relevance information must not be defined merely from whichever retriever
happens to return a chunk.

The initial benchmark is curated because the corpus contains only 14 papers.

---

## ADR-010 — Experiment Runner

Status: ACCEPTED

Experiments are executed through a retrieval-strategy-agnostic
`ExperimentRunner`.

The runner records:

* configuration
* raw traces
* metrics
* artifacts

Aggregate metrics should remain traceable to per-query results.

---

## ADR-011 — Runtime and Experiment Separation

Status: ACCEPTED

Runtime querying and experimental evaluation remain separate concerns.

Runtime:

```text
Query
 ↓
Retriever
 ↓
Context Builder
 ↓
Generator
 ↓
Answer
```

Experiment:

```text
Evaluation Dataset
 ↓
Retriever
 ↓
Generator
 ↓
Evaluation
 ↓
Metrics + Traces + Artifacts
```

---

## ADR-012 — Provenance Preservation

Status: ACCEPTED

Provenance must survive from source PDF through:

```text
PDF
 ↓
Document
 ↓
Element
 ↓
Chunk
 ↓
Retrieval Result
 ↓
Context
 ↓
Answer Sources
```

No component should silently discard source identity when it is still
available.

---

## ADR-013 — Structured Data Contracts

Status: ACCEPTED

Important component boundaries use structured schemas.

Pydantic is used for project data contracts.

Examples include:

* Document
* Element
* Chunk
* RetrievalResult
* RetrievalResponse
* GenerationRequest
* GenerationResult
* Experiment configuration
* Evaluation records

---

## ADR-014 — Deterministic Testing

Status: ACCEPTED

Unit and integration tests should avoid unnecessary dependence on live
external APIs.

External providers should be mocked/faked where appropriate.

Live provider execution belongs to explicit validation or experiment runs.

---

## ADR-015 — Stage-Based Processing

Status: ACCEPTED

The main data pipeline is explicitly staged:

```text
PDF
 ↓
Document
 ↓
Chunk
 ↓
Embedding
 ↓
Index
 ↓
Retrieval
 ↓
Context
 ↓
Generation
```

Intermediate stages should remain inspectable and reproducible where practical.

---

## ADR-016 — No Core RAG Framework Dependency

Status: ACCEPTED

The project maintains its own component interfaces and architecture.

LangChain/LlamaIndex are not core architectural dependencies.

Additional frameworks should not be introduced unless there is a clear need.

---

## ADR-017 — Phase Boundaries

Status: ACCEPTED

Only the functionality belonging to the active phase should be implemented.

Current state:

```text
Phase 1 → COMPLETE
Phase 2 → COMPLETE
Phase 3 → COMPLETE
Phase 4 → COMPLETE
Phase 5 → IMPLEMENTED (live benchmark pending)
```

Future functionality must not be implemented prematurely.

---

## ADR-018 — Independent BM25 Retrieval Strategy and Indexing Architecture

Status: ACCEPTED

Phase 3 introduces BM25 as an independent lexical retrieval strategy:

* BM25 indexes the **same canonical chunk corpus** as dense retrieval
  (`data/processed/chunks/*.chunks.jsonl`) without modifying it or assuming a
  fixed chunk count.
* `BM25Retriever` and `DenseRetriever` are fully decoupled: neither imports,
  calls, nor falls back to the other (architecture-guard enforced).
* Both conform to the common `Retriever` protocol and return the same
  `RetrievalResponse` contract, so evaluation remains strategy-agnostic.
* Scores are **native, unnormalized Okapi BM25 scores** (`k1=1.2`, `b=0.75`,
  non-negative Robertson IDF); no artificial normalization is applied.
* Preprocessing is deterministic and symmetric between indexing and querying
  (Unicode NFKD, diacritic stripping, lowercase, alphanumeric tokens).
* The BM25 index persists as JSON under `storage/bm25/` with a
  `corpus_version` staleness guard; a missing or corrupt index fails fast with
  typed errors, while a valid query with no lexical overlap returns
  `status="no_results"`.
* `retrieval_method` and `retriever_version` remain strictly aligned
  (`dense ⇔ dense_v1`, `bm25 ⇔ bm25_v1`) across configuration, execution
  traces, and manifests.
* No hybrid fusion/RRF, reranking, or routing logic is included; results are
  structured for later Phase 4 consumption.

---

## ADR-019 — Hybrid Retrieval as Rank Fusion of Existing Retrievers

Status: ACCEPTED

Phase 4 adds hybrid retrieval as a **composition layer**, not a new retrieval
engine:

* `HybridRetriever` receives two objects that already satisfy the `Retriever`
  protocol and queries them. It holds no index, no embedding model, and no
  tokenizer, and it does not modify `retrieval/dense.py` or
  `retrieval/bm25.py`. The existing dense/BM25 isolation guard remains
  unchanged and is the proof that composition did not leak backward.
* Fusion is **Reciprocal Rank Fusion**:
  `score(chunk) = Σ_lists 1 / (rrf_k + rank_in_that_list)`. RRF is rank-based
  only, so cosine similarity and BM25 magnitudes are never mixed and no score
  normalization is introduced. The canonical RRF constant is `rrf_k = 60`.
  Weighted, CombSUM, and CombMNZ variants are deliberately **not**
  implemented; there is no `FusionStrategy` base class and no registry.
* Deduplication is by `chunk_id`, never by text. A document missing from a
  list contributes nothing: no padding, no fabricated rank, no fabricated
  score. Result ordering is fully deterministic with the tie-break
  `(-rrf_score, best_single_source_rank, chunk_id)`, so output never depends on
  input or dict iteration order.
* When a chunk appears in both branches, the dense result supplies
  `text` / `metadata` / `provenance` and only `score` is replaced; a chunk found
  only by BM25 keeps BM25's payload. `RetrievalResult` itself is unchanged — no
  per-result diagnostic fields were added, so `score` is the RRF score and
  `rank` is the fused rank.
* `RetrievalMetadata.latency_ms` is the **full end-to-end hybrid cost** (both
  constituent calls plus fusion), because it is the single latency the
  `EfficiencyEvaluator` consumes. Per-branch latencies, per-branch candidate
  counts, and fusion time are reported as additional optional fields.
* Constituent exceptions propagate **unchanged** — no re-wrapping, no
  `try`/`except` in the module. A failing branch produces
  `status="retrieval_failed"` with the original error type; hybrid can never be
  silently downgraded to a dense-only or BM25-only answer.
* Hybrid does **not** forward `score_threshold` to either branch: a cosine
  threshold and a BM25 threshold are on incomparable scales, and applying one
  to both would silently truncate a single candidate list. The configured value
  is echoed into metadata for traceability only. `filters` *are* forwarded to
  both branches.
* Both branches are queried at `candidate_k` (default 20) — deeper than the
  returned `top_k` — and truncation happens only after fusion, so fusion always
  has enough material to work with.
* `rrf_k`, `candidate_k`, and `top_k` live inside `RetrievalConfig` and are
  covered by `config_hash`, and `hybrid ⇔ hybrid_v1` follows the same
  `retrieval_method` / `retriever_version` alignment rule as Phases 2 and 3.
* Hybrid reuses the existing Qdrant collection and persisted BM25 index. There
  is no third index, no `storage/hybrid/`, and no build script. Because the
  dense branch embeds every query, a hybrid run requires `AICREDITS_API_KEY`
  even with `--no-judge --no-generation`; unlike `--retriever bm25`, it is not
  offline-capable.

Measured outcome (`experiments/20260929T175539Z-hybrid_baseline_v1`, 20
examples, corpus `corpus_6c416f423920385d`): hybrid beats BM25 on every
retrieval-quality metric, and beats dense on Recall@5 (+0.0166) and Recall@10
(+0.0167) while losing on Recall@1 (−0.2000), Precision@1 (−0.2000), and MRR
(−0.1300). Fusion itself costs 0.65 ms mean against a 721 ms end-to-end total,
so hybrid's latency is effectively dense's.

The rank-flattening behavior is the expected consequence of the rank-based
method chosen above, and it is the reason rank-fusion-only is accepted as a
Phase 4 baseline rather than as a final answer: a chunk that only one branch
ranks highly accumulates a single small term and falls below chunks both
branches agree on, so fusion improves coverage and degrades the single best
guess. Correcting top-rank ordering is left to later phases.

---

## ADR-020 — Reranking as an Independent Second Stage

Status: ACCEPTED

Phase 5 adds second-stage cross-encoder reranking as a **composition layer**,
not as a new retrieval strategy of its own:

* The `Reranker` protocol (`reranking/base.py`) is deliberately isolated from
  every retriever, index, and corpus artifact: it receives a query and passages
  and returns one score per passage. It holds no vector store, no BM25 index, no
  embedding model, and no corpus handle.
* `RerankedRetriever` wraps **any** first-stage `Retriever` — dense, BM25, or
  hybrid — and only reorders what that retriever already returned. It never
  imports `retrieval.dense`, `retrieval.bm25`, `retrieval.hybrid`, or
  `retrieval.fusion`, and `retrieval/dense.py`, `retrieval/bm25.py`,
  `retrieval/hybrid.py`, and `retrieval/fusion.py` remain unmodified.
* `retrieval_method` gains `dense_rerank`, `bm25_rerank`, and `hybrid_rerank`,
  each paired with a matching `retriever_version`. A manifest therefore names
  the whole pipeline, not just its head, and all six Phase 1–5 configurations
  are unambiguous in a comparison.
* **The runtime is ONNX Runtime over a pre-exported cross-encoder**
  (`Xenova/ms-marco-MiniLM-L-6-v2`, ~22 M params, ~90 MB ONNX). No torch, no
  `transformers`, and `sentence-transformers` remains a banned dependency, so
  the heavy-dependency ban from ADR-016 stands unweakened. `onnxruntime` and
  `tokenizers` are imported lazily inside the backend's loader, so importing the
  module never touches the network and offline unit tests never depend on a
  model download.
* `score` holds the rerank score, mirroring Phase 4's rule that `score` is the
  signal that produced the current order. The first-stage signal is preserved
  separately as `retrieval_score` and the pre-rerank position as
  `retrieval_rank`. Rerank scores are **never** averaged, normalized, or blended
  with cosine / BM25 / RRF values; they are model-specific ranking signals, not
  calibrated probabilities.
* Ordering is `(-rerank_score, retrieval_rank, chunk_id)`. The pre-rerank rank
  and then the chunk id make the order total, so equal scores never depend on
  input order and repeated runs are stable. Scoring is per-pair with no
  cross-pair interaction, so output is batch-size invariant — asserted in tests
  rather than assumed.
* **Failure is fail-visible by default.** A reranker exception propagates
  unchanged and becomes `status="retrieval_failed"` with the original error type.
  The only `try`/`except` in `reranked.py` is gated on `config.rerank_fallback`,
  and an AST guard asserts the handler both re-raises and references that flag,
  so silent degradation is structurally impossible. When the fallback *is* taken,
  it sets `rerank_fallback=True`, which surfaces in every trace, in
  `rerank_fallback_count`, and in the manifest. An empty candidate pool is
  handled separately and is not a failure: no model invocation happens at all.
* Cost is recorded per stage. `RetrievalMetadata`, `ExperimentTrace`, and
  `EfficiencyEvaluator` all carry the candidate-generation / rerank split, and
  the new efficiency metrics are emitted **only** when a trace carries them, so
  existing runs' `metrics_efficiency.json` is byte-identical. All new trace and
  retrieval-metadata fields default to `None`, so previously recorded run
  directories still validate against the extended schemas.
* `score_threshold` is **not** forwarded to the base retriever: a first-stage
  score threshold and a second-stage ranker are different concerns, and applying
  one here would silently shrink the candidate pool before the ranker ever sees
  it. The configured value is echoed into metadata for traceability only,
  mirroring the Phase 4 §4.5 decision. `--rerank-candidate-k` also raises
  hybrid's own `candidate_k` so one flag controls depth end-to-end.
* `compare_retrievers.py` stays frozen, and Phase 5 tooling lives in a new
  `compare_reranking.py` that never auto-discovers a rerank run by glob — a
  rerank run is only comparable against the exact baseline it was built from.
* `RerankedRetriever` is deliberately **not** conditional on anything. It never
  decides whether reranking is worth it, for which query type, or at what
  candidate depth. Those decisions belong to Phase 6, which remains guard-banned
  (`adaptive_rout`, `query_classif`, `strategy_select`).

---

## ADR-021 — Phase 6 Is an Adaptive Decision and Orchestration Layer

Status: ACCEPTED

Phase 6 decides **when** each existing retrieval capability is used and when
additional retrieval computation is justified. It does not:

* rewrite dense retrieval, BM25, hybrid/RRF, or the reranker
* introduce a new vector database or a new retrieval algorithm
* introduce a general-purpose agent framework or unbounded retrieval loops
* read ground-truth relevance labels at query time

`src/adaptive_rag/retrieval/adaptive.py` contains **zero** `try`/`except`
blocks and no `retrieval/dense.py`, `.bm25`, `.hybrid`, `.reranked`, or
`.fusion` imports; `src/adaptive_rag/routing/` may not import `.indexing`,
`.ingestion`, `.chunking`, or `.embeddings`. Each constraint is enforced by an
AST/import guard in `tests/test_architecture_guards.py`, not by convention.

Routing decisions are **bounded**: at most one escalation, chosen from a total
order. This keeps per-query cost measurable and prevents the adaptive layer from
degenerating into an unbounded retrieval agent.

---

## ADR-022 — Routing Is a Structured Decision, and `adaptive` Is a Pipeline Identity

Status: ACCEPTED

A router returns a `RoutingDecision`, never a bare strategy name. It carries the
selected strategy, confidence, per-strategy evidence with weighted contributions,
`candidate_k`, `final_top_k`, the reranking requirement, the router/analyzer
versions, enabled and disabled signal groups, the cost weight, and routing
metadata. The decision is therefore reproducible, loggable, and comparable
between router implementations.

`RoutingConfig` holds the rule weights, the measured cost table, the sufficiency
thresholds, and the escalation ladder. Everything that can change a routing
outcome lives in the hashed configuration, so a run is reproducible from its
recorded `config.json`.

`RetrievalResponse.retrieval_method` gains the `adaptive` literal, paired with
`retriever_version="adaptive_v1"`, preserving the existing ⇔ invariant. As with
Phase 5's `hybrid_rerank`, the method names the **whole pipeline** rather than its
head: results, ranks, scores, provenance, and the rerank diagnostics come from
the winning stage unchanged, while the routing trace records what actually ran.
The response's `latency_ms` is the true end-to-end cost — analysis, routing, and
every stage executed — so `EfficiencyEvaluator` prices an adaptive run with no
special-casing.

The router decides **what** runs; the selected retriever decides **how**. The
runner change is a single line copying `metadata.routing` into the trace.

---

## ADR-023 — Routing Confidence Is Not a Calibrated Probability, and Does Not Gate Sufficiency

Status: ACCEPTED

`RoutingDecision.confidence` is the normalized margin between the winning and
runner-up strategy scores, in `[0, 1]`. It expresses how strongly the available
evidence supports the selected strategy. It is **not** a probability, no
calibration is performed, and it must not be treated as one. A single available
strategy yields `0.0`, because there was no evidence to discriminate on.

Confidence never triggers escalation on its own. Escalation requires a separate,
evidence-based judgement from `SufficiencyChecker`, which inspects the retrieved
results themselves. Keeping the two separate is deliberate: a router can be
confidently wrong, and retrieval can fail quietly regardless of how confident the
router was.

---

## ADR-024 — Sufficiency Is Judged at Query Time from Retrieved Evidence Alone

Status: ACCEPTED

`SufficiencyChecker` answers "was the retrieved evidence enough?" using signals
available at query time — result count, lexical coverage of query content terms in
the returned text, and top-1 coverage. It never reads benchmark labels, held-out
relevance judgements, reference answers, or evaluation metrics; those do not exist
outside an experiment, and a check that depended on them would look excellent
offline while being meaningless in production. `tests/test_architecture_guards.py`
enforces this by banning label identifiers in `routing/sufficiency.py`.

There is **no global score floor** by default. BM25 magnitudes, cosine
similarity, RRF scores, and cross-encoder logits live on incomparable scales, so
a single threshold across strategies would be an invented number rather than a
measured one. `RoutingConfig.score_floor` remains available for a per-strategy
threshold that has actually been measured.

The default `sufficiency_threshold` (0.5) is a reasoned default, **not** a tuned
value. Calibrating it against labels is Phase 7 work (Ablation 3); Phase 6
records the limitation rather than implying the threshold is optimal.

---

## ADR-025 — Escalation Follows a Total-Order Ladder Drawn from Configuration

Status: ACCEPTED

The escalation ladder is a total order over strategies, taken from
`RoutingConfig.escalation_ladder` rather than scattered per-strategy transition
rules. A strategy therefore has exactly one successor, or none at the top rung.

This makes the forbidden

```text
BM25 → Dense → Hybrid → BM25 → …
```

impossible **by construction**, not merely untested. `RoutingConfig` rejects a
ladder containing duplicates, naming an unavailable strategy, or exceeding the
reachable range, and `EscalationPolicy` re-validates it at construction.

`max_escalation_steps` defaults to **1**: one bounded escalation per query. An
escalated query returns the stronger stage's results and does **not** merge the
two rankings — merging two rankings is a new fusion algorithm belonging to Phase
4's concerns, not to an orchestration layer. A one-rung ladder has nowhere to
escalate, so the validator couples the ladder length to the step bound.

---

## ADR-026 — The Router Is an Interface, and the Learned Router Is Deferred

Status: ACCEPTED

`Router.route(features, *, top_k) -> RoutingDecision` takes `QueryFeatures`
rather than the raw query string. A learned router can therefore implement the
same protocol and be substituted without changing retrieval, sufficiency, or
escalation — which is what makes the rule-based-versus-learned comparison an
experiment about the router alone.

Phase 6 implements only the rule-based router, and **no ML classifier is
trained**. `QueryFeatureAnalyzer` is deliberately not linguistic analysis: no
stemming, part-of-speech tagging, or parsing. It counts surface cues with frozen,
version-stamped lexicons, keeping analysis cheap, deterministic, and
interpretable. Learned-router training and routing-dataset generation belong to
Phase 7.
