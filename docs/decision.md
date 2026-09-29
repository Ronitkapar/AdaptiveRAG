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
