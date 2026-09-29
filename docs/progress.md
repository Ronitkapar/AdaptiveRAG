# AdaptiveRAG — Project Progress

## Project

AdaptiveRAG — Query-Aware Retrieval Optimization for Efficient RAG

Core research/engineering question:

> When does a query actually need expensive retrieval, and can we automatically
> choose an appropriate retrieval strategy?

The eventual system will compare:

* Dense retrieval
* BM25 / lexical retrieval
* Hybrid retrieval
* Reranked retrieval
* Adaptive retrieval

Evaluation will consider:

* Retrieval quality
* Answer quality
* Latency
* Token usage
* Estimated cost

---

# Phase Status

| Phase                              | Status                                      |
| ---------------------------------- | ------------------------------------------- |
| Phase 1 — Corpus Foundation        | COMPLETE                                    |
| Phase 2 — Fixed Dense-RAG Baseline | IMPLEMENTATION COMPLETE / OFFLINE VALIDATED |
| Phase 3 — BM25                     | COMPLETE                                    |
| Phase 4 — Hybrid                   | COMPLETE                                    |
| Phase 5 — Reranking                | FUTURE                                      |
| Phase 6 — Adaptive Routing         | FUTURE                                      |
| Phase 7 — Evaluation & Ablations   | FUTURE                                      |

---

# Phase 1 — Corpus Foundation

Status: COMPLETE

Completed:

* 14 research papers collected
* Canonical metadata manifest
* Deterministic downloader
* PDF validation
* SHA-256 validation
* Corpus tests
* Corpus documentation
* Raw PDFs excluded from Git

The corpus covers:

* Foundational RAG
* Dense retrieval
* Sparse retrieval
* Hybrid retrieval
* Reranking
* Advanced RAG
* Adaptive retrieval

Phase 1 deliberately did not perform:

* text extraction
* chunking
* embedding
* indexing
* retrieval
* generation

---

# Phase 2 — Fixed Dense-RAG Baseline

Status:

* Implementation: COMPLETE
* Offline validation: COMPLETE
* Live provider execution: PENDING USER-SIDE VALIDATION

Phase 2 established the fixed Dense-RAG baseline that later retrieval
strategies will be compared against.

## Implemented

### Ingestion

* PDF ingestion using `pdfplumber`
* Hierarchical Document representation
* Structure-aware element extraction
* Provenance preservation
* Heading classification and guards

### Chunking

* Structure-aware chunking
* Frozen chunking configuration
* Configurable overlap
* Hard-capped splitting
* Undersized sibling merging
* Atomic handling of tables/figures/equations where appropriate

Observed frozen corpus:

* 713 chunks
* Mean: approximately 408 tokens
* Median: approximately 447 tokens
* 106 atomic chunks
* Maximum: approximately 891 tokens

The largest chunk is intentionally retained because it is an unsplit table.

The initial chunking attempt produced excessive fragmentation and was
recalibrated before freezing the current corpus.

### Embeddings

* AICredits provider
* `text-embedding-3-large`
* Provider abstraction
* Content-addressed SQLite embedding cache

### Indexing

* Local Qdrant
* Cosine similarity
* Vector dimension validation
* Index metadata/versioning
* Compatibility handling for current Qdrant client behavior

### Retrieval

* `DenseRetriever`
* Fixed `top_k`
* Structured retrieval results
* Scores and provenance preserved
* Explicit failure semantics

### Generation

* Groq API
* Provider abstraction
* Context Builder
* Versioned prompts
* Structured `GenerationResult`
* Source/chunk IDs preserved
* Token usage and latency recorded

### Evaluation

* Independent evaluation dataset
* Retrieval metrics:

  * Recall@k
  * Precision@k
  * Hit@k
  * MRR
  * nDCG
* Generation metrics:

  * lexical F1
  * ROUGE-L
  * cached Groq judge
* Efficiency:

  * retrieval latency
  * generation latency
  * total latency
  * token usage
  * estimated cost
* Raw experiment traces
* Reproducible experiment configuration
* Experiment artifacts

### Validation

* 42 pytest tests pass
* Phase 1's 5 unittest tests pass
* Offline end-to-end smoke test passes
* Provider checks fail closed without credentials
* `.env` is ignored
* No secrets are committed
* Experiment run directories are ignored
* Guards confirm no BM25/hybrid/reranking/adaptive implementation was
  introduced during Phase 2

---

# Phase 3 — BM25 Lexical Retrieval

Status: COMPLETE

Implemented:

* Deterministic symmetric tokenizer (NFKD + diacritic stripping + lowercase +
  alphanumeric tokens)
* `BM25Index` — inverted index, non-negative Robertson IDF, Okapi BM25
  scoring (`k1=1.2`, `b=0.75`), JSON persistence with corpus-version staleness
  guard (`storage/bm25/bm25_index.json`)
* `BM25Retriever` — conforms to the `Retriever` protocol, native BM25 scores,
  1-indexed ranks, full metadata/provenance, typed failure vs. empty semantics
* Configuration consistency: `retrieval_method` ⇔ `retriever_version`
  (`dense ⇔ dense_v1`, `bm25 ⇔ bm25_v1`) enforced by validator and stamped
  into experiment `component_versions`
* `scripts/build_bm25_index.py` — indexes all 713 canonical chunks (no
  hard-coded count; 21,406-term vocabulary)
* `scripts/run_experiment.py --retriever {dense,bm25}` plus `--no-generation`
  for fully offline retrieval-only runs
* `scripts/compare_retrievers.py` — dense vs BM25 metrics table

Validation:

* 57 offline pytest tests pass (14 BM25 tests + 1 isolation guard added;
  all Phase 1/2 tests green)
* Architecture guards updated: `"bm25"` removed from forbidden tokens;
  dense/BM25 isolation enforced; future-phase bans retained
* BM25 baseline run persisted at `experiments/bm25_baseline_v1/`:
  Recall@5 = 0.7167, MRR = 0.6771, retrieval latency mean 1.48 ms
  (p50 1.30 ms, p95 2.39 ms), 0 retrieval failures over the 20-example
  `dense_eval_v1` benchmark
* Dense live baseline at `experiments/dense_baseline_v1/`: 20/20 traces `ok`,
  Recall@5 = 0.8417, MRR = 0.95, retrieval latency mean 751.51 ms
* Side-by-side comparison generated via `scripts/compare_retrievers.py`
  (corpus version, trace count, and retrieval method all consistent)
* Full results: `docs/phases/phase-3.md`

Not implemented (future phases): reranking, adaptive routing,
query classification.

---

# Phase 4 — Hybrid Retrieval (Dense + BM25 + RRF)

Status: COMPLETE

Implemented:

* `retrieval/fusion.py` — Reciprocal Rank Fusion, isolated and unit-testable
  with no retriever, index, or credential. Rank-based only
  (`1/(rrf_k + rank)`), dedup by `chunk_id`, deterministic tie-break
  `(-rrf_score, best_single_source_rank, chunk_id)`
* `retrieval/hybrid.py` — `HybridRetriever`, conforms to the `Retriever`
  protocol, composes the existing `DenseRetriever` and `BM25Retriever` without
  modifying either. Both branches are queried at `candidate_k` (20) and the
  fused list is truncated to `top_k` with ranks renumbered `1..n`
* `RetrievalConfig` support for `retrieval_method="hybrid"`, `rrf_k=60`,
  `candidate_k=20`, plus a dedicated `HybridRetrievalConfig`; the validator
  keeps `hybrid ⇔ hybrid_v1` aligned and rejects `candidate_k < top_k`
* `RetrievalMetadata` fusion diagnostics (`fusion_method`, `rrf_k`,
  `candidate_k`, per-branch candidate counts and latencies, `fusion_latency_ms`),
  all optional so dense/BM25 responses are unchanged on the wire
* `experiments/config.py` hybrid branch; the dense `vector_store` stays in the
  CLI's index slot so `count()` / `close()` keep working
* `scripts/run_experiment.py --retriever hybrid` with a
  `hybrid_baseline_v1` default name and a dual-index emptiness check
* `scripts/compare_retrievers.py --hybrid-run` — opt-in third column; output is
  byte-identical to today when the flag is absent
* No changes to `retrieval/dense.py`, `retrieval/bm25.py`, `indexing/*`,
  `evaluation/*`, `experiments/runner.py`, `rag.py`, or `scripts/ask.py`

Validation:

* 86 offline pytest tests pass (57 pre-existing unchanged in intent + 29 new
  Phase 4 tests), no credentials required
* RRF mathematics asserted exactly; dedup, single-source, unequal-length, and
  empty-list cases covered
* Failure semantics asserted: a failing branch propagates its original
  exception, and the runner records `retrieval_failed` with `retrieval: null` —
  never a dense-only or BM25-only result
* End-to-end offline integration through `ExperimentRunner` +
  `RetrievalEvaluator` + `EfficiencyEvaluator` with a real dense retriever
  (in-memory Qdrant + fake embeddings) and a real BM25 retriever over the same
  chunks
* Canonical-corpus regression: "Robertson", "ColBERT", and "BM25" surface the
  expected documents after fusion, and the fused list is a superset of neither
  branch alone
* Architecture guards: `"hybrid"` removed from `FORBIDDEN_STRATEGY_TOKENS`
  (Phase 5+ bans retained); the dense/BM25 isolation guard is unchanged and
  still holds; a new guard asserts `hybrid.py` orchestrates only and contains
  no `try`/`except`, and that `fusion.py` couples to neither branch
* Hybrid baseline run persisted at
  `experiments/20260929T175539Z-hybrid_baseline_v1/`: 20/20 traces, 0
  retrieval failures, Recall@5 = 0.8583, MRR = 0.8200, retrieval latency
  mean 721.00 ms (p50 516.65, p95 1975.13), `rrf_k=60`, `candidate_k=20`
* Per-branch cost from the traces: dense 688.96 ms, BM25 3.47 ms, fusion
  0.65 ms mean — fusion overhead is negligible and latency is dominated by the
  dense query embedding
* Three-way comparison generated via
  `scripts/compare_retrievers.py --hybrid-run` (corpus version, trace count,
  and retrieval method consistent across all three runs). Hybrid beats BM25 on
  every retrieval-quality metric, beats dense on Recall@5 (+0.0166) and
  Recall@10 (+0.0167) and ties on Hit@5, but loses to dense on Recall@1
  (−0.2000), Precision@1 (−0.2000), and MRR (−0.1300) — the expected RRF
  ranking-flattening trade-off on a 20-example benchmark
* Full results: `docs/phases/phase-4.md`

---

# Live Validation Status

Phase 1–4 live validation is now **complete**: credentials loaded
from `.env`, providers verified, embeddings generated (713 chunks), Qdrant
index built, dense baseline executed, BM25 baseline executed, hybrid baseline
executed, and the comparison tables generated.

Sequence used:

```text
check_providers.py
        ↓
embed_chunks.py → build_index.py → run_experiment.py --retriever dense
        ↓
build_bm25_index.py → run_experiment.py --retriever bm25
        ↓
compare_retrievers.py
```

Notes from live execution:

* Groq intermittently returned `429` during generation; the SDK's built-in
  retry with backoff recovered on every call (0 generation failures).
* The LLM judge initially failed on harder examples because `max_tokens=512`
  left no room for the reasoning model's internal reasoning before the JSON
  body; `evaluation/judge.py` now uses `max_tokens=4096`. Judge results are
  cached in `storage/eval_cache.sqlite3`, so repeats are free.

Phase 4 live validation is also **complete**: the hybrid baseline was executed
on a networked machine (20/20 traces, 0 retrieval failures) and the three-way
comparison was generated.

Sequence used:

```text
check_providers.py
        ↓
embed_chunks.py → build_index.py → run_experiment.py --retriever dense
        ↓
build_bm25_index.py → run_experiment.py --retriever bm25
        ↓
compare_retrievers.py
        ↓
run_experiment.py --retriever hybrid --no-judge --no-generation
        ↓
compare_retrievers.py --hybrid-run experiments/20260929T175539Z-hybrid_baseline_v1
```

Notes from live execution:

* Groq intermittently returned `429` during generation; the SDK's built-in
  retry with backoff recovered on every call (0 generation failures).
* The LLM judge initially failed on harder examples because `max_tokens=512`
  left no room for the reasoning model's internal reasoning before the JSON
  body; `evaluation/judge.py` now uses `max_tokens=4096`. Judge results are
  cached in `storage/eval_cache.sqlite3`, so repeats are free.
* The hybrid run inherits the dense branch's network cost and is **not**
  offline-capable, unlike `--retriever bm25`: it needs `AICREDITS_API_KEY`
  even with `--no-judge --no-generation`, because the dense branch embeds
  every query remotely.

---

# Current Project State

Phases 1–4 are complete as implementation, offline-validation, and live
validation milestones. All three fixed retrieval strategies (dense, BM25,
hybrid) operate over the same canonical 713-chunk corpus and are consumed
unchanged by the same strategy-agnostic runner and evaluators.

Recorded live metrics over `dense_eval_v1` (20 examples, corpus
`corpus_6c416f423920385d`):

| Metric     | Dense   | BM25    | Hybrid  |
| ---------- | ------- | ------- | ------- |
| Recall@1   | 0.8417  | 0.5417  | 0.6417  |
| Recall@5   | 0.8417  | 0.7167  | 0.8583  |
| Recall@10  | 0.8583  | 0.7833  | 0.8750  |
| Precision@1| 0.9500 | 0.6000  | 0.7500  |
| Hit@5      | 0.9500  | 0.8000  | 0.9500  |
| MRR        | 0.9500  | 0.6771  | 0.8200  |
| nDCG@5     | 0.8908  | 0.6631  | 0.7788  |
| Latency mean (ms) | 751.51 | 1.48 | 721.00 |

Summary of what the three fixed strategies show on this benchmark: dense leads
on top-rank precision and MRR; BM25 is ~500× cheaper but clearly weaker on
quality; hybrid dominates BM25 on every quality metric, edges dense on deep
recall (Recall@5/@10) while losing to it on rank-1 precision and MRR, and costs
essentially the same as dense because fusion itself takes 0.65 ms.

This is the fixed-strategy evidence base the adaptive system will be measured
against. The next development phase is:

> Phase 5 — Reranking

Do not implement Phase 5+ functionality until the phase is explicitly started.
