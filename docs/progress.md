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
| Phase 4 — Hybrid                   | NEXT                                        |
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

Not implemented (future phases): hybrid/RRF, reranking, adaptive routing,
query classification.

---

# Live Validation Status

Phase 2 and Phase 3 live validation is now **complete**: credentials loaded
from `.env`, providers verified, embeddings generated (713 chunks), Qdrant
index built, dense baseline executed, BM25 baseline executed, and the
side-by-side comparison generated.

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

---

# Current Project State

Phases 1–3 are complete as implementation, offline-validation, and live
validation milestones. Both fixed retrieval baselines (dense and BM25) are
established over the same canonical 713-chunk corpus, evaluated by the same
strategy-agnostic pipeline, and compared side-by-side
(`scripts/compare_retrievers.py`: dense MRR 0.95 vs BM25 MRR 0.6771;
dense Recall@5 0.8417 vs BM25 0.7167; dense latency 751.51 ms vs BM25 1.48 ms).

The next development phase is:

> Phase 4 — Hybrid Retrieval (e.g. Reciprocal Rank Fusion over dense + BM25)

Do not implement Phase 4+ functionality until the phase is explicitly started.
