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
| Phase 5 — Reranking                | COMPLETE                                    |
| Phase 6 — Adaptive Routing         | IMPLEMENTATION COMPLETE / OFFLINE VALIDATED |
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
against.

---

# Phase 5 — Reranking (Second-Stage Cross-Encoder)

Status: COMPLETE — live benchmark executed

Full record: [`docs/phases/phase-5.md`](phases/phase-5.md).
Decision: ADR-020 in [`docs/decision.md`](decision.md).

Implemented:

* `reranking/base.py` — the `Reranker` protocol, deliberately isolated from
  every retriever, index, and corpus artifact. Input order is preserved, the
  returned length must equal the input length, and scores are documented as
  model-specific ranking signals rather than calibrated probabilities
* `reranking/onnx_backend.py` — `OnnxCrossEncoderReranker`, the only module that
  runs a model. ONNX Runtime over a pre-exported ms-marco cross-encoder
  (`Xenova/ms-marco-MiniLM-L-6-v2`, ~22 M params, ~90 MB ONNX). No torch, no
  transformers, and `sentence-transformers` stays banned. Batched (default 16),
  truncation at 512, explicit `cpu`/`cuda`/`auto` provider selection with a hard
  error rather than a silent CPU downgrade, `num_labels`-driven output decoding.
  The artifact is fetched on first use into `storage/reranker/`, and
  `onnxruntime`/`tokenizers` are imported lazily so importing the module never
  touches the network
* `retrieval/reranked.py` — `RerankedRetriever`, conforming to `Retriever`,
  wrapping **any** base retriever (dense, BM25, or hybrid). Queries the base at
  `rerank_candidate_k`, re-scores, sorts by
  `(-rerank_score, retrieval_rank, chunk_id)`, truncates to `top_k`, and
  renumbers ranks. `text`, `metadata`, and `provenance` pass through untouched;
  `retrieval_score` and `retrieval_rank` preserve the first-stage signal and
  position
* `retrieval_method` gains `dense_rerank` / `bm25_rerank` / `hybrid_rerank` with
  matching `retriever_version`s. `RetrievalConfig` gains eight `rerank_*`
  fields, all covered by `config_hash`; the validator enforces
  `rerank_candidate_k >= top_k` and, for hybrid, `candidate_k >=
  rerank_candidate_k`. A dedicated `RerankerConfig` model mirrors
  `HybridRetrievalConfig`
* `RetrievalMetadata` second-stage diagnostics and `ExperimentTrace` latency /
  count split, all optional with `None` defaults so existing run directories
  still validate
* `experiments/config.py` wraps the base retriever when `rerank_enabled`, with
  hybrid's fusion depth coupled to the rerank depth; the dense
  `vector_store` stays in the CLI's index slot so `count()` / `close()` keep
  working, and `run_experiment.py`'s hybrid index check unwraps through
  `retriever.base_retriever`
* `evaluation/efficiency.py` reports the split — but only when a trace carries
  it, so non-reranked runs' output is byte-identical
* `scripts/run_experiment.py --rerank` plus `--reranker-model`,
  `--reranker-revision`, `--rerank-candidate-k`, `--rerank-device`,
  `--rerank-batch-size`, `--rerank-max-length`, `--rerank-fallback`, and
  `{dense,bm25,hybrid}_rerank_v1` run-name defaults
* `scripts/compare_reranking.py` — base-vs-reranked table plus a depth-ablation
  table. It never auto-discovers a rerank run by glob; every run is supplied
  explicitly, and it exits 2 on corpus, trace-count, or method mismatch.
  `compare_retrievers.py` is untouched

Failure semantics:

* An empty query raises `InvalidQueryError` before the base retriever is
  touched
* An empty candidate pool is **not** a failure: `status="no_results"`,
  `rerank_latency_ms == 0.0`, and the reranker is never invoked
* A reranker exception propagates unchanged by default and becomes
  `status="retrieval_failed"` with the original error type. The single
  `try`/`except` in `reranked.py` is gated on `config.rerank_fallback`, and an
  AST guard asserts the handler both re-raises and references that flag. When
  the opt-in fallback is taken it sets `rerank_fallback=True`, visible in every
  trace, in `rerank_fallback_count`, and in the manifest
* A score/candidate length mismatch raises `RerankingError` rather than silently
  truncating the result list

Validation: **136 offline deterministic tests pass**, 2 deselected
(`integration`). No credentials, no network, no live indexes. Covers scoring and
ordering, `top_k` behaviour, candidate depth (12/20/40) and hybrid depth
coupling, the empty-pool no-model path, deterministic tie-breaking, batch-size
invariance, metadata/provenance/text passthrough, latency accounting, all four
failure paths, protocol conformance, score-contract enforcement, config
validation and hash coverage, offline integration through `ExperimentRunner`
for all three strategies, canonical-corpus regression, byte-identical output
with reranking disabled, old run artifacts still validating, the comparison CLI
driven as a subprocess, and four new architecture guards — each verified to fail
against a deliberately broken variant rather than passing vacuously.
`retrieval/dense.py`, `retrieval/bm25.py`, `retrieval/hybrid.py`, and
`retrieval/fusion.py` are unmodified.

Live benchmark: **COMPLETE.** Executed on a networked, CPU-only host. The
declared reranking dependencies installed cleanly (`onnxruntime` 1.30.0,
`tokenizers` 0.23.2, `huggingface-hub` 1.33.0), and the documented blocking
assumption was confirmed rather than assumed: `Xenova/ms-marco-MiniLM-L-6-v2`
resolves at revision `a09144355adeed5f58c8ed011d209bf8ee5a1fec` with
`onnx/model.onnx`, `tokenizer.json`, and `config.json`. No model substitution.

Running the integration-marked ONNX test first, as the phase plan advised, paid
for itself by surfacing three latent defects that the 136 offline tests could not
reach, because the live path had never actually executed:

1. `tests/conftest.py` added an unconditional `skip` marker to every
   integration-marked item, so the documented `pytest -m integration
   tests/test_reranking.py` reported `2 skipped` on any machine. A command-line
   `-m` overrides the `addopts` `-m 'not integration'`, so the guard now yields
   when the caller opts in, and still skips for `-m "not integration"`. Default
   runs are unchanged at `136 passed, 2 deselected`.
2. The live relevance assertion compared a score with itself against a zero
   baseline (`scores[...] > scores[0] * 0`), so it could never rank anything.
   This model's head is a single raw regression logit, negative for non-matches.
   It now asserts the ordering its comment described.
3. `snapshot_download` was unwrapped, leaking a raw `huggingface_hub` exception
   on a missing repository instead of the documented `RerankerModelError`;
   download failures are now translated into the project's typed error with the
   original chained as `__cause__`.

After the fixes the integration tests pass against a real ONNX session:
**2 passed**, offline **136 passed**.

Results, recorded **as observed**:

* The six primary configurations and the k = 10 / 20 / 40 depth ablation all ran
  retrieval-only on corpus `corpus_6c416f423920385d` and the 20-example
  benchmark; 20/20 traces each, `retrieval_failed: 0` and
  `rerank_fallback_count: 0` in every run. `compare_reranking.py` verified
  corpus, trace count, and retrieval method across all three pairs.
* **Reranking did not pay for itself on this corpus.** Recall@10 improved or held
  on all three strategies (dense +0.0167, BM25 +0.0250, hybrid +/-0.0000), but
  every head metric fell on all three: MRR -0.0833 / -0.0609 / -0.1274, nDCG@5
  -0.1270 / -0.1484 / -0.1507, Precision@5 -0.1600 / -0.1500 / -0.1500. In the
  hybrid run the reranker pulled a candidate from beyond the first-stage top-10 in
  20/20 traces, yet kept the first-stage #1 in only 3/20, so it is genuinely
  reordering with a poorly-calibrated signal rather than no-op'ing. The likely
  cause is domain mismatch: a general-web MS-MARCO cross-encoder scoring dense
  academic RAG prose. That is a finding about the model choice, not the
  composition-layer design, and substituting a domain-specific cross-encoder is
  out of Phase 5 scope.
* **The depth curve runs the wrong way for the usual second-stage story.** R@5
  falls 0.7917 -> 0.7583 -> 0.7333 as the candidate pool widens 10 -> 20 -> 40
  while rerank latency climbs 3.00 s -> 3.49 s -> 7.67 s. `k=10` is both the
  cheapest and the most accurate reranked configuration measured.
* **The documented cost estimate was wrong by two orders of magnitude.** ~155 ms
  per candidate pair on CPU (10 pairs 1 519 ms, 20 pairs 3 209 ms, 40 pairs
  6 221 ms) against a predicted "tens of milliseconds". The 20-candidate
  configuration costs ~4.0 s end to end versus BM25's 1.7 ms, and roughly 11x
  dense's own 621 ms. The cost is linear in candidate count and intrinsic to CPU
  inference on 512-token sequences; batch-size invariance is asserted in tests
  and no fallback fired.
* The §7.3 latency comparability caveat is resolved: dense and BM25 were re-run
  retrieval-only (`dense_baseline_ro_v1`, `bm25_baseline_ro_v1`) and both
  reproduce their original retrieval metrics exactly, so the confound was the
  generation/judge harness rather than retrieval. All six columns are now
  retrieval-only. The hybrid baseline was already retrieval-only on the same
  corpus with the same 20 traces and `top_k=10`, so it was reused unchanged.
* Full results, per-run provenance, and interpretation:
  `docs/phases/phase-5.md` §8.

This is the fixed-strategy evidence base the adaptive system will be measured
against. The next development phase is:

> Phase 6 — Adaptive Routing

Do not implement Phase 6+ functionality until the phase is explicitly started.
Query classification, strategy selection, and adaptive routing remain
architecture-guard banned (`adaptive_rout`, `query_classif`, `strategy_select`).
`RerankedRetriever` is deliberately not conditional on anything — it never
decides whether reranking is worth it, for which query type, or at what
candidate depth. Those questions are Phase 6's.

Phase 5's measured result sharpens exactly those questions. Reranking is
uniformly unprofitable on this corpus at every depth measured, and its cost is
strongly depth-dependent, so "should we rerank?" and "at what depth?" are
routinely *no* here. Phase 6 must decide that from query characteristics rather
than assuming a second stage helps.

---

# Phase 6 — Adaptive Routing

Status: IMPLEMENTATION COMPLETE / OFFLINE VALIDATED

Phase 6 adds the adaptive decision layer that Phases 2–5 were built to be
measured by. It decides *which* existing retrieval strategy a query needs, and
whether the returned evidence justifies spending more.

## Implemented

### Routing package (`src/adaptive_rag/routing/`)

* `base.py` — `QueryAnalyzer` and `Router` protocols. `Router.route` consumes
  `QueryFeatures`, not the raw string, which is what makes a learned router
  substitutable without changing retrieval.
* `analyzer.py` — `QueryFeatureAnalyzer`: a deterministic, dependency-free
  extractor over frozen, version-stamped lexicons. Query length, content terms,
  lexical density, entity/exact-match indicators (quoted spans, acronyms,
  CamelCase, identifiers), technical terminology, semantic indicators, question
  type (10-way), concept count, comparison indicators, and a bounded
  `complexity_score`. No stemming, no POS tagging, no model, no network.
* `rule_based.py` — `RuleBasedRouter`: `score = Σ(weight × signal) −
  cost_weight × normalised_cost`, argmax, deterministic tie-break on
  `STRATEGY_ORDER`. Every score decomposes into recorded contributions.
* `sufficiency.py` — `SufficiencyChecker`: `result_count`, `lexical_coverage`,
  `top1_coverage`, optional per-strategy `score_floor`. Label-free by
  construction and guard-enforced.
* `escalation.py` — `EscalationPolicy`: a total-order ladder from configuration.
  `max_escalation_steps` defaults to 1.

### Adaptive retriever

`retrieval/adaptive.py` — `AdaptiveRetriever`, conforming to the existing
`Retriever` protocol so the runner and both existing evaluators consume it
unchanged. Orchestrates the four collaborators; owns no ranking logic and holds
**zero** `try`/`except` blocks, so a failing strategy surfaces as
`retrieval_failed` exactly like a failing hybrid branch.

### Contracts and wiring

* `schemas/routing.py` — `QueryFeatures`, `StrategyEvidence`, `RoutingDecision`,
  `SufficiencySignal`, `SufficiencyDecision`, `EscalationDecision`, `RoutingTrace`.
* `RoutingConfig` — rule weights, measured cost table, sufficiency thresholds,
  escalation ladder, and per-group ablation switches, all inside the hashed config.
* `retrieval_method="adaptive"` ⇔ `adaptive_v1`, as `model_copy` preserves the
  winning stage's provenance, scores, and rerank diagnostics while layering on
  the routing trace and adaptive diagnostics.
* `ExperimentTrace.routing`; one line in the runner copies it.
* `RoutingEvaluator`, emitting metrics **only** when traces carry routing, so
  fixed-strategy runs stay byte-identical.
* `run_experiment.py --retriever adaptive` plus routing flags;
  `--retriever adaptive --rerank` is rejected, because reranking is now a routing
  decision rather than a flag.

## Validation

* **221 offline deterministic tests pass** (140 pre-existing + 83 new), 2
  integration-marked deselected, **14/14 architecture guards** green.
* A real offline adaptive run over the canonical 713-document BM25 index with no
  credentials and no network: 20/20 traces, `retrieval_failed: 0`, Recall@5
  **0.7167** — identical to the fixed BM25 baseline — at **1.48 ms** mean
  retrieval latency plus **0.34 ms** routing overhead. The escalation path was
  separately verified against the same index and correctly fires only when the
  threshold judges real evidence insufficient.
* **Backward compatibility:** all **220** pre-Phase-6 traces across the recorded
  Phase 1–5 runs still validate with `routing is None`, and the recorded
  `metrics.json` files contain no routing section.

## Findings and limitations

* Thresholds (`sufficiency_threshold=0.5`, `cost_weight=0.25`) are reasoned
  defaults, **not** tuned values. Calibrating them against labels is Phase 7
  Ablation 3 work; Phase 6 records the limitation rather than implying optimality.
* The dense, hybrid, and hybrid+reranker arms are **unvalidated here** — this
  environment has no network and those arms need `AICREDITS_API_KEY` for query
  embeddings, as in Phases 4–5. Only the BM25-only arm was executed.
* The rule table is hand-designed. A learned router is designed for but not
  implemented; training belongs to Phase 7.

## Next steps

> Phase 7 — Evaluation & Ablations

Phase 7 owns the adaptive-vs-fixed comparison, the ablation program
(fixed vs adaptive, query-only vs feedback, no-escalation vs bounded escalation,
rule-based vs learned, and signal-group ablations), failure-mode analysis, and
threshold calibration. Phase 6 deliberately implements none of it: the current
phase is the adaptive mechanism itself — correct, testable, observable, and
runnable.
