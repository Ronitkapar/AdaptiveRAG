# AdaptiveRAG — Project Progress

## Final Status

```text
Status:
Research investigation completed.
Adaptive-routing experimentation paused.
No final optimal routing policy claimed.
Project consolidated for documentation, presentation, and future research.
```

Phases 1–15 are complete. The central research record is
[`ADAPTIVERAG_REPORT.md`](ADAPTIVERAG_REPORT.md); the phase-by-phase chronology
is [`history/experiment_timeline.md`](history/experiment_timeline.md). The
sections below are the full chronological project log, preserved as written
during the investigation.

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
| Phase 7 — Evaluation & Ablations   | **CLOSED** (gates complete; DoD items disclosed as limitations) |
| Phase 8 — Corpus Fix Study         | QUALITY STUDY COMPLETE / COST MEASURED, NOT FROZEN |
| Phase 9 — Strategy Sensitivity     | COMPLETE — verdict `PROMISING_SIGNAL` (Gates 9.1–9.5) |
| Phase 10 — Post-Dense Escalation   | COMPLETE — verdict `FAILURE` (honest negative; frozen policy underperforms dense-only on test) |
| Phase 11 — Missing-Signal Analysis | COMPLETE — outcome `B` (mechanism found, evidence insufficient; no candidate, no intervention) |
| Phase 12 — Research Boundary      | COMPLETE — recommendation `STOP CURRENT ADAPTIVE-ROUTING TRACK` (position C; re-entry conditions stated) |
| Phase 13 — Next Direction         | COMPLETE — selection `NEW RESEARCH TRACK`: cheap-first adaptive retrieval (BM25 → Dense need prediction) |
| Phase 14 — Cheap-First Experiment  | COMPLETE — verdict `INSUFFICIENT EVIDENCE` (frozen `bm25_slope ≥ −0.915` rule; +0.05–0.08 over BM25-only at 22% call rate; TP 5/11 per arm; mixed random-ablation replication) |
| Phase 15 — Powered Confirmation    | COMPLETE — verdict `INSUFFICIENT EVIDENCE` 4/5 bars (frozen rule on 110 fresh queries/arm; P_comb=0.001 selection win replicated; dense margin missed by 0.004) |

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

---

# Phase 7 — Evaluation & Ablations

Status: **CLOSED** — E-series executed and both pre-registered closure gates complete

Phase 7 evaluates whether adaptive routing actually earns its complexity. This
entry records the foundation (7.0a–7.0c), the executed E-series (E1–E8), and
the two closure gates. The **final** Phase 7 result record is
[`docs/phase_7_closure_results.md`](phase_7_closure_results.md); the E-series
report is [`docs/phase_7_results.md`](phase_7_results.md); the phase record is
[`docs/phases/phase-7.md`](phases/phase-7.md). The Definition of Done is **not**
fully met — the items that remain partially met are **disclosed limitations, not
unfinished work** (see the honest accounting at the end of this entry).

**Final answer, in one paragraph.** The evaluation found limited query-level
retrieval heterogeneity: an oracle can improve over the best fixed strategy on a
small subset of queries (+0.0218 / +0.0343 recall@5, CIs excluding zero, carried
by 4–6 of 107 queries). This headroom is **not reliably identifiable** from the
currently observable query and retrieval-feedback signals in out-of-sample
evaluation (AUC ≈ 0.534, 0/24 features surviving Holm, on both corpus arms).
Consequently, the experiments **do not establish a practical adaptive routing
policy** under the evaluated conditions. Neither "adaptive RAG does not work"
nor "adaptive RAG works" is the finding.

## 7.0a Environment gate — COMPLETE

`scripts/validate_environment.py` probes all five arms against the canonical
corpus before any measurement is trusted, writing `experiments/phase7/gate.json`.

Result on this machine (commit `9db5054`): **5/5 PASS, gate open.** Every arm
returned well-formed, deterministic responses — `bm25`, `dense`, `hybrid`,
`hybrid_rerank`, and `adaptive` — with empty `problems` and identical rankings
across two identical calls.

The gate checks response *format* rather than quality: non-empty results, finite
and non-negative latency clocks, matching `top_k`, and populated
`corpus_version`/`retriever_version`. A missing clock is a hard failure rather
than a `0.0`, because a zero would silently flatter whichever arm produced it in
any later percentile.

This also retired the standing limitation carried from Phases 4–6 that the
dense, hybrid, and reranked arms were unvalidated for want of network. They are
now exercised live. `experiments/phase7/` is gitignored like every other run
output, so provenance travels inside the artifact rather than in the repo.

## 7.0b Strategy-cost re-measurement — COMPLETE

### Why

`RoutingConfig.strategy_cost_ms` is the router's price list, and
`RuleBasedRouter._normalised_costs` divides each strategy's cost by the maximum
across available strategies. The router therefore consumes the **ratios**, which
makes the table unusually sensitive to measurement error. The Phase 5 seed came
from per-query means over 20 examples in a single pass, with no warm-up and no
repetitions, and the gate then observed dense retrieval swinging between roughly
### Protocol

Defined in `evaluation/measurement.py`, executed by
`scripts/measure_strategy_cost.py`:

1. Build every component before any clock starts.
2. Warm up each arm and **discard** those samples, paying the ONNX session's
   first-call cost.
3. Run 5 repetitions of all 20 queries per arm, rotating both arm order and
   query order between repetitions so neither machine drift nor query position
   loads onto one strategy.
4. Aggregate with `statistics.median`; p95 is nearest-rank
   (`evaluation.base.percentile`, no interpolation), so at n=100 it is literally
   the 95th observed sample.

The adaptive arm is measured and reported but deliberately excluded from the
table: its cost is a per-query mixture of the others decided at runtime.

### Results (n=100 per arm)

| Strategy | p50 (ms) | p95 (ms) | stdev | p95/p50 |
| --- | --- | --- | --- | --- |
| `bm25` | 2.25 | 5.14 | 1.60 | 2.28 |
| `dense` | 451.78 | 744.64 | 129.13 | 1.65 |
| `hybrid` | 455.97 | 923.42 | 1373.66 | 2.03 |
| `hybrid_rerank` | 3854.41 | 4389.73 | 1068.51 | 1.14 |

Frozen into `RoutingConfig.strategy_cost_ms` after review, replacing the Phase 5
seed. Raw samples travel with the artifact so every aggregate can be recomputed
without a re-run.

### Findings

* **The embedding API dominates the "retrieval" cost.** The per-stage breakdown
  shows dense's 451.78 ms median is ~426 ms of live `text-embedding-3-large`
  call and only ~24 ms of actual vector search. Hybrid is the same shape
  (~450 ms dense branch, ~29 ms BM25). The frozen table therefore prices a
  *network round-trip*, not a retrieval algorithm, and its ratios move if the
  provider's latency or quota changes.
* **Reranking dominates everything else**, as expected: `hybrid_rerank` is
  ~8.5x the cost of the fused arm, with ~3.37 s of its 3.85 s median inside the
  cross-encoder.
* **The median earned its place.** `hybrid` carries a 12.0 s outlier — a single
  rate-limited embedding call — which inflates its stdev to 1373.66 while leaving
  the median at 455.97. The old mean-based seed would have absorbed that outlier
  into the price list.
* **The measured ratios differ substantially from the seed.** Every arm moved:
  dense −27%, hybrid −37%, `hybrid_rerank` −13%, `bm25` +32%. dense and hybrid
  are now near-identical in cost, which is the opposite of what the seed claimed
  and follows directly from the seed's means absorbing network variance into one
  arm but not the other.

### Two defects found and fixed en route

* **`instantiate_components` ignored an injected Qdrant client** in the adaptive
  branch and opened a second one. Local Qdrant takes an exclusive lock, so any
  caller building several arms at once — exactly what this sweep does — failed
  with `AlreadyLocked`. The gate never hit it because it builds one arm at a
  time. Two regression tests cover both directions; the reuse test fails without
  the fix.
* **The sweep tripped the embedding provider's rate limit** at repetition 5/5 on
  a first attempt. Three of the four costed strategies embed every query
  remotely, so the protocol issues a burst of live calls that the adapter's
  `2 ** attempt` backoff over `max_retries=4` could not ride out. The sweep now
  paces queries on provider-calling arms, sleeping *between* timed retrievals so
  it cannot enter any `latency_ms`. The pacing is recorded in the artifact.

## Limitations

* **Latency is hardware-, load-, and provider-dependent.** This table describes
  one machine against one provider over one afternoon. It is not portable, and
  the `experiments/phase7/` artifacts are gitignored, so a re-run is the only way
  to refresh it.
* **Dense and hybrid are nearly free of retrieval cost relative to their network
  cost.** Comparing them on this table mostly compares two API round-trips.
* **The median understates worst case.** `hybrid`'s p95 is 2x its p50 and its
  worst sample was 12 s; a router optimising against the median will not avoid
  the tail.
* **The cost table remains untuned against quality.** `cost_weight=0.25` and
  `sufficiency_threshold=0.5` are still reasoned defaults; calibrating them is
  Phase 7 ablation work.

## 7.0c Suite execution & reporting infrastructure — COMPLETE

Infrastructure for the E-series. **No experiment has been run and no result is
claimed here**; this is the machinery the studies will execute on.

Four new modules, one per deliverable:

* `experiments/registry.py` — an append-only registry keyed by
  `(experiment_id, variant)`. Provenance is read back out of each run's own
  `manifest.json` and `config.json` rather than being re-entered, so an entry
  cannot disagree with the run it describes. Re-registering a pair raises; a
  *different* variant under the same study is the normal way E3 accumulates one
  arm per disabled feature group. `extra="forbid"` throughout.
* `evaluation/rows.py` — one flat row per (query, system): recall/precision/hit
  at k in {1, 5, 10}, MRR, nDCG@5, the latency decomposition, the routing
  decision, the pre-escalation evidence E8 needs, and the cost fields, exported
  to CSV and JSONL with a frozen column order. **Every metric value comes from
  calling `RetrievalEvaluator`'s own helpers** — no recall/precision/MRR
  arithmetic is reimplemented, so a row cannot disagree with the mean published
  in the same run's `metrics.json`. Failed retrievals are rows with a `status`,
  not dropped queries.
* `evaluation/suite.py` — the driver. Plans the arms a study implies, verifies
  the shared conditions *before* anything runs, executes each through
  `ExperimentRunner`, exports rows, and registers the result.
* `scripts/run_phase7_suite.py` — the CLI (`--dataset`, `--split`, `--experiments`,
  `--arms`, `--out`, `--pace`, `--strict`, `--resume`, `--no-registration`).

Three properties the driver is built around:

* **One Qdrant client for the whole suite.** Local mode takes an exclusive lock on
  the storage folder, and building several arms at once is exactly what a suite
  does. The suite opens one client and passes the same handle to every arm, and
  additionally *caches component trees by config identity*: E1's `adaptive` arm
  and E2's shipped-config variant are the same system and open the index once.
  A regression test pins both the shared handle and the sharing.
* **Pacing outside the clock.** `PacedRetriever` sleeps *before* delegating on
  provider-calling arms (`dense`, `hybrid`, `hybrid_rerank`, `adaptive`), so the
  sleep cannot enter any `latency_ms`; the default matches
  `measure_strategy_cost.py`. The value is recorded in `suite.json` and on each
  registry entry. A test asserts the trace latency is bit-identical to the
  fake's with pacing on.
* **A failing arm is reported, not dropped.** It becomes a failure record in
  `suite.json` and in the returned result, and is never registered. `--strict`
  aborts on the first failure *after* persisting the work that did run.

E6, E7 and E8 have no arm of their own: they are analyses over the runs E1-E3
produce. Requesting one is answered with that explanation rather than an
invented run.

## Definition of Done for 7.0a/7.0b/7.0c

- [x] Five-arm environment gate, 5/5 deterministic, artifact with provenance
- [x] Cost table re-measured under a defined, warm-up-and-rotation protocol
- [x] `n=100` per arm, complete artifact, no rerank fallbacks
- [x] Medians reviewed and frozen into `RoutingConfig`, `adaptive` excluded
- [x] Injected-Qdrant-client defect fixed and regression-tested
- [x] Rate-limit pacing added, recorded in the artifact
- [x] Suite driver, registry, row export, and CLI in place and tested offline
- [x] Adaptive-vs-fixed comparison (E-series) — *executed; see 7.1-7.3 below*
- [x] Ablation program and threshold calibration — *executed; see 7.1-7.3 below*

## 7.1-7.3 E-series — EXECUTED

All eight studies ran against the frozen 107-record `phase7_eval_v1` benchmark
(47 calibration / 60 test, sha256 prefix `f0695189903c8ef8`, sanity gates S1-S5
and dataset gates G1-G7 pass, `gate_open=true`, `grounding_verified=true`, 0
empty and 0 contaminated evidence quotes across 293 extractions). Every run was
retrieval-only. Artifacts, tables, figures, and traces live under
`experiments/phase7/`, which is gitignored, so provenance travels inside the
artifact rather than in the repository.

### What was measured

* **E1 — five-arm comparison, 107 paired queries.** Recall@5 / MRR / median
  latency (ms): bm25 `0.7788 / 0.7264 / 1.88`; dense
  `0.9283 / 0.8604 / 472.84`; hybrid `0.8723 / 0.8576 / 465.65`;
  hybrid_rerank `0.7165 / 0.5851 / 3491.69`; adaptive `0.8723 / 0.8564 /
  524.85`. Against bm25 with Holm step-down per metric, **15 of the 16 evaluable
  comparisons are significant**; the single non-significant result is
  `recall_at_5` for `hybrid_rerank` (adj 0.1267). The other four requested
  comparisons (`estimated_cost_usd`) have `n_pairs = 0` because no run was
  generation-bearing.
* **E2 — escalation ablation A/B/C, 47 calibration.** Quality is tied end to
  end (p_adj 1.0 on recall_at_5 / MRR / nDCG@5). The one significant result is
  B-vs-C latency: **+58.91 ms median, p = 0.00309, p_adj 0.00618**. Escalation
  bought latency, not quality.
* **E3 — leave-one-out over the six feature groups, 47 calibration.**
  **0 of 24 evaluable comparisons significant.** Recall@5 dips only for
  `without_complexity` and `without_question_type` (0.8617 vs 0.8830), and
  neither survives Holm adjustment.
* **E4 — sufficiency-threshold sweep, 47 calibration.** `0.3 / 0.4 / 0.5 / 0.6`
  are identical (Recall@5 0.8830, MRR 0.8547, 0.0% escalation — the gate never
  fires). `0.7` escalates **6.4%** (3 of 47), MRR 0.8377, median 556.71 ms.
* **E5 — cost-weight sweep, 47 calibration.** `0.0 / 0.25` give Recall@5 0.8830;
  `0.5 / 0.75 / 1.0` give 0.8617 and nDCG@5 0.6486 -> 0.6330. Escalation is
  0.0% at every weight, and the router still commits to hybrid's embedding call
  on ≥94% of queries, so raising the weight costs quality with no median-latency
  benefit.
* **E6 — routing decision overhead, n=107.** `routing_latency_ms` mean 0.85 ms,
  median 0.81 ms — 0.71% of total latency per query by mean, 0.16% by median.
  `reranking_latency_ms`, `candidate_generation`, `generation`, and
  `query_embedding` each have **n = 0**, because the escalation path never ran.
* **E7 — per-category breakdown, adaptive, n=107 pooled.** Recall@5 by category:
  terminology 0.9500, conceptual 0.9310, factual 0.9231, fine_grained 0.8824,
  comparative 0.6250, multi_document 0.4762. Escalation is 0.0% in every
  category.
* **E8 — escalation analysis.** `n_escalated = 0` and
  `n_transitions_observed = 0` across all 107 routed rows. Observed sufficiency
  never falls below **0.6364** (test) / **0.65** (calibration), so the shipped
  0.5 bar cannot open. Escalation *was* observed — 3 times, all in the 0.7
  calibration arm — so the correct claim is that the shipped gate is too tight
  on this dataset, not that escalation never happens.

### Headline negative result

**At the shipped Phase 6 defaults adaptive routing earns nothing.** The
sufficiency gate never fires, so `adaptive` collapses onto hybrid: Recall@5 and
Hit@5 are identical on 107/107, retrieved chunk-ids are identical on 104/107,
and MRR differs on exactly one query. That equivalence costs ~59 ms of median
latency, of which only 0.81 ms is the routing decision itself — the rest is the
hybrid stage the router selects. No composite quality/cost score is computed
anywhere, and none may be derived: the result is a trade-off to be reasoned
about, not an ordering.

Two findings sit underneath that:

* **The cross-encoder reranker is net-negative on this corpus.** It is last on
  every E1 quality metric and 7.5x hybrid's median latency, and it is
  significantly *worse* than bm25 on MRR (adj 0.00037) and nDCG@5 (adj
  0.000117) — a significant difference in the wrong direction. It is also the
  terminal rung of the escalation ladder, so the gate can only ever escalate
  *into* it. Full defect-isolation evidence, including an exact replay of the
  stored logits, is in `docs/phase_7_results.md` §11.
* **The sufficiency threshold sits above the observed score range**, so the
  escalation path is unreachable as configured.

Both are recorded as recommendations in ADR-028. **Neither has been acted on.**
Changing the escalation ladder or the threshold is a Phase 6 change and is out
of Phase 7 scope.

### Figures

Four figures rendered: `e1_quality_vs_latency`, `e1_strategy_distribution`,
`e3_feature_ablation`, `e4_threshold_sensitivity`. Two were skipped and both
skips are recorded in `experiments/phase7/figures/figures_manifest.json`:
`quality_vs_cost` (no arm carries both `recall_at_5` and `estimated_cost_usd`,
because every run was retrieval-only) and `escalation_transitions` (zero
transitions in the 107 frozen-configuration adaptive rows supplied, so there is
nothing to plot; an all-zero chart would instead claim the router declined to
escalate, which those rows cannot establish — three escalations do exist in the
0.7 calibration arm).

### Validation

* **729 offline deterministic tests pass, 2 deselected** (`.venv/bin/python -m
  pytest -q`), one matplotlib `Axes3D` import warning. No credentials, no
  network, no live index.
* **14/14 architecture guards green** (`tests/test_architecture_guards.py`).
* No change to `src/`, `tests/`, or `experiments/` was needed to produce this
  entry or the results report.

### Definition of Done — PARTIAL

Met:

* [x] Five-arm environment gate, cost table re-measured and frozen, harness,
      registry, row export, and CLI in place (7.0a-7.0c, above)
* [x] Adaptive-vs-fixed comparison executed — E1 over 107 paired queries, with
      Holm-corrected paired statistics against bm25
* [x] Ablation program executed against the built harness — E2, E3
* [x] Threshold calibration executed — E4 and E5 sweeps over the built harness
* [x] E6, E7, E8 analyses produced from the E1-E3 runs

Not met, or only partially met:

* [ ] **Escalation is measured but not characterised.** E8 produced **zero
  transitions** at the frozen threshold, so what escalation buys per transition
  is unmeasured on this run; the escalation-transitions figure and table are
  empty by observation, not by omission.
* [ ] **E4's `max_escalation_steps` sweep is unanalysable, not null.** The 188
  rows were excluded with the limitation flagged: the 0.5 gate never fires, and
  `allows_escalation()` is called once with a hardcoded `steps_taken=0` in a
  loop-free `retrieve()`, so steps 1/2/3 are structurally identical.
* [ ] **E6 measures the decision layer only.** Four stage clocks
  (`reranking_latency_ms`, `candidate_generation`, `generation`,
  `query_embedding`) have n = 0 because no query escalated.
* [ ] **No cost dollar figures.** `estimated_cost_usd` has no data
  (`n_pairs = 0`) in every study; all runs were retrieval-only.
* [ ] **Calibration was swept one axis at a time, not on a joint grid**, so the
  per-axis best was never validated in combination. The shipped default is used
  because it is the only setting exercised end to end.
* [ ] **E7's `comparative` and `multi_document` cells are below `min_cell=5`**
  on each split individually; only the pooled cell is reported, descriptively.
* [ ] **No adaptive-vs-hybrid p-value exists.** E1 tests every arm against
  bm25; "adaptive == hybrid" rests on identical Recall@5/Hit@5 on 107/107, not
  on a paired test.
* [ ] **nDCG@5 is not textbook nDCG** — its IDCG denominator is derived from the
  trace's own retrieved chunks, so it is not comparable to a textbook value.

## Next steps

> Not a new phase. Two Phase 6 changes are recommended and neither is in scope
> here.

1. **Decide the rerank rung.** ADR-028 recommends removing `hybrid_rerank` from
   the router's strategy set and terminating the ladder on hybrid. That is a
   Phase 6 router change.
2. **Recalibrate the sufficiency gate** inside the observed score range
   (0.6-0.65), validated on a **joint grid with `cost_weight`** rather than the
   independent sweeps used here, and report on the held-out test split.
3. **Then re-run E1/E8.** Until the gate can open and the ladder terminates on
   a quality-positive rung, "adaptive == hybrid" is an artifact of a gate that
   never fires, not evidence that routing is unnecessary.

The measurement foundation is frozen, the harness is in place, and the honest
negative result is recorded. Full numbers, provenance, and per-study
limitations: `docs/phase_7_results.md`.

---

# Phase 8 — Corpus Fix Study

Status: QUALITY STUDY COMPLETE / COST MEASURED, NOT FROZEN

## What was done

A two-column extraction defect in `ingestion/normalizer.py` was found to be
splicing the left and right columns of every two-column page together mid-sentence.
It was fixed with column detection and gutter-aware line clustering, then measured
as a **before/after study** over two corpora built from the same PDFs and the same
chunking config, differing only in whether the fix ran:

| | Path | Chunks |
|---|---|---|
| before | `data/processed_phase8_before/` | 618 |
| after | `data/processed_phase8_after/` | 613 |

Validation gates all passed: residual interleaving 0, 77 of 285 pages detected as
two-column, 7009 word pairs repaired, single-column extraction byte-identical to
the previous behaviour, reading order verified.

## What was found

1. **The fix changed no quality metric significantly.** Across all five E1 arms,
   no recall@5 or MRR before/after delta survives Holm correction at n=107. Four
   arms nudge up, `dense` nudges down, none is distinguishable from zero.
2. **The reranker did not improve — and that is the finding.** `hybrid_rerank`
   moved by -0.0016 recall@5 (99 of 107 queries tied) and its penalty against
   `hybrid` grew slightly. The corpus defect was not the mechanism behind Phase 7's
   rerank result. Both pre-registered outcomes that would have *rescued* the
   reranker are ruled out by measurement.
3. **Two apparent latency findings were withdrawn.** `dense` (+411 ms) and
   `hybrid_rerank` (+112 ms) reached significance, but both sit in the
   query-embedding API call, which embeds the query rather than the corpus. The
   `bm25` control arm makes no API call and moved 1.83 -> 1.85 ms; the reranker's
   local ONNX stage moved -2.3 ms. The corpus fix's latency effect is ~0, and the
   paired tests had detected provider variance.
4. **`nDCG@5` and E8 are not measured.** The gold section labels are themselves
   corrupt: 47 of 140 spliced, 40 unresolvable on the fixed corpus. Reported as
   not measured rather than computed on labels of unknown provenance.
5. **A measurement-integrity failure nearly inverted the headline.** 13 of 29 arms
   in the first sweep silently lost embedding calls; every suite still reported
   `ok` with a full trace count. The contaminated reranker arm read 0.6384 where
   its true value is 0.352 and its clean re-run is 0.7072 — inverting the sign of
   the reranker's before/after result.

## ADR-028

Revised. The rerank-rung removal recommendation **stands** and no longer depends on
the defect being fixed, but its stated cause — model mismatch with column-interleaved
prose — is **withdrawn**. See `docs/decision.md`.

## Routing ceiling (offline, added after Phase 8)

Question: does routing by query characteristic have *any* headroom on this
benchmark, or is Phase 7's `adaptive ≡ hybrid` the whole story? Measured offline
from the clean E1 arms over the four **selectable** strategies — no index, no
API, deterministic across runs (`experiments/phase8/oracle_ceiling_v2.json`, §8
of the results doc and `docs/phase7-closure-evidence.md` §8).

| | `after` (shipping) | `before` (replication) |
|---|---|---|
| best fixed | `dense` 0.8988 | `dense` 0.9143 |
| oracle (per-query) | 0.9206 | 0.9486 |
| delta | **+0.0218** | **+0.0343** |
| frontier(0.01) latency saving | −842.11 ms | −398.21 ms |
| gate verdict | `routing_has_headroom` | `routing_has_headroom` |

Three qualifications, all of which travel with the number:

1. **The gain rides on 4 of 107 queries** (`before`: 6). `recall_at_5` takes only
   four distinct values because 92 of 107 queries have a single relevant
   document, so the five strategies tie at the maximum on 98 queries and the
   after-arm delta clears its +0.02 threshold by 0.0018 — about two queries.
2. **The ceiling is not detectable.** 8 queries need `dense`, 99 do not, and
   nothing separates them: `sufficiency_score` p = 0.4989, `coverage` p = 0.9078,
   `category` p = 0.0567, no sufficiency signal at α = 0.05. Both scalars run the
   *wrong way* — sufficiency is higher where a cheaper strategy already suffices.
   The gate's latency branch therefore fails despite the −842.11 ms headline.
3. **The oracle's MRR is worse** than the best fixed strategy's (0.8224 vs
   0.8645) — but that is a property of the *latency* tie-break, not of the
   ceiling. The same ceiling re-scored with an `mrr` tie-break carries MRR 0.9420
   (+0.0775 vs best-fixed), and a rank-aware frontier at ε = 0.01 keeps **72.7%**
   of the latency saving while holding that MRR. So rank is a choice the gate
   leaves unpriced, not a cost routing necessarily pays.

**Gate 2 correction (selectable set).** The ceiling above was originally computed
over five arms including `adaptive`, which is the router's *output*, not a
strategy it can select. Re-run over the four selectable arms
(`experiments/phase8/oracle_ceiling_v2.json`), **recall@5, MRR and both gate
verdicts are unchanged** — `adaptive` was credited on 6 (`after`) / 1 (`before`)
queries but only as a tie at the recall maximum, so it never raised the ceiling.
Only credited-arm latencies moved (frontier saving −845.84 → −842.11 ms `after`).
The correction also added a paired-bootstrap CI for the delta: recall@5
[0.0031, 0.0467] `after` and [0.0093, 0.0670] `before`, both excluding zero. The
leak was methodological, not numerical: it is fixed for correctness and it
*confirms* the published numbers. Record: `docs/phase7-closure-evidence.md` §8.

Net: routing headroom is **non-zero but not bankable** by the shipped
mechanism. Phase 7's equivalence is not evidence that routing is worthless —
it is evidence that the router never routed.

### Nuance carried by the Gate 2 correction (recorded explicitly)

1. The oracle recall headroom **survives** the selectable-set correction:
   recall@5 0.9206/0.9486 and both `routing_has_headroom` verdicts are unchanged.
   Excluding `adaptive` was a correctness fix that *confirms* the numbers, so
   the original leakage concern was methodological rather than the source of the
   observed gain.
2. The confidence interval on the recall delta **excludes zero** —
   [0.0031, 0.0467] `after`, [0.0093, 0.0670] `before`. The *sign* is solid.
3. The effect nevertheless remains **concentrated in a handful of queries** —
   4 of 107 (`after`), 6 of 107 (`before`). The interval excludes zero while the
   magnitude remains fragile; both statements are true and neither licenses a
   router on its own.
4. Oracle MRR is **tie-break sensitive** (0.8224 under `latency` vs 0.9420 under
   `mrr`, a 0.1195 swing that flips its sign against best-fixed). It must not be
   summarised as an unconditional "the oracle costs MRR" finding; that was a
   property of the latency tie-break alone.
5. The rank-aware frontier is a useful **secondary efficiency/quality
   observation** — 72.7% of the latency saving retained at equal MRR — and is
   **not** evidence that a deployable router works.

Stage 2 prerequisite, answered: `build_experiment_config` **can** express the
reduced strategy set, but `run_phase7_suite.py` exposes no routing flags, so the
re-run cannot be configured through it as the plan's command assumes. Flagged,
not changed.

## Gate 3 — full observable signal exhaustion (offline, `no_out_of_sample_signal`)

Question: can anything a router can *observe* tell it, out of sample, which of
the 107 queries are the few where a different strategy is actually beneficial?
`scripts/gate3_signal_exhaustion.py` →
`experiments/phase8/gate3_signal_exhaustion.json`. Full record: §9.

| | `after` | `before` |
|---|---|---|
| positives (`needed`) / negatives | **8 / 99** | **8 / 99** |
| positives in calibration / test | 2 / 6 | 3 / 5 |
| out-of-sample AUC (fit on calibration, scored on test) | **0.5340** | **0.5345** |
| bootstrap 95% CI | [0.2469, 0.8086] | [0.2000, 0.8691] |
| permutation p (whole procedure, 10k) | 0.4016 | 0.4087 |
| features surviving Holm (of 24) | **0** | **0** |
| min detectable AUC at 80% power | 0.825 | 0.850 |

All four pre-registered conditions fail on both arms. Four things matter more
than the p-values:

1. **The overfitting signature.** Fit-split AUC is **1.000** (`after`) / **0.962**
   (`before`) against out-of-sample **0.534** on both. With 2–3 positives the
   model separates the fit split perfectly and generalises to nothing. Reading the
   fit-split number as the result would have inverted the conclusion entirely.
2. **Power is binding, so this is a power statement.** At 6 positives vs 54
   negatives a true AUC of 0.60 has 12% power to clear chance and 0.80 has 76%;
   only ≥0.825 reaches 80%. The correct reading is "no reliable out-of-sample
   signal **at this sample size**", not "no signal exists".
3. **The retrieval-feedback signal runs the wrong way on both arms.**
   `top1_coverage` is the strongest feedback feature and is *inversely*
   associated with needing `dense` (AUC 0.346 / 0.357). Mean `coverage` on the 8
   positives is 0.803 vs 0.808 on the 99 negatives — the sufficiency check says
   "terms found" on exactly the queries where the reference was still the only
   acceptable answer. Same inversion Gate 2 recorded, now measured out of sample.
4. **The strongest feature is not stable across arms.** `content_term_count` leads
   `after` (AUC 0.711, raw p = 0.045) and collapses to 0.562 in `before`. The 8
   positives span six categories with no common feature.

**Verdict: `no_out_of_sample_signal`.** Oracle headroom exists (Gate 2) but is
**not identifiable** from the currently available observable signals. Gate 4 and
a learned router are **not** licensed: the rule-based router's failure is not
evidence a learned one would succeed, and a learned router here would be fitted
on 8 positives and inherit the fit-split AUC of 1.0 with none of it surviving
held-out data. Per-query evidence for all 107 queries is preserved in the
artifact.

Scope honoured: no learned router trained or deployed, shipped router unmodified,
corpus not rebuilt, `strategy_cost_ms` unchanged, ADR-028 not promoted, the 42
spliced section paths kept out of the primary routing closure experiment.

## Next steps

1. **Decide whether to freeze `strategy_cost_ms`** — recommendation now made,
   awaiting a user call: **do not freeze**. Full evidence in
   `docs/strategy-cost-freeze-decision.md`. Summary: the two arms disagree by
   25% on `dense`, but that is provider variance inside the query-embedding call
   (the `bm25` control moved 2.48 → 2.12 ms), and on the *normalised* ratio the
   router actually consumes (`rule_based.py:148-157`) the disagreement is 5.3% —
   a 0.008 shift in the score penalty against rule weights of 0.2–1.5. Freezing
   would pin the unstable representation (absolute ms) and leave the stable one
   (ratios) unpinned. `strategy_cost_ms` remains untouched.
2. **`nDCG@5` / E8 unblocking — route measured, found blocked.** Only **22** of
   the 47 corrupt labels are safely repairable, not the 37 the Step 4 audit
   called high-confidence (`experiments/phase8/gold_label_repair.json`). 15 of
   those 37 have no clean reconstruction; 3 resolve only to paths that are
   themselves spliced. Root cause is upstream: the **fixed corpus still carries
   42 spliced section paths** (down from 52), so label repair alone cannot make
   the metric trustworthy. Fix the corpus's section paths, then repair labels and
   read the remainder. No labels were rewritten.
3. **Act on ADR-028** — a Phase 6 router change, out of scope here.
4. **Add a no-API control arm to any future latency comparison** on this hardware;
   `total_latency_ms` alone cannot distinguish provider variance from system cost.
5. **Decide whether to run Stage 2 at all**, and if so through a runner that can
   actually pass the routing override (the suite runner cannot — see above).
   Recommendation: not as a routing test. Under ADR-028's rung removal,
   escalation from `hybrid` is a no-op by construction, so it would measure an
   unreachable path.
6. **Treat the 4–6 ceiling-carrying queries as the only real target.** Any future
   router work is a `detect-those` problem, not a `route-everything` problem —
   and 4–6 positives is too few to fit anything without a held-out split
   (ADR-027).

Full numbers, provenance, and limitations: `docs/phase-8-results.md`.
Phase plan and per-step detail: `docs/phases/phase-8.md`.

---

# Phase 9 — Strategy Sensitivity: Mechanism, Not Correlation

Status: **COMPLETE** — Gates 9.1–9.5 executed; verdict `PROMISING_SIGNAL`
Plan of record: [`docs/phases/phase-9.md`](phases/phase-9.md)
Results of record: [`docs/phase9_results.md`](phase9_results.md)

Phase 9 is an **explanatory measurement phase**. It runs no retrieval, builds no
index, and makes no provider call in Gates 9.1–9.2. Nothing under
`src/adaptive_rag/retrieval/` changes.

## 9.0 Motivation — three findings, and one of them is a Phase 7 limitation

1. **Gate 3's target label is disjoint from the headroom queries.** `needed`
   means no alternative comes within ε of the reference; carrying oracle gain
   means some strategy *beats* it. These are mutually exclusive, so all 8 of
   Gate 3's positives carried `oracle_gain == 0.0` and all 4–6 gain-carrying
   queries sat in its negative class. Gate 3's AUC ≈ 0.53 is a statement about
   that label. **Phase 7's conclusion stands as written** and is not revised
   here; Phase 9 observes only that the label does not answer the question Phase
   7 posed. Recorded as a limitation, per the instruction not to alter Phase 7.
2. **The residual target is larger than 8.** Dispersion exceeds zero on 34/107
   (`after`) and 41/107 (`before`), which supports rank-correlation testing at
   n = 107 in a way an 8-positive binary label never could.
3. **Most 4-arm dispersion is the reranker dragging** — 26/34 (`after`) and 30/41
   (`before`) — which ADR-028 already adjudicated. The honest target is
   dispersion over `{bm25, dense, hybrid}`: **21/107** and **29/107**.

## 9.1 Targets — measured and verified

`evaluation/dispersion.py`, offline over the frozen `rows.jsonl` of both corpus
arms. Reproduces the pre-registered expectations exactly:

| | `after` | `before` |
|---|---|---|
| queries complete under all four selectable arms | 107 | 107 |
| `T-disp3` (recall dispersion, sound three) > 0 | **21** | **29** |
| `T-disp3-mrr` (MRR dispersion, sound three) > 0 | 46 | 47 |
| `T-disp4` (four arms) > 0 | 34 | 41 |
| reranker strictly worst | 26 | 30 |
| reranker strictly best | 0 | 1 |
| tie-break artefacts (H7 control) | 73 | 66 |
| `T-needed` (Gate 3 control) | 8 | 8 |

**The decomposition counts overlap and do not partition** — 26 + 21 exceeds 34,
because a query can be dispersed among the sound three *and* have the reranker
strictly last. Caught by `test_reranker_decomposition_counts_overlap`, which
corrected an earlier claim in the module docstring that they did partition.

## 9.2 Signal table

`evaluation/agreement.py` + `evaluation/signals.py`. `experiments/phase9/` now
holds `signal_table_phase8_{after,before}.jsonl`, 107 rows each, byte-identical
across re-runs (verified by checksum). Verified end-to-end *through the written
artifact* rather than the build path:

| Signal vs `T-disp3` | `after` | `before` | pilot §7.4 |
|---|---|---|---|
| `jaccard@5 (bm25, dense)` | **−0.454** | **−0.520** | −0.454 / −0.518 |
| `jaccard@5 (bm25, dense)` vs `T-disp3-mrr` | −0.575 | −0.502 | −0.575 / −0.502 |
| `jaccard@5 (dense, hybrid)` | −0.431 | −0.373 | −0.431 / −0.372 |
| `distinct_doc_ratio@5 (bm25)` — **single-arm** | +0.226 | +0.211 | — |

Mean Jaccard@5 (bm25, dense) reproduces at **0.5173 / 0.4977**.

**One number moved, and it matters.** The single-arm proxy read +0.327 in the
pilot and reads **+0.226** against the honest three-arm target. The pilot figure
was inflated by reranker-driven dispersion. The only candidate in the family
that a router could actually compute pre-decision is *weaker* than the pilot
suggested. Gate 9.3 must use the honest number.

## Outcome

Gates 9.1–9.5 are executed. Verdict: **`PROMISING_SIGNAL`**, with a
methodological audit applied afterwards (see
[`docs/phase9_results.md`](phase9_results.md)).

**Five survivors** (§2.7 replication on both arms, Holm denominator 12):

| Signal | `after` | `before` |
|---|---|---|
| `jaccard@bm25\|dense` | −0.454 | −0.520 |
| `jaccard@dense\|hybrid` | −0.431 | −0.373 |
| `union_concentration` | +0.454 | +0.520 |
| `distinct_doc_ratio@dense` | +0.343 | +0.406 |
| `top1_agreement@bm25\|dense` | −0.409 | −0.492 |

**Deployability:** 0 pre-routing, 1 single-arm (`distinct_doc_ratio@dense`),
4 post-hoc. The §2.10 out-of-sample criterion is met by
`distinct_doc_ratio@dense` alone — §2.4-registered, calibration +0.254,
**test +0.410**, no fitting. No query-only feature reaches |ρ| ≥ 0.30 on `test`
(best `lexical_density` +0.194).

**Audit corrections.** The Ridge analysis (test ρ +0.369) was never
preregistered and is **withdrawn as evidence**; it does not affect the verdict,
which rests on the single unfitted signal above. §2.10's claim that verdicts are
"emitted by `evaluate_gate()`" is **not implemented for Phase 9** — no such
function exists. §2.10 also carries an unresolved internal tension between its
pass condition (which admits post-dense single-arm) and its question cell
("usable at a cheap decision point?", which post-dense escalation does not
satisfy).

**Boundary.** A pre-routing = **NO**. B post-dense escalation = **SIGNAL
EXISTS**. C outcome improvement = **NOT ESTABLISHED**. Phase 9 ran no
intervention, and `T-gain` has only 4/6 members with §2.7 forbidding fitting on
it, so no per-query gain oracle exists to validate a policy against.

No router has been built, trained, tuned or deployed. `ADR-026` keeps the learned
router deferred and this phase does not revisit it. Phase 10 has not begun.

---

# Phase 10 — Post-Dense Escalation via `distinct_doc_ratio@dense`

Status: **COMPLETE** — verdict `FAILURE` (honest negative result)
Plan of record: [`docs/phases/phase-10.md`](phases/phase-10.md) (frozen before
any split-separated Phase 10 number was computed)
Results of record: [`docs/phase10_results.md`](phase10_results.md)

Phase 10 asked whether the Phase 9 signal supports a useful adaptive decision:
after paying for Dense, escalate to Hybrid iff `distinct_doc_ratio@dense ≥ t`.
It built the per-query gain oracle Phase 9 recorded as missing (dense vs
hybrid outcomes joined offline from `experiments/phase8/combined/p8{a,b}_e1`,
DDR integrity 107/107 per arm, 47/60 split, no new retrieval, no network).

## Oracle

`oracle_escalate ⟺ recall@5(hybrid) − recall@5(dense) ≥ 0.01` (frozen).
Per arm: **helps 4, harms 9–12, neutral ~92** of 107 — always-escalate is
net-negative by construction, so only selection could win. Incremental
escalation latency (hybrid's non-embedding stages): median 31.0 ms both arms,
fallback count 0. No USD figure exists for retrieval-only runs; latency is the
labelled proxy and API call count is flat across policies.

## Signal analysis (calibration, n=47)

2 oracle-positives per arm. Shipping: ROC-AUC **0.55** (≈ chance),
PR-AUC 0.27, Spearman(DDR, Δrecall) −0.13 (wrong way); the two positives sit
at DDR 0.2 and 0.8. Frozen rule selected **t = 0.8, direction high**
(non-degenerate, 19.1% calibration escalation); the control family did not
clear the 1-SE bar.

## Test evaluation (one pass, n=60, frozen policy)

| Arm | Dense R@5 | Adaptive R@5 | Δ (CI95) | Esc. rate | TP/FP/FN/TN | P(random ≥ adaptive) |
| --- | --- | --- | --- | --- | --- | --- |
| after | 0.9139 | 0.8639 | −0.050 [−0.117, 0.000] | 0.133 | 0/8/2/50 | 1.000 |
| before | 0.9167 | 0.8833 | −0.033 [−0.083, 0.000] | 0.183 | 0/11/2/47 | 0.954 |

No primary/secondary comparison significant; point estimates negative vs
dense on both arms; adaptive pays +4–6 ms mean latency for lower quality —
dominated by dense-only and by random escalation at equal spend.

## Mechanism reading

Phase 9's correlation stands (DDR tracks dispersion), but this benchmark's
dispersion is dominated by dense succeeding where RRF fusion fails — DDR
marks disagreement without marking escalation value. Correlation with
dispersion ≠ prediction of gain. A Phase 9 limitation (§9.4) thus resolves
against usefulness. One above-dense operating point exists in the
non-pre-selected direction (+0.0056, inside noise, unreplicated); recorded as
a non-finding per protocol, not acted on.

## Deliverables

`src/adaptive_rag/evaluation/escalation.py`
(`phase10_escalation_v1`), `scripts/phase10_{build_oracle,signal_analysis,
evaluate_policy,figures}.py`, `tests/test_escalation_oracle.py` (25 tests),
`experiments/phase10/` (oracle tables + provenance, calibration analyses,
`frozen_policy.json`, test evaluations + per-query rows, 5 figures +
manifest). Full suite green including 14/14 architecture guards; no change to
any Phase 9 artifact, retrieval, routing, or runner code.

---

# Phase 11 — Missing Signal Behind Useful Retrieval Escalation

Status: **COMPLETE** — outcome `B` (insufficient evidence)
Plan of record: [`docs/phases/phase-11.md`](phases/phase-11.md) (frozen before
any Phase 11 computation)
Results of record: [`docs/phase11_results.md`](phase11_results.md)

Phase 11 is the mechanism autopsy of the Phase 10 failure. It reuses the
frozen Phase 10 oracle unchanged (helps == oracle YES, asserted in code),
joins it with frozen Phase 9 signal values and E1 trace scores, and asks
what distinguishes the 5 unique help queries from the 9–12 harm queries per
arm. No threshold fitted, no classifier trained, no intervention run (gate
pre-registered closed: ≤3 helps per split/arm cannot support a freezable
policy).

## Findings

* **What helps (Q1):** dense misses relevant evidence (recall 0–0.33) while
  BM25 holds/ranks it; RRF keeps it in the top-5 (sources `bm25_only` in 4/5
  unique helps). Dense top-1 low (0.40–0.56), disagreement high.
* **What harms (Q2), two sub-mechanisms:** H1 — intruder promotion
  (BM25-confident irrelevant docs displace relevant ones); H2 — pure
  re-ranking among shared docs with *no new document entering* (relevant
  chunks fall below the cutoff; e.g. hybrid top-5 all one wrong doc). All
  harms are State 3 (dense held evidence, recall > 0).
* **Distinguishability (Q3):** directionally consistent hints (helps =
  weaker dense top-1 + higher disagreement on both arms) but ranges overlap
  and the bounded 6-feature separation check hits 1/6 on one calibration
  arm and 0/6 on the other — the signature of fitting ~5 points. Not
  established.
* **Adequacy (Q4) / intervention (Q5):** 5 unique helps cannot support
  supervised routing; no new adaptive experiment is justified. Gate: NO-GO.
* **Conclusion (Q7):** dense-only stands; DDR is a dispersion diagnostic;
  disagreement ≠ insufficiency ≠ escalation value (demonstrated per query).
  Carry-forward: three-state framing, evidence-insufficiency hypothesis,
  H1/H2 taxonomy, and a future-design recipe (dozens of positives, power
  analysis upfront, pre-registered confidence×disagreement rule, rung
  re-justification since RRF harms 2–3× what it helps).

## Deliverables

`src/adaptive_rag/evaluation/mechanism.py` (`phase11_mechanism_v1`),
`scripts/phase11_mechanism.py`, `scripts/phase11_figures.py`,
`tests/test_mechanism.py` (12 tests), `experiments/phase11/` (per-query
mechanism tables + summary with all non-tie case blocks, 4 figures +
manifest; 5th figure omitted per protocol — no candidate). Full suite green
including 14/14 architecture guards; no Phase 8–10 artifact, retrieval,
routing, or runner code touched.

---

# Phase 12 — Research Boundary of Adaptive Retrieval

Status: **COMPLETE** — recommendation `STOP CURRENT ADAPTIVE-ROUTING TRACK`
Plan of record: [`docs/phases/phase-12.md`](phases/phase-12.md) (frozen
before any analysis beyond the Phase 8–11 audit)
Results of record: [`docs/phase12_results.md`](phase12_results.md)

Phase 12 is the boundary analysis over the frozen Phases 8–11 chain:
offline only, no retrieval, no routing experiment, no threshold sweep; the
Phase 10 oracle and Phase 11 labels are unchanged.

## Evidence matrix (one row per question, exact repository evidence)

Disagreement exists and DDR tracks it incl. out-of-sample (P9, established);
DDR predicts gain at chance level (P10 AUC 0.55, falsified); the frozen DDR
policy is dominated by dense-only and by random escalation (P10 test,
falsified); helps are BM25 rescues of dense-missed evidence (5 unique, P11);
harms are fusion corrupting dense-sufficient results — H1 intruder promotion
and H2 pure re-rank with no new doc entering — all State 3, exact both arms
(P11); distinguishability has directional hints only with overlapping ranges
and a 1/6 + 0/6 separation check (not established); no intervention is
justified (NO-GO: Phase 13 gate fails on positives and rung upside).

## Boundary (derived from evidence)

Disagreement is measurable and real, but under the tested setting
(107-query academic-RAG benchmark, dense strongest fixed arm, RRF-hybrid
rung, post-dense retrieval-state-only decision) it does not reliably
indicate insufficiency, and no available retrieval-state information
reliably predicted help vs harm. The tested escalation is net-negative
(harms 2–3× helps): no rung worth routing to. Position C adopted; B true
as limitation; A rejected (24 + 12 + 6 features exhausted, 4–8 positives
in all framings); D rejected (helps exist; D1–D5 families untested).

## Limiting factors (judged, not assumed)

Supported by measurement: positive scarcity (A), dense already strong (E),
net-negative rung (F), decision point exposes disagreement but not
insufficiency (G — three independent failures: Gate 3 AUC ≈ 0.53, P10 AUC
0.55, P11 separation 1/6+0/6). Reported as measured properties, not
defects: useful disagreement is ~4% of queries (B/C); query-mix diversity
unproven either way (D).

## Deliverables

`docs/phases/phase-12.md` (frozen plan + Phase 13 gate), `docs/phase12_results.md`
(matrix, four-way classification, boundary, A–G judgments, A–D adjudication,
Directions 1–5 assessment as hypotheses, 12 framework answers, contribution
analysis, arrow-verified unified story), this entry. No source, test,
experiment, retrieval, routing, or runner change; no new tests (no reusable
analysis code introduced). Re-entry conditions (new research, not
continuation): dozens of oracle-positives with upfront power analysis, a
rung beating dense somewhere substantial, a pre-registered
confidence×disagreement rule, ADR-027 split discipline, and a gate that can
say no again.

---

# Phase 13 — Next Research Direction After the Adaptive-Routing Track

Status: **COMPLETE** — selection `NEW RESEARCH TRACK`
Plan of record: [`docs/phases/phase-13.md`](phases/phase-13.md) (discipline
held: design/selection only; only read-only frozen-row counts)
Results of record: [`docs/phase13_results.md`](phase13_results.md)

Phase 13 audited the whole project (system, all four fixed strategies +
adaptive layer, evaluators + suite harness, both datasets, dual-corpus E1
rows, Phases 8–12, ADR-026–028) and selected the next step from evidence.
The old track (retrieval-state → predict Dense→Hybrid gain) stays closed;
no DDR/classifier/rung-reuse proposal was entertained.

## New feasibility evidence (read-only, frozen rows)

Cheap-first oracle (escalate BM25→Dense ⟺ dense−BM25 ≥ 0.01, recall@5):
**16 positives after (calib 5 / test 11), 21 before (10 / 11)** — 4–5×
every old-track framing — with 86–91 queries where BM25 already suffices
or ties, dense beating BM25 by +0.11–0.14 R@5 (first EV-positive rung in
the project), and the decision sitting **before** the embedding API call
(~2 ms local vs ~450 ms provider): savings in API calls avoided, not
milliseconds of fusion. Fully offline-evaluable from frozen rows.

## Resolved sub-questions

* Sufficiency is operationalizable (label-derived recall-at-max oracle +
  query/BM25-state inputs, no circularity) — as the new track's framing.
* Stopping is genuinely different as measurement (no counterfactual rung
  outcome needed) but needs an actionable rung — provided here by Dense.
* Intervention inventory: only BM25→Dense has headroom; documented as the
  project constraint (adaptive retrieval = deciding when to pay for Dense).
* Benchmark: current 107-query set supports cheap-first (16–21 positives)
  and cannot support post-dense routing — same corpus, different question;
  benchmark expansion is the named fallback, not a prerequisite.

## Selection

Five options scored (matrix in results): cheap-first adopted; stopping
folded in as framing; answerability deferred (new cost + weak power);
benchmark-track as fallback; premature stop rejected. Phase 14 design
specified (question, falsifiable hypothesis, T0/T1 decision point,
observables, oracle, baselines incl. random-at-rate, leakage controls,
success/failure criteria, step-0 rung screening) — design only, no
implementation. Preserved narrative: P9 dispersion → P10 non-prediction →
P11 mechanism → P12 stop → P13 original question restored at the pre-cost
decision point with an EV-positive rung.

## Deliverables

`docs/phases/phase-13.md`, `docs/phase13_results.md`, this entry. No
source, test, experiment, retrieval, routing, or runner change; no new
tests (no reusable analysis code introduced).

---

# Phase 14 — Cheap-First Adaptive Retrieval (BM25 → Dense)

Status: **COMPLETE** — verdict `INSUFFICIENT EVIDENCE`
Plan of record: [`docs/phases/phase-14.md`](phases/phase-14.md) (frozen
before any split-separated Phase 14 number was computed)
Results of record: [`docs/phase14_results.md`](phase14_results.md)

Phase 14 executed the Phase 13 track: after cheap BM25, can pre-Dense
information predict whether the embedding call is worthwhile? Fully
offline over frozen rows/traces/tables; test opened exactly once, after
the freeze; replication arm reused the frozen rule with no refit.

## Oracle

`oracle_escalate ⟺ recall@5(dense) − recall@5(bm25) ≥ 0.01` (ties/harm →
stop). YES 16 after / 21 before (5+10 calib, 11+11 test) — the Phase 13
counts reproduced exactly. Integrity: DDR/gap/slope recomputed from
frozen rows/traces match 107/107 per arm; T0 features present 107/107;
splits 47/60; dataset sha matches.

## Frozen rule

Six bounded single-feature rules (4 BM25-state + 2 T0 query), decile
grids, pooled 1-SE selection, control precedence not triggered:
**`bm25_slope ≥ −0.915`** (flat BM25 decay → escalate), rate 0.191 on
after-calib (TP 2/FP 7/FN 3/TN 35), consistency on before-calib holds
(0.8759 ≥ BM25-only 0.7695, TP 6). 35 rules within 1 SE — flat
landscape, declared limitation. DDR@bm25 (AUC 0.586) loses the family
to BM25-slope (descriptive AUC 0.757).

## Test (one pass, n=60)

Adaptive +0.078/+0.050 R@5 over BM25-only at 21.7% dense-call rate
(78.3% calls avoided), TP 5/FN 6 both arms, but −0.081/−0.086 vs
dense-only; random ablation P = 0.029 after / 0.210 before; no comparison
Holm-significant. Success criterion (a) fails outright; no FAILURE bar
met → INSUFFICIENT EVIDENCE. Mechanism: BM25-side flat-decay signal
finds half the positives, misses half — selection on 5 positives is
noise-exposed, and the protocol's guards prevented reading it as signal.

## Deliverables

`src/adaptive_rag/evaluation/cheapfirst.py`
(`phase14_cheapfirst_v1`; pre-Dense allowlist enforced in code),
`scripts/phase14_{build_oracle,signal_analysis,evaluate_policy,figures}.py`,
`tests/test_cheapfirst_oracle.py` (20 tests),
`experiments/phase14/` (oracle tables + provenance, calibration
analyses, `frozen_policy.json`, test evaluations + per-query rows, 4
figures + manifest). Full suite green including architecture guards; no
Phase 8–13 artifact, retrieval, routing, or runner code touched. The new
track stays open but unproven; the named sequel is a powered-up
cheap-first test, not a re-sweep.

---

# Phase 15 — Powered Cheap-First Confirmation (BM25 → Dense)

Status: **COMPLETE** — verdict `INSUFFICIENT EVIDENCE` (4 of 5 frozen
success bars; the dense-margin bar missed by 0.004, no failure bar met)
Plan of record: [`docs/phases/phase-15.md`](phases/phase-15.md) (frozen
before any powered retrieval ran)
Results of record: [`docs/phase15_results.md`](phase15_results.md)

Phase 15 re-tested the exact frozen Phase 14 rule (`bm25_slope ≥
−0.915…`, Option A, no re-estimation) on 110 freshly curated queries per
arm (all test; C4 benchmark-expansion fallback from Phase 13).

## Powered benchmark

`data/evaluation/phase15_eval_v1.jsonl` + ledger (new data files, not
code): 110 blind-curated queries (category quotas exact; all 14 paper
minimums met; 0 duplicate flags vs 266 existing queries). Dataset gate
green with G7 enabled (0 contaminated); `single_split_test_only`
recorded via the pre-registered `--allow-test-only` flag (default gate
behavior unchanged). Curation → gate → freeze → retrieval order held.
Corpus note: `pyserini_lin_2021` chunks hold a biomedical-KG survey, not
the Pyserini paper — curated against actual content. Oracle positives on
fresh data: 14 after / 11 before (shortfall vs ~20 expected; reported,
not repaired, per protocol).

## Powered test (one pass, frozen rule)

After: BM25 0.8545 / adaptive 0.9129 / dense 0.9432 (Δ +0.058/−0.030),
rate 0.282, TP 9/14, P_random 0.008. Before: 0.8795 / 0.9242 / 0.9424
(Δ +0.045/−0.018), rate 0.318, TP 9/11, P_random 0.019. MRR likewise
between arms (adaptive-vs-BM25 significant both arms). Combined:
adaptive 0.9186 vs dense 0.9428 (gap −0.0242 vs −0.02 bar);
P_comb = 0.001; cluster CI +0.052 [0.011, 0.098]. ~70% of Dense calls
avoided. No recall@5 paired comparison Holm-significant (coarse metric,
~90% ties); bootstrap CIs exclude 0 on the benefit direction.

## What changed vs Phase 14

The open question — mixed random ablation — is closed affirmatively:
the rule beats random spending decisively on both arms and combined.
Catch rate improved (18/25 vs 10/22) at higher spend (~30% vs 22%).
The verdict still turns on the dense margin, missed by 0.004.

## Deliverables

`src/adaptive_rag/evaluation/powered.py` (combined randomization test,
cluster bootstrap, curation screens; pure functions),
`scripts/phase15_{build_oracle,evaluate,figures}.py`,
`tests/test_powered_confirmation.py` (20 tests incl. Phase 14 oracle
equivalence), `--allow-test-only` gate flag + test,
`experiments/phase15/` (gate artifact, 6 E1 runs 110/110 ok, oracle +
provenance, evaluations + per-query rows, combined test, 5 figures +
manifest). Full suite + guards below; no retrieval, routing, runner,
evaluation-library, or Phase 8–14 artifact touched. Per the §29 tree:
the remaining question is the claim margin, not statistical power —
further work needs a new information source, not another run on this
benchmark.

# Presentation Frontend (post-research, no new experiments)

Streamlit demo over the frozen project (`frontend/`, `pip install -e ".[frontend]"`,
`streamlit run frontend/app.py`). Eight pages: Overview, Query Playground (live
retrieval over the 14-paper corpus, BM25 offline; provider arms when
`AICREDITS_API_KEY` is set), Strategy Comparison (sequential, one shared Qdrant
client, paced provider calls, Jaccard overlap), Adaptive Decision (live routing
trace viewer), Benchmarks (frozen E1 / Phase 8 / oracle-ceiling / Phase 15 graphs
from recorded artifacts), Experiment Explorer (on-disk `rows.jsonl`, working
tree only), Research Journey (15-phase timeline + Phase 10/11 failure analysis),
Findings. Committed `frontend/data/{headline,phases}.json` projections of
already-committed results make the demo render on a fresh clone where
`storage/` and `experiments/` are gitignored; every number is tagged LIVE,
FROZEN or WORKING-TREE and nothing is simulated. `tests/test_frontend/` (24
offline tests incl. an overclaiming-vocabulary guard). No retrieval, routing,
evaluation or experiment code touched; full suite 1056 + 24 green.
