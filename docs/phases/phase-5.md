# Phase 5 — Reranking (Second-Stage Cross-Encoder)

**Status:** COMPLETE — live benchmark executed; reranking measured and recorded as observed
**Phase:** 5
**Purpose:** Add a second-stage cross-encoder that re-scores the candidate list
produced by any first-stage retriever, without modifying that retriever, and
evaluate the quality/cost trade-off on the identical benchmark.

---

## 1. Objective

Implement a second-stage reranking strategy that:

1. Composes any existing `Retriever` — dense, BM25, or hybrid — instead of
   reimplementing candidate generation.
2. Re-scores candidates with a real cross-encoder rather than a heuristic, and
   reorders on that signal alone.
3. Conforms to the existing `Retriever` protocol and `RetrievalResponse`
   contract, so the runner and both evaluators are unchanged.
4. Fails loudly by default: a broken reranker produces
   `status="retrieval_failed"`, never a silently un-reranked answer.
5. Records its own cost separately, so the quality gain can be priced against
   the latency it costs.
6. Remains a composition layer — no tokenization of the corpus, no index, no
   candidate-depth heuristics, and no decision about *when* to rerank.

---

## 2. Scope

### Included

* `reranking/base.py` — the `Reranker` protocol, isolated from every retriever,
  index, and corpus artifact
* `reranking/onnx_backend.py` — `OnnxCrossEncoderReranker`, the only module that
  performs model inference
* `retrieval/reranked.py` — `RerankedRetriever`, conforming to `Retriever`
* `RetrievalConfig` support for `dense_rerank` / `bm25_rerank` /
  `hybrid_rerank`, plus the `rerank_*` parameters; `RerankerConfig` dedicated
  model
* `RetrievalResult` provenance fields (`retrieval_score`, `retrieval_rank`) and
  `RetrievalMetadata` second-stage diagnostics
* `ExperimentTrace` latency/count split, so `EfficiencyEvaluator` can report
  rerank cost without a parallel evaluator
* `instantiate_components()` second-stage wrap, with hybrid fusion depth coupled
  to the rerank depth
* `run_experiment.py --rerank` plus explicit overrides
* `compare_reranking.py` (base-vs-reranked table + depth ablation)
* Unit, integration, regression, comparison-CLI, and architecture-guard tests

### Excluded

* Adaptive routing, query classification, strategy selection (Phase 6)
* Any change to `retrieval/dense.py`, `retrieval/bm25.py`,
  `retrieval/hybrid.py`, `retrieval/fusion.py`, `indexing/*`,
  `evaluation/retrieval.py`, `evaluation/generation.py`, `generation/*`,
  `rag.py`, `scripts/ask.py`, or `scripts/compare_retrievers.py`
* Confidence-based or value-based decisions about *whether* to rerank — Phase 5
  reranks unconditionally whenever it is enabled
* Dynamic candidate depth — the depth is a configured constant
* A new index, `storage/hybrid/`, or `storage/reranked/`
* Caching of reranker outputs; `storage/reranker/` holds only the downloaded
  model artifact
* Score mixing: rerank scores are never blended with first-stage scores

---

## 3. Architecture

```text
        Canonical Chunk Corpus
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 DenseRetriever        BM25Retriever            (unchanged, untouched)
       │                   │
       └─────────┬─────────┘
                 ▼
        HybridRetriever (RRF)                   (unchanged, untouched)
                 │  RetrievalResponse  ≤ N candidates
                 ▼
       ┌─────────────────────┐
       │   RerankedRetriever  │   candidate_generation_latency_ms
       │                     │   ← base response latency, reused verbatim
       │   1. guard empty query → InvalidQueryError
       │   2. base.retrieve(query, top_k=rerank_candidate_k)
       │   3. empty candidates → status="no_results",
       │                          rerank_latency_ms = 0.0,
       │                          the model is never invoked
       │   4. reranker.score(query, [c.text ...])   ──► rerank_latency_ms
       │   5. sort by (-rerank_score, retrieval_rank, chunk_id)
       │   6. truncate to top_k, rank = 1..n
       │   7. latency_ms = candidate_generation + rerank
       └─────────────────────┘
                 │  RetrievalResponse  retrieval_method="<base>_rerank"
                 ▼
        ExperimentRunner (unchanged except the trace copy)
                 │
      ┌──────────┼───────────────┐
      ▼          ▼               ▼
 RetrievalEvaluator  GenerationEvaluator  EfficiencyEvaluator
 (unchanged)         (unchanged)          (+ conditional rerank metrics)
```

```text
  RerankedRetriever
        │ holds only
        ├── base_retriever: Retriever      (the candidate generator)
        └── reranker: Reranker             (query + passages → one score each)

  It holds no vector store, no BM25 index, no embedding model, and no corpus
  handle. It never imports adaptive_rag.retrieval.{dense,bm25,hybrid,fusion},
  adaptive_rag.indexing, or adaptive_rag.embeddings.
```

The wrapper carries the base response's `RetrievalMetadata` through with
`model_copy(update=...)` rather than rebuilding it. Phase 4 added fusion
diagnostics and Phase 2/3 added dense/lexical fields; hand-listing them would
silently drop whichever set grew last.

---

## 4. Implementation

### 4.1 `Reranker` protocol

```python
@runtime_checkable
class Reranker(Protocol):
    version: str
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage, in input order."""
```

Contract, enforced rather than assumed:

* returned length **equals** input length; a mismatch is a typed
  `RerankingError`, never a silent `zip` truncation
* `scores[i]` corresponds to `passages[i]`
* scores are model-specific ranking signals, **not** calibrated probabilities

The protocol takes the whole batch in one call. Batching lives inside the
backend, which is what makes batch-size invariance a property of the
implementation rather than a caller convention.

### 4.2 `OnnxCrossEncoderReranker`

The only module that runs a model. `onnxruntime` and `tokenizers` are imported
**inside** `_load_artifacts`, and the artifact is fetched on first `score()`
call rather than at construction — so importing the module never touches the
network and offline unit tests never fail on a missing download.

* Artifact cached at `storage/reranker/<model_id>/` via
  `huggingface_hub.snapshot_download(..., local_dir=...)`; reused afterwards
* Requires `onnx/model.onnx`, `tokenizer.json`, `config.json`; a missing file
  raises `RerankerModelError` naming the exact path
* Pairs encoded as `(query, passage)`; truncation at `max_length` (default 512),
  padding within a batch only
* Providers: `cpu` → `["CPUExecutionProvider"]`; `cuda` →
  `["CUDAExecutionProvider", "CPUExecutionProvider"]` with an explicit
  `RerankerUnavailableError` if CUDA is requested but absent — never a silent
  CPU downgrade; `auto` → CUDA when available else CPU, logged once
* Output decoding from `config.json` `id2label`: `num_labels == 1` → raw logit;
  `num_labels == 2` → softmax, taking the `label_1` / `"relevant"` index
* Batched in `batch_size` chunks (default 16). Scoring is per-pair with no
  cross-pair interaction, so output is **batch-size invariant** — asserted in
  tests rather than assumed
* Default model `Xenova/ms-marco-MiniLM-L-6-v2` (~22 M params, ~90 MB ONNX);
  id and revision pinned in config and hashed

### 4.3 `RerankedRetriever`

`retrieve()` flow, in order:

1. Empty or whitespace query → `InvalidQueryError`, **before** the base
   retriever is touched.
2. `base.retrieve(query, top_k=config.rerank_candidate_k, score_threshold=None,
   filters=filt)` → ≤ N candidates. `candidate_generation_latency_ms` is the
   **base response's own** `latency_ms`, not a fresh `perf_counter` around the
   call: reusing the base number is what makes a reranked run's
   `retrieval_latency_ms` directly comparable to a non-reranked run's, rather
   than differing by wrapper overhead.
3. Empty candidate set is **not** a failure: `status="no_results"`,
   `results=[]`, `rerank_latency_ms == 0.0`, and the reranker is **not called
   at all**. Distinct from a reranker exception, which propagates.
4. Reranker call timed into `rerank_latency_ms`.
5. Sort by `(-rerank_score, retrieval_rank, chunk_id)`. The pre-rerank rank and
   then the chunk id make the order total, so equal scores never depend on input
   order and repeated runs are stable. Mirrors the tie-break in
   `retrieval/fusion.py`.
6. Truncate to `top_k`, renumber `rank = 1..n`, and rewrite each result with
   `score = rerank_score`, `retrieval_score = original score`,
   `retrieval_rank = original rank`. `text`, `metadata`, and `provenance` are
   carried through untouched.
7. `latency_ms = candidate_generation_latency_ms + rerank_latency_ms` (full
   end-to-end, the Phase 4 hybrid rule). `search_latency_ms` passes through from
   the base response.
8. **Failure.** Any reranker exception propagates unchanged. The only
   `try`/`except` in the module is gated on `config.rerank_fallback`; when taken
   it returns the candidates truncated to `top_k` with `score` and
   `retrieval_score` both equal to the original retrieval score, and sets
   `RetrievalMetadata.rerank_fallback = True` so it surfaces in every trace, the
   `rerank_fallback_count` metric, and the manifest.

The ordering rule lives in a module-level function, `apply_rerank_scores`, so it
is unit-testable with a fake scorer and no retriever:

```python
scored = [
    result.model_copy(update={
        "score": float(rr_score),
        "retrieval_score": result.score,
        "retrieval_rank": result.rank,
    })
    for result, rr_score in zip(results, scores, strict=True)
]
scored.sort(key=lambda r: (-r.score, r.retrieval_rank or 0, r.chunk_id))
selected = [r.model_copy(update={"rank": i}) for i, r in enumerate(scored[:top_k], 1)]
```

---

## 5. Configuration

```python
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
```

The same fields live inside `RetrievalConfig` prefixed `rerank_*`, so one hashed
object covers the whole retrieval pipeline — the same choice Phase 4 made for
`rrf_k` / `candidate_k`. All are covered by `config_hash`.

`RetrievalConfig` additions:

| Field | Default | Meaning |
| --- | --- | --- |
| `rerank_enabled` | `False` | Wrap the base retriever in `RerankedRetriever` |
| `rerank_model_id` | `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder artifact |
| `rerank_model_revision` | `None` | Pinned revision for reproducibility |
| `rerank_device` | `"auto"` | `auto` / `cpu` / `cuda` |
| `rerank_batch_size` | `16` | Scoring batch size |
| `rerank_max_length` | `512` | Tokenizer truncation length |
| `rerank_candidate_k` | `20` | First-stage depth fed to the reranker |
| `rerank_fallback` | `False` | Opt-in degradation instead of failing |

Validator additions to `validate_method_and_version`:

* `default_version` gains `dense_rerank → dense_rerank_v1`,
  `bm25_rerank → bm25_rerank_v1`, `hybrid_rerank → hybrid_rerank_v1`
* when `rerank_enabled`: `rerank_candidate_k >= top_k`, else `ValueError`
* when `retrieval_method == "hybrid_rerank"`: `candidate_k >=
  rerank_candidate_k`, so the fused pool can actually feed the requested depth

`build_experiment_config()` adds `component_versions["reranker"]` **only** when
reranking is enabled, so existing manifests are byte-identical.

---

## 6. CLI

```bash
# second stage on
scripts/run_experiment.py --retriever dense  --rerank --name dense_rerank_v1  --no-judge --no-generation
scripts/run_experiment.py --retriever bm25   --rerank --name bm25_rerank_v1   --no-judge --no-generation
scripts/run_experiment.py --retriever hybrid --rerank --name hybrid_rerank_v1 --no-judge --no-generation

# depth ablation, final top_k held at 10
for k in 10 20 40; do
  scripts/run_experiment.py --retriever hybrid --rerank \
      --rerank-candidate-k $k --name hybrid_rerank_k${k}_v1 --no-judge --no-generation
done

scripts/compare_reranking.py \
    --dense-run  experiments/<dense_baseline_v1> \
    --dense-rerank-run  experiments/<id-dense_rerank_v1> \
    --bm25-run   experiments/<bm25_baseline_v1> \
    --bm25-rerank-run   experiments/<id-bm25_rerank_v1> \
    --hybrid-run experiments/<hybrid_baseline_v1> \
    --hybrid-rerank-run experiments/<id-hybrid_rerank_v1>
```

`--rerank-candidate-k` also raises hybrid's own `candidate_k` to match, so one
flag controls depth end-to-end. The hybrid dual-index emptiness check reaches
the BM25 index through `retriever.base_retriever` when the wrapper is present.

`compare_reranking.py` performs no globbing: every run directory is supplied
explicitly, because a rerank run is only comparable against the exact baseline
it was built from. It exits 2 on corpus, trace-count, or `retrieval_method`
mismatch.

---

## 7. Evaluation

### 7.1 Retrieval quality — no evaluator change required

`evaluation/retrieval.py` reads `chunk_id`, `metadata.document_id`, `rank`, and
list order. Reranking improves *ordering*, and the evaluator consumes order, so
no retrieval metric needed modification.

`_reciprocal_rank` uses `result.rank`, not list position. This is why
renumbering `rank = 1..n` after the reorder is load-bearing rather than
cosmetic: it is the renumbering that moves MRR.

### 7.2 Cost — additive, conditional metrics

`ExperimentTrace` gains five optional `None`-defaulted fields
(`candidate_generation_latency_ms`, `rerank_latency_ms`,
`rerank_candidate_count`, `rerank_result_count`, `rerank_fallback`), copied from
retrieval metadata in `_run_example`. The `None` defaults are required: the
Phase 4 run directories on disk must still validate against the extended schema.

`EfficiencyEvaluator` emits nine new metrics — but **only** when at least one
trace carries the split, so a non-reranked run's `metrics_efficiency.json` is
unchanged:

```
candidate_generation_latency_ms_{mean,p50,p95}
rerank_latency_ms_{mean,p50,p95}
rerank_candidate_count_mean
rerank_result_count_mean
rerank_fallback_count
```

### 7.3 Methodology and the latency comparability caveat

> `dense_baseline_v1` on disk was a *full* run with generation and an LLM judge;
> `bm25_baseline_v1` and `20260929T175539Z-hybrid_baseline_v1` were
> retrieval-only. Because reranking changes what reaches the generator, the
> clean comparison is retrieval-only across all six configurations.
> Re-run dense and BM25 with `--no-judge --no-generation` for a strictly
> like-for-like latency table, and state in §8 which run supplied each column.

**Resolved in §8.4.** Dense and BM25 were re-run retrieval-only as
`dense_baseline_ro_v1` and `bm25_baseline_ro_v1`; both reproduce their original
retrieval metrics exactly, so the confound was the generation/judge harness
rather than retrieval. All six columns in §8 are now retrieval-only.

Reranked dense and hybrid runs need `AICREDITS_API_KEY` (query embeddings, as
in Phase 4). **All** reranked runs need network on first use to download the
ONNX artifact; after that they are offline.

Outcome is recorded **as observed**. This phase does not assume reranking wins.

### 7.4 Known consequence: raw logits in the prompt

`ContextBuilder.build()` sorts by `rank` and interpolates `score=` into the
prompt text. Because Phase 5 sets `score` to the rerank score, a reranked run's
generation prompt shows raw cross-encoder logits. This is consistent with the
decision that `score` holds the signal that produced the current order (the
same rule Phase 4 used for RRF scores), and Phase 4 already surfaced RRF
values the same way.

---

## 8. Results

**MEASURED.** All six configurations and the depth ablation were executed on a
CPU-only host (12 cores, no GPU), retrieval-only, against the canonical corpus
`corpus_6c416f423920385d` and the 20-example `dense_eval_v1` benchmark. The
reranker was pinned to `Xenova/ms-marco-MiniLM-L-6-v2` at revision
`a09144355adeed5f58c8ed011d209bf8ee5a1fec`, `batch_size=16`, `max_length=512`,
`device=auto` resolving to CPU.

Latency columns are per-query means in milliseconds. `retr. lat` is end-to-end
retrieval; `cand. gen` and `rerank` are its two stages, so
`retr. lat` = `cand. gen` + `rerank`. Baseline rows carry no split, so those two
cells read `--` by design rather than as missing data.

| Strategy | R@1 | R@5 | R@10 | MRR | nDCG@10 | retr. lat (ms) | cand. gen (ms) | rerank (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 0.8417 | 0.8417 | 0.8583 | 0.9500 | 0.9308 | 621.35 | -- | -- |
| dense_rerank | 0.6917 | 0.8583 | 0.8750 | 0.8667 | 0.8646 | 7708.96 | 881.96 | 6826.99 |
| bm25 | 0.5417 | 0.7167 | 0.7833 | 0.6771 | 0.7378 | 1.71 | -- | -- |
| bm25_rerank | 0.4667 | 0.6583 | 0.8083 | 0.6162 | 0.6713 | 4014.21 | 3.81 | 4010.40 |
| hybrid | 0.6417 | 0.8583 | 0.8750 | 0.8200 | 0.8557 | 721.00 | -- | -- |
| hybrid_rerank | 0.4667 | 0.7583 | 0.8750 | 0.6926 | 0.7567 | 4447.68 | 960.60 | 3487.09 |

Depth ablation (`hybrid_rerank`, `top_k = 10` held constant):

| candidate_k | R@5 | MRR | rerank latency (ms) | cand. gen (ms) |
| --- | --- | --- | --- | --- |
| 10 | 0.7917 | 0.7921 | 3003.67 | 836.84 |
| 20 | 0.7583 | 0.6926 | 3487.09 | 960.60 |
| 40 | 0.7333 | 0.7058 | 7667.37 | 926.99 |

### 8.1 Reranking did not pay for itself on this benchmark

**Recorded as observed: reranking was a net loss on every strategy here.** The
result is consistent across the three first stages rather than noisy. Reranking
*bought* a little tail recall and *spent* more head precision.

* Recall@10 improved or held on all three: dense +0.0167, BM25 +0.0250, hybrid
  +/-0.0000. The reranker does surface relevant chunks the first stage missed.
* Every head metric moved the other way, on all three strategies. MRR fell
  -0.0833 (dense), -0.0609 (BM25), -0.1274 (hybrid); nDCG@5 fell -0.1270,
  -0.1484, -0.1507; Precision@5 fell -0.1600, -0.1500, -0.1500.
* The same direction is monotone in candidate depth (§8.2): deeper pools give
  worse R@5 and worse MRR at sharply rising cost.

Per-trace inspection explains the shape. In the hybrid run the reranker pulled a
candidate from *beyond* the first-stage top-10 in **20/20** traces, so it is
genuinely reordering rather than no-op'ing; yet the reranked #1 equals the
first-stage #1 in only **3/20** traces. The cross-encoder applies a real but
poorly-calibrated ordering signal to this corpus, discarding strong first-stage
rankings near the top in favour of deeper candidates it scores marginally
higher.

The likely cause is a **domain mismatch** the phase design anticipated but could
not avoid. `ms-marco-MiniLM-L-6-v2` is a ~22 M-parameter general-web MS-MARCO
cross-encoder, scored here against 14 dense academic RAG papers whose chunks
average ~1 870 characters (~470 tokens, right at the `max_length=512`
truncation boundary). Short, citation-heavy academic prose is a different
distribution from the MS-MARCO passages the model was trained to rank. Because
reranking is a *composition layer* by design, this is a finding about the model
choice rather than a defect in the mechanism; substituting a domain-specific
cross-encoder is out of Phase 5 scope.

### 8.2 Depth ablation: more candidates, worse and slower

The depth curve runs against the usual second-stage story. R@5 falls
0.7917 -> 0.7583 -> 0.7333 as the pool widens 10 -> 20 -> 40, while rerank
latency climbs 3.00 s -> 3.49 s -> 7.67 s. `k=10` is simultaneously the cheapest
*and* the most accurate reranked configuration measured.

A deeper pool gives the cross-encoder more opportunities to promote a spurious
candidate, and on this corpus its judgements degrade faster than the extra recall
is worth. Cost per unit of quality is monotonically unfavourable beyond `k=10`.

### 8.3 Cost: the predicted shape was wrong by two orders of magnitude

§4 and §7 expected "tens of milliseconds" for ~40 pairs and treated the reranker
as nearly free next to dense's ~750 ms network embedding. **Measurement refutes
that expectation.** A warm-session micro-benchmark over real corpus chunks gives
~155 ms per pair (10 pairs 1 519 ms, 20 pairs 3 209 ms, 40 pairs 6 221 ms), so
the 20-candidate configuration costs **~4.0 s** end to end against BM25's
**1.7 ms** -- roughly a 2 300x slowdown over the cheapest first stage, and about
11x dense's own 621 ms, which is itself dominated by the query-embedding round
trip.

That cost is linear in candidate count and is intrinsic to CPU inference on
512-token sequences rather than to this implementation: batch-size invariance is
asserted in tests, the split is recorded per stage, and no fallback fired
(`rerank_fallback_count = 0` in every run). A GPU deployment would change the
absolute numbers; the *ranking* conclusion would not, since that is a quality
result.

Recording the split per stage is what made this legible: the total alone would
have read as "reranking is expensive" without separating a 3-7 s cross-encoder
from a 1-3 ms first stage.

### 8.4 Run provenance

Every row above is retrieval-only (`--no-judge --no-generation`), so the six
configurations are like-for-like. `empty: 20` in each manifest is the runner's
label for a retrieval-only run with no generator, not a retrieval failure; every
run recorded 20/20 traces with `retrieval_failed: 0` and `rerank_fallback_count: 0`.

| Column | Run directory |
| --- | --- |
| dense | `20260930T170100Z-dense_baseline_ro_v1` |
| dense_rerank | `20260930T170129Z-dense_rerank_v1` |
| bm25 | `20260930T170121Z-bm25_baseline_ro_v1` |
| bm25_rerank | `20260930T170537Z-bm25_rerank_v1` |
| hybrid | `20260929T175539Z-hybrid_baseline_v1` |
| hybrid_rerank / k=20 | `20260930T170835Z-hybrid_rerank_v1` |
| k=10 | `20260930T171229Z-hybrid_rerank_k10_v1` |
| k=40 | `20260930T171504Z-hybrid_rerank_k40_v1` |

Dense and BM25 were re-run retrieval-only to resolve the §7.3 comparability
caveat: the `dense_baseline_v1` run on disk was a *full* run with generation and
an LLM judge, so its latency was not comparable to the other five. The retrieval
metrics reproduce exactly (dense R@5 0.8417, BM25 R@5 0.7167), confirming
retrieval parity; only the harness differed. The hybrid baseline was already
retrieval-only on the same corpus with the same 20 traces and `top_k=10`, and
`compare_reranking.py` verified corpus, trace count, and retrieval method across
all three pairs before emitting the tables.

The k=20 ablation row reuses the `hybrid_rerank_v1` run because its config is
identical (default `rerank_candidate_k=20`); a separate k=20 run would be the
same experiment twice.

---


---

## 9. Validation

**136 offline deterministic tests pass, 2 deselected (`integration`).** No
credentials, no network, no live indexes.

* **Scoring and ordering**: rerank score drives order (an inverting scorer
  produces the inverted output); `score == rerank_score`,
  `retrieval_score == original`, `retrieval_rank == original` for every result;
  original ranks survive renumbering in a genuine inversion.
* **top_k**: 20 candidates at `top_k=5` → exactly 5 results, ranks 1–5,
  `candidate_count=20`, `result_count=5`, latency recorded; `top_k` above the
  candidate count returns all results with no padding.
* **Candidate depth**: `StubRetriever` records the `top_k` it received;
  parametrized over 12 / 20 / 40, the base is asked for `rerank_candidate_k`,
  never the final `top_k`.
* **Hybrid depth coupling**: with `hybrid_rerank` at `rerank_candidate_k=40`,
  both fused branches are queried at 40, not 20.
* **Empty candidates**: `status="no_results"`, `results=[]`,
  `rerank_latency_ms == 0.0`, and the fake scorer's call count is **0** — no
  model invocation on an empty pool.
* **Determinism**: an all-equal scorer falls back to `retrieval_rank` then
  `chunk_id`; two identical runs produce identical ids and scores; output at
  `batch_size=1` and `64` is identical.
* **Passthrough fidelity**: `document_id`, `doc_title`, `section_path`,
  `headings`, `element_ids`, `page_start`/`page_end`, `token_count`,
  `provenance.document_id`/`pages`/`source_sha256`, and byte-identical `text`.
* **Latency accounting**: `latency_ms == candidate_generation_latency_ms +
  rerank_latency_ms` and `>=` the base's `latency_ms`.
* **Failure**: the original exception type and message propagate; through
  `ExperimentRunner` the trace is `status="retrieval_failed"`,
  `retrieval is None`, `error.stage == "retrieval"`, `error_type` preserved and
  surfaced in the manifest.
* **Fallback**: off by default (a raising scorer never yields a response); when
  enabled, results are truncated to `top_k` with `score == retrieval_score`,
  `retrieval_rank == rank`, and `rerank_fallback is True`.
* **Empty query**: `InvalidQueryError` with both the base's and the scorer's
  call counts at 0.
* **Protocol**: `isinstance(retriever, Retriever)`, `isinstance(reranker,
  Reranker)`, and `method` equals the expected `<base>_rerank` for all three
  base strategies.
* **Score contract**: a scorer returning 2 scores for 3 candidates raises
  `RerankingError` rather than silently dropping a chunk.
* **Config**: version alignment for all three methods;
  `rerank_candidate_k < top_k` raises; `hybrid_rerank` with `candidate_k <
  rerank_candidate_k` raises; `config_hash` changes when `rerank_model_id`,
  `rerank_candidate_k`, or `rerank_device` changes.
* **Offline integration, real components**: dense → rerank over in-memory
  Qdrant + `FakeEmbeddingModel`, BM25 → rerank over an in-memory `BM25Index`
  built from the same chunks, and hybrid → rerank over real dense + real BM25 +
  `HybridRetriever` — all driven through `ExperimentRunner` with
  `RetrievalEvaluator` + `EfficiencyEvaluator`, asserting artifacts written and
  `rerank_latency_ms_mean` non-null. The hybrid case additionally asserts that
  `fusion_method`, `rrf_k`, `dense_candidate_count`, `bm25_candidate_count`, and
  `fusion_latency_ms` all survive into the reranked metadata.
* **Canonical-corpus regression** (auto-skips when chunks are absent):
  "Robertson", "ColBERT", and "BM25" still surface the expected documents after
  reranking, the reranked set is a subset of the first-stage pool, and ranks are
  renumbered 1–10 — a reranker reorders, it never removes what retrieval found.
* **Regression, reranking disabled**: `instantiate_components()` returns the
  plain `DenseRetriever` / `BM25Retriever` / `HybridRetriever` (asserted via
  `type(...) is`, not `isinstance`) and no `reranker` key appears in
  `component_versions`; a non-reranked run's efficiency report gains no new
  metric names; the existing `bm25_baseline_v1` and hybrid run directories on
  disk still validate against the extended `ExperimentTrace`.
* **Comparison CLI**: driven as a subprocess against fixture run directories —
  base-vs-reranked table, corpus mismatch → exit 2, method mismatch → exit 2,
  missing baseline → exit 1, and no auto-discovery of rerank runs. A separate
  test pins that `compare_retrievers.py` still contains neither `rerank` nor the
  Phase 5 header.
* **Architecture guards**: `"rerank"` removed from
  `FORBIDDEN_STRATEGY_TOKENS` while `adaptive_rout`, `query_classif`, and
  `strategy_select` stay banned — **Phase 6 remains guard-banned**.
  `FORBIDDEN_HEAVY_DEPS` is unchanged and now also asserts `onnxruntime` and
  `tokenizers` are declared while `torch` / `transformers` appear nowhere in
  `src/`. New guards: `test_reranker_has_no_index_access`,
  `test_reranked_retriever_only_composes` (exactly one `ast.Try`, and its
  handler must both re-raise and reference `config.rerank_fallback`),
  `test_hybrid_is_independent_of_reranking`, and
  `test_reranking_does_not_leak_into_retrievers`. Each was verified to fail
  against a deliberately broken variant rather than merely passing vacuously.
* **Integration-marked** (deselected by default via the existing
  `addopts = "-m 'not integration'"`): `test_onnx_cross_encoder_backend_runs`
  asserts deferred loading, session construction, a 4-passage batch, batch-size
  invariance against `batch_size=1`, and no NaN logits;
  `test_onnx_backend_rejects_a_missing_artifact` asserts a `RerankerModelError`
  naming the missing path. Run with `pytest -m integration tests/test_reranking.py`
  on the networked machine.

### Live experiment

**COMPLETE.** The benchmark was executed on a networked, CPU-only host. The
declared dependencies installed cleanly (`onnxruntime` 1.30.0, `tokenizers`
0.23.2, `huggingface-hub` 1.33.0) and the blocking assumption above was
**confirmed rather than assumed**: `Xenova/ms-marco-MiniLM-L-6-v2` resolves at
revision `a09144355adeed5f58c8ed011d209bf8ee5a1fec` and provides
`onnx/model.onnx` (90 992 115 bytes), `tokenizer.json`, and `config.json`. No
model substitution was made and ADR-020 needed no change.

Running the integration-marked ONNX test first, as advised, paid for itself: it
surfaced two latent defects that the offline suite could not reach, both now
fixed and both with regression coverage.

**Fix 1 — the documented integration command could not run.**
`tests/conftest.py` added an unconditional `skip` marker to every
integration-marked item, so `pytest -m integration tests/test_reranking.py`
reported `2 skipped` and never executed the ONNX session, on this machine or
any other. A command-line `-m` overrides the `addopts` `-m 'not integration'`, so
the caller had already opted in and the guard had to yield. The guard now skips
only when no `-m` expression requests integration, and still skips for
`-m "not integration"`. The default run is unchanged at `136 passed, 2
deselected`.

**Fix 2 — two live-path assertions were unfalsifiable.** The live relevance
assertion read `scores[...] > scores[0] * 0`, which compares a score with itself
against a zero baseline and never ranks anything; this model's head is a single
raw regression logit whose scale is entirely negative for non-matches, so it
could never hold. It now asserts the ordering the comment describes (the BM25
passage outranks the unrelated fusion passage). Separately, `snapshot_download`
was unwrapped, so a missing repository leaked a raw `huggingface_hub` error
instead of the documented `RerankerModelError`; download failures are now
translated into the project's own typed error, with the original exception
chained as `__cause__`.

After the fixes: `pytest -m integration tests/test_reranking.py` → **2 passed**,
and the offline suite → **136 passed, 2 deselected**.

Live benchmark: **COMPLETE.** Six primary configurations plus the k = 10 / 20 / 40
depth ablation, all retrieval-only, on corpus `corpus_6c416f423920385d` and the
20-example benchmark; 20/20 traces each, `retrieval_failed: 0`,
`rerank_fallback_count: 0` in every run. `compare_reranking.py` verified corpus,
trace count, and retrieval method across all three pairs before emitting both
tables. Full results and interpretation in §8; the headline is that reranking
did **not** pay for itself on this corpus, and that the documented cost estimate
was wrong by two orders of magnitude.

---

## 10. Definition of Done

- [x] **Composition, not reimplementation**: `RerankedRetriever` wraps any
      `Retriever`; `dense.py`, `bm25.py`, `hybrid.py`, and `fusion.py` are
      unmodified, and a guard enforces it.
- [x] **Real cross-encoder**: ONNX Runtime over a pre-exported ms-marco
      cross-encoder. No torch, no transformers, `sentence-transformers` still
      banned.
- [x] **Contract compliance**: `Retriever` protocol, `RetrievalResponse` with
      `retrieval_method="<base>_rerank"`, 1-indexed ranks, rerank `score`,
      preserved provenance and first-stage signal.
- [x] **Configuration consistency**: `retrieval_method` ⇔ `retriever_version`
      (`dense_rerank ⇔ dense_rerank_v1`, and likewise for bm25 and hybrid);
      every `rerank_*` field inside the hashed config; `reranker` component
      version present only when enabled.
- [x] **Failure vs. empty handling**: empty query → `InvalidQueryError`;
      reranker failure → `retrieval_failed` with the original error type and no
      fallback by default; empty candidate pool → `no_results` with no model
      invocation; fallback, when opted into, is visible in every trace.
- [x] **Evaluation parity**: consumed unchanged by `ExperimentRunner` and
      `RetrievalEvaluator`; `EfficiencyEvaluator`'s new metrics are additive and
      conditional, so existing runs' output is unchanged.
- [x] **Testing**: 136 offline deterministic tests pass, including guards, plus
      the 2 integration-marked ONNX tests against a real session.
- [x] **Documentation**: `phase-5.md`, `progress.md`, `decision.md` (ADR-020),
      `architecture.md`, `README.md`, `experiments/README.md` updated.
- [x] **Live evaluation**: six primary configurations plus the 10/20/40 depth
      ablation run on a networked machine, compared via
      `compare_reranking.py`, and §8 filled in as observed. Recorded outcome:
      reranking **did not** improve this corpus (§8.1) and cost ~155 ms per
      candidate pair on CPU, two orders of magnitude above the estimate in §4
      (§8.3).