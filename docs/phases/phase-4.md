# Phase 4 — Hybrid Retrieval (Dense + BM25 + RRF)

**Status:** COMPLETE
**Phase:** 4
**Purpose:** Add a rank-fusion hybrid retrieval strategy that composes the
existing dense and BM25 retrievers, without modifying either of them, and
evaluate it on the identical benchmark.

---

## 1. Objective

Implement a hybrid retrieval strategy that:

1. Composes the Phase 2 `DenseRetriever` and the Phase 3 `BM25Retriever`
   instead of reimplementing either one.
2. Fuses the two rankings with Reciprocal Rank Fusion (RRF), a rank-based
   method that never mixes incomparable score scales.
3. Conforms to the existing `Retriever` protocol and `RetrievalResponse`
   contract, so the runner and evaluators are unchanged.
4. Fails loudly: a broken branch produces `status="retrieval_failed"`, never a
   silent single-strategy answer.
5. Remains a composition layer — no tokenization, embedding, index access,
   reranking, or routing.

---

## 2. Scope

### Included

* `retrieval/fusion.py` — the RRF function, isolated and unit-testable with no
  retriever, index, or credential
* `retrieval/hybrid.py` — `HybridRetriever`, conforming to `Retriever`
* `RetrievalConfig` support for `retrieval_method="hybrid"`, `rrf_k`,
  `candidate_k`; `HybridRetrievalConfig` dedicated model
* `RetrievalMetadata` fusion diagnostics
* `instantiate_components()` hybrid branch
* `run_experiment.py --retriever hybrid`
* `compare_retrievers.py --hybrid-run` (opt-in third column)
* Unit, integration, regression, and architecture-guard tests

### Excluded

* Reranking (Phase 5)
* Adaptive routing, query classification, strategy selection (Phase 6)
* Alternative fusion functions (weighted, CombSUM, CombMNZ)
* Any change to `retrieval/dense.py`, `retrieval/bm25.py`, `indexing/*`,
  `evaluation/*`, `experiments/runner.py`, `rag.py`, or `scripts/ask.py`
* A new index or `storage/hybrid/` — hybrid reuses both existing indexes

---

## 3. Architecture

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
      reciprocal_rank_fusion(ranked_lists, rrf_k)
                 ▼
        truncate → top_k, renumber ranks
                 ▼
     RetrievalResponse(retrieval_method="hybrid")
                 ▼
      ExperimentRunner + evaluators (unchanged)
```

`HybridRetriever` holds no index, no embedding model, and no tokenizer. It
receives two objects that already satisfy `Retriever` and calls them.

---

## 4. Implementation

### 4.1 RRF (`src/adaptive_rag/retrieval/fusion.py`)

```python
score(chunk) = Σ_lists 1 / (rrf_k + rank_in_that_list)
```

* **Rank, never score.** Cosine similarity and BM25 magnitudes never meet.
* **Dedup key is `chunk_id`**, never text.
* **Missing document ⇒ 0 contribution.** No padding, no fabricated rank.
  Unequal list lengths and empty lists just make the loop shorter.
* **Payload carrier:** the first list wins. The dense result supplies
  `text` / `metadata` / `provenance` when a chunk appears in both branches;
  a chunk found only by BM25 keeps BM25's payload. Only `score` is replaced.
* **Tie-break:** `(-rrf_score, best_single_source_rank, chunk_id)`. Ties are
  genuinely possible, and `chunk_id` is the final guarantee that ordering never
  depends on input or dict iteration order.
* Negative `rrf_k` raises `ConfigurationError`.

No `FusionStrategy` base class, no registry, no alternative fusion variants.

### 4.2 `HybridRetriever` (`src/adaptive_rag/retrieval/hybrid.py`)

* `method = "hybrid"`; signature matches the `Retriever` protocol exactly
* Empty/whitespace query → `InvalidQueryError("Query cannot be empty")`,
  validated **before** either constituent is called
* Each branch is called with `top_k = candidate_k` (default 20), never the
  caller's `top_k`; truncation to `top_k` and rank renumbering `1..n` happen
  only after fusion
* `filters` are forwarded to both branches
* Constituent exceptions propagate **unchanged** — no re-wrapping, no
  `try`/`except` anywhere in the module (architecture-guard enforced via AST)
* `status="no_results"` only when the fused list is empty

### 4.3 Configuration (`schemas/config.py`)

```python
class RetrievalConfig:
    retrieval_method: Literal["dense", "bm25", "hybrid"] = "dense"
    rrf_k: int = 60
    candidate_k: int = 20
```

The validator keeps `retrieval_method` ⇔ `retriever_version` aligned
(`dense ⇔ dense_v1`, `bm25 ⇔ bm25_v1`, `hybrid ⇔ hybrid_v1`) and rejects
`candidate_k < top_k` for hybrid. `HybridRetrievalConfig` mirrors
`DenseRetrievalConfig` / `BM25RetrievalConfig`.

`rrf_k`, `candidate_k`, `top_k`, `retrieval_method`, and `retriever_version`
all live inside `RetrievalConfig`, which `build_experiment_config()` hashes
into `config_hash` and mirrors into `manifest.json`
(`retrieval_method`, `component_versions.retrieval = "hybrid_v1"`).

### 4.4 Retrieval metadata (`schemas/retrieval.py`)

`RetrievalResponse.retrieval_method` now accepts `"hybrid"`. `RetrievalMetadata`
gained eight **optional** fusion fields, so dense and BM25 responses are
unchanged on the wire:

| Field | Meaning |
| ----- | ------- |
| `fusion_method` | `"rrf"` |
| `rrf_k`, `candidate_k` | fusion parameters actually used |
| `dense_candidate_count`, `bm25_candidate_count` | candidates each branch returned |
| `dense_latency_ms`, `bm25_latency_ms` | each branch's self-reported latency |
| `fusion_latency_ms` | fusion + truncation time |

`latency_ms` is the **full end-to-end hybrid cost** (both branch calls plus
fusion), because it is the single latency the `EfficiencyEvaluator` consumes.
`search_latency_ms` is the sum of the two branches' search latencies.

### 4.5 Deliberate call: no `score_threshold` forwarding

A cosine threshold and a BM25 threshold are on incomparable scales, and
applying one to both would silently truncate a single candidate list. Hybrid
therefore does not forward `score_threshold` to either branch;
`instantiate_components()` hands each constituent a threshold-free copy of the
config. The configured value is still echoed into
`RetrievalMetadata.score_threshold` for traceability. The default is `None`, so
the baseline is unaffected.

### 4.6 Experiment integration

`instantiate_components()` gains a third branch that builds
`DenseRetriever` + `BM25Retriever` from the same `config` and composes them.
The **dense `vector_store` is returned in slot 2**, so the CLI's
`count()` / `close()` calls keep working unchanged; the BM25 index is validated
at `BM25Index.load()` time, which already raises `IndexUnavailableError` on a
missing/corrupt file and `IndexConfigMismatchError` on corpus drift.

`ExperimentRunner` and both evaluators require **zero** changes:
`runner.py` calls `retriever.retrieve(query, top_k=...)` and reads only
`retrieval_metadata.latency_ms`; `RetrievalEvaluator` reads
`results[*].chunk_id / metadata.document_id / rank`; `EfficiencyEvaluator`
reads `trace.retrieval_latency_ms`.

`run_experiment.py` accepts `--retriever {dense,bm25,hybrid}` with a
per-strategy default run name (`dense_baseline_v1` / `bm25_baseline_v1` /
`hybrid_baseline_v1`), and checks **both** indexes for emptiness when hybrid is
selected, naming the missing side.

`compare_retrievers.py` adds an optional `--hybrid-run`. With the flag absent
the printed table is byte-identical to the dense-vs-BM25-only table; with it
present, a `hybrid` column and two deltas are added and the method check is
relaxed to expect `dense` / `bm25` / `hybrid` respectively.

### 4.7 CLI

```bash
# Requires AICREDITS_API_KEY (the dense branch embeds each query remotely)
# and the persisted storage/bm25/bm25_index.json
.venv/bin/python scripts/run_experiment.py --retriever hybrid --no-judge --no-generation
.venv/bin/python scripts/compare_retrievers.py --hybrid-run experiments/<run-id>
```

> **Hybrid is not offline-capable.** Unlike `--retriever bm25`, the hybrid run
> embeds every query through the embedding provider, so it needs
> `AICREDITS_API_KEY` even with `--no-judge --no-generation`. This is the
> expected behavior of the dense branch, not a defect.

---

## 5. Evaluation Results

Run: `experiments/20260929T175539Z-hybrid_baseline_v1/` over the 20 examples of
`data/evaluation/dense_eval_v1.jsonl` (corpus `corpus_6c416f423920385d`,
retriever `hybrid_v1`, `rrf_k=60`, `candidate_k=20`, `top_k=10`,
retrieval-only run with `--no-judge --no-generation`).

| Metric   | @1     | @3     | @5     | @10    |
| -------- | ------ | ------ | ------ | ------ |
| Recall   | 0.6417 | 0.7417 | 0.8583 | 0.8750 |
| Precision| 0.7500 | 0.7500 | 0.7500 | 0.7000 |
| Hit      | 0.7500 | 0.8500 | 0.9500 | 0.9500 |
| nDCG     | 0.7500 | 0.7572 | 0.7788 | 0.8557 |

* **MRR:** 0.8200
* **Retrieval latency (ms):** mean 721.00 · p50 516.65 · p95 1975.13
* Status counts: 20 traces, 0 retrieval failures, 0 errors
  (`empty: 20` is the expected retrieval-only status when generation is
  skipped, identical to the BM25 baseline run)
* 126 unique chunks surfaced across the 20 queries
* All 20 traces carry `fusion_method="rrf"`, `rrf_k=60`, `candidate_k=20`,
  `retriever_version="hybrid_v1"`, `index_id="hybrid_dense_bm25_v1"`, and 20
  candidates from each branch

### Per-branch cost (from `RetrievalMetadata`)

| Stage | mean (ms) | p50 (ms) | p95 (ms) |
| ----- | --------- | -------- | -------- |
| Dense branch (embedding + Qdrant) | 688.96 | 509.86 | 1677.30 |
| BM25 branch (inverted index) | 3.47 | 3.61 | 4.74 |
| Fusion + truncation | 0.65 | 0.57 | 1.00 |
| **Total end-to-end** | **721.00** | **516.65** | **1975.13** |

Fusion itself is negligible (0.65 ms mean). The ~32 ms gap between the total and
the dense branch is BM25 search plus the deeper Qdrant search at
`candidate_k=20` instead of `top_k=10` — not fusion overhead.

### Three-way comparison (identical benchmark, `dense_eval_v1`, same corpus)

Produced by `scripts/compare_retrievers.py --hybrid-run`. Corpus version, trace
count, and retrieval method are consistent across all three manifests.

| Metric           | Dense   | BM25    | Hybrid  | Δhyb−dense | Δhyb−bm25 |
| ---------------- | ------- | ------- | ------- | ---------- | --------- |
| Recall@1         | 0.8417  | 0.5417  | 0.6417  | −0.2000    | +0.1000   |
| Recall@3         | 0.8417  | 0.6167  | 0.7417  | −0.1000    | +0.1250   |
| Recall@5         | 0.8417  | 0.7167  | 0.8583  | **+0.0166**| +0.1416   |
| Recall@10        | 0.8583  | 0.7833  | 0.8750  | **+0.0167**| +0.0917   |
| Precision@1      | 0.9500  | 0.6000  | 0.7500  | −0.2000    | +0.1500   |
| Precision@5      | 0.8600  | 0.6200  | 0.7500  | −0.1100    | +0.1300   |
| Hit@1            | 0.9500  | 0.6000  | 0.7500  | −0.2000    | +0.1500   |
| Hit@5            | 0.9500  | 0.8000  | 0.9500  |  0.0000    | +0.1500   |
| MRR              | 0.9500  | 0.6771  | 0.8200  | −0.1300    | +0.1429   |
| nDCG@5           | 0.8908  | 0.6631  | 0.7788  | −0.1120    | +0.1157   |
| nDCG@10          | 0.9308  | 0.7378  | 0.8557  | −0.0751    | +0.1179   |
| Latency mean (ms)| 751.51  | 1.48    | 721.00  | −30.51     | +719.52   |
| Latency p50 (ms) | 651.06  | 1.30    | 514.70  | −136.37    | +513.39   |
| Latency p95 (ms) | 1192.58 | 2.39    | 1975.13 | +782.55    | +1972.74  |

### Observations

**Hybrid beats BM25 on every single retrieval-quality metric.** It is strictly
the better lexical-inclusive strategy, and it does so at a latency cost of
~720 ms mean, essentially all of it the dense query embedding.

**Hybrid beats dense on deep recall but loses on top-rank precision.** It is
the only strategy to exceed dense at Recall@5 (+0.0166) and Recall@10
(+0.0167), and it ties dense on Hit@5. But it gives up Recall@1 (−0.20),
Precision@1 (−0.20), and MRR (−0.13). This is the expected RRF trade-off on a
20-example benchmark: reciprocal rank fusion flattens the ranking. A chunk that
only one branch ranks highly accumulates one small term and is pushed below
chunks that both branches agree on, so the fused list is better at *coverage*
and worse at *leading with the single best guess*. With n=20 the Recall@5/10
gains (+0.0166/+0.0167) are one document's worth of evidence and should not be
over-read; the Recall@1 and MRR losses are the larger and more consistent
effect.

**Latency is dominated by the dense branch, not by fusion.** Hybrid's mean
(721.00 ms) is close to dense's (751.51 ms); BM25 contributes 3.47 ms and
fusion 0.65 ms. The p95 gap (hybrid 1975 ms vs dense 1192 ms) is embedding
network variance on 20 samples, not a structural cost — hybrid makes exactly
one embedding call per query, the same as dense.

### Traceability

`config.json`, `manifest.json` (git commit, corpus version,
`component_versions.retrieval: "hybrid_v1"`), `traces.jsonl`, `metrics.json`,
`metrics_{retrieval,generation,efficiency}.json`, and `report.md` are all
persisted under `experiments/20260929T175539Z-hybrid_baseline_v1/`. Run
directories are gitignored; force-add `config.json`, `manifest.json`, and
`metrics.json` explicitly if the run is to be published.

---

## 6. Failure Semantics

Any exception from either branch propagates untouched. `runner.py` records it as
`status="retrieval_failed"` with the original `error_type` and a `null`
retrieval payload. Hybrid is never downgraded to a dense-only or BM25-only
answer — that is impossible by construction, since the module contains no
`try`/`except` and the guard test enforces it on the AST.

Observed live during Phase 4 implementation: with the embedding provider
unreachable, `run_experiment.py --retriever hybrid` produced 20/20
`retrieval_failed` traces with `error_types: ["EmbeddingAPIError"]`,
`retrieval_method: "hybrid"`, and `component_versions.retrieval: "hybrid_v1"`
in the manifest — the correct fail-closed outcome.

---

## 7. Validation

* **86 offline pytest tests pass** (57 pre-existing, all unchanged in intent,
  plus 29 Phase 4 tests), no credentials required. Suite is stable across
  repeated runs.
* `tests/test_hybrid.py` covers:
  * exact RRF values and derived ordering for dense `[A,B,C]` / BM25 `[B,C,D]`
  * duplicate at dense rank 1 / lexical rank 4 → `1/61 + 1/64`, appearing once
  * dedup by `chunk_id`, not text (identical text under two ids survives twice)
  * single-source chunks with a one-term contribution
  * unequal lengths (20 vs 7 → 27 results, no padding artifacts)
  * empty branch and both-empty cases, including `status="no_results"` with
    latency still recorded
  * constituent failure propagating with the original exception identity, and
    the runner recording `retrieval_failed` with `retrieval: null`
  * metadata/provenance preservation, with the dense payload preferred on
    overlap and the BM25 payload kept for lexical-only chunks
  * `top_k` truncation to 5 with ranks `1..5` and non-increasing scores
  * candidate depth: both branches receive `candidate_k=20`, not `top_k`
  * fusion diagnostics in `RetrievalMetadata`
  * latency accounting (`latency_ms >= dense + bm25` on real components)
  * protocol conformance and `method == "hybrid"`
  * empty-query guard with both constituent call counts at 0
  * determinism across runs and exact-tie resolution by `chunk_id`
  * config loading, `candidate_k < top_k` rejection, and `config_hash`
    sensitivity to `rrf_k` / `candidate_k`
  * end-to-end integration: real `DenseRetriever` (in-memory Qdrant +
    `FakeEmbeddingModel`) + real `BM25Retriever` over the same chunks, driven
    through `ExperimentRunner` with `RetrievalEvaluator` + `EfficiencyEvaluator`
  * regression on the canonical corpus: "Robertson", "ColBERT", and "BM25"
    surface the expected documents after fusion, and the fused list is a
    superset of neither branch alone (auto-skips if corpus artifacts are absent)
* **Architecture guards:** `"hybrid"` removed from `FORBIDDEN_STRATEGY_TOKENS`
  (Phase 5+ bans retained). The existing dense/BM25 isolation guard is
  unchanged and still holds — neither retriever file was touched, which is the
  proof that composition did not leak backward. New
  `test_hybrid_composes_existing_retrievers` asserts `hybrid.py` contains no
  rerank/route/adaptive/alternative-fusion/index/tokenization logic and no
  `ast.Try` node, and that `fusion.py` couples to neither branch.
* `compare_retrievers.py` output verified byte-identical before and after the
  change when `--hybrid-run` is omitted; the three-column path, the
  `retrieval_method` mismatch path, and the missing-run path were all exercised.
  A regression test drives the real CLI against fixture run directories and
  pins the two properties: a hybrid run sitting on disk does **not** change the
  default two-run table (the hybrid column is resolved from `--hybrid-run` only,
  never auto-discovered), and a hybrid run whose manifest says
  `retrieval_method != "hybrid"` exits 2.

### Live experiment

**COMPLETE.** Run `20260929T175539Z-hybrid_baseline_v1`, executed on a
networked machine after the implementation environment proved unable to reach
the embedding provider. Results are in §5. The pre-network portion (config
stamping, both-index resolution, corpus match) had already been verified during
implementation:

```text
Loaded 20 evaluation examples
Experiment config: name=hybrid_baseline_v1 corpus=corpus_6c416f423920385d
                   retriever=hybrid top_k=10 judge=False
Hybrid retrieval active: dense collection 'adaptiverag_dense_v1' (713 points)
                         + BM25 index (713 docs), rrf_k=60 candidate_k=20
```

The completed run returned 20/20 traces with 0 retrieval failures, and
`compare_retrievers.py --hybrid-run` reported MATCH on corpus version, trace
count, and retrieval method across all three runs.

---

## 8. Definition of Done

- [x] **Composition, not reimplementation**: `HybridRetriever` reuses
      `DenseRetriever` and `BM25Retriever`; neither file was modified.
- [x] **Rank-based fusion**: RRF uses only `1/(rrf_k + rank)`; no score mixing,
      no normalization, no alternative fusion functions.
- [x] **Contract Compliance**: `Retriever` protocol, `RetrievalResponse` with
      `retrieval_method="hybrid"`, 1-indexed ranks, RRF `score`, full
      provenance.
- [x] **Configuration Consistency**: `retrieval_method` ⇔ `retriever_version`
      (`hybrid ⇔ hybrid_v1`) in config, traces, and manifests; `rrf_k` and
      `candidate_k` are inside the hashed config.
- [x] **Failure vs. Empty Handling**: empty query → `InvalidQueryError`; branch
      failure → `retrieval_failed` with the original error type and no
      fallback; no overlap → `status="no_results"`.
- [x] **Evaluation Parity**: consumed unchanged by `ExperimentRunner` and both
      evaluators; driven by `run_experiment.py --retriever hybrid`.
- [x] **Testing**: 86 offline deterministic tests pass, including guards.
- [x] **Live Evaluation**: hybrid run over `dense_eval_v1` recorded at
      `experiments/20260929T175539Z-hybrid_baseline_v1/` (20/20 traces, 0
      retrieval failures, Recall@5 0.8583) and compared via
      `compare_retrievers.py --hybrid-run`.
- [x] **Documentation**: `phase-4.md`, `progress.md`, `decision.md` (ADR-019),
      `architecture.md`, `experiments/README.md` updated.
