# Phase 3 — BM25 Lexical Retrieval

**Status:** COMPLETE
**Phase:** 3
**Purpose:** Establish a fixed, fully lexical BM25 retrieval baseline over the
same canonical chunk corpus, independently evaluable against the Phase 2 dense
baseline.

---

## 1. Objective

Implement an Okapi BM25 retrieval strategy that:

1. Indexes the canonical Phase 1/2 chunk corpus (`data/processed/chunks/*.chunks.jsonl`,
   713 chunks) without modifying or re-chunking it.
2. Retrieves chunks with native, unnormalized BM25 scores.
3. Conforms to the existing `Retriever` protocol and `RetrievalResponse` contract.
4. Runs the strategy-agnostic evaluation pipeline on the identical benchmark
   (`dense_eval_v1.jsonl`) for direct comparison with the dense baseline.
5. Remains fully independent of dense retrieval — no imports, no fallback, no
   cascading between the two strategies.

---

## 2. Scope

### Included

* Deterministic text normalization + tokenization (symmetric for index and query)
* Inverted index construction with precomputed non-negative Robertson IDF
* Okapi BM25 scoring (`k1=1.2`, `b=0.75`)
* JSON index persistence with corpus-version staleness guard
* `BM25Retriever` implementing the `Retriever` protocol
* CLI index builder (`scripts/build_bm25_index.py`)
* `--retriever {dense,bm25}` experiment selection
* `scripts/compare_retrievers.py` side-by-side metrics table
* Unit, integration, regression, and architecture-guard tests

### Excluded

* Hybrid retrieval / Reciprocal Rank Fusion (Phase 4)
* Reranking (Phase 5)
* Adaptive routing, query classification, strategy selection (Phase 6)
* Any change to Dense baseline internals

---

## 3. Architecture

```text
               Canonical Chunk Corpus (713 chunks)
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
         Dense Index (Qdrant)                      BM25 Index (JSON)
               │                                         │
               ▼                                         ▼
         DenseRetriever                            BM25Retriever
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                          Common RetrievalResponse
                                    ▼
                          Strategy-Agnostic Eval
```

Both retrievers:

* conform to `adaptive_rag.retrieval.base.Retriever`
* return `RetrievalResponse` with `retrieval_method ∈ {"dense", "bm25"}`
* preserve `ChunkMetadata` and `ChunkProvenance` (including `source_sha256`)
  end-to-end
* are consumed unchanged by `ExperimentRunner` and the evaluators

---

## 4. Implementation

### 4.1 Paths & Configuration

* `config/paths.py`: `BM25_INDEX_DIR = storage/bm25`,
  `BM25_INDEX_PATH = storage/bm25/bm25_index.json` (created by
  `ensure_directories()`; gitignored via `storage/`).
* `schemas/config.py`:
  * `DenseRetrievalConfig` and `BM25RetrievalConfig` dedicated models.
  * `RetrievalConfig.retrieval_method: Literal["dense", "bm25"]` with `k1`/`b`
    fields and a model validator that keeps `retrieval_method` and
    `retriever_version` aligned (`bm25 ⇔ bm25_v1`, `dense ⇔ dense_v1`).
* `experiments/config.py`: `build_experiment_config()` stamps the active
  retriever version into `component_versions["retrieval"]`; `instantiate_components()`
  branches on `retrieval_method` — `"bm25"` loads the persisted `BM25Index`
  (with corpus-version staleness check) and never touches embeddings/Qdrant.

### 4.2 Preprocessing (symmetric, deterministic)

`indexing/bm25.py::tokenize_text`:

1. Unicode `NFKD` normalization with combining marks (diacritics) stripped
2. Lowercasing
3. Alphanumeric token extraction (`[a-z0-9]+`)

Exactly the same pipeline runs on chunk text at index time and on queries at
search time. The raw query text is preserved untouched in
`RetrievalResponse.query`.

### 4.3 Index (`src/adaptive_rag/indexing/bm25.py`)

* Inverted index: `term → [(doc_index, term_frequency)]`
* Document lengths + `avgdl`
* Robertson IDF: `ln((N − df + 0.5)/(df + 0.5) + 1)` lower-bounded at `0`
* Score:
  `Σ_q IDF(q) · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))`
* Ranking: descending score, ties broken by document insertion order
  (deterministic)
* `save()`/`load()` with canonical JSON; `load()` accepts
  `expected_corpus_version` and raises `IndexConfigMismatchError` on staleness,
  `IndexUnavailableError` on missing/corrupt file
* Index metadata persisted: `index_version`, `corpus_version`, `k1`, `b`,
  `total_docs`, `avgdl`

### 4.4 Retriever (`src/adaptive_rag/retrieval/bm25.py`)

* `method = "bm25"`; signature matches the `Retriever` protocol exactly
* Empty/whitespace query → `InvalidQueryError("Query cannot be empty")`
* Valid query with no lexical overlap → `status="no_results"`, `results=[]`
  (never falls back to dense)
* Scores are native Okapi BM25 values (≥ 0, unnormalized); ranks are 1-indexed
* `filters` accepted for protocol conformity; applied as simple payload
  equality matching when provided
* `RetrievalMetadata` carries `k1`/`b`; dense-only embedding fields are
  optional (`None`) and BM25-only fields are `None` for dense

### 4.5 CLI

```bash
.venv/bin/python scripts/build_bm25_index.py            # → storage/bm25/bm25_index.json
.venv/bin/python scripts/run_experiment.py --retriever bm25 --no-judge --no-generation
.venv/bin/python scripts/compare_retrievers.py          # dense vs bm25 table
```

`run_experiment.py --retriever bm25` requires **no embedding credentials**
(BM25 loads the persisted index directly). `--no-generation` enables fully
offline retrieval-only evaluation; generation still works normally when Groq
credentials are present.

---

## 5. Evaluation Results

Run: `experiments/bm25_baseline_v1/` over the 20 examples of
`data/evaluation/dense_eval_v1.jsonl` (corpus `corpus_6c416f423920385d`,
retriever `bm25_v1`, `k1=1.2`, `b=0.75`, `top_k=10`, retrieval-only run).

| Metric   | @1     | @3     | @5     | @10    |
| -------- | ------ | ------ | ------ | ------ |
| Recall   | 0.5417 | 0.6167 | 0.7167 | 0.7833 |
| Precision| 0.6000 | 0.6167 | 0.6200 | 0.5550 |
| Hit      | 0.6000 | 0.7000 | 0.8000 | 0.8500 |
| nDCG     | 0.6000 | 0.6413 | 0.6631 | 0.7378 |

* **MRR:** 0.6771
* **Retrieval latency (ms):** mean 1.48 · p50 1.30 · p95 2.39
* Status counts: 20 traces, 0 retrieval failures, 0 errors

Traceability: `config.json`, `manifest.json` (git commit, corpus version,
component versions with `retrieval: bm25_v1`), `traces.jsonl`, `metrics.json`,
`report.md` all persisted under `experiments/bm25_baseline_v1/`.

### Dense vs BM25 side-by-side (identical benchmark, `dense_eval_v1`)

The live dense baseline completed at `experiments/dense_baseline_v1/`
(20/20 traces `ok`, corpus `corpus_6c416f423920385d` on both runs):

| Metric          | Dense   | BM25    | Delta (bm25 − dense) |
| --------------- | ------- | ------- | -------------------- |
| Recall@1        | 0.8417  | 0.5417  | −0.3000              |
| Recall@5        | 0.8417  | 0.7167  | −0.1250              |
| Recall@10       | 0.8583  | 0.7833  | −0.0750              |
| Precision@1     | 0.9500  | 0.6000  | −0.3500              |
| Precision@5     | 0.8600  | 0.6200  | −0.2400              |
| Hit@5           | 0.9500  | 0.8000  | −0.1500              |
| MRR             | 0.9500  | 0.6771  | −0.2729              |
| nDCG@5          | 0.8908  | 0.6631  | −0.2277              |
| nDCG@10         | 0.9308  | 0.7378  | −0.1930              |
| Latency mean (ms)| 751.51 | 1.48    | −750.03              |

Observations: Dense retrieval achieved higher retrieval quality than BM25 on
this evaluation benchmark, while BM25 provided substantially lower retrieval
latency (mean 1.48 ms vs 751.51 ms, no credential/API dependency). The gap
narrows as k grows (Recall@10: 0.858 vs 0.783).

Reproducibility: `scripts/compare_retrievers.py` verifies corpus-version,
trace-count, and method consistency between both manifests (all MATCH) and
emits a byte-identical table across invocations. The BM25 run reproduces
identical metrics on re-execution; the dense run is reproducible from the
persisted embeddings cache, Qdrant index, and trace artifacts.

---

## 6. Validation

* **57 offline pytest tests pass** (14 new BM25 tests + 1 new retriever
  isolation guard + all Phase 1/2 tests unchanged and green), no credentials
  required:
  * tokenization/normalization consistency (incl. diacritic stripping)
  * index build, save/load round-trip, staleness + missing-file guards
  * term-frequency monotonicity and `b` length-normalization invariants
  * rank ordering, top-k truncation, score ordering
  * empty query → `InvalidQueryError`; vocabulary mismatch → `no_results`
  * filters applied via payload matching
  * provenance/metadata integrity end-to-end (build → persist → load → retrieve)
  * `Retriever` protocol conformance (`@runtime_checkable`)
  * config consistency (`retrieval_method` ⇔ `retriever_version` in config and
    `component_versions`)
  * regression: "Robertson", "ColBERT", "BM25" surface the expected documents
    at top ranks on the canonical corpus (auto-skips if corpus artifacts are
    absent)
* **Architecture guards**: `"bm25"` removed from `FORBIDDEN_STRATEGY_TOKENS`;
  new isolation guard asserts dense and BM25 never reference each other;
  future-phase bans (`hybrid`, `rerank`, `adaptive_rout`, `query_classif`,
  `strategy_select`) retained.
* `build_bm25_index.py` indexes all **713 canonical chunks** (no hard-coded
  count) → 21,406-term vocabulary, avgdl ≈ 192.3 tokens.

---

## 7. Definition of Done

- [x] **Corpus Parity**: BM25 indexes the canonical corpus without modifying it
      or assuming a fixed chunk count.
- [x] **Independence**: BM25 does not import, call, or fall back to dense;
      dense internals unchanged (guard-enforced).
- [x] **Contract Compliance**: `BM25Retriever` conforms to `Retriever` and
      returns native scores, 1-indexed ranks, and full provenance.
- [x] **Configuration Consistency**: `retrieval_method` and `retriever_version`
      aligned without drift (validator + component-version stamping).
- [x] **Failure vs. Empty Handling**: missing index fails fast;
      empty query raises `InvalidQueryError`; zero match returns
      `status="no_results"`.
- [x] **Testing**: 57 offline deterministic tests pass, including guards.
- [x] **Evaluation Parity**: evaluated on `dense_eval_v1.jsonl` with
      Recall/Precision/Hit/MRR/nDCG@k and latency percentiles.
- [x] **Traceability & Reproducibility**: traces + manifest under
      `experiments/bm25_baseline_v1/`.
- [x] **Future Compatibility**: `RetrievalResponse` (incl. `retrieval_method`)
      and traces are ready for Phase 4 RRF consumption; no Phase 4 code exists.
- [x] **Documentation**: `phase-3.md`, `progress.md`, `decision.md`,
      `architecture.md` updated.

