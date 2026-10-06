# Phase 5 — Reranking (Second-Stage Cross-Encoder)

## A. Repository Assessment

Phase 1–4 are complete and give Phase 5 every structural precondition. Confirmed
by reading the code:

- `src/adaptive_rag/retrieval/base.py:14` — `Retriever` is a `@runtime_checkable`
  `Protocol`: `method: str` + `retrieve(query, top_k, score_threshold, filters)`.
  `RerankedRetriever` conforms with **no protocol change**.
- `src/adaptive_rag/retrieval/hybrid.py:26` is already the composition precedent:
  a wrapper that holds two `Retriever` objects, orchestrates, and emits a
  `RetrievalResponse`. `RerankedRetriever` is the same pattern applied to a
  second stage. `dense.py`, `bm25.py`, and `hybrid.py` are **not modified**.
- `src/adaptive_rag/schemas/retrieval.py` is `extra="forbid"` on all three models.
  Required edits: `RetrievalResponse.retrieval_method` Literal, two optional
  `RetrievalResult` fields, ~11 optional `RetrievalMetadata` fields.
- `src/adaptive_rag/experiments/runner.py:161` calls
  `retriever.retrieve(query=..., top_k=config.retrieval.top_k)` and reads only
  `retrieval_metadata.latency_ms`. It already converts any exception into
  `status="retrieval_failed"` with `retrieval: null`, so **fail-visible is free**.
- `src/adaptive_rag/evaluation/retrieval.py` is strategy-agnostic: it reads
  `chunk_id`, `metadata.document_id`, `rank`, and list order only. Reranking
  needs **zero** retrieval-metric changes — it improves ordering, and the
  evaluator consumes order.
- `src/adaptive_rag/evaluation/efficiency.py` reads only
  `trace.retrieval_latency_ms`. New split metrics need additive trace fields.
- `src/adaptive_rag/experiments/config.py:110` `instantiate_components()` is the
  single construction site and already branches on `retrieval_method`.
- `tests/test_architecture_guards.py:17` — `FORBIDDEN_STRATEGY_TOKENS` contains
  `"rerank"` and scans every file under `src/`. **The guard must be updated
  first or the suite becomes unsatisfiable.**
- `tests/test_architecture_guards.py:26` — `FORBIDDEN_HYBRID_TOKENS` contains
  `"rerank"` and `"route"`. Keep both; they become the proof that `hybrid.py`
  never learned about the reranker.
- `tests/test_architecture_guards.py:23` — `sentence-transformers` is banned as
  a dependency. The ONNX decision keeps that ban intact.

### Environment facts (verified this session)

- 12 CPU cores, 7.8 GB RAM, **no GPU** (`/dev/nvidia` absent), 262 GB free disk.
- `.venv` (Python 3.12, uv-managed, editable install) has **no** torch,
  transformers, onnxruntime, or tokenizers.
- No outbound network from this session. `~/.cache/huggingface/hub` holds only
  bi-encoder snapshots (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`) — no
  cross-encoder. So the real ONNX artifact **cannot be fetched or benchmarked
  here**; live runs happen on a networked machine, as Phase 4's did.

### Decisions taken with the user

| Decision | Choice |
| --- | --- |
| Reranker runtime | ONNX Runtime + pre-exported ONNX cross-encoder. No torch, no `sentence-transformers`; the existing heavy-dep ban stands. |
| Pipeline identity | New `retrieval_method` literals `dense_rerank` / `bm25_rerank` / `hybrid_rerank` with matching `retriever_version` values. Keeps the ⇔ alignment invariant and makes all six runs unambiguous in manifests. |
| Result schema | `score` = rerank score (the signal that produced the current order, mirroring Phase 4's RRF rule); new optional `retrieval_score` (original cosine/BM25/RRF) and `retrieval_rank` (pre-rerank position). |
| Latency split | `RetrievalMetadata` + `ExperimentTrace` + `EfficiencyEvaluator` all gain the split; `latency_ms` = candidate generation + rerank. |
| CLI | Boolean `--rerank` plus explicit overrides; existing `dense`/`bm25`/`hybrid` invocations unchanged. |
| Hybrid depth | `--rerank-candidate-k` raises hybrid's own `candidate_k` to match, so one flag controls depth end-to-end. |
| Failure | Fail-visible default; `--rerank-fallback` is opt-in and recorded in `RetrievalMetadata.rerank_fallback`. |
| Default model | `Xenova/ms-marco-MiniLM-L-6-v2` (~22 M params, ~90 MB ONNX). Model id and revision pinned in config. |
| Comparison tooling | New `scripts/compare_reranking.py`; `compare_retrievers.py` stays frozen so its Phase 4 regression test is untouched. |

---

## B. Files to Create

### 1. `src/adaptive_rag/reranking/__init__.py` + `base.py`

`Reranker` protocol, isolated from every retriever, index, and corpus:

```python
@runtime_checkable
class Reranker(Protocol):
    version: str
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage, in input order."""
        ...
```

Rules: input order is preserved and the returned list length equals the input
length (a mismatch is a typed error, never a silent zip-truncation); scores are
model-specific ranking signals, not calibrated probabilities; the module holds
no index, no retriever, no corpus access.

### 2. `src/adaptive_rag/reranking/onnx_backend.py`

`OnnxCrossEncoderReranker` — the only file that touches ONNX.

- `__init__(model_id, model_revision, device, batch_size, max_length, model_dir)`
- Downloads to `storage/reranker/<model_id>/` on first use via
  `huggingface_hub.snapshot_download(..., local_dir=...)`; reuses the local copy
  afterwards. Add `storage/reranker/` to `ensure_directories()` in
  `config/paths.py`.
- Requires `onnx/model.onnx`, `tokenizer.json`, `config.json`. Missing file →
  typed `RerankerModelError` naming the exact path.
- Tokenizes with `tokenizers.Tokenizer.from_file`, `enable_truncation(max_length)`,
  `enable_padding()` within a batch. Pairs are encoded as `[query, passage]`.
- Session providers: `device="cpu"` → `["CPUExecutionProvider"]`; `"cuda"` →
  `["CUDAExecutionProvider", "CPUExecutionProvider"]` with an explicit error if
  the CUDA provider is unavailable (fail-visible, never a silent CPU downgrade);
  `"auto"` → CUDA when available, else CPU, logged once at construction.
- **Batching**: encode in `batch_size` chunks, `session.run` per chunk. Scores
  from different batches never interact, so output is **batch-size invariant** —
  a property to assert in tests.
- **Output decoding**: read `id2label` from `config.json`. `num_labels == 1` →
  raw logit. `num_labels == 2` → softmax, take the index of `label_1` /
  `"relevant"`. Documented as a ranking signal, not a probability.
- `max_length` default 512, `batch_size` default 16 — both configurable.

### 3. `src/adaptive_rag/retrieval/reranked.py`

`RerankedRetriever(base_retriever: Retriever, reranker: Reranker, config, corpus_version)`.

`method` is one of `dense_rerank` / `bm25_rerank` / `hybrid_rerank`, derived from
`base_retriever.method`.

`retrieve()` flow:

1. Empty/whitespace query → `InvalidQueryError("Query cannot be empty")`,
   **before** the base retriever is called.
2. `base.retrieve(query, top_k=config.rerank_candidate_k, filters=filt)` →
   candidates, timing it into `candidate_generation_latency_ms`.
   `score_threshold` is **not** forwarded: a retrieval-stage threshold and a
   reranker are different concerns, and applying the retriever's threshold here
   would silently shrink the candidate pool. The configured value is echoed into
   metadata for traceability, exactly as hybrid does.
3. **Empty candidate set is not a failure**: return `status="no_results"`,
   `results=[]`, `rerank_latency_ms=0.0`, and **do not call the reranker at all**
   (no model invocation on an empty pool). Distinct from a reranker exception,
   which propagates.
4. Time the reranker into `rerank_latency_ms`.
5. Sort by `(-rerank_score, retrieval_rank, chunk_id)`. `retrieval_rank` then
   `chunk_id` guarantee a total order, so repeated runs are byte-stable and
   equal scores never depend on input order.
6. Truncate to `top_k`, renumber `rank = 1..n`, and rewrite each result with
   `score=<rerank_score>`, `retrieval_score=<original score>`,
   `retrieval_rank=<original rank>`. `text`, `metadata`, and `provenance` are
   carried through untouched — the reranker only supplies a number per chunk.
7. `latency_ms = candidate_generation_latency_ms + rerank_latency_ms` (full
   end-to-end, mirroring the Phase 4 hybrid rule). `search_latency_ms` is passed
   through from the base response.
8. **Failure**: any reranker exception propagates unchanged. The only
   `try/except` in the module is gated on `config.rerank_fallback`; when taken it
   returns the candidates truncated to `top_k` with `score` and `retrieval_score`
   both equal to the original retrieval score, and sets
   `RetrievalMetadata.rerank_fallback = True` so it is visible in every trace,
   metric, and manifest. The AST guard asserts the `except` cannot run unless
   `config.rerank_fallback` is set.

Holds no vector store, no BM25 index, no corpus handle, and never imports
`retrieval.dense`, `retrieval.bm25`, `indexing.*`, or `experiments.*`.

### 4. `tests/test_reranking.py`

Phase 5 unit + integration suite (§H). Uses `FakeReranker` and the existing
`StubRetriever` / in-memory Qdrant / in-memory `BM25Index` fixtures — no
credentials, no model download.

### 5. `scripts/compare_reranking.py`

Reads run directories and prints two tables:

- **Base vs reranked**, one row per metric per strategy pair, with deltas.
- **Depth ablation** — rerank latency and quality at each candidate depth, with
  `top_k` held constant, so the depth/cost curve is visible.

Reuses the consistency checks from `compare_retrievers.py`: corpus version, trace
count, and expected `retrieval_method` per column. Exits 2 on mismatch.
`compare_retrievers.py` is **not touched**.

---

## C. Files to Modify

| Path | Required change | Reason |
| --- | --- | --- |
| `src/adaptive_rag/schemas/retrieval.py` | `RetrievalResult` += `retrieval_score: float \| None`, `retrieval_rank: int \| None`. `RetrievalResponse.retrieval_method` Literal += the three `_rerank` values. `RetrievalMetadata` += `rerank_enabled`, `reranker_version`, `reranker_model_id`, `reranker_device`, `rerank_candidate_k`, `rerank_top_k`, `candidate_generation_latency_ms`, `rerank_latency_ms`, `candidate_count`, `result_count`, `rerank_fallback` — **all optional** | Reranked responses must serialize into the common contract without changing dense/BM25/hybrid output |
| `src/adaptive_rag/schemas/config.py` | `RerankerConfig` (mirrors `HybridRetrievalConfig`). `RetrievalConfig` += `rerank_enabled: bool = False`, `rerank_model_id`, `rerank_model_revision`, `rerank_device: Literal["auto","cpu","cuda"]`, `rerank_batch_size: int = 16`, `rerank_max_length: int = 512`, `rerank_candidate_k: int = 20`, `rerank_fallback: bool = False`; Literal += 3 methods; validator gains the `_rerank ⇔ _rerank_v1` alignment, `rerank_candidate_k >= top_k` when enabled, and `hybrid`+rerank ⇒ `candidate_k >= rerank_candidate_k` | Configurable, hashed, reproducible |
| `src/adaptive_rag/schemas/experiment.py` | `ExperimentTrace` += `candidate_generation_latency_ms`, `rerank_latency_ms`, `rerank_candidate_count`, `rerank_result_count`, `rerank_fallback` — all optional `None` | Flat per-query latency/count split for the efficiency evaluator. **No new required field on `ExperimentConfig`** — that would break every existing construction site |
| `src/adaptive_rag/schemas/__init__.py` | Export `RerankerConfig` | Keep the flat export convention |
| `src/adaptive_rag/retrieval/__init__.py` | Export `RerankedRetriever` | ADR-007 strategy set |
| `src/adaptive_rag/reranking/__init__.py` | Export `Reranker`, `OnnxCrossEncoderReranker` | — |
| `src/adaptive_rag/config/paths.py` | Add `RERANKER_DIR = storage/reranker` to `paths.py` + `ensure_directories()` | Model cache location |
| `src/adaptive_rag/errors.py` | `RerankingError(AdaptiveRAGError)`, `RerankerModelError(RerankingError)`, `RerankerUnavailableError(RerankingError)` | Typed failures, consistent with the existing hierarchy |
| `src/adaptive_rag/experiments/config.py` | After building the base retriever for `dense` / `bm25` / `hybrid`, wrap it when `config.retrieval.rerank_enabled`; for hybrid, build the `HybridRetriever` with a `candidate_k` raised to `rerank_candidate_k`. Add `component_versions["reranker"]` only when reranking is enabled | Single construction site; keeps CLI slot-2 (`count()` / `close()`) working |
| `src/adaptive_rag/experiments/runner.py` | In `_run_example`, copy the four new metadata values onto the trace when present. Nothing else changes | Flat trace fields |
| `src/adaptive_rag/evaluation/efficiency.py` | Emit `candidate_generation_latency_ms_{mean,p50,p95}`, `rerank_latency_ms_{mean,p50,p95}`, `rerank_candidate_count_mean`, `rerank_result_count_mean`, `rerank_fallback_count` — **only when at least one trace carries them**, so existing efficiency output stays unchanged | Phase 5 cost accounting without a parallel evaluator |
| `scripts/run_experiment.py` | `--rerank` plus `--reranker-model`, `--reranker-revision`, `--rerank-candidate-k`, `--rerank-device`, `--rerank-batch-size`, `--rerank-max-length`, `--rerank-fallback`; `DEFAULT_RUN_NAMES` += `{dense,bm25,hybrid}_rerank_v1`; hybrid dual-index check must unwrap via `retriever.base_retriever` | Strategy selection + depth ablation from the CLI |
| `tests/fakes.py` | `FakeReranker` (canned score table or per-chunk callable, records call count and batch sizes, can be set to raise) | Offline deterministic reranking tests |
| `tests/test_architecture_guards.py` | Remove `"rerank"` from `FORBIDDEN_STRATEGY_TOKENS`; add Phase 5 guards (§H) | Otherwise the suite fails immediately |
| `pyproject.toml` | Add `onnxruntime>=1.19.0`, `tokenizers>=0.20.0`, `huggingface-hub>=0.25.0` to `dependencies`, matching the existing unpinned-minimum style | Existing manually managed uv workflow; no competing env tooling |
| `docs/phases/phase-5.md` (new) | Objective, architecture, config, failure semantics, methodology, measured results, DoD | Documentation is source of truth |
| `docs/progress.md` | Phase 5 status, results table, next steps | — |
| `docs/decision.md` | ADR-020 (reranking as an independent second stage); ADR-017 phase table update | — |
| `docs/architecture.md` | New reranking section; `retrieval_method` set updated | — |
| `README.md`, `experiments/README.md` | Phase 5 section, setup, run commands | — |

**Explicitly not modified**: `retrieval/dense.py`, `retrieval/bm25.py`,
`retrieval/hybrid.py`, `retrieval/fusion.py`, `indexing/*`, `evaluation/retrieval.py`,
`evaluation/generation.py`, `generation/*`, `rag.py`, `scripts/ask.py`,
`scripts/compare_retrievers.py`. No new index, no `storage/hybrid/`, no
`storage/reranked/`. No caching of reranker outputs in the baseline.

---

## D. Data / Control Flow

```text
run_experiment.py --retriever {dense|bm25|hybrid} [--rerank ...]
  → RetrievalConfig(retrieval_method="<base>_rerank", rerank_enabled=True,
                    rerank_candidate_k=N, rerank_top_k=config.top_k)
      validator: retriever_version → "<base>_rerank_v1"
                 rerank_candidate_k >= top_k
                 hybrid ⇒ candidate_k >= rerank_candidate_k
  → instantiate_components(config)
      → DenseRetriever | BM25Retriever | HybridRetriever   (unchanged code)
      → if rerank_enabled: OnnxCrossEncoderReranker(model_id, revision, device, ...)
      → RerankedRetriever(base, reranker, config)
  → ExperimentRunner.run(config, dataset)                 (unchanged except trace copy)
      per example:
        RerankedRetriever.retrieve(query, top_k=10)
          ├─ guard: empty query → InvalidQueryError
          ├─ t0
          ├─ base.retrieve(query, top_k=N)         → ≤N candidates
          │     (hybrid: both branches at candidate_k ≥ N, fused → N)
          ├─ candidates empty → status="no_results", reranker NOT called, return
          ├─ reranker.score(query, [c.text ...])    → N scores, batched
          ├─ sort (-rerank_score, retrieval_rank, chunk_id)
          ├─ truncate to top_k, rank = 1..n,
          │     score=rerank, retrieval_score=orig, retrieval_rank=orig
          └─ RetrievalResponse(retrieval_method="<base>_rerank",
                               RetrievalMetadata{ candidate_generation_latency_ms,
                                                 rerank_latency_ms,
                                                 latency_ms = sum, ... })
        exception anywhere → status="retrieval_failed", retrieval=null,
                            error_type = original   (no fallback by default)
        RetrievalEvaluator / EfficiencyEvaluator consume unchanged
      → traces.jsonl, metrics.json, manifest.json, report.md

Failure path (default): reranker raises → propagates → runner records
`retrieval_failed` with the original `error_type` and `retrieval: null`.
With `--rerank-fallback`: un-reranked candidates returned, truncated to `top_k`,
`rerank_fallback=True` in every trace's metadata and surfaced in the manifest.
```

---

## E. Reranking Algorithm

```python
# reranked.py — the ordering rule, isolated so it is unit-testable with a fake reranker
scored = [
    result.model_copy(update={
        "score": float(rr_score),
        "retrieval_score": result.score,
        "retrieval_rank": result.rank,
    })
    for result, rr_score in zip(results, scores, strict=True)   # length mismatch ⇒ RerankingError
]
scored.sort(key=lambda r: (-r.score, r.retrieval_rank or 0, r.chunk_id))
selected = [r.model_copy(update={"rank": i}) for i, r in enumerate(scored[:top_k], start=1)]
```

- **`chunk_id` is the final tiebreak.** Ties are real (a cross-encoder can
  emit identical logits, and stubbed tests will force them), and Python's stable
  sort would otherwise make ordering depend on input order.
- **Batch-size invariance.** Because scoring is per-pair with no cross-pair
  interaction, `batch_size=1` and `batch_size=64` must produce identical output.
  This is a determinism property worth asserting.
- **No score mixing.** Rerank scores (raw logits, unbounded, model-specific) are
  never averaged, normalized, or blended with cosine/BM25/RRF values.
  `retrieval_score` is preserved purely as provenance.
- **No fallback in the default path.** `zip(..., strict=True)` converts a
  score/candidate length mismatch into a typed `RerankingError` rather than
  silently truncating the result list.

---

## F. Configuration

```python
class RerankerConfig(BaseModel):
    """Configuration specifically for second-stage cross-encoder reranking."""
    model_config = ConfigDict(extra="forbid")

    reranker_version: str = "onnx_cross_encoder_v1"
    model_id: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_revision: str | None = None      # pinned for reproducibility
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 16
    max_length: int = 512
    candidate_k: int = 20
    top_k: int = 10
    fallback_to_retrieval: bool = False
```

The same fields live inside `RetrievalConfig` (prefixed `rerank_*`) so that a
single hashed object covers the whole retrieval pipeline — the same choice Phase
4 made for `rrf_k` / `candidate_k`. All of them are covered by `config_hash`.

Validator additions to `RetrievalConfig.validate_method_and_version`:

- `default_version` gains `dense_rerank → dense_rerank_v1`,
  `bm25_rerank → bm25_rerank_v1`, `hybrid_rerank → hybrid_rerank_v1`.
- When `rerank_enabled`: `rerank_candidate_k >= top_k`, else `ValueError`.
- When `retrieval_method == "hybrid_rerank"`: `candidate_k >= rerank_candidate_k`.

`build_experiment_config()` adds `component_versions["reranker"]` **only** when
reranking is enabled, so existing manifests are unaffected. `--rerank` in the
CLI sets `retrieval_method = f"{base}_rerank"`, `rerank_enabled=True`, and the
CLI overrides set the corresponding `rerank_*` fields.

**Model caching is not part of the critical path.** `storage/reranker/` holds
the downloaded artifact only; reranker *outputs* are never cached, so baseline
latency measurements are not contaminated by cache hits. This is the same rule
Phase 2 applied to the embedding cache (content-addressed, for reproducibility,
not to game latency).

---

## G. Experiment Integration

`experiments/config.py::instantiate_components` — after the existing base-retriever
branch, wrap when `config.retrieval.rerank_enabled`:

```python
retriever = <dense | bm25 | hybrid>          # existing code, unchanged
if config.retrieval.rerank_enabled:
    retriever = RerankedRetriever(
        base_retriever=retriever,
        reranker=OnnxCrossEncoderReranker(
            model_id=config.retrieval.rerank_model_id,
            model_revision=config.retrieval.rerank_model_revision,
            device=config.retrieval.rerank_device,
            batch_size=config.retrieval.rerank_batch_size,
            max_length=config.retrieval.rerank_max_length,
            model_dir=RERANKER_DIR,
        ),
        config=config.retrieval,
        corpus_version=config.corpus_version,
    )
```

Slot 2 still returns the dense `vector_store` (or the `BM25Index` for BM25), so
`run_experiment.py`'s `count()` / `close()` are untouched. The hybrid dual-index
emptiness check must reach through `retriever.base_retriever.bm25_retriever.index`
when reranking is on.

### Runs to execute (networked machine)

Baseline runs already on disk and reusable as-is: `dense_baseline_v1`,
`bm25_baseline_v1`, `20260929T175539Z-hybrid_baseline_v1` (the latter is
retrieval-only). Six primary configurations:

```bash
# --rerank off: re-run only if a like-for-like retrieval-only dense/bm25 run is needed
.venv/bin/python scripts/run_experiment.py --retriever dense --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever bm25  --no-judge --no-generation

.venv/bin/python scripts/run_experiment.py --retriever dense --rerank --name dense_rerank_v1  --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever bm25  --rerank --name bm25_rerank_v1   --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever hybrid --rerank --name hybrid_rerank_v1 --no-judge --no-generation

# depth ablation, final top_k held at 10
for k in 10 20 40; do
  .venv/bin/python scripts/run_experiment.py --retriever hybrid --rerank \
      --rerank-candidate-k $k --name hybrid_rerank_k${k}_v1 --no-judge --no-generation
done

.venv/bin/python scripts/compare_reranking.py \
    --dense-rerank-run experiments/<id-dense_rerank_v1> \
    --bm25-rerank-run  experiments/<id-bm25_rerank_v1> \
    --hybrid-rerank-run experiments/<id-hybrid_rerank_v1>
```

> **Latency comparability caveat to record.** `dense_baseline_v1` was a *full*
> run with generation and LLM judge, while the BM25 and hybrid baselines were
> retrieval-only. Because reranking changes what reaches the generator, the
> clean comparison is retrieval-only across all six. Re-run dense and BM25 with
> `--no-judge --no-generation` if a strictly like-for-like latency table is
> wanted; state explicitly in `phase-5.md` which runs supplied each column.

Reranked dense and hybrid runs need `AICREDITS_API_KEY` (query embeddings, as in
Phase 4). **All reranked runs need network on first use** to download the ONNX
artifact; after that they are offline. Record the exact revision in the run's
`config.json` so every result is traceable.

Outcome is recorded **as observed**. The plan does not assume reranking wins.

---

## H. Testing Plan

`tests/test_reranking.py` — all offline and deterministic, using `FakeReranker`:

**Unit — scoring and ordering**
1. Rerank score drives ordering: a `FakeReranker` that inverts the candidate
   order produces exactly the inverted output.
2. `score == rerank_score`; `retrieval_score == original score`;
   `retrieval_rank == original rank`, for every returned result.
3. Original `rank` (1..n) is preserved in `retrieval_rank` even though `rank` is
   renumbered after reranking.
4. `top_k` behavior: 20 candidates at `top_k=5` → exactly 5 results, `rank` 1..5,
   `rerank_latency_ms` recorded, `candidate_count=20`, `result_count=5`.
5. `top_k` greater than the candidate count → returns all, no padding.
6. **Candidate depth**: `StubRetriever` records the `top_k` it received; assert it
   got `rerank_candidate_k` (e.g. 40), never the final `top_k`. Repeat for
   10 / 20 / 40.
7. **Hybrid depth coupling**: with `hybrid_rerank` and
   `rerank_candidate_k=40`, the wrapped `HybridRetriever`'s branches are queried
   at 40, not 20.
8. **Empty candidates** → `status="no_results"`, `results=[]`,
   `rerank_latency_ms == 0.0`, and the fake reranker's call count is **0**.
9. **Deterministic tie-breaking**: a fake reranker returning all-equal scores →
   output ordered by `retrieval_rank`, then `chunk_id` for equal ranks; two
   identical runs produce identical `chunk_id` order and identical scores.
10. **Batch-size invariance**: same fake reranker at `batch_size=1` and 64 →
    identical output.
11. **Metadata preservation**: `document_id`, `doc_title`, `section_path`,
    `headings`, `element_ids`, `page_start/end`, `token_count`.
12. **Provenance preservation**: `provenance.document_id`, `pages`,
    `source_sha256` identical to the base retriever's.
13. **Text preservation**: returned `text` is byte-identical to the base result.
14. **Latency accounting**: `latency_ms == candidate_generation_latency_ms +
    rerank_latency_ms` (within tolerance) and `>=` the base's `latency_ms`.
15. **Reranker failure** → original exception type propagates, message not
    rewritten; via `ExperimentRunner`, `status == "retrieval_failed"`,
    `retrieval is None`, `error.stage == "retrieval"`, and the base error type is
    preserved.
16. **Fallback off by default**: a raising reranker with default config never
    yields a `RetrievalResponse`.
17. **Fallback on, observable**: with `rerank_fallback=True`, results come back
    truncated to `top_k` with `score == retrieval_score`,
    `retrieval_rank == rank`, and `retrieval_metadata.rerank_fallback is True`.
18. **Empty-query guard**: `InvalidQueryError` raised with both the base
    retriever's call count and the fake reranker's call count at 0.
19. **Protocol conformance**: `isinstance(retriever, Retriever)`; `method` is the
    expected `<base>_rerank`.
20. **Score-length mismatch** in the fake reranker → `RerankingError`, no silent
    truncation.
21. **Config**: `RetrievalConfig(retrieval_method="dense_rerank")` →
    `retriever_version == "dense_rerank_v1"`; `rerank_candidate_k < top_k` with
    reranking enabled raises; `hybrid_rerank` with
    `candidate_k < rerank_candidate_k` raises; `config_hash` differs when
    `rerank_model_id`, `rerank_candidate_k`, or `rerank_device` changes.

**Integration (offline, real components)**
22. Dense → Rerank: real `DenseRetriever` (in-memory Qdrant + `FakeEmbeddingModel`)
    + `FakeReranker`, through `ExperimentRunner` with `RetrievalEvaluator` +
    `EfficiencyEvaluator`; assert artifacts written and `rerank_latency_ms_mean`
    non-null.
23. BM25 → Rerank: real `BM25Retriever` over an in-memory `BM25Index` built from
    the same chunks.
24. Hybrid → Rerank: real dense + real BM25 + `HybridRetriever` + `FakeReranker`;
    assert both index-emptiness concerns hold and fusion diagnostics survive into
    the reranked metadata.
25. Canonical-corpus regression (`@pytest.mark.skipif` when chunks are absent):
    "Robertson" / "ColBERT" / "BM25" still surface the expected documents after
    reranking — a reranker must not destroy retrieval correctness.

**Regression — reranking disabled**
26. Dense, BM25, and Hybrid outputs are **byte-identical** to their pre-Phase-5
    forms with `rerank_enabled=False`: assert via `canonical_json(model_dump())`
    against the same components without the wrapper.
27. `instantiate_components()` returns the plain `DenseRetriever` /
    `BM25Retriever` / `HybridRetriever` (not a `RerankedRetriever`) when
    reranking is off.
28. Existing `experiments/bm25_baseline_v1` and hybrid runs still validate against
    the current schemas — i.e. new optional fields do not break old artifacts.

**Architecture guards (`test_architecture_guards.py`)**
29. Remove `"rerank"` from `FORBIDDEN_STRATEGY_TOKENS`; keep `adaptive_rout`,
    `query_classif`, `strategy_select` — **Phase 6 stays banned**.
30. Keep `FORBIDDEN_HEAVY_DEPS` unchanged: `sentence-transformers` and
    `llama-index` remain banned; assert `onnxruntime`/`tokenizers` are declared.
31. **New** `test_reranker_has_no_index_access`: `reranking/base.py` and
    `reranking/onnx_backend.py` must not import `adaptive_rag.retrieval`,
    `adaptive_rag.indexing`, `adaptive_rag.ingestion`, or `adaptive_rag.chunking`,
    and must not contain `qdrant`, `bm25`, `idf`, `collection`.
32. **New** `test_reranked_retriever_only_composes`: `reranked.py` must not
    contain `qdrant`, `bm25`, `tokenize`, `idf`, `embedding_model`; must not
    import `dense`/`bm25`/`hybrid` modules; and must contain no unconditional
    `ast.Try` (the single `except` must be guarded by `config.rerank_fallback`).
33. **New** `test_hybrid_is_independent_of_reranking`: `hybrid.py` must not
    contain `rerank`, `route`, or `router`, and must not import
    `adaptive_rag.reranking` — the existing `FORBIDDEN_HYBRID_TOKENS` guard
    extended rather than replaced.
34. **New** `test_reranking_does_not_leak_into_retrievers`: `dense.py`, `bm25.py`,
    `hybrid.py`, and `fusion.py` must all remain free of `rerank`.

Full suite must stay green: currently **86** tests, all unchanged in intent,
plus the new Phase 5 tests.

**Live / integration-marked**
35. `@pytest.mark.integration` `test_onnx_cross_encoder_backend_runs`: download
    `Xenova/ms-marco-MiniLM-L-6-v2`, load the ONNX session, assert the session
    input/output names and output shape are as expected, score a 4-passage
    batch, assert batch-size invariance against `batch_size=1`, and assert the
    model id + revision resolve. Skipped by default (the repo already
    deselects `integration` via `addopts`); run explicitly on the networked
    machine.

---

## I. Risks

| Risk | Assessment / Mitigation |
| --- | --- |
| Guard bans `rerank` across `src/` | Known and expected. Step 1 of implementation is updating `FORBIDDEN_STRATEGY_TOKENS`, before any rerank source file exists. |
| `sentence-transformers` ban vs. a real cross-encoder | Resolved by the ONNX backend — no banned dependency, and the guard test asserts it stays unbanned. |
| ONNX artifact may not exist as assumed | Cannot be verified offline. Mitigation: `onnx_backend.py` raises a typed `RerankerModelError` naming the exact missing path; `Xenova/ms-marco-MiniLM-L-12-v2` is a documented drop-in via `--reranker-model`; `onnxruntime` is a well-maintained exporter path for ms-marco cross-encoders. Verify in test 35 before running the benchmark. |
| `extra="forbid"` on all retrieval schemas | Every new field is declared and optional, so dense/BM25/hybrid output is byte-identical. Pinned by regression tests 26–28. |
| Runner reads only `latency_ms` | Kept as the full end-to-end total (hybrid rule), so `EfficiencyEvaluator`'s existing metrics keep their meaning; the split is additive. |
| Efficiency output changes | New metrics are emitted **only** when a trace carries the rerank fields, so existing runs' `metrics_efficiency.json` is unchanged. |
| `run_experiment.py` hybrid index check | Must unwrap `retriever.base_retriever`. Explicitly listed in §C. |
| Rerank latency swamps the comparison | With MiniLM-L-6 at `batch_size=16`, ~40 pairs is tens of ms on 12 cores — small next to dense's ~750 ms network embedding, which is the point the phase must measure. Recorded per-stage, not just as a total. |
| Benchmark noise on n=20 | Latency deltas are directional. `EfficiencyEvaluator` percentiles plus the separate rerank-latency series make the reranker's own cost readable independently of embedding noise. Report as observed with the caveat stated. |
| Silent fallback contamination | Impossible by default: the `except` is gated on `config.rerank_fallback` (AST guard 32), and when taken it sets `rerank_fallback=True`, which surfaces in traces, in `rerank_fallback_count`, and in the manifest. |
| Non-deterministic ties | Explicit `(-rerank_score, retrieval_rank, chunk_id)` key. Batch-size invariance (test 10) proves batching cannot perturb ordering. |
| Scores mistaken for probabilities | Documented as ranking signals; `retrieval_score` retained separately and never blended with them. |
| Phase 6 leakage | `adaptive_rout`, `query_classif`, `strategy_select` stay guard-banned. `RerankedRetriever` is deliberately *not* conditional on anything — it never decides whether reranking is worth it. That decision is Phase 6's. |

---

## J. Phase 5 File / Change Summary

| File | Action | Purpose |
| ---- | ------ | ------- |
| `src/adaptive_rag/reranking/base.py` | create | `Reranker` protocol, index-free by construction |
| `src/adaptive_rag/reranking/onnx_backend.py` | create | ONNX Runtime cross-encoder backend, batched, device-explicit |
| `src/adaptive_rag/reranking/__init__.py` | create | Package exports |
| `src/adaptive_rag/retrieval/reranked.py` | create | `RerankedRetriever` composing any base retriever + a reranker |
| `tests/test_reranking.py` | create | Scoring, `top_k`, depth, empty, ties, provenance, failure, integration, regression, guards |
| `scripts/compare_reranking.py` | create | Six-config comparison + depth-ablation tables |
| `src/adaptive_rag/schemas/retrieval.py` | modify | Rerank literals, result/metadata fields |
| `src/adaptive_rag/schemas/config.py` | modify | `RerankerConfig`, `rerank_*` fields, validator |
| `src/adaptive_rag/schemas/experiment.py` | modify | Optional trace fields for the latency split |
| `src/adaptive_rag/schemas/__init__.py` | modify | Export `RerankerConfig` |
| `src/adaptive_rag/retrieval/__init__.py` | modify | Export `RerankedRetriever` |
| `src/adaptive_rag/errors.py` | modify | Reranking error hierarchy |
| `src/adaptive_rag/config/paths.py` | modify | `RERANKER_DIR` |
| `src/adaptive_rag/experiments/config.py` | modify | Wrap base retriever when reranking enabled; hybrid depth coupling |
| `src/adaptive_rag/experiments/runner.py` | modify | Copy split latencies/counts onto the trace |
| `src/adaptive_rag/evaluation/efficiency.py` | modify | Rerank latency/count metrics, emitted conditionally |
| `scripts/run_experiment.py` | modify | `--rerank` + overrides, run-name defaults, index-check unwrap |
| `tests/fakes.py` | modify | `FakeReranker` |
| `tests/test_architecture_guards.py` | modify | Unban `rerank`; add four Phase 5 guards |
| `pyproject.toml` | modify | `onnxruntime`, `tokenizers`, `huggingface-hub` |
| `docs/phases/phase-5.md` | create | Phase 5 record |
| `docs/progress.md`, `docs/decision.md`, `docs/architecture.md`, `README.md`, `experiments/README.md` | modify | Status, ADR-020, architecture, runbook |

Not touched: `retrieval/{dense,bm25,hybrid,fusion}.py`, `indexing/*`,
`evaluation/{retrieval,generation,judge}.py`, `generation/*`, `rag.py`,
`scripts/ask.py`, `scripts/compare_retrievers.py`. No query classification, no
heuristic or learned routing, no confidence-based reranking decisions, no
adaptive weights, no dynamic candidate depth.

---

## K. Implementation Order

1. **Guards first** — remove `"rerank"` from `FORBIDDEN_STRATEGY_TOKENS`, add the
   four Phase 5 guard tests (they pass trivially until the modules exist). Also
   add `FakeReranker` to `tests/fakes.py`.
2. **Schemas** — `schemas/retrieval.py`, `schemas/config.py` (+ `RerankerConfig`),
   `schemas/experiment.py`, `schemas/__init__.py`, `errors.py`. Verify with
   `pytest tests/test_schemas.py tests/test_config.py tests/test_bm25.py
   tests/test_determinism.py tests/test_evaluation.py tests/test_experiment_runner.py`
   before touching retrieval — these must still pass unchanged in intent.
3. **`reranking/base.py`** + the ordering rule, then unit tests 1–3, 9, 20.
4. **`retrieval/reranked.py`** + tests 4–8, 10–19, 21.
5. **`reranking/onnx_backend.py`** — written but not exercised offline; covered by
   the `integration`-marked test 35.
6. **Experiment integration** — `experiments/config.py` wrap + hybrid depth
   coupling, `runner.py` trace copy, `evaluation/efficiency.py` metrics, and
   `run_experiment.py` flags. Integration tests 22–24.
7. **Regression** — tests 26–28: dense/BM25/hybrid byte-identical with reranking
   off, and old run artifacts still validating.
8. **`scripts/compare_reranking.py`** with fixture run directories (mirroring
   `test_hybrid.py`'s `_write_fake_run` approach), asserting exit 2 on a method
   or corpus mismatch.
9. **Full suite green** — 86 existing + all new Phase 5 tests, no credentials.
10. **Live benchmark on the networked machine** — run the `integration` ONNX test
    first (test 35); if the artifact does not resolve, fix the model id before
    spending a benchmark run. Then the six primary configurations, then the
    10/20/40 depth ablation, then `compare_reranking.py`. Record results **as
    observed**, including if reranking does not pay for itself.
11. **Documentation** — `docs/phases/phase-5.md`, `progress.md`, `decision.md`
    (ADR-020), `architecture.md`, `README.md`, `experiments/README.md`.
12. **Report** — changed files, architectural decisions, configuration options,
    test results, experiment configurations, measured retrieval/latency results,
    unresolved issues, and the questions deliberately left open for Phase 6.

---

## Open Questions for Phase 6 (deliberately unresolved here)

- Does reranking help enough to justify its cost, and for which query types?
  Phase 5 measures the aggregate; it does **not** classify queries.
- Should the reranker be a distinct selectable strategy, or a modifier applied
  to whatever strategy the router picks? Phase 5 makes it composable but does
  not decide.
- Is there a candidate depth at which reranking stops paying for itself? The
  10/20/40 ablation is designed to surface this; the decision belongs to Phase 6.
- Should a reranker failure fall back in production, or stay fail-closed? Phase 5
  ships fail-closed by default with an observable opt-in fallback; the policy
  choice is Phase 6's.

## Blocking Assumption

Step 10 depends on a networked machine and on
`Xenova/ms-marco-MiniLM-L-6-v2` resolving to a repo containing `onnx/model.onnx`,
`tokenizer.json`, and `config.json`. Steps 1–9 and 11–12 are fully offline; step
10 is marked PENDING until that artifact is confirmed.
