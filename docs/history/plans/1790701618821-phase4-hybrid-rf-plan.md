# Phase 4 — Hybrid Retrieval (Dense + BM25 + RRF)

## A. Repository Assessment

The repo already satisfies every structural precondition Phase 4 needs:

- `src/adaptive_rag/retrieval/base.py:14` — `Retriever` is a `@runtime_checkable`
  `Protocol` with `method: str` and
  `retrieve(query, top_k, score_threshold, filters) -> RetrievalResponse`.
  `HybridRetriever` can conform with no protocol change.
- `src/adaptive_rag/retrieval/dense.py:25` and `.../bm25.py:22` are two
  independent concrete implementations of that protocol, each producing
  1-indexed `rank`, native `score`, and full `ChunkMetadata` /
  `ChunkProvenance` (incl. `source_sha256`).
- `src/adaptive_rag/schemas/retrieval.py` — `RetrievalResult`,
  `RetrievalMetadata`, `RetrievalResponse` are `extra="forbid"` Pydantic models.
  Two edits are required: `RetrievalResponse.retrieval_method` must accept
  `"hybrid"`, and `RetrievalMetadata` must gain optional hybrid fields.
- `src/adaptive_rag/experiments/runner.py:161` — `ExperimentRunner` only calls
  `retriever.retrieve(query=..., top_k=config.retrieval.top_k)` and reads
  `retrieval.retrieval_metadata.latency_ms`. **No runner change is required**
  for Hybrid; failure semantics (exception → `status="retrieval_failed"`,
  never a silent fallback) are already correct.
- `src/adaptive_rag/evaluation/retrieval.py` and `.../efficiency.py` are
  fully strategy-agnostic (they read `trace.retrieval.results[*].chunk_id`,
  `.metadata.document_id`, `.rank`, and `trace.retrieval_latency_ms`).
  **No evaluator change is required.**
- `src/adaptive_rag/experiments/config.py:109` `instantiate_components()` is the
  single construction site; it branches on `config.retrieval.retrieval_method`.
- `scripts/run_experiment.py:36` — `--retriever {dense,bm25}`; `scripts/compare_retrievers.py`
  — the Dense/BM25 table. Both are the extension points.
- `tests/test_architecture_guards.py:15` — `FORBIDDEN_STRATEGY_TOKENS` still
  contains `"hybrid"`, and the guard scans every file under `src/`. **Phase 4
  must update this guard or the whole suite fails.**

Two conventions the implementation must follow:

1. `retrieval_method ⇔ retriever_version` is kept aligned by the model validator
   in `schemas/config.py:105` and stamped into `component_versions["retrieval"]`
   by `build_experiment_config()`. Hybrid gets `"hybrid"` ⇔ `"hybrid_v1"`.
2. `ExperimentTrace.retrieval_latency_ms` is the single latency the efficiency
   evaluator consumes, so `RetrievalMetadata.latency_ms` must be the **full
   end-to-end** hybrid cost, not just fusion time.

Decisions taken with the user:

- **No per-result diagnostic fields.** `RetrievalResult` is untouched; `score`
  is the RRF score, `rank` is the fused rank. Fusion diagnostics live in
  `RetrievalMetadata`.
- **Tie-break:** `(-rrf_score, best_single_source_rank, chunk_id)`.
- **Comparison:** extend `compare_retrievers.py` with an optional `--hybrid-run`
  and a third column.
- **Failure:** constituent exceptions propagate unchanged (no re-wrapping).
- **Empty query:** validated up front in `HybridRetriever` before delegating.

---

## B. Files to Create

### 1. `src/adaptive_rag/retrieval/hybrid.py`
**Purpose:** the Phase 4 fusion retriever.

Responsibilities:
- `HybridRetriever(dense_retriever: Retriever, bm25_retriever: Retriever, config, corpus_version, index_id)`.
- `method = "hybrid"`.
- Up-front empty-query guard raising `InvalidQueryError` (same message as the
  constituents).
- Call `dense_retriever.retrieve(query, top_k=candidate_k, filters=filt)` and
  `bm25_retriever.retrieve(query, top_k=candidate_k, filters=filt)`, timing each.
- RRF fusion (pure function below), truncate to `top_k`, renumber ranks `1..n`.
- Build `RetrievalMetadata` incl. hybrid diagnostics; return
  `RetrievalResponse(retrieval_method="hybrid", status="ok"|"no_results")`.
- No tokenization, no embedding, no score mixing, no fallback, no try/except
  around constituents.

### 2. `src/adaptive_rag/retrieval/fusion.py`
**Purpose:** the RRF function, isolated from orchestration so it is unit-testable
without any retriever, index, or credential.

Responsibilities:
- `reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[RetrievalResult]], rrf_k: int) -> list[RetrievalResult]`
  — returns fused results with `score` = RRF score, `rank` set by the caller.
- Deterministic tie-break as decided.
- No `FusionStrategy` base class, no registry, no weighted/CombSUM variants.

### 3. `tests/test_hybrid.py`
**Purpose:** the Phase 4 test suite (§34.1–34.13). All offline and deterministic.

---

## C. Files to Modify

| Path | Current responsibility | Required change | Reason |
| --- | --- | --- | --- |
| `src/adaptive_rag/schemas/retrieval.py` | Retrieval contracts | `retrieval_method: Literal["dense","bm25","hybrid"]`; add optional hybrid fields to `RetrievalMetadata` | Hybrid must serialize into the common response |
| `src/adaptive_rag/schemas/config.py` | Config models | `RetrievalConfig.retrieval_method` += `"hybrid"`; add `rrf_k: int = 60`, `candidate_k: int = 20`; extend validator for `hybrid ⇔ hybrid_v1`; add `HybridRetrievalConfig` | Configurable, reproducible hybrid params |
| `src/adaptive_rag/schemas/__init__.py` | Package exports | Export `HybridRetrievalConfig` | Keep the flat export surface convention |
| `src/adaptive_rag/retrieval/__init__.py` | Package exports | Export `HybridRetriever` (and `reciprocal_rank_fusion` if desired) | ADR-007 strategy set |
| `src/adaptive_rag/experiments/config.py` | Component construction | New `elif retrieval_method == "hybrid"` branch: build `DenseRetriever` + `BM25Retriever` from the same `config`, compose into `HybridRetriever`; return the dense `vector_store` in slot 2 so `run_experiment.py` keeps working | Single place where retrievers are built |
| `scripts/run_experiment.py` | Experiment CLI | `--retriever {dense,bm25,hybrid}`; default name `hybrid_baseline_v1`; index-emptiness check branches for hybrid (both indexes) | Strategy selection via the existing mechanism |
| `scripts/compare_retrievers.py` | Comparison CLI | Add `--hybrid-run` (optional), third column, hybrid-aware consistency check | §38/§22, user decision |
| `tests/test_architecture_guards.py` | Guards | Remove `"hybrid"` from `FORBIDDEN_STRATEGY_TOKENS`; add Phase 4 guards (§35) | Otherwise the suite fails; guards must be re-pointed, not deleted |
| `tests/fakes.py` | Offline test doubles | Add a `StubRetriever` that returns a canned `RetrievalResponse` and can be made to raise | Failure / empty-result / unequal-length tests need a controllable retriever |
| `docs/phases/phase-4.md` (new) | Phase record | Objective, architecture, RRF spec, config, validation, DoD | Documentation is source of truth |
| `docs/progress.md` | Status | Phase 4 → COMPLETE, results, next steps | — |
| `docs/decision.md` | ADRs | New ADR-019 (hybrid fusion), ADR-017 phase table update | — |
| `docs/architecture.md` | System architecture | New hybrid/RRF section, `retrieval_method` set updated | — |
| `experiments/README.md` | Runbook | `--retriever hybrid` invocation | — |

Explicitly **not** modified: `retrieval/dense.py`, `retrieval/bm25.py`,
`indexing/*`, `evaluation/*`, `experiments/runner.py`, `generation/*`,
`rag.py`, `scripts/ask.py`, `config/paths.py` (no `storage/hybrid/`).

---

## D. Data / Control Flow

```text
run_experiment.py --retriever hybrid
  → build_experiment_config(retrieval=RetrievalConfig(retrieval_method="hybrid",
                                                      rrf_k=60, candidate_k=20, top_k=10))
      (validator aligns retriever_version → "hybrid_v1"; config_hash covers rrf_k/candidate_k)
  → instantiate_components(config)
      → DenseRetriever(embedding_model, vector_store, config, corpus_version, index.collection_name)
      → BM25Retriever(index=BM25Index.load(expected_corpus_version=...), config, corpus_version)
      → HybridRetriever(dense_retriever, bm25_retriever, config, corpus_version)
  → ExperimentRunner.run(config, dataset)
      per example:
        retriever.retrieve(query, top_k=10)
          ├─ guard: empty query → InvalidQueryError
          ├─ t0
          ├─ dense.retrieve(query, top_k=20)          → 20 results (or fewer/empty)
          ├─ bm25.retrieve(query,  top_k=20)          → 20 results (or fewer/empty)
          ├─ reciprocal_rank_fusion([[...], [...]], rrf_k=60)   → ≤40 unique results
          ├─ truncate to 10, renumber rank 1..10, score = rrf
          └─ RetrievalResponse(retrieval_method="hybrid", status, RetrievalMetadata{...})
              latency_ms = dense + bm25 + fusion
        RetrievalEvaluator / EfficiencyEvaluator consume the response unchanged
      → traces.jsonl, metrics.json, manifest.json, report.md
```

Failure path: any exception from either constituent propagates untouched →
`runner.py:167` records `status="retrieval_failed"` with the original
`error_type`. Hybrid is never silently downgraded to dense or BM25.

---

## E. RRF Algorithm

```python
def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse ranked result lists with Reciprocal Rank Fusion (rank-based only)."""
    if rrf_k < 0:
        raise ConfigurationError("rrf_k must be non-negative")

    # chunk_id -> accumulation, in first-seen order (never relied on for ordering)
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    source: dict[str, RetrievalResult] = {}   # canonical payload carrier
    MISSING = 1 << 30

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            cid = result.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if rank < best_rank.get(cid, MISSING):
                best_rank[cid] = rank
            # dense is always ranked_lists[0]; prefer it as the payload carrier
            if cid not in source or results is ranked_lists[0]:
                source[cid] = result

    fused = [
        source[cid].model_copy(update={"score": score})
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r.score, best_rank[r.chunk_id], r.chunk_id))
    return fused
```

Notes on the decisions encoded above:

- **Rank, never score.** `1/(k+rank)` only. Dense cosine and BM25 magnitudes
  never meet.
- **Deduplication key is `chunk_id`**, never text.
- **Missing document ⇒ 0 contribution.** No padding, no fabricated rank, no
  fabricated score. Unequal list lengths (20 vs 7) and empty lists just make
  the accumulation loop shorter.
- **Payload carrier.** When a chunk appears in both lists, the dense
  `RetrievalResult` is reused verbatim for `text`/`metadata`/`provenance`
  (dense rebuilds these from the Qdrant payload; BM25 returns the stored
  canonical objects, which are the same values). Only `score` is replaced.
  A chunk present only in BM25 keeps its BM25 payload.
- **Deterministic tie-break:** `(-rrf_score, best_single_source_rank, chunk_id)`.
  Ties are genuinely possible (e.g. rank 1 + absent vs absent + rank 1 mirror
  images), so this is required, not cosmetic. `chunk_id` is the canonical stable
  identifier and is the final guarantee; nothing depends on set or dict
  iteration order.
- **No `fusion.py` abstraction layer beyond this one function** — no
  `FusionStrategy`, no registry (§37).

---

## F. Configuration Changes

`src/adaptive_rag/schemas/config.py`:

```python
class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "dense_v1"
    retrieval_method: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None
    k1: float = 1.2
    b: float = 0.75
    # Hybrid / RRF-specific parameters
    rrf_k: int = 60
    candidate_k: int = 20

    @model_validator(mode="after")
    def validate_method_and_version(self) -> "RetrievalConfig":
        default_version = {"dense": "dense_v1", "bm25": "bm25_v1", "hybrid": "hybrid_v1"}
        if self.retriever_version in default_version.values():
            object.__setattr__(self, "retriever_version",
                               default_version[self.retrieval_method])
        if self.retrieval_method == "hybrid" and self.candidate_k < self.top_k:
            raise ValueError("hybrid candidate_k must be >= top_k")
        return self
```

Plus a sibling `HybridRetrievalConfig` mirroring `DenseRetrievalConfig` /
`BM25RetrievalConfig`, exported from `schemas/__init__.py`.

Reproducibility: `rrf_k`, `candidate_k`, `top_k`, `retrieval_method`, and
`retriever_version` are all inside `RetrievalConfig`, which
`build_experiment_config()` hashes into `config.json` / `config_hash` and
mirrors into `manifest.json` (`retrieval_method`, `component_versions.retrieval`
= `"hybrid_v1"`). No new config framework, no YAML files — the project uses
pydantic models assembled in code, and Phase 4 follows that.

**Deliberate call on `score_threshold`:** Hybrid does **not** forward
`score_threshold` to either constituent. A cosine threshold and a BM25
threshold are on incomparable scales, and applying one to both would silently
truncate one candidate list. The value is still echoed into
`RetrievalMetadata.score_threshold` for traceability. Default is `None`
anyway, so the baseline is unaffected. `filters` **are** forwarded to both
(§19).

---

## G. Experiment Integration

`experiments/config.py::instantiate_components` gains a third branch:

```python
if config.retrieval.retrieval_method == "hybrid":
    embedding_model = AICreditsEmbeddingModel(config=config.embedding)
    vector_store = QdrantVectorStore(config=config.index)
    bm25_index = BM25Index.load(expected_corpus_version=config.corpus_version)
    dense = DenseRetriever(embedding_model=embedding_model, vector_store=vector_store,
                           config=config.retrieval, corpus_version=config.corpus_version,
                           index_id=config.index.collection_name)
    bm25 = BM25Retriever(index=bm25_index, config=config.retrieval,
                         corpus_version=config.corpus_version, index_id="adaptiverag_bm25_v1")
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25,
                                config=config.retrieval, corpus_version=config.corpus_version)
    return (embedding_model, vector_store, retriever, context_builder,
            generator, chunker, ingestion_pipeline)
```

Returning `vector_store` in slot 2 keeps `run_experiment.py`'s
`index_or_store.count()` / `.close()` calls valid. Hybrid therefore needs
**both** `AICREDITS_API_KEY` and the persisted BM25 index — the branch raises
`IndexUnavailableError` if the BM25 JSON is missing, which is the correct
"fail, don't fall back" behavior.

`scripts/run_experiment.py`:
- `--retriever` choices → `["dense", "bm25", "hybrid"]`.
- Default experiment name mapping: `bm25` → `bm25_baseline_v1`, `hybrid` →
  `hybrid_baseline_v1` (generalize the existing `if args.retriever == "bm25"`
  special case).
- Emptiness guard: dense → `vector_store.count()`; bm25 → `index.total_docs`;
  hybrid → both, with an error naming the missing side.

`scripts/compare_retrievers.py`:
- Add `--hybrid-run` (optional). When absent, behaviour is byte-identical to
  today (dense vs bm25 only).
- When present: add a `hybrid` column and relax the method check to expect
  `dense`/`bm25`/`hybrid` respectively. Corpus-version and trace-count
  consistency checks extend to the third run unchanged.

Post-implementation run (same benchmark, same corpus version `corpus_6c416f423920385d`):

```bash
.venv/bin/python scripts/run_experiment.py --retriever hybrid --name hybrid_baseline_v1 --no-judge --no-generation
.venv/bin/python scripts/compare_retrievers.py --hybrid-run experiments/hybrid_baseline_v1
```

Artifacts land in `experiments/hybrid_baseline_v1/`:
`config.json`, `manifest.json`, `traces.jsonl`, `metrics.json`,
`metrics_{retrieval,efficiency}.json`, `report.md`.

Outcome is recorded as observed — the plan does not assume Hybrid wins.

---

## H. Testing Plan

New `tests/test_hybrid.py` (all offline; fakes + in-memory Qdrant + in-memory
`BM25Index`; no credentials):

1. **RRF mathematics** — dense `[A,B,C]`, bm25 `[B,C,D]`, `rrf_k=60`; assert
   exact scores: `A=1/61`, `B=1/61+1/61`, `C=1/62+1/62`, `D=1/62`, and the
   resulting order `B, A, C, D` (or the exact computed order — assert the
   values, derive the order from them).
2. **Duplicate handling** — same chunk at dense rank 1 / bm25 rank 4 appears
   exactly once with `1/61 + 1/64`.
3. **Dedupe by chunk_id, not text** — two distinct chunk_ids with identical
   text both survive.
4. **Single-source chunks** — chunks present in only one list still appear,
   with the one-term contribution.
5. **Unequal lengths** — dense 20 / bm25 7; all 27 (or overlap-adjusted) ids
   present with correct contributions; no padding artifacts.
6. **Empty result sets** — `dense=[] , bm25=[...]` and the mirror case.
7. **Both empty** → `status="no_results"`, `results == []`, latency still
   recorded.
8. **Constituent failure** — a `StubRetriever` set to raise; `pytest.raises`
   the original exception type, and the message is not rewritten. Assert via
   `ExperimentRunner` that `status == "retrieval_failed"` and that the
   response is *not* a dense-only or bm25-only result.
9. **Metadata/provenance preservation** — fused results keep
   `chunk_id`, `document_id`, `section_path`, `pages`, `source_sha256`,
   `text`; a chunk present only in BM25 keeps BM25's payload.
10. **`top_k` truncation** — `top_k=5` with 20/20 candidates returns exactly 5,
    ranks 1..5, scores non-increasing.
11. **Candidate depth** — `StubRetriever` records the `top_k` it received;
    assert both constituents received `candidate_k` (20), not the final `top_k`.
12. **Hybrid score semantics** — `score` equals the RRF score, not the dense
    cosine or the BM25 magnitude; `RetrievalMetadata` reports
    `fusion_method="rrf"`, `rrf_k`, `candidate_k`, and the per-branch counts.
13. **Latency accounting** — `latency_ms >= dense_latency_ms + bm25_latency_ms`
    and includes fusion; `search_latency_ms` is the sum of the two constituent
    search latencies.
14. **Protocol conformance** — `isinstance(retriever, Retriever)`,
    `retriever.method == "hybrid"`.
15. **Empty-query guard** — `InvalidQueryError` raised before either
    constituent is called (assert call counts are 0).
16. **Determinism** — two identical runs produce byte-identical fused
    `chunk_id` order and identical scores; a constructed exact-tie input yields
    the chunk_id-ascending order.
17. **Config loading** — `RetrievalConfig(retrieval_method="hybrid")` →
    `retriever_version == "hybrid_v1"`; `HybridRetrievalConfig` defaults
    `rrf_k=60`, `candidate_k=20`; `candidate_k < top_k` raises.
18. **Integration** — real `DenseRetriever` (in-memory Qdrant +
    `FakeEmbeddingModel`) + real `BM25Retriever` (in-memory `BM25Index` built
    from the same chunks) → `HybridRetriever` → `RetrievalResponse`, through
    `ExperimentRunner` with `RetrievalEvaluator` + `EfficiencyEvaluator`,
    asserting artifacts and a non-null Recall@5.
19. **Regression on the canonical corpus** — `@pytest.mark.skipif` when
    `data/processed/chunks/*.chunks.jsonl` is absent, mirroring
    `test_bm25.py::test_regression_lexical_keywords_on_canonical_corpus`;
    asserts hybrid surfaces the expected documents for known keywords, and that
    the fused union is a superset of neither ranking.

`tests/test_architecture_guards.py` changes:
- Remove `"hybrid"` from `FORBIDDEN_STRATEGY_TOKENS` (keep `"rerank"`,
  `"adaptive_rout"`, `"query_classif"`, `"strategy_select"`).
- Existing `test_retriever_isolation_between_dense_and_bm25` is left as-is: it
  asserts `dense.py` contains no `"bm25"` and `bm25.py` contains no
  `"dense"`/`"vector_store"`/`"embedding"` — Phase 4 does not change either
  file, so the guard still holds and remains the proof that composition did
  not leak backward.
- New `test_hybrid_composes_existing_retrievers`: `hybrid.py` must not contain
  `"rerank"`, `"router"`, `"route"`, `"adaptive"`, `"comb_sum"`, `"comb_mnz"`,
  `"weighted"`, `"qdrant"`, `"tokenize"`, or `"idf"` — i.e. it orchestrates and
  never reimplements dense, BM25, fusion-alternative, routing, or reranking
  logic. Also assert `hybrid.py` contains no `try:` around constituent calls.

Regression: full suite must stay green — currently **57 tests**; the 57 must
all still pass unchanged in intent, plus the new Phase 4 tests.

---

## I. Risks

| Risk | Assessment / Mitigation |
| --- | --- |
| `RetrievalResponse` Literal rejects `"hybrid"` | Known; the first schema edit. Guarded by test 18 and by an explicit `retrieval_method == "hybrid"` assertion. |
| `RetrievalMetadata` is `extra="forbid"` | Any new field must be declared on the model; add only the §32 fields, all optional, so dense/BM25 responses stay byte-identical on the wire. |
| Result identity across indexes | `DenseRetriever` takes `chunk_id` from the Qdrant payload and `BM25Retriever` from the persisted index; both derive from the same canonical `Chunk.chunk_id`, so fusion keys align. Verified by test 3 and the regression test. |
| Metadata drift between the two branches | Dense rebuilds `ChunkMetadata` from a Qdrant payload; BM25 returns the stored canonical objects. Values should be identical, but the payload carrier rule (prefer dense) plus test 9 make any drift visible rather than silent. |
| Nondeterministic ties | Explicit `(-rrf_score, best_rank, chunk_id)` key. Python's sort is stable, but relying on stability would depend on input order, which is why `chunk_id` is the final tiebreaker. |
| Failure semantics | No try/except around constituents; the decision is to propagate. Test 8 asserts both the raised type and the runner's `retrieval_failed` status. Silent dense-only downgrade is impossible by construction. |
| Candidate depth leakage | Hybrid must pass `candidate_k` (20) down and apply the caller's `top_k` only after fusion. Test 11 pins the value each constituent receives. |
| Latency under-reporting | `latency_ms` must be measured around *both* constituent calls *and* fusion, not copied from either branch. Test 13 asserts the inequality. |
| Config compatibility | `extra="forbid"` means any old config object still validates; the new fields are defaulted. The rewritten validator is behavior-preserving for `dense`/`bm25` — existing `test_bm25_config_validation` and the dense path in `test_determinism.py` cover this. |
| Experiment reproducibility | `rrf_k`/`candidate_k` are inside the hashed `RetrievalConfig`; `config.json` and `manifest.json` record `retrieval_method`, `retriever_version="hybrid_v1"`, and the corpus version. |
| `instantiate_components` tuple shape | Hybrid needs two indexes; the dense `vector_store` is returned in slot 2 so the CLI's `count()`/`close()` keep working. The BM25 index is validated at load time. |
| `compare_retrievers.py` output stability | Hybrid column is opt-in; with `--hybrid-run` absent the printed table must be byte-identical to today (existing determinism expectations). |
| Architecture guard breakage | `"hybrid"` in `FORBIDDEN_STRATEGY_TOKENS` would fail the suite immediately; the guard update is step 1 of implementation, before any hybrid source file is created. |
| Benchmark noise | Hybrid inherits dense's ~750 ms/query network latency; with only 20 examples the latency comparison is directional. `EfficiencyEvaluator` percentiles already capture this; the report should state it. |

---

## J. Phase 4 File / Change Summary

| File | Action | Purpose |
| ---- | ------ | ------- |
| `src/adaptive_rag/retrieval/fusion.py` | create | Rank-based RRF function with deterministic tie-break |
| `src/adaptive_rag/retrieval/hybrid.py` | create | `HybridRetriever` composing dense + BM25 |
| `tests/test_hybrid.py` | create | RRF math, dedupe, empty, failure, metadata, config, integration, determinism |
| `src/adaptive_rag/schemas/retrieval.py` | modify | Allow `"hybrid"`; optional hybrid diagnostics on `RetrievalMetadata` |
| `src/adaptive_rag/schemas/config.py` | modify | `hybrid` method, `rrf_k`, `candidate_k`, version alignment, `HybridRetrievalConfig` |
| `src/adaptive_rag/schemas/__init__.py` | modify | Export `HybridRetrievalConfig` |
| `src/adaptive_rag/retrieval/__init__.py` | modify | Export `HybridRetriever` |
| `src/adaptive_rag/experiments/config.py` | modify | Build dense + BM25 + hybrid retriever |
| `scripts/run_experiment.py` | modify | `--retriever hybrid`, name default, dual-index guard |
| `scripts/compare_retrievers.py` | modify | Optional `--hybrid-run` third column |
| `tests/test_architecture_guards.py` | modify | Unban `hybrid`; add Phase 4 composition guards |
| `tests/fakes.py` | modify | `StubRetriever` for failure/empty/unequal tests |
| `docs/phases/phase-4.md` | create | Phase 4 record |
| `docs/progress.md` | modify | Status + results |
| `docs/decision.md` | modify | ADR-019, ADR-017 phase table |
| `docs/architecture.md` | modify | Hybrid/RRF section |
| `experiments/README.md` | modify | Hybrid run command |

No new index, no `storage/hybrid/`, no build script, no change to dense/BM25
retrieval behavior, no reranking, no routing.

---

## K. Implementation Order

1. **Guards first** — remove `"hybrid"` from `FORBIDDEN_STRATEGY_TOKENS` in
   `tests/test_architecture_guards.py`, and add the new Phase 4 guard
   `test_hybrid_composes_existing_retrievers` (it will pass trivially until
   `hybrid.py` exists). Rationale: the current guard makes the suite
   unsatisfiable the moment any hybrid symbol lands.
2. **Schema edits** — `schemas/retrieval.py` (`"hybrid"` literal + optional
   hybrid metadata fields) and `schemas/config.py` (`rrf_k`, `candidate_k`,
   `hybrid` method, validator, `HybridRetrievalConfig`) + `schemas/__init__.py`.
   Verify with `pytest tests/test_schemas.py tests/test_bm25.py
   tests/test_determinism.py` before touching retrieval.
3. **RRF + tests** — write `fusion.py` and the §34.1–34.4 pure-function tests;
   get exact-score and dedupe/unequal-length tests green.
4. **`HybridRetriever` + tests** — `hybrid.py`, then empty/failure/top_k/
   candidate-depth/metadata/latency/protocol tests (§34.5–34.11).
5. **Config + packaging** — `retrieval/__init__.py` export; `HybridRetrievalConfig`
   loading tests.
6. **Experiment integration** — `experiments/config.py` hybrid branch;
   `scripts/run_experiment.py` `--retriever hybrid`.
7. **Integration test** — real dense (in-memory Qdrant + fake embeddings) + real
   BM25 (in-memory index) → hybrid → `ExperimentRunner` → evaluators
   (§34.12), plus the canonical-corpus regression test.
8. **Comparison tooling** — `--hybrid-run` in `compare_retrievers.py`; assert
   the dense-vs-bm25-only output is unchanged when the flag is absent.
9. **Full regression suite** — `pytest` must be green (57 existing + new).
10. **Live experiment** — `--retriever hybrid --no-judge --no-generation`, then
    `compare_retrievers.py --hybrid-run ...` against the existing
    `dense_baseline_v1` and `bm25_baseline_v1` runs. Record the observed
    outcome, including if Hybrid underperforms.
11. **Documentation** — `docs/phases/phase-4.md`, `progress.md`, `decision.md`
    (ADR-019), `architecture.md`, `experiments/README.md`.

## Open Questions

None blocking. One assumption to confirm at step 10: the hybrid run needs
`AICREDITS_API_KEY` (for dense query embeddings) plus the persisted
`storage/bm25/bm25_index.json`; the run is otherwise fully offline with
`--no-judge --no-generation`.
