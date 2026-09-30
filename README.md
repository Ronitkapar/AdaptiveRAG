# AdaptiveRAG: Query-Aware Retrieval Optimization for Efficient RAG

AdaptiveRAG investigates query-aware retrieval optimization: dynamically evaluating whether an incoming query warrants expensive dense retrieval, multi-hop search, and cross-encoder reranking, or whether simpler lexical retrieval (BM25) or single-stage retrieval suffices.

---

## Phase 1: Research Paper Corpus

Phase 1 establishes the canonical research-paper corpus serving as the dataset for all downstream ingestion, indexing, and retrieval evaluations.

### Corpus Summary
* **Total Documents**: 14 peer-reviewed / preprint foundational papers in IR and NLP.
* **Format**: Original authoritative open-access PDFs (arXiv, university repositories).
* **Metadata Manifest**: `data/metadata/papers.json` with document IDs, full bibliographic metadata, key concept taxonomies, verified SHA-256 digests, and file sizes.

### Paper Categories Covered
1. **Foundational RAG**: RAG (Lewis et al.), REALM (Guu et al.), RETRO (Borgeaud et al.)
2. **Dense Retrieval**: DPR (Karpukhin et al.), Contriever (Izacard et al.)
3. **Sparse / Lexical Retrieval**: BM25 (Robertson & Zaragoza), SPLADE v2 (Formal et al.)
4. **Hybrid Retrieval**: Reciprocal Rank Fusion (Cormack et al.), Pyserini (Lin et al.)
5. **Reranking**: monoBERT (Nogueira & Cho), ColBERT (Khattab & Zaharia)
6. **Advanced RAG**: Self-RAG (Asai et al.), Corrective RAG (CRAG) (Yan et al.)
7. **Adaptive Retrieval**: Adaptive-RAG (Jeong et al.)

Detailed descriptions, rationales, and concept mappings are documented in [`docs/corpus.md`](docs/corpus.md).

---

## Repository Structure

```
AdaptiveRAG/
├── data/
│   ├── raw/                  # Downloaded raw PDF papers (gitignored)
│   ├── processed/            # Placeholder for extracted/processed text in Phase 2
│   └── metadata/
│       └── papers.json       # Canonical corpus metadata manifest
├── docs/
│   └── corpus.md             # Corpus documentation and taxonomy rationale
├── scripts/
│   ├── download_corpus.py    # Deterministic downloader with retries and hash validation
│   └── validate_corpus.py    # Manifest and file integrity validator
├── tests/
│   └── test_corpus.py        # Automated test suite
├── pyproject.toml            # Project configuration
└── README.md
```

---

## Reproducing the Corpus

### 1. Download Corpus Documents
The corpus can be fully reproduced and validated using pure Python standard library:

```bash
python3 scripts/download_corpus.py
```

This script:
* Reads `data/metadata/papers.json`
* Downloads missing documents from authoritative sources (e.g. arXiv)
* Validates PDF magic bytes (`%PDF-`) and SHA-256 integrity
* Skips existing valid files (use `--force` to force re-download)

### 2. Validate the Manifest & Files
Verify the metadata schema, URLs, unique IDs, category completeness, and PDF file status:

```bash
python3 scripts/validate_corpus.py
```

To validate only the JSON schema without checking local PDFs:
```bash
python3 scripts/validate_corpus.py --manifest-only
```

### 3. Run Automated Tests
```bash
.venv/bin/pytest -q
```

57 offline tests cover schemas, config, ingestion, chunking, embeddings cache,
indexing, dense and BM25 retrieval, context, generation, evaluation, the
experiment runner, determinism, and architecture guards. Live-provider checks
live outside pytest: `.venv/bin/python scripts/check_providers.py` (needs
`.env` keys).

---

## Phase 2: Fixed Dense-RAG Baseline

Phase 2 established the controlled comparison point for all future retrieval
strategies: at Phase 2 **only** dense retrieval was implemented — no hybrid,
reranking, adaptive routing, query classification, or frontend. (BM25 was
added later as Phase 3; see below.)

### Pipeline

```text
14 PDFs → Ingestion → Structure-Aware Chunking → text-embedding-3-large (AICredits)
→ local Qdrant → DenseRetriever (top_k=10) → ContextBuilder (5 chunks)
→ Groq openai/gpt-oss-120b → Answer + Sources → ExperimentRunner → Evaluators
```

### Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
cp .env.example .env   # then fill in AICREDITS_API_KEY and GROQ_API_KEY
```

### Reproduce the baseline

```bash
.venv/bin/python scripts/validate_corpus.py      # Phase 1 integrity gate
.venv/bin/python scripts/ingest_corpus.py        # → data/processed/documents/
.venv/bin/python scripts/build_chunks.py         # → data/processed/chunks/ (713 chunks)
.venv/bin/python scripts/report_chunk_stats.py   # calibration report
.venv/bin/python scripts/embed_chunks.py         # → data/processed/embeddings/ (+ SQLite cache)
.venv/bin/python scripts/build_index.py          # → storage/qdrant/
.venv/bin/python scripts/run_experiment.py --name dense_baseline_v1
.venv/bin/python scripts/ask.py "What is RAG-Sequence?"
```

### Repository Structure (Phase 2 additions)

```text
AdaptiveRAG/
├── src/adaptive_rag/       # schemas, config, ingestion, chunking, embeddings,
│                           # indexing, retrieval, generation, evaluation, experiments
├── data/evaluation/        # dense_eval_v1.jsonl (20 curated Q/A pairs, committed)
├── storage/                # local-only: embedding cache, embedded Qdrant (gitignored)
├── experiments/            # run dirs: config/manifest/traces/metrics/report (gitignored)
├── docs/
│   ├── architecture.md     # locked decisions + component map
│   ├── dense_baseline.md   # frozen configuration + reproduction
│   └── evaluation.md       # metric definitions
├── scripts/                # one CLI per pipeline stage (+ ask, run_experiment, check_providers)
└── tests/                  # offline suite + fakes + synthetic-PDF fixtures
```

### Phase 2 Definition of Done

- [x] all 14 Phase 1 papers ingested with hierarchy + provenance (`ingestion_v2`)
- [x] deterministic structure-aware chunking, 713 chunks, atomics preserved
- [x] `text-embedding-3-large` via AICredits with SQLite embedding cache
- [x] vectors indexed in local Qdrant with dim-guarded collection
- [x] `DenseRetriever` with fixed top-k; scores + provenance in every result
- [x] `ContextBuilder` → structured `GenerationRequest`; Groq generation with `[Source N]` mapping
- [x] typed failures (`retrieval_failed`/`generation_failed`) distinct from `no_results`/`empty`
- [x] `dense_eval_v1` dataset (20 examples); raw `traces.jsonl` persisted per run
- [x] retrieval (Recall/Precision/Hit/MRR/nDCG), generation (lexical F1/ROUGE-L + cached LLM judge),
      and efficiency (latency/tokens/estimated cost) metrics from traces only
- [x] reproducible config (`config_hash` + `corpus_version`) + 43 offline tests + guards
- [x] at Phase 2 completion, no BM25 / hybrid / rerank / adaptive code in `src/`
      (machine-checked; BM25 was added later in Phase 3, hybrid/rerank/adaptive
      remain guard-banned)
- [x] live baseline run (`embed → index → run_experiment` executed with `.env` keys)

---

## Phase 3: BM25 Lexical Baseline

Phase 3 adds an independent lexical retrieval strategy over the **same**
canonical 713-chunk corpus, sharing the retrieval protocol and the
strategy-agnostic evaluation pipeline.

### Pipeline

```text
713 chunks → BM25Index (inverted index, deterministic tokenizer)
→ storage/bm25/bm25_index.json → BM25Retriever (k1=1.2, b=0.75)
→ ContextBuilder → ExperimentRunner → Evaluators
```

### Reproduce the BM25 baseline (offline, no embedding credentials)

```bash
.venv/bin/python scripts/build_bm25_index.py          # → storage/bm25/bm25_index.json
.venv/bin/python scripts/run_experiment.py --retriever bm25 --no-judge --no-generation
.venv/bin/python scripts/compare_retrievers.py        # dense vs BM25 table (once both runs exist)
```

### Definition of Done

- [x] indexes the canonical corpus without modification or hard-coded counts
- [x] `BM25Retriever` conforms to the `Retriever` protocol; native BM25 scores,
      1-indexed ranks, full metadata/provenance
- [x] zero coupling with dense retrieval (guard-enforced); no fallback logic
- [x] `retrieval_method` ⇔ `retriever_version` consistency without drift
- [x] missing index fails fast; empty query → `InvalidQueryError`;
      zero lexical match → `status="no_results"`
- [x] evaluated on `dense_eval_v1` with Recall/Precision/Hit/MRR/nDCG@k +
      latency (Recall@5 = 0.7167, MRR = 0.6771); run artifacts in
      `experiments/bm25_baseline_v1/`
- [x] live dense-vs-BM25 comparison over the same corpus/benchmark
      (`scripts/compare_retrievers.py`: dense MRR 0.95 vs BM25 0.6771,
      dense Recall@5 0.8417 vs BM25 0.7167, dense 751.51 ms vs BM25 1.48 ms)
- [x] 57 offline tests pass; Phase 4+ still guard-banned
- [x] docs updated (`docs/phases/phase-3.md`, `progress.md`, `decision.md`,
      `architecture.md`)

---

## Phase 5: Reranking (Second-Stage Cross-Encoder)

Phase 5 adds a second stage that re-scores the candidate list produced by any
first-stage retriever. It reorders what retrieval already found — it never
reimplements candidate generation, and it never decides *whether* to run (that
is Phase 6's decision).

### Pipeline

```text
DenseRetriever | BM25Retriever | HybridRetriever      (all unchanged)
        ↓  ≤ N candidates (rerank_candidate_k, default 20)
RerankedRetriever
  └─ OnnxCrossEncoderReranker (ONNX Runtime, Xenova/ms-marco-MiniLM-L-6-v2)
        ↓  sort (-rerank_score, retrieval_rank, chunk_id) → truncate → rank = 1..n
RetrievalResponse(retrieval_method="<base>_rerank")
        ↓
ExperimentRunner + evaluators (unchanged; efficiency gains a conditional
candidate-generation / rerank latency split)
```

### Reproduce the reranked baselines

```bash
# Confirm the ONNX artifact resolves before spending a benchmark run
.venv/bin/pytest -m integration tests/test_reranking.py

# Reranked runs (retrieval-only). dense/hybrid need AICREDITS_API_KEY for query
# embeddings; bm25_rerank needs no credentials. All reranked runs need network
# on first use to download the artifact.
.venv/bin/python scripts/run_experiment.py --retriever dense  --rerank --name dense_rerank_v1  --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever bm25   --rerank --name bm25_rerank_v1   --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever hybrid --rerank --name hybrid_rerank_v1 --no-judge --no-generation

# Candidate-depth ablation, top_k held at 10
for k in 10 20 40; do
  .venv/bin/python scripts/run_experiment.py --retriever hybrid --rerank \
      --rerank-candidate-k $k --name hybrid_rerank_k${k}_v1 --no-judge --no-generation
done

# Compare each baseline against its reranked variant + the depth table
.venv/bin/python scripts/compare_reranking.py \
    --dense-run experiments/<dense_baseline_v1> \
    --dense-rerank-run experiments/<id-dense_rerank_v1>
```

Overrides: `--reranker-model`, `--reranker-revision` (pin it — recorded in
`config.json`), `--rerank-candidate-k`, `--rerank-device {auto,cpu,cuda}`,
`--rerank-batch-size`, `--rerank-max-length`, `--rerank-fallback`.

### Definition of Done

- [x] `RerankedRetriever` composes any base `Retriever`;
      `dense.py` / `bm25.py` / `hybrid.py` / `fusion.py` unmodified and
      guard-enforced
- [x] real cross-encoder on ONNX Runtime — no torch, no transformers;
      `sentence-transformers` still banned
- [x] `score` = rerank score; `retrieval_score` / `retrieval_rank` preserve the
      first-stage signal. Scores are never mixed or normalized
- [x] deterministic ordering `(-rerank_score, retrieval_rank, chunk_id)` and
      batch-size invariance
- [x] fail-closed by default; the single `except` is gated on
      `rerank_fallback` and AST-guarded, and when taken it is visible in every
      trace and metric
- [x] empty candidate pool → `no_results` with no model invocation
- [x] `retrieval_method` ⇔ `retriever_version` for all three `_rerank` variants;
      every `rerank_*` field inside the hashed config
- [x] retrieval metrics consumed unchanged; efficiency split additive and
      conditional, so baseline artifacts are byte-identical
- [x] 136 offline tests pass (2 `integration` deselected), including four new
      architecture guards verified to fail against broken variants
- [x] docs updated (`docs/phases/phase-5.md`, `progress.md`, `decision.md`
      ADR-020, `architecture.md`, `experiments/README.md`)
- [ ] **live benchmark** — six primary configurations plus the 10/20/40 depth
      ablation, run on a networked machine and recorded **as observed**. Blocked
      here: the implementation environment has no outbound network and no GPU,
      so the ONNX artifact cannot be downloaded and `onnxruntime` / `tokenizers`
      / `huggingface-hub` could not be installed.

Full record: [`docs/phases/phase-5.md`](docs/phases/phase-5.md).
