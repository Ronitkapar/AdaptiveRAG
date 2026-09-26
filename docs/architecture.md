# AdaptiveRAG Architecture — Phase 2 Dense-RAG Baseline

Locked conceptual pipeline (see plan: every stage is a deliberately separated component):

```text
14 PDFs → Ingestion → Hierarchical Documents → Structure-Aware Chunking → Structured Chunks
→ text-embedding-3-large (AICredits) → Embedding Cache → Qdrant (local) → DenseRetriever
→ ContextBuilder → Groq Generator → Answer + Sources → ExperimentRunner → Evaluators
```

## Component map (`src/adaptive_rag/`)

| Module | Responsibility | Provider isolation |
|---|---|---|
| `schemas/` | Pydantic data contracts only (no business logic) | — |
| `config/` | env/secrets (pydantic-settings), paths, hashing, pricing | secrets never persisted |
| `ingestion/` | pdfplumber extraction → normalizer → `Document` + report | extractor protocol for future backends |
| `chunking/` | `StructureAwareChunker` (only chunker in Phase 2) | `Chunker` protocol for future strategies |
| `embeddings/` | `AICreditsEmbeddingModel` + SQLite cache + pipeline | `EmbeddingModel` protocol |
| `indexing/` | `QdrantVectorStore` (embedded path default) | `VectorStore` protocol |
| `retrieval/` | `DenseRetriever` (fixed top-k, scores + provenance) | `Retriever` protocol for BM25/hybrid/adaptive later |
| `generation/` | `ContextBuilder`, `GroqGenerator`, versioned prompts | generation ↔ embeddings fully decoupled |
| `evaluation/` | retrieval / generation (lexical + GroqLLMJudge) / efficiency | independent observer over traces only |
| `experiments/` | config assembly, corpus fingerprint, `ExperimentRunner` | strategy-agnostic |
| `rag.py` | runtime `Query → Retrieve → Context → Generate` composition | no evaluation inside |

## Locked decisions (do not reopen without a design note)

- **Extraction:** pdfplumber (MIT). PyMuPDF deliberately avoided (AGPL). Figures/tables/equations
  are preserved as representations + metadata — Phase 2 performs no semantic/visual understanding.
- **Chunking:** structure-aware over section hierarchy. Naive fixed-size chunking was evaluated and
  rejected (median 158 tokens, 40% sub-100-token fragments); `structure_aware_v1`
  (target 500 / max 800 / min 150 / overlap 80 + undersized sibling merge) yields median 447 tokens
  over 713 chunks. Atomic elements (table/figure/equation) are never split and never dropped.
- **Embeddings:** `text-embedding-3-large` via AICredits OpenAI-compatible gateway, L2-normalized,
  `Distance.COSINE` in Qdrant. Dimension is probed from the provider, never hardcoded.
- **Vector store:** qdrant-client embedded path mode (`storage/qdrant/`). Single-process lock is a
  known constraint: scripts and `ask.py` must not run concurrently.
- **Generation:** Groq `openai/gpt-oss-120b`, temperature 0.0, max 1024 tokens, citations as
  `[Source N]` mapped to canonical `chunk_id`s. Judge (offline path in metrics): `openai/gpt-oss-20b`.
- **Retriever contract:** retrieval top_k=10 (so Recall@10 is measurable) while the context builder
  passes max_chunks=5 to the generator — both frozen. Filters exist at the interface level only.
- **Failure semantics:** infrastructure failures raise typed exceptions and are recorded as
  `trace.error` (`retrieval_failed`/`generation_failed`); they are never converted to "no results"
  or empty answers (`no_results` / `empty` / `empty_answer` statuses exist separately).
- **Determinism:** canonical artifacts contain no timestamps; ingest → chunk → embed → index is
  byte-reproducible for fixed inputs (guarded by `tests/test_determinism.py`).
