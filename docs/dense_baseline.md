# Dense Baseline — Frozen Configuration (`dense_baseline_v1`)

Reproducibility anchor for Phase 2. Any rerun with the same inputs must produce the same
`config_hash`, `corpus_version`, chunk ids, and metric definitions below.

- Corpus: 14 Phase-1 papers → `corpus_6c416f423920385d` (fingerprint of `papers.json` + raw PDF hashes)
- Ingestion: `ingestion_v2` (pdfplumber; heading classifier with numeric-noise / reference /
  axis-label guards; frontmatter + hierarchical sections; canonical element ids)
- Chunking: `structure_aware_v1` — target 500 / max 800 / min 150 / overlap 80 tokens,
  section-header prefixes, atomic element preservation, undersized sibling merge.
  Observed distribution: **713 chunks, mean 408 / median 447 tokens**, min 8 (heading-only
  boundaries), max 891 (intentionally unsplit table, flagged `oversized_atomic`).
- Embeddings: `text-embedding-3-large` via AICredits, L2-normalized, batch 50 with 413 auto-split
  and 429/5xx backoff; dimension probed live; SQLite cache keyed by
  `(text, model, host, dimensions, normalize, pipeline_version)`.
- Index: local embedded Qdrant, collection `adaptiverag_dense_v1`, cosine distance, dim-guarded
  ensure; `storage/qdrant/index_meta.json` records the full identity.
- Retrieval: `dense_v1`, **top_k=10** (Recall@10 measurable), score threshold unused, filters
  interface-only.
- Context: `context_v1` template, **max_chunks=5, max_context_tokens=6000**, retrieval order
  preserved, whole chunks only, dropped ids recorded.
- Generation: `groq_v1` + `generation_prompt_v1`, Groq `openai/gpt-oss-120b`, temperature 0.0,
  max 1024 completion tokens, `[Source N]` → `chunk_id` mapping (no citation verification).
- Evaluation: `eval_v1` dataset `dense_eval_v1` (20 examples); k-grid {1,3,5,10};
  judge `judge_prompt_v1` on `openai/gpt-oss-20b` (cached; `--no-judge` for lexical-only runs);
  pricing snapshot `PRICING_VERSION` in `config/pricing.py`.

## Reproduce end-to-end (needs `.env` keys)

```bash
.venv/bin/python scripts/validate_corpus.py      # Phase 1 integrity gate
.venv/bin/python scripts/ingest_corpus.py        # → data/processed/documents/
.venv/bin/python scripts/build_chunks.py         # → data/processed/chunks/
.venv/bin/python scripts/embed_chunks.py         # → data/processed/embeddings/ + cache
.venv/bin/python scripts/build_index.py          # → storage/qdrant/
.venv/bin/python scripts/run_experiment.py --name dense_baseline_v1
.venv/bin/python scripts/ask.py "Your question"  # interactive runtime demo
```
