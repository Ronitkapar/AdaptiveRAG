# Evaluation Design (`eval_v1`)

Evaluation is an **independent observer**: evaluators only ever read persisted
`ExperimentTrace`s. Aggregate metrics are recomputable from `traces.jsonl` at any time.

## Retrieval (`retrieval_eval_v1`)

- **Recall@k** — document-level: fraction of `relevant_documents` present in top-k chunks.
- **Precision@k** — chunk-level: fraction of top-k chunks whose document is relevant
  (and whose section matches a `relevant_sections` prefix when section labels exist).
- **Hit@k** — 1 when any relevant document appears in top-k, else 0.
- **MRR** — reciprocal rank of the first relevant document.
- **nDCG@k** — binary chunk relevance, standard log2 discount.
- k-grid: {1, 3, 5, 10}. `n` counts paired traces; unpaired/failed traces appear in the
  `errors` aggregate, never silently dropped.

## Generation (`generation_eval_v1`)

- `answer_count`, `empty_answer_rate`, `generation_failure_rate`, `citation_coverage`
  (answers with ≥1 parsed `[Source N]` → `chunk_id` mapping).
- **token-F1 / ROUGE-L** — deterministic lexical overlap against `reference_answer`
  (stdlib implementation, no extra dependencies).
- **LLM judge** (1–5: correctness, answer relevance, faithfulness) via Groq
  `openai/gpt-oss-20b` + `judge_prompt_v1`, JSON mode, temperature 0.0, cached in
  `storage/eval_cache.sqlite3` and cost-tracked separately. `--no-judge` disables it
  (fully deterministic, offline).

## Efficiency (`efficiency_eval_v1`)

- Latency: retrieval / generation / total mean, p50, p95 (client-measured ms).
- Tokens: input / output / total sums from provider-reported `usage`.
- Cost: estimated USD from `config/pricing.py` (`PRICING_VERSION`), reported total,
  per-query, and per-1000-queries. Marked **estimated**, never billed.

Relevance labels live in `data/evaluation/dense_eval_v1.jsonl` (see its README for
schema, coverage, and the retriever-independent chunk-relevance rule).
