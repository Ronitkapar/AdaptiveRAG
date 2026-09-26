# Evaluation Dataset — `dense_eval_v1`

20 curated question/answer pairs over the 14-paper Phase 1 corpus, stored as one
`EvaluationExample` per JSONL line.

## Schema

- `example_id` — stable id (`eval_001` … `eval_020`)
- `query` — the benchmark question
- `reference_answer` — curator-written ground truth from the paper text
- `relevant_documents` — primary relevance labels (`document_id`s from `papers.json`)
- `relevant_sections` — optional section-path prefixes for finer granularity
- `relevant_chunks` — intentionally empty in v1 (chunk ids depend on chunking config)
- `category` — one of `factual | terminology | conceptual | comparative | multi_document | fine_grained`
- `requires_multi_hop` — true when the answer synthesizes multiple papers
- `notes` — section pointer used during curation
- `dataset_version` — `dense_eval_v1`

## Coverage

- 15 single-paper questions (≥1 per corpus paper, incl. all 7 taxonomy categories)
- 5 cross-paper comparative / multi-hop questions

## Validation rules (enforced by `evaluation.dataset`)

1. `example_id`s are unique and non-empty `relevant_documents` is required.
2. Every `relevant_documents` id must exist in `data/metadata/papers.json`.
3. Chunk-level relevance is **derived at evaluation time** from document (+ optional
   section-prefix) labels with a documented rule, so labels never depend on
   whichever retriever happens to retrieve them.

To extend the benchmark, append validated lines and bump `dataset_version`.
