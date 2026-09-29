# AdaptiveRAG: Baseline Experiments

Experiment runs produce one directory per run: `experiments/<experiment_id>/`.

Each run directory contains:

- `config.json` — frozen experiment configuration (with `config_hash`)
- `manifest.json` — reproducibility manifest (git commit, corpus/index ids, counts, errors)
- `traces.jsonl` — one raw `ExperimentTrace` per evaluation example (the source of truth)
- `metrics.json` — aggregate metrics derived from traces (retrieval, generation, efficiency)
- `metrics_<evaluator>.json` — per-evaluator reports
- `report.md` — human-readable summary

Run directories are **gitignored**; only this README is tracked. To publish a run,
force-add its `config.json`, `manifest.json`, and `metrics.json` explicitly.

## Running the baselines

```bash
# Dense baseline — full run with LLM judge (needs GROQ_API_KEY)
.venv/bin/python scripts/run_experiment.py --retriever dense --name dense_baseline_v1

# Dense baseline — deterministic lexical-only run (no judge calls)
.venv/bin/python scripts/run_experiment.py --retriever dense --name dense_baseline_v1 --no-judge

# BM25 baseline — fully offline (build the index first; no credentials needed)
.venv/bin/python scripts/build_bm25_index.py
.venv/bin/python scripts/run_experiment.py --retriever bm25 --no-judge --no-generation

# Hybrid baseline — RRF over dense + BM25 (needs AICREDITS_API_KEY for query
# embeddings, plus the persisted BM25 index; not offline-capable)
.venv/bin/python scripts/run_experiment.py --retriever hybrid --no-judge --no-generation

# Resume an interrupted run
.venv/bin/python scripts/run_experiment.py --retriever bm25 --run-id <id> --resume

# Compare dense vs BM25 metrics side-by-side (once both runs exist)
.venv/bin/python scripts/compare_retrievers.py

# Add a third column for a hybrid run (corpus version, trace count, and
# retrieval method are checked across all three)
.venv/bin/python scripts/compare_retrievers.py --hybrid-run experiments/<run-id>
```

`--name` defaults to `<strategy>_baseline_v1`, so omitting it keeps each
strategy in its own run directory. Pass `--name` to override.

Hybrid requires **both** the Qdrant collection (`scripts/build_index.py`) and
the BM25 index (`scripts/build_bm25_index.py`); the CLI checks both and names
the missing one. A failing branch is recorded as `retrieval_failed` with the
original error type — hybrid never silently degrades to a single strategy.
