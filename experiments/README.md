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

---

## Running the reranked baselines (Phase 5)

`--rerank` wraps the chosen first-stage retriever in a second-stage
cross-encoder. The ONNX artifact is downloaded to `storage/reranker/` on first
use and reused afterwards, so only the first reranked run needs network access.

```bash
# Confirm the artifact resolves before spending a benchmark run.
.venv/bin/pytest -m integration tests/test_reranking.py

# Second-stage runs (retrieval-only). dense and hybrid need AICREDITS_API_KEY
# for query embeddings, as in Phase 4; bm25_rerank needs no credentials.
.venv/bin/python scripts/run_experiment.py --retriever dense  --rerank --name dense_rerank_v1  --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever bm25   --rerank --name bm25_rerank_v1   --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever hybrid --rerank --name hybrid_rerank_v1 --no-judge --no-generation

# Candidate-depth ablation, final top_k held at 10
for k in 10 20 40; do
  .venv/bin/python scripts/run_experiment.py --retriever hybrid --rerank \
      --rerank-candidate-k $k --name hybrid_rerank_k${k}_v1 --no-judge --no-generation
done

# Compare each baseline against its reranked variant, plus the depth table.
# Every run directory is explicit — compare_reranking.py never auto-discovers.
.venv/bin/python scripts/compare_reranking.py \
    --dense-run experiments/<dense_baseline_v1> \
    --dense-rerank-run experiments/<id-dense_rerank_v1> \
    --hybrid-run experiments/<hybrid_baseline_v1> \
    --hybrid-rerank-run experiments/<id-hybrid_rerank_v1> \
    --ablation-run experiments/<id-hybrid_rerank_k10_v1> \
    --ablation-run experiments/<id-hybrid_rerank_k20_v1> \
    --ablation-run experiments/<id-hybrid_rerank_k40_v1>
```

`--name` defaults to `<strategy>_rerank_v1` when `--rerank` is set, so rerank
runs never land in a baseline directory.

`--rerank-candidate-k` also raises hybrid's fusion `candidate_k` to match, so
one flag controls depth end-to-end.

Other overrides: `--reranker-model`, `--reranker-revision` (pin it —
`config.json` records it), `--rerank-device {auto,cpu,cuda}`,
`--rerank-batch-size`, `--rerank-max-length`, and `--rerank-fallback`.

## What a reranked run adds to its artifacts

- `retrieval_method` is `dense_rerank` / `bm25_rerank` / `hybrid_rerank`, and
  `component_versions["reranker"]` appears in the manifest
- each result carries `retrieval_score` and `retrieval_rank` alongside the
  rerank `score`
- `traces.jsonl` gains `candidate_generation_latency_ms`, `rerank_latency_ms`,
  `rerank_candidate_count`, `rerank_result_count`, and `rerank_fallback`
- `metrics_efficiency.json` gains the split-latency and count metrics — but only
  for reranked runs, so baseline artifacts are unchanged

Reranking **fails closed by default**: a reranker error is recorded as
`retrieval_failed` with the original error type. `--rerank-fallback` opts into
returning un-reranked candidates instead, and sets `rerank_fallback=True` in
every trace so the degradation is visible rather than silent.

`compare_retrievers.py` is unchanged by Phase 5 and remains the dense-vs-BM25
(plus optional hybrid) table. Use `compare_reranking.py` for reranked runs.

---

## Running adaptive retrieval (Phase 6)

`--retriever adaptive` routes each query to BM25, dense, hybrid, or
hybrid+reranker, then escalates **once** if the retrieved evidence looks
insufficient. It uses the same config, runner, and evaluators as the fixed
strategies, and adds a `metrics_routing.json` report.

```bash
# BM25-only adaptive run — fully offline, no credentials needed
# (build the index first; this is the one arm that needs nothing else)
.venv/bin/python scripts/build_bm25_index.py
.venv/bin/python scripts/run_experiment.py --retriever adaptive \
    --routing-strategies bm25 --no-judge --no-generation --name adaptive_bm25_v1

# Full four-strategy run — needs AICREDITS_API_KEY for query embeddings,
# exactly as Phase 4/5's dense and hybrid runs do
.venv/bin/python scripts/run_experiment.py --retriever adaptive \
    --no-judge --no-generation --name adaptive_v1

# Ablation switches (Phase 6 exposes the levers; Phase 7 runs the study)
.venv/bin/python scripts/run_experiment.py --retriever adaptive --no-escalation \
    --name adaptive_no_escalation_v1 --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever adaptive --no-sufficiency \
    --name adaptive_no_sufficiency_v1 --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever adaptive \
    --disable-feature-group semantic --disable-feature-group entity \
    --name adaptive_no_signals_v1 --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever adaptive \
    --routing-cost-weight 0.0 --name adaptive_pure_evidence_v1 --no-judge --no-generation
```

`--routing-strategies` restricts both what the router may select *and* which
indexes are built, so narrowing it to `bm25` keeps a run fully offline. The
escalation ladder narrows to match automatically unless `--routing-ladder` is
given.

`--retriever adaptive --rerank` is rejected on purpose: second-stage scoring is
one of the strategies the router selects, so the flag would silently override the
routing decision.

Routing metrics appear only in adaptive runs:

```text
escalation_rate             fraction of queries that escalated
rerank_rate                 fraction whose final stage included reranking
insufficient_initial_rate   fraction whose first stage looked insufficient
routing_confidence_mean     normalized score margin (not a probability)
retrieval_stages_mean       mean number of stages executed
routing_latency_ms_mean     analysis + routing overhead
escalated_retrieval_latency_ms_mean / non_escalated_retrieval_latency_ms_mean
initial_strategy_distribution / final_strategy_distribution
```
