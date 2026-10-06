# Reproduction

How to run and re-verify the project. Full report:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md); architecture:
[`../architecture/system_overview.md`](../architecture/system_overview.md).

## 0. Requirements

* Python 3.10+ (the project pins 3.12 in `.python-version`)
* Dependencies from `pyproject.toml` (`pip install -e ".[dev,plots]"`)
* **Credentials are optional for everything except** the `dense`, `hybrid`,
  `hybrid_rerank` and non-`bm25` adaptive arms, which need `AICREDITS_API_KEY`
  in `.env` (copy `.env.example`). `bm25`, BM25-only adaptive, and every phase
  9–15 offline analysis run without any network access.

## 1. Offline test suite (start here)

```bash
.venv/bin/pytest -q                      # 1056 offline tests, 2 deselected
.venv/bin/pytest -m integration          # 2 live-artifact tests (needs ONNX artifact)
```

The suite is deterministic and offline by design; the `integration` marker is
excluded by default via `addopts`.

## 2. Corpus and indexes (rebuildable from raw PDFs)

```bash
.venv/bin/python scripts/validate_corpus.py --manifest-only   # manifest only, no files
.venv/bin/python scripts/validate_corpus.py                    # + PDF/SHA-256 checks
.venv/bin/python scripts/download_corpus.py                    # fetch raw PDFs (network)
.venv/bin/python scripts/ingest_corpus.py                      # → data/processed/documents/
.venv/bin/python scripts/build_chunks.py                       # → data/processed/chunks/
.venv/bin/python scripts/report_chunk_stats.py                  # calibration report
.venv/bin/python scripts/embed_chunks.py                       # → data/processed/embeddings/
.venv/bin/python scripts/build_index.py                        # → storage/qdrant/
.venv/bin/python scripts/build_bm25_index.py                   # → storage/bm25/
```

Both Phase 8 corpus arms are built by pointing `--docs-dir` / `--out-dir` at the
`data/processed_phase8_{before,after}/` namespaces; each keeps its own BM25
index file and Qdrant collection so the arms are distinguishable (see Caveats).

## 3. Baseline and adaptive runs

```bash
# BM25 baseline — fully offline, no credentials
.venv/bin/python scripts/build_bm25_index.py
.venv/bin/python scripts/run_experiment.py --retriever bm25 --no-judge --no-generation

# Dense / hybrid — need AICREDITS_API_KEY for query embeddings
.venv/bin/python scripts/run_experiment.py --retriever dense  --no-judge --no-generation
.venv/bin/python scripts/run_experiment.py --retriever hybrid --no-judge --no-generation

# Adaptive, BM25-only arm — fully offline
.venv/bin/python scripts/run_experiment.py --retriever adaptive \
    --routing-strategies bm25 --no-judge --no-generation --name adaptive_bm25_v1

# Compare arms
.venv/bin/python scripts/compare_retrievers.py [--hybrid-run <run-id>]
```

`--retriever adaptive --rerank` is rejected by design: reranking is a routing
decision, not a flag.

## 4. Phase 7 suite and environment gates

```bash
.venv/bin/python scripts/validate_environment.py            # 5-arm gate, must pass 5/5
.venv/bin/python scripts/run_phase7_suite.py                 # E-series driver
.venv/bin/python scripts/run_phase7_analysis.py              # tables
.venv/bin/python scripts/make_phase7_plots.py                # figures
```

The cost sweep (`scripts/measure_strategy_cost.py`) re-measures per-strategy
latency; it needs `AICREDITS_API_KEY` because three of four costed strategies
embed every query remotely, and it paces provider arms by default (`--pace`).

## 5. Phase 7/8 closure gates (offline, over frozen rows)

```bash
.venv/bin/python scripts/oracle_routing_ceiling.py           # Gate 2: oracle ceiling
.venv/bin/python scripts/gate3_signal_exhaustion.py          # Gate 3: signal search
.venv/bin/python scripts/compare_phase8_arms.py              # before/after arm comparison
.venv/bin/python scripts/audit_phase8_gold_labels.py        # label integrity audit
```

Both gate scripts reuse the Phase 8 arm loader, so the contamination guard and
the query-set equality assertion apply automatically: a short, contaminated or
mismatched arm **aborts** the analysis rather than producing a number. Both are
deterministic and byte-identical across runs apart from `generated_at`.

## 6. Phase 9–15 analyses (offline over frozen rows)

```bash
# Phase 9
.venv/bin/python scripts/gate9_mechanism_screening.py
.venv/bin/python scripts/gate9_deployability_proxy.py

# Phase 10 (post-dense escalation) — inputs default to experiments/phase8/combined
.venv/bin/python scripts/phase10_build_oracle.py
.venv/bin/python scripts/phase10_signal_analysis.py
.venv/bin/python scripts/phase10_evaluate_policy.py
.venv/bin/python scripts/phase10_figures.py

# Phase 11 (mechanism)
.venv/bin/python scripts/phase11_mechanism.py
.venv/bin/python scripts/phase11_figures.py

# Phase 14 (cheap-first)
.venv/bin/python scripts/phase14_build_oracle.py
.venv/bin/python scripts/phase14_signal_analysis.py
.venv/bin/python scripts/phase14_evaluate_policy.py
.venv/bin/python scripts/phase14_figures.py

# Phase 15 (powered confirmation) — pass the two suite roots and the dataset
.venv/bin/python scripts/phase15_build_oracle.py \
    --suite-after experiments/phase15/p15_after \
    --suite-before experiments/phase15/p15_before \
    --dataset data/evaluation/phase15_eval_v1.jsonl
.venv/bin/python scripts/phase15_evaluate.py
.venv/bin/python scripts/phase15_figures.py
```

All Phase 9–15 scripts read frozen rows/traces and write to
`experiments/phase*/`; none performs new retrieval. Only the Phase 15 suite
builds required fresh retrieval runs, which is why its oracle builder takes
`--suite-after` / `--suite-before` explicitly.

## 7. Artifacts and provenance

* Every run directory carries `config.json` (with `config_hash`), `manifest.json`
  (commit, corpus/index ids, counts, errors), `traces.jsonl` (the source of
  truth), and per-evaluator metric files.
* `experiments/*/` and `storage/` are **gitignored**; the hashes that make a
  Phase 7/8 run auditable are recorded in
  [`../phase7-closure-evidence.md`](../phase7-closure-evidence.md).
* Superseded run directories are kept under `experiments/archive/` (Phase 7
  intermediate reruns; Phase 8 smoke runs) rather than deleted.

## Caveats when re-running

* **Arm identity.** `ensure_collection` reuses an existing Qdrant collection by
  name, so the two corpus arms must be told apart by namespace directory, not by
  `corpus_version` (both Phase 8 arms report the same string).
* **The working-tree `data/processed/` corpus is a later rebuild.** It does not
  text-match the BM25 index that Phases 2–7 evaluated (713 chunks, manifest
  `max_tokens: 600`, vs the indexed mean 408 / median 447 / max 891 tokens).
  The historical results are authoritative in the storage indexes plus the
  documented figures; rebuilding the corpus changes `corpus_version` and would
  invalidate the frozen results, so this is documented rather than "fixed".
* **Latency needs a control arm.** The `bm25` arm makes no API call and is the
  control for provider-latency comparisons.
* **Provider numbers are hardware-specific**; the ONNX reranker ran on CPU.

---

*Commands and reproducibility notes. Report:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md).*