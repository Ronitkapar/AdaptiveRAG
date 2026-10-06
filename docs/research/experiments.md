# Experiments

A catalog of every experiment run, with its source of record. Narrative:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md) §5; chronology:
[`../history/experiment_timeline.md`](../history/experiment_timeline.md).

Experiments are grouped by the question they answer. Each row states the
hypothesis, the test, the result, and where the full record lives.

## Foundations (Phases 1–6)

| # | Phase | Experiment | Result | Source |
| --- | --- | --- | --- | --- |
| 1 | Corpus | Collect 14 papers; deterministic download; SHA-256 + PDF validation | 14-doc canonical corpus, reproducible | `docs/corpus.md` |
| 2 | Dense | Fixed bi-encoder baseline + ingestion/chunking/embedding/indexing | Dense baseline (R@5 0.8417, MRR 0.95 on the 20-query set) | `docs/dense_baseline.md` |
| 3 | BM25 | Independent lexical strategy over the same corpus | BM25 baseline (R@5 0.7167); ~500× cheaper than dense | `docs/phases/phase-3.md` |
| 4 | Hybrid | RRF rank-fusion of dense + BM25 | Hybrid R@5 0.8583; fusion flattens (mixed from the start) | `docs/phases/phase-4.md` |
| 5 | Rerank | Cross-encoder second stage on all three first-stages | **Reranking did not pay**: every head metric fell; ~155 ms/candidate-pair | `docs/phases/phase-5.md` |
| 6 | Adaptive | Rule-based router + sufficiency + bounded escalation | Implemented, offline-validated; **gate never fires** at shipped defaults | `docs/phases/phase-6.md` |

## Phase 7 — evaluation, ablation, calibration (E-series)

| # | Experiment | Design | Result |
| --- | --- | --- | --- |
| E1 | Five-arm comparison | 107 paired queries | **adaptive ≡ hybrid** on quality; ~59 ms overhead; dense leads |
| E2 | Escalation ablation A/B/C | 47 calibration | Quality tied; escalation costs +58.91 ms (p = 0.003) |
| E3 | Feature-group ablation | leave-one-out, 6 groups | **0 / 24** significant |
| E4 | Sufficiency-threshold sweep | 0.3 … 0.7 | 0.3–0.6 identical, 0% escalation; 0.7 escalates 6.4%, loses MRR |
| E5 | Cost-weight sweep | 0.0 … 1.0 | identical recall (gate closed ⇒ cost term inert) |
| Gate 2 | Oracle ceiling (selectable strategies) | offline join | +0.0218 / +0.0343 over best fixed, on 4–6 / 107 queries |
| Gate 3 | Observable-signal search | ridge as measuring instrument | AUC 1.000 fit → **0.534 OOS; 0/24 Holm** |
| Closure | | | headroom exists; not detectable; router not established |

Source: `docs/phase_7_results.md`, `docs/phase_7_closure_results.md`.

## Phase 8 — corpus fix study

| Experiment | Design | Result |
| --- | --- | --- |
| Column-extraction fix + re-baseline | two arms (before/after), all five strategies | quality unchanged on every arm; reranker −0.0016 (99/107 ties) |
| Latency before/after | paired, Holm | **withdrawn** — provider variance, no-API control moved 1.83→1.85 ms |
| Gold-label audit / repair | 140 section labels | 47 corrupt, only 22 repairable; nDCG blocked |
| Run-integrity re-sweep | 29 arms | 13 silently invalid → re-run; trace-status guard added |

Source: `docs/phase-8-results.md`; cost decision in
`docs/strategy-cost-freeze-decision.md`.

## Phase 9 — signal discovery (mechanism, not correlation)

Five gates (9.1–9.2 audit, 9.3 deployability, 9.4 mechanism, 9.5 proxy).
Hypothesis family of 12 (after H5 withdrawal), Holm denominator 12. **Verdict:
`PROMISING_SIGNAL`** — DDR (Dense distinct-document ratio) tracks dispersion
(ρ +0.343/+0.406; OOS +0.410); zero query-only pre-routing signals exist
(best abs(ρ) = 0.194). Source: `docs/phase9_results.md`.

## Phase 10 — post-dense escalation (the decisive `FAILURE`)

| Element | Value |
| --- | --- |
| Hypothesis | `DDR@dense` decides whether Dense→Hybrid escalation is worth it |
| Oracle | escalate iff Δrecall ≥ 0.01 |
| Calibration AUC | **0.550** (chance); Spearman(DDR, Δ) = −0.129 |
| Test Δ vs dense-only | **−0.050 / −0.033** |
| True positives | **0 of 2 per arm** |
| Random-at-rate ablation | **P(random ≥ adaptive) = 1.000 / 0.954** |
| Verdict | **`FAILURE`** — dominated by dense-only *and* by random |

Source: `docs/phase10_results.md`.

## Phase 11 — mechanism analysis (outcome B)

Per-query decomposition of the 4 helps and 9–12 harms per arm. Helps =
**BM25-side rescue** of Dense-missed evidence (State 2); harms = **fusion
corrupting a correct Dense result** (State 3), via intruder promotion or
re-ranking among shared docs. Separation check over pre-declared features:
**1/6 and 0/6** — the two cases cannot be told apart before paying. Source:
`docs/phase11_results.md`.

## Phase 12 — research boundary

Adopted Position **C — STOP CURRENT ADAPTIVE-ROUTING TRACK**, with re-entry
conditions (§9 of the results doc). Source: `docs/phase12_results.md`.

## Phases 13–15 — cheap-first track (BM25 → Dense)

| # | Experiment | Result |
| --- | --- | --- |
| 13 | Feasibility counts over frozen rows | cheap-first positives **16 / 21**; 91/86 need no Dense call; rung is EV-positive |
| 14 | Frozen rule from 6 pre-Dense signals, one-pass test | rule `bm25_slope ≥ −0.915`; +0.078/+0.050 over BM25, −0.081/−0.086 vs dense; 5/11 caught; random-ablation mixed → **`INSUFFICIENT EVIDENCE`** |
| 15 | Powered confirmation, 110 fresh queries/arm, rule frozen | **18/25 caught at ~30% spend; beats random (P = 0.008/0.019, combined 0.001);** but −0.018/−0.030 vs dense, missed the 0.02 bar by 0.004 → **`INSUFFICIENT EVIDENCE` (4/5 bars)** |

Source: `docs/phase13_results.md`, `docs/phase14_results.md`,
`docs/phase15_results.md`.

## Experiment infrastructure

The suite driver (`scripts/run_phase7_suite.py`), registry, statistics
(`src/adaptive_rag/evaluation/statistics.py`: Holm, bootstrap, permutation) and
the analysis modules (dispersion, signals, escalation, mechanism, cheapfirst,
powered) are the reusable harness behind every phase above. Artifacts land
under `experiments/phase*/` and carry their own provenance.

---

*Catalog of what was run. Interpretation: [`findings.md`](findings.md).
Chronology: [`../history/experiment_timeline.md`](../history/experiment_timeline.md).*