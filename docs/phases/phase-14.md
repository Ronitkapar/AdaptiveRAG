# Phase 14 — Cheap-First Adaptive Retrieval (BM25 → Dense)

Status: **COMPLETE** — verdict `INSUFFICIENT EVIDENCE`. Final report:
`docs/phase14_results.md`. The body below remains the frozen pre-registration,
written before any split-separated Phase 14 number was computed (only the
Phase 13 oracle counts, already published, are referenced). Phases 8–13 are
frozen and unchanged by this phase.

## 1. Research question

> After running cheap BM25 retrieval, can decision-time-available
> information available **before the Dense embedding call** predict whether
> paying for Dense retrieval is worthwhile, such that the adaptive policy
> obtains useful Dense improvements while avoiding unnecessary Dense calls?

This is a **pre-routing, cheap-first** problem. The closed track
(retrieval-state → predict Dense→Hybrid gain) is not revisited: different
decision point (T0/T1, pre-cost), different rung direction (BM25→Dense),
different information (no dense/hybrid output), different currency
(embedding calls avoided).

## 2. Decision point

```text
Query → BM25 Retrieval (local, ~2 ms, no API call) → DECISION (T0 query-only / T1 BM25-state)
    ├── Stop → BM25 result (no embedding call)
    └── Escalate → Dense result (one embedding call)
```

The decision is computed strictly before the T2 embedding call. Any
candidate signal requiring dense embeddings, dense/hybrid/reranker
results, or any post-Dense information is invalid for this experiment.

## 3. Rung (BM25 → Dense, justified)

Dense beats BM25 by +0.11–0.14 recall@5 on both frozen arms (Phase 13 §2:
0.8988/0.7866 after, 0.9143/0.7757 before) — the only EV-positive rung in
the measured stack, so the first rung worth routing to. Hybrid and
`hybrid_rerank` are excluded (net-negative vs dense; ADR-028). No new
retrieval method is introduced.

## 4. Oracle definition (frozen)

Primary: `recall_at_5`. Secondary: `mrr` (reported separately, never
selected post-hoc). `ndcg_at_5` excluded (corrupt section labels, as in
Phases 8/10).

```text
ΔQuality(q) = recall_at_5(dense, q) − recall_at_5(bm25, q)
oracle_escalate(q) = YES  iff  ΔQuality(q) ≥ EPSILON (= 0.01)
                      NO   otherwise
```

`EPSILON = 0.01` (project constant). Ties → NO (a call with no benefit);
harm → NO. Oracle version: `phase14_cheapfirst_v1`.

## 5. Latency and cost model (frozen)

* **Adaptive latency:** `bm25.retrieval_latency_ms + (dense.retrieval_latency_ms
  iff escalated)`, per query from the frozen E1 rows. The dense trace
  latency already contains its embedding call; it is charged only when the
  call happens. Decision cost is a threshold comparison, recorded as 0 —
  stated, not hidden.
* **Primary currency:** Dense-call rate = fraction escalated; calls avoided
  = 1 − rate. Dense-only = 100%, BM25-only = 0%.
* **Cost:** no USD figure exists for retrieval-only runs (null on every
  phase-8 row); latency + call rate is the labelled proxy throughout, same
  convention as Phase 10.

## 6. Data (frozen inputs, no new retrieval)

* `experiments/phase8/combined/p8a_e1` (shipping/`after`) and `p8b_e1`
  (replication/`before`): bm25 + dense E1 rows/traces, 107 queries each.
* BM25-state signals from frozen `experiments/phase9/
  signal_table_phase8_{after,before}.jsonl`: `distinct_doc_ratio.bm25`,
  `score_gap_top1_top2.bm25`, `score_decay_slope.bm25`, plus `bm25_top1`
  recomputed from frozen bm25 traces (`retrieval.results[0].
  retrieval_score`). Cross-arm quantities (jaccard, top-1 agreement,
  union_concentration, overlap, rank-correlation) are excluded by
  construction — dense has not run at decision time.
* T0 query features from frozen adaptive-arm traces
  (`routing.features`, numeric fields only): `complexity_score` and
  `content_term_count` as the two pre-declared representatives. Labels
  (`relevant_documents`, metrics) score rules only, never enter them.
* Splits: ADR-027 frozen `calibration` (n=47) / `test` (n=60) from
  `data/evaluation/phase7_eval_v1.jsonl`, joined on `query_id`.

## 7. Candidate signals (bounded, frozen)

Six single-feature threshold rules. Primary direction follows the
insufficiency mechanism (escalate when BM25 looks weak); the control
direction is the exact opposite:

| Signal | Source | Primary (escalate iff) | Rationale |
| --- | --- | --- | --- |
| `ddr_bm25` | signal table | HIGH (`≥ t`) | diffuse coverage → weak (mirrors Phase 10) |
| `bm25_top1` | traces | LOW (`≤ t`) | weak top evidence → need dense |
| `bm25_gap` | signal table | LOW (`≤ t`) | small margin → uncertain top |
| `bm25_slope` | signal table | HIGH (`≥ t`) | flat decay (≈0) → diffuse (code semantics, `agreement.py`) |
| `complexity_score` | adaptive traces | HIGH (`≥ t`) | harder query → need dense |
| `content_term_count` | adaptive traces | HIGH (`≥ t`) | more content → need dense |

Threshold grids are the deciles of the `after`-calibration signal
distribution (label-free, deterministic; at most 9 per rule). BM25 raw
scores are uncalibrated across queries (scale with query length/IDF) —
recorded as a limitation of `bm25_top1`, which the normalized `bm25_gap`
partly addresses. No AUC-as-evidence on ≤6 positives (Phase 11
discipline): ranking metrics are descriptive only. No classifier, no
feature combination, no second family.

## 8. Rule selection (frozen)

On `after`-calibration only:

1. Consider primary-direction rules with escalation rate in [5%, 95%]
   (non-degenerate).
2. Among rules within **one standard error** of the family's maximum
   calibration mean recall@5, select the one with the **lowest escalation
   rate** (SE = sample std / √n over the 47 calibration adaptive recalls).
3. The control direction replaces the primary only if its best calibration
   mean recall@5 exceeds the primary best by more than 1 SE — declared now
   so direction cannot flip post-hoc.
4. If no rule satisfies (1), the maximum-recall rule is frozen anyway and
   the degeneracy is reported (expected verdict then INSUFFICIENT or
   FAILURE, not a re-sweep).

**Cross-arm consistency check (frozen):** the frozen rule's `before`-
calibration adaptive mean recall@5 must be ≥ `before`-calibration
BM25-only mean. If it fails, the test is still evaluated exactly once
(protocol integrity) but the verdict is capped at INSUFFICIENT EVIDENCE.
The `before` arm reuses the frozen rule with no refitting.

The selected (signal, threshold, direction) is written to
`frozen_policy.json` with rationale and provenance. The `test` split is
evaluated exactly once per arm against it.

## 9. Evaluation (test, one pass)

Arms: **A** BM25-only, **B** always-escalate (= dense rows),
**C** adaptive (frozen rule), **D** random escalation at C's test
escalation rate (fixed-count subsets, 1000 draws, seed 20250101,
distribution reported).

* Per query: bm25 / dense / adaptive outcomes; TP/FP/FN/TN vs oracle;
  dense-call indicator; latencies.
* Primary (recall@5, test, shipping): paired C-vs-A and C-vs-B via
  `evaluation/stats.py::compare_paired`, Holm-adjusted within the pair;
  bootstrap 95% CI on mean paired C−A difference and on escalation rate.
  C-vs-D descriptive (adaptive recall vs random-draw distribution;
  `P(random ≥ adaptive)` reported as in Phase 10).
* Secondary: MRR C-vs-A and C-vs-B, Holm within the pair; call-rate and
  latency paired effects (descriptive).
* Frontier: quality vs dense-call rate over A/B/C (+ sweep points).
* Replication: same frozen rule on the `before` arm's test split.

## 10. Verdict mapping (frozen)

* **SUCCESS** — all of: (a) `after`-test adaptive R@5 ≥ dense-only R@5 −
  0.02 (pre-registered margin = the project's headroom quantum);
  (b) escalation rate ≤ 50% (avoids at least half the calls);
  (c) TP ≥ 1 on test; (d) adaptive exceeds the random distribution's 95th
  percentile (`P(random ≥ adaptive)` < 0.05); (e) `before`-test replicates
  the benefit direction (adaptive − BM25-only R@5 ≥ 0) at rate ≤ 50%.
* **FAILURE** — any of: (i) `after`-test TP = 0 with rate > 5% (spends
  without catching); (ii) adaptive R@5 < BM25-only R@5 on `after`-test
  (dominated: quality down, cost up — the Phase 10 signature);
  (iii) `P(random ≥ adaptive)` ≥ 0.5 on `after`-test; (iv) no candidate
  beats BM25-only on calibration and test confirms no benefit.
* **INSUFFICIENT EVIDENCE** — otherwise: rule frozen non-degenerate and
  calibration-consistent, but test inconclusive (wide CI, TP small,
  replication mixed). Includes the consistency-check-failure cap from §8.

Known limitation, declared in advance: `after`-calibration holds ~5
oracle positives (Phase 13), so selection is noise-exposed by
construction — the 1-SE rule, the consistency check, the one-pass test,
and the random ablation exist to prevent reading noise as signal.

## 11. Leakage and reproducibility protocol

* Selection reads `after`-calibration rows only; the analysis script has
  no code path that loads the test split. Test rows are first opened by
  the evaluation script, after the policy is frozen.
* Oracle, metric, epsilon, families, grids, selection rule, stats family,
  seeds, and verdict mapping are fixed in this document before any
  split-separated Phase 14 number is computed.
* Every artifact records: dataset path + sha256, corpus arm, split sizes,
  oracle definition + version, rule + threshold + direction, cost
  assumptions, seeds, software version (git commit), timestamp.
* All plots derive from written artifacts, never from in-memory
  intermediates.

## 12. Files

New module: `src/adaptive_rag/evaluation/cheapfirst.py`
(`NEED_ORACLE_VERSION = "phase14_cheapfirst_v1"`; pure functions;
allowlisted pre-Dense inputs only — no dense/hybrid/reranker-derived
quantities enter any signal path).
New scripts: `phase14_build_oracle.py`, `phase14_signal_analysis.py`,
`phase14_evaluate_policy.py`, `phase14_figures.py`.
New tests: `tests/test_cheapfirst_oracle.py` (incl. a pre-routing guard:
signal code paths accept only allowlisted keys).
Artifacts (gitignored, provenance inside): `experiments/phase14/`.
Docs: this file (pre-registration), `docs/phase14_results.md`,
`docs/progress.md` entry. No changes under `retrieval/`, `routing/`,
`experiments/runner.py`, or any Phase 8–13 artifact.
