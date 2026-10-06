# Phase 10 — Post-Dense Escalation via `distinct_doc_ratio@dense`

Status: **COMPLETE** — verdict `FAILURE` (honest negative result). Final report:
`docs/phase10_results.md`. The body below remains the frozen pre-registration,
written before any split-separated Phase 10 number was computed. Phase 9 is
frozen and unchanged by this phase.

## 1. Research question

> After paying for Dense retrieval, can `distinct_doc_ratio@dense` decide whether
> escalating to a stronger retrieval strategy is worthwhile, such that the
> resulting adaptive policy improves the quality–latency–cost trade-off relative
> to fixed retrieval strategies?

This is a **post-dense escalation problem**, not a pre-routing problem. Phase 9's
verdict (`PROMISING_SIGNAL`: test ρ = +0.410 against `T-disp3`) established a
retrieval-dispersion signal, not a router. Phase 10 tests whether the signal
supports a useful decision. A negative result is valid and pre-declared possible.

## 2. Decision point

```text
Query → Dense Retrieval → COMPUTE distinct_doc_ratio@dense → DECISION
    ├── Stop → Dense result
    └── Escalate → Hybrid result (RRF of dense@20 + BM25@20, top_k=10)
```

## 3. Escalation action (single rung, justified)

**Dense → Hybrid.** Hybrid is the only quality-competitive stronger strategy:
dense is the best fixed arm on recall@5 (0.8988 `after`), hybrid trades
recall@5 (0.8583) for MRR (0.8678 vs 0.8645); the reranker is net-negative on
every metric at 7.5x latency, and ADR-028 recommends removing that rung, so
`hybrid_rerank` is excluded. No new retrieval method is introduced. BM25 is a
de-escalation, not an escalation, and is excluded.

## 4. Oracle definition (frozen)

Primary quality metric: `recall_at_5` (continuity with Phase 9's `T-disp3`).
Secondary: `mrr` (pre-registered secondary; reported separately, never
selected post-hoc). `ndcg_at_5` is excluded: the Phase 8 gold-label audit found
the section labels corrupt (`docs/phase-8-results.md` §0), same exclusion as
`scripts/oracle_routing_ceiling.py`.

```text
ΔQuality(q) = recall_at_5(hybrid, q) − recall_at_5(dense, q)
oracle_escalate(q) = YES  iff  ΔQuality(q) ≥ EPSILON (= 0.01)
                     NO   otherwise
```

* `EPSILON = 0.01` is the project's established constant
  (`evaluation/dispersion.py:48`, Phase 8 frontier).
* Ties (Δ = 0) → NO: escalation costs compute with no benefit.
* Worse (Δ < 0) → NO: escalation actively harms.
* The latency penalty is not inside the rule because the measured incremental
  escalation cost (~tens of ms, §5) is < 5% of dense latency; a
  latency-penalized utility is a pre-registered sensitivity (recorded, not
  primary), not a second chance at the primary rule.

## 5. Latency and cost model (frozen)

* **Incremental escalation latency (primary):**
  `ΔLatency(q) = bm25_latency_ms + fusion_latency_ms + search_latency_ms`,
  taken per query from the hybrid arm's trace metadata. The query embedding is
  already paid at the dense stage and reused, so it is not double-counted.
  Fallback: if any stage is missing, use the full hybrid `retrieval_latency_ms`
  and record the fallback count.
* **Adaptive latency:** `dense.retrieval_latency_ms + (ΔLatency if escalated)`.
* **Cost:** no USD figure exists for retrieval-only runs (`estimated_cost_usd`
  is null on every phase-8 row). The project's established proxy is latency plus
  the frozen `strategy_cost_ms` medians (bm25 2.25, dense 451.78, hybrid 455.97
  ms; `schemas/config.py`). Labelled as a proxy throughout. API call count is
  flat (1 embedding call per query under every policy), so monetary cost does
  not differentiate arms.

## 6. Data (frozen inputs, no new retrieval)

* Shipping arm: `experiments/phase8/combined/p8a_e1` (`phase8_after`), dense +
  hybrid E1 rows/traces (107 queries, 107/107 `ok`).
* Replication arm: `experiments/phase8/combined/p8b_e1` (`phase8_before`),
  same arms. This is the exact row set Phase 9's signal tables were built over.
* Signal: frozen `experiments/phase9/signal_table_phase8_{after,before}.jsonl`,
  field `distinct_doc_ratio.dense`. Recomputed from dense rows at build time;
  the build aborts on any mismatch.
* Splits: ADR-027 frozen `calibration` (n=47) / `test` (n=60) from
  `data/evaluation/phase7_eval_v1.jsonl`, joined on `query_id`.

## 7. Signal analysis (calibration only)

On `calibration`, shipping arm:
* (A) DDR distribution for `oracle_escalate = YES` vs `NO`.
* (B) Ranking ability: ROC-AUC and PR-AUC of DDR against the oracle label, plus
  Spearman(DDR, ΔQuality). No direction assumed by the metrics; the threshold
  families below test both.
* (C) Threshold sweep over t ∈ {0.2, 0.4, 0.6, 0.8, 1.0} (DDR's full value set)
  in two pre-registered families: primary `escalate iff DDR ≥ t` (justified by
  Phase 9's positive ρ), control `escalate iff DDR ≤ t`. Per threshold:
  escalation rate, adaptive recall@5/MRR means, mean latency, TP/FP/FN/TN and
  oracle agreement.

## 8. Threshold selection rule (frozen)

1. Consider primary-family thresholds with escalation rate in [5%, 95%]
   (non-degenerate; t = 0.2 is always-escalate by construction).
2. Among thresholds within **one standard error** of the family's maximum
   calibration mean recall@5, select the one with the **lowest escalation
   rate** (SE = sample std / √n over the 47 calibration adaptive recalls).
3. The control family is selected only if its best calibration mean recall@5
   exceeds the primary family's best by more than 1 SE — declared now so the
   direction choice cannot become a post-hoc flip.
4. If no threshold satisfies (1), the maximum-recall threshold is selected
   anyway and the degeneracy is reported as a limitation (expected verdict then
   PARTIAL or FAILURE, not a re-sweep).

The selected (threshold, direction) is written to `frozen_policy.json` with its
rationale. The `test` split is evaluated exactly once against it. The
replication arm reuses the frozen policy with no refitting.

## 9. Evaluation (test split, one pass)

Arms: **A** dense-only, **B** always-escalate (= hybrid rows),
**C** adaptive (frozen policy), **D** random escalation at C's test escalation
rate (fixed-count subsets, 1000 draws, seed 20250101, distribution reported).

* Per query: dense / escalated / adaptive outcomes and
  true-escalation / false-escalation / missed-escalation / correct-stop counts.
* Pre-registered primary family (recall@5, test, shipping): paired C-vs-A and
  C-vs-B via `evaluation/stats.py::compare_paired` (Wilcoxon + sign test +
  paired effect), Holm-adjusted within the pair. C-vs-D is descriptive
  (adaptive recall vs the random-draw distribution).
* Pre-registered secondary: MRR C-vs-A and C-vs-B, Holm within the pair;
  bootstrap 95% CI on the mean paired recall@5 difference (C−A) and on the
  escalation rate; latency paired effect (descriptive).
* Frontier: quality vs latency, quality vs cost-proxy, quality vs escalation
  rate over A/B/C (+ sweep points).
* Replication: same frozen policy on the `before` arm's test split;
  direction-of-effect confirmation required for SUCCESS.

## 10. Verdict mapping (frozen)

* **SUCCESS** — on `test` (shipping): adaptive recall@5 exceeds dense-only with
  the 95% bootstrap CI on the mean paired difference excluding 0, escalation
  rate well under 100%, and the `before` arm replicates the direction of
  effect. (Significance on top is welcome but, given the expected ~handful of
  oracle-positive queries, the CI-excludes-zero bar is the binding one.)
* **PARTIAL SUCCESS** — DDR ranks oracle-positives above chance (AUC materially
  > 0.5 on calibration) but the system-level test delta is too small or its CI
  includes 0, or adaptive is indistinguishable from random escalation at the
  same rate.
* **FAILURE** — AUC ≈ chance, or no threshold beats dense-only on calibration,
  or adaptive underperforms dense-only on test. Reported honestly; the
  methodology is not adjusted to force a positive.

Known binding limitation, declared in advance: the oracle-positive class is
expected to be small (Phase 8's ceiling gain rides on 4–6 queries over four
strategies; the {dense, hybrid} subset is smaller), so this study is
power-limited by construction — the Gate 3 power statement applies here too.

## 11. Leakage and reproducibility protocol

* Threshold selection reads calibration rows only; the analysis script has no
  code path that loads the test split. Test rows are first opened by the
  evaluation script, after the policy is frozen.
* Oracle rule, metric, epsilon, families, selection rule, stats family, and
  verdict mapping are fixed in this document before any split-separated Phase
  10 number is computed.
* Every artifact records: dataset path + sha256, corpus arm, split sizes,
  retrieval config (top_k/candidate_k/rrf_k from the frozen configs),
  oracle definition + version, threshold + direction, cost assumptions, seeds,
  software version (git commit), timestamp.
* All plots derive from written artifacts, never from in-memory intermediates.

## 12. Files

New module: `src/adaptive_rag/evaluation/escalation.py`
(`ESCALATION_ORACLE_VERSION = "phase10_escalation_v1"`).
New scripts: `phase10_build_oracle.py`, `phase10_signal_analysis.py`,
`phase10_evaluate_policy.py`, `phase10_figures.py`.
New tests: `tests/test_escalation_oracle.py`.
Artifacts (gitignored, provenance inside): `experiments/phase10/`.
Docs: this file (pre-registration), `docs/phase10_results.md`,
`docs/progress.md` entry. No changes under `retrieval/`, `routing/`,
`experiments/runner.py`, or any Phase 9 artifact.
