# Phase 10 Results — Post-Dense Escalation via `distinct_doc_ratio@dense`

Verdict: **FAILURE** (honest negative result; the methodology was not adjusted
to force a positive). Plan of record: `docs/phases/phase-10.md` (frozen before
any split-separated Phase 10 number was computed).

## 1. Question and answer

> After paying for Dense retrieval, can `distinct_doc_ratio@dense` decide
> whether escalating to Hybrid is worthwhile, such that the adaptive policy
> improves the quality–latency–cost trade-off over fixed strategies?

No — not on this benchmark, under the pre-registered protocol. The signal does
not identify the queries where escalation helps; the frozen policy escalates
13–18% of test queries, catches **0 of 2** oracle-positive test queries per
arm, underperforms dense-only on recall@5 (−0.050 / −0.033), and is worse than
random escalation at the same rate (P(random ≥ adaptive) = 1.000 / 0.954).
The failure replicates in direction on the `before` corpus arm.

## 2. Setup (no new retrieval; fully offline)

* Escalation rung: Dense → Hybrid only (ADR-028 excludes the reranker rung).
  Hybrid = RRF(dense@20, BM25@20) → top-10; frozen config
  top_k=10, candidate_k=20, rrf_k=60 (asserted at build time).
* Oracle inputs: dense + hybrid E1 rows from
  `experiments/phase8/combined/p8a_e1` (shipping) and `p8b_e1` (replication) —
  the exact row set Phase 9 measured — plus the frozen Phase 9 signal tables
  and the ADR-027 47/60 calibration/test split.
* Oracle rule (frozen §4): escalate iff
  `recall@5(hybrid) − recall@5(dense) ≥ 0.01`. Ties and harm → NO.
* Latency model (frozen §5): incremental escalation cost = hybrid's
  BM25 + fusion + vector-search stages per query (embedding already paid);
  measured median **31.0 ms** both arms (p95 41.0/43.9), fallback count **0**.
  No USD figure exists for retrieval-only runs; latency is the established
  proxy and API call count is flat (1 embedding/query under every policy).
* Integrity: DDR recomputed from dense rows matches the frozen signal table
  on **107/107** queries per arm; both arms 107/107 `ok`; dataset sha256
  prefix `f0695189903c8ef8` matches the frozen benchmark.

## 3. Oracle headroom (the decision problem's size)

| Arm | helps (Δ≥0.01) | harms (Δ≤−0.01) | neutral | YES total |
| --- | --- | --- | --- | --- |
| after | 4 | 9 | 94 | 4/107 (2 calib + 2 test) |
| before | 4 | 12 | 91 | 4/107 (2 calib + 2 test) |

Escalation helps on 4 queries per arm but harms on 9–12: always-escalate is
net-negative by construction, so a useful policy must *select*, not merely
spend. The selectable headroom is 4 queries — the power limitation declared
in advance (§10 of the plan) binds.

## 4. Signal analysis (calibration only, n=47)

| Arm | YES / NO | DDR mean YES vs NO | ROC-AUC | PR-AUC | Spearman(DDR, Δrecall) |
| --- | --- | --- | --- | --- | --- |
| after | 2 / 45 | 0.50 vs 0.43 | **0.550** | 0.272 | −0.129 |
| before | 2 / 45 | — | 0.861 | 0.583 | −0.164 |

* The two shipping-arm positives sit at DDR 0.2 and 0.8 — split across the
  range. AUC 0.55 is chance; with 2 positives the `before` AUC of 0.86 is
  small-sample noise, not replication (its Spearman runs the *wrong way*,
  as does shipping's).
* Threshold sweep (shipping, primary family `DDR ≥ t`): every threshold's
  mean recall sits within 1 SE (~0.04) of every other's; the frozen rule
  selected **t = 0.8, direction high** (cheapest within 1 SE of the max:
  rate 0.191, tp/fp/fn/tn = 1/8/1/37). Not degenerate. The control family did
  not clear the 1-SE bar, so the direction stands as pre-registered.
* Calibration fixed-arm means are themselves unstable: after-calib hybrid
  (0.8830) edges dense (0.8794) by 0.0036, while before-calib dense (0.9113)
  beats hybrid (0.8511) by 0.06. Nothing here generalizes without a held-out
  check — which is what the test split is for.

## 5. Test evaluation (one pass, n=60, frozen policy t=0.8 high)

| Arm | Dense-only R@5 | Always-esc. R@5 | Adaptive R@5 | Dense MRR | Adaptive MRR | Esc. rate | Confusion (TP/FP/FN/TN) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after | 0.9139 | 0.8389 | **0.8639** | 0.8557 | 0.8268 | 0.133 | 0/8/2/50 |
| before | 0.9167 | 0.8472 | **0.8833** | 0.8833 | 0.8673 | 0.183 | 0/11/2/47 |

* Primary family (recall@5, Holm within pair): adaptive−dense = **−0.050**,
  95% CI [−0.117, 0.000], adj p = 0.228 (ns); adaptive−hybrid = +0.025, CI
  includes 0 (ns). Replication: adaptive−dense = −0.033, CI [−0.083, 0.000].
* Secondary (MRR, Holm within pair): no comparison significant on either arm;
  point estimates negative vs dense on both.
* Latency: adaptive pays +4.0 / +6.0 ms mean over dense (sign test
  significant — structurally guaranteed, since escalation only adds
  compute). Quality down, cost up: the policy is dominated.
* Ablation D (random escalation at the same rate, 1000 draws): random mean
  recall 0.9039 / 0.9045 vs adaptive 0.8639 / 0.8833;
  **P(random ≥ adaptive) = 1.000 / 0.954**. The benefit, if any, does not come
  from DDR-based selection — random spending beats it, because most
  escalation targets on this benchmark are queries where hybrid *harms*.

## 6. The one above-dense operating point (diagnostic, non-finding)

The test operating curve (figure 4, computed post-freeze, no selection) shows
the control direction at t=0.2 reaching 0.9194 vs dense 0.9139 on `after`
(+0.0056 — one-third of a single single-document query flip). On `before` the
same point reads exactly dense (0.9167, Δ=0.0). It is tiny, inside noise
(SE≈0.04), unreplicated across arms, and in the non-pre-selected direction.
Per the frozen protocol it changes nothing; it is recorded so a future study
can state a prior, not as evidence.

## 7. Mechanism reading

Phase 9's correlation stands (DDR tracks cross-strategy *dispersion*), but the
dispersion on this benchmark is dominated by dense succeeding where RRF fusion
fails (dense is the best fixed arm; escalation harms 9–12 queries while
helping 4). DDR therefore marks disagreement without marking *escalation
value*: the two phenomena Phase 9 could not separate (proxy-vs-distinct
quantity, §9.4) resolve here against usefulness — the signal fires on queries
where the dense result is diffuse, and those are disproportionately queries
where fusion degrades the ranking. Correlation with dispersion ≠ prediction of
gain. This is the distinction §21.10 required preserving, and it is preserved.

## 8. Limitations

* Oracle-positive class is 4/107 per arm (2 per split): power-limited by
  construction; CIs are wide and significance was unlikely either way.
* recall@5 takes four distinct values; DDR takes five: both the target and
  the policy are coarse.
* Latency is provider/hardware-dependent (the dense test mean carries a
  ~2.4 s rate-limited embedding outlier); the incremental model is local
  stages only and is unaffected.
* No USD cost (proxy labelled throughout); API call count flat across arms.
* MRR secondary; nDCG excluded (corrupt section labels, as in Phase 8).
* One operating point in a non-pre-selected direction is noted descriptively
  (§6) and was not acted on.

## 9. Verdict mapping (frozen §10, checked mechanically)

* `adaptive_beats_dense_ci_excludes_zero`: **false** (CI [−0.117, 0.000]).
* Adaptive underperforms dense-only on test (−0.050): meets the FAILURE
  condition directly.
* Shipping calibration AUC ≈ chance (0.55).
* Replication confirms the direction (adaptive−dense −0.033, tp=0).
* Random escalation beats the policy at the same spend on both arms.

**FAILURE.** `distinct_doc_ratio@dense` does not reliably identify queries for
which escalation to Hybrid is worthwhile, and the adaptive policy provides no
advantage over fixed strategies — it is dominated by dense-only (higher
quality, lower cost) and by random escalation at equal spend.

## 10. Artifacts (all under `experiments/phase10/`, gitignored)

`oracle_phase8_{after,before}.jsonl` (107 rows each: signal, both outcomes,
deltas, latency model, oracle label), `oracle_provenance.json`,
`signal_analysis_{after,before}.json` (calibration only),
`frozen_policy.json` (t=0.8, high, non-degenerate),
`policy_eval_{after,before}.json` + `policy_rows_{arm}.jsonl` (test, one
pass), `policy_evaluation.json` (incl. mechanical verdict inputs),
`figures/` (5 PNGs + `figures_manifest.json` with sha256).
Code: `src/adaptive_rag/evaluation/escalation.py`
(`ESCALATION_ORACLE_VERSION = "phase10_escalation_v1"`),
`scripts/phase10_{build_oracle,signal_analysis,evaluate_policy,figures}.py`,
`tests/test_escalation_oracle.py` (25 tests). No change to any Phase 9
artifact, retrieval, routing, or runner code.
