# Phase 14 Results — Cheap-First Adaptive Retrieval (BM25 → Dense)

Verdict: **INSUFFICIENT EVIDENCE** (frozen mapping applied mechanically;
rule non-degenerate and calibration-consistent, test improves over
BM25-only but neither approaches dense-only nor replicates the random
ablation win).
Plan of record: [`docs/phases/phase-14.md`](phases/phase-14.md) (frozen
before any split-separated Phase 14 number was computed).

## 1. Question and answer

> After running cheap BM25 retrieval, can decision-time information
> available before the Dense embedding call predict whether paying for
> Dense is worthwhile?

On this benchmark, under the pre-registered protocol: not reliably. The
frozen rule (`bm25_slope ≥ −0.915`, flat BM25 score decay → escalate)
catches 5/11 oracle positives per test arm at a 21.7% dense-call rate,
beats BM25-only by +0.078/+0.050 recall@5, but trails dense-only by
−0.081/−0.086, beats random-at-rate on one arm (P = 0.029) and not the
other (P = 0.21). Success criterion (a) — within 0.02 of dense-only —
fails outright; no FAILURE bar is met. The new track's first experiment
is therefore inconclusive, not negative and not positive.

## 2. Setup (no new retrieval; fully offline)

* Rung: BM25 → Dense (dense beats BM25 +0.11–0.14 R@5 both arms; the only
  EV-positive rung; hybrid/reranker excluded per ADR-028 and the Phase 13
  constraint).
* Oracle inputs: bm25 + dense E1 rows/traces from
  `experiments/phase8/combined/p8a_e1` (shipping) and `p8b_e1`
  (replication), frozen Phase 9 BM25-state scalars (DDR/gap/slope, each
  recomputed from frozen rows/traces and matched 107/107 per arm), frozen
  adaptive-trace T0 query features (present 107/107 per arm; zero missing
  gap/slope values), ADR-027 47/60 split, dataset sha256 prefix
  `f0695189903c8ef8`.
* Oracle rule (frozen §4): escalate iff
  `recall@5(dense) − recall@5(bm25) ≥ 0.01`. Ties and harm → stop.
* Latency model (frozen §5): adaptive = BM25 clock + dense clock iff
  escalated; decision cost recorded as 0. Primary currency: dense-call
  rate. No USD figure exists for retrieval-only runs.
* Pre-routing integrity: signal code serves only the six allowlisted
  pre-Dense scalars and refuses all else (`signal_value` allowlist +
  contract tests); cross-arm quantities excluded by construction.

## 3. Oracle headroom

| Arm | YES (Δ≥0.01) | NO | YES calib/test |
| --- | --- | --- | --- |
| after | 16 | 91 | 5 / 11 |
| before | 21 | 86 | 10 / 11 |

4–5× the positives of any old-track framing, as Phase 13 counted
(16/21 reproduced exactly). 86–91 queries need no Dense call.

## 4. Calibration (after, n=47; test untouched)

Descriptive operating numbers (AUC oriented to primary direction;
descriptive-only per protocol at 5 positives):

| Signal | YES mean vs NO mean | AUC |
| --- | --- | --- |
| `bm25_slope` ↑ | −1.078 vs −2.176 (flatter → need) | 0.757 |
| `bm25_gap` ↓ | 0.092 vs 0.141 | 0.643 |
| `content_term_count` ↑ | 20.2 vs 14.7 | 0.624 |
| `bm25_top1` ↓ | 31.3 vs 32.4 | 0.595 |
| `complexity_score` ↑ | 0.427 vs 0.394 | 0.586 |
| `ddr_bm25` ↑ | 0.56 vs 0.49 | 0.586 |

All six directions run the mechanism way; 35 rules sit within 1 SE of the
family maximum (flat landscape — the 5-positive limitation, declared in
advance). Frozen rule (pooled 1-SE, cheapest rate, control precedence
not triggered): **`bm25_slope ≥ −0.915`** — rate 0.191, TP/FP/FN/TN =
2/7/3/35, mean quality 0.8617. Consistency on before-calib (no refit):
adaptive 0.8759 ≥ BM25-only 0.7695, TP 6 — holds, so no verdict cap.

## 5. Test evaluation (one pass, n=60, frozen rule)

| Arm | BM25 R@5 | Adaptive R@5 | Dense R@5 | Δ adapt−BM25 (CI95) | Δ adapt−dense (CI95) | Rate | TP/FP/FN/TN | P(rand ≥ adapt) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| after | 0.7556 | 0.8333 | 0.9139 | +0.078 [0.017, 0.150] | −0.081 [−0.161, −0.014] | 0.217 | 5/8/6/41 | 0.029 |
| before | 0.7806 | 0.8306 | 0.9167 | +0.050 [−0.022, 0.128] | −0.086 [−0.167, −0.019] | 0.217 | 5/8/6/41 | 0.210 |

MRR likewise between arms (after +0.085/−0.049, before +0.082/−0.111;
no comparison Holm-significant in either family). No primary/secondary
comparison is significant after Holm on either arm. Adaptive avoids
78.3% of dense calls (rate CI95 [0.117, 0.317] after). Latency is
descriptive (means carry dense-trace outliers, incl. the known
rate-limited embedding call): adaptive mean 271/108 ms vs BM25 1.9 ms;
the call rate, not the mean, is the cost finding.

## 6. Verdict mapping (frozen §10, checked mechanically)

* SUCCESS needs all of (a)–(e): (a) adaptive ≥ dense − 0.02 → **fails**
  (−0.081/−0.086); (b) rate ≤ 50% holds; (c) TP ≥ 1 holds (5/5);
  (d) P < 0.05 holds after (0.029), fails before (0.21);
  (e) before benefit direction holds (+0.050). → not SUCCESS.
* FAILURE needs any of (i)–(iv): TP = 0 (no: 5), adaptive < BM25-only
  (no: +0.078/+0.050), P ≥ 0.5 (no: 0.029/0.21), calibration-degenerate
  with confirming test (no: consistent, TP > 0). → not FAILURE.
* **INSUFFICIENT EVIDENCE.** The policy demonstrably spends Dense calls
  better than chance on one arm and better than BM25-only on both, but
  the evidence cannot support a routing claim: 5/11 positives caught per
  arm, replication mixed on the ablation, CIs wide, dense-only far above.

## 7. Mechanism reading

The frozen signal is BM25-side, not disagreement-side: flat score decay
(opaque, low-margin BM25 ranking) predicts Dense's marginal value better
than chance on calibration (AUC 0.757, descriptive) and catches nearly
half the test positives at a 22% spend — while DDR@bm25, the old track's
protagonist, manages only 0.586 and loses the family selection outright.
But the landscape is flat (35/≈100 rules within 1 SE) and the test
record is symmetric in the wrong way for a claim: TP = 5 with FN = 6 on
*both* arms, i.e. the rule finds the same number it misses. Selection
among near-tied rules on 5 positives is noise-exposed by construction;
the protocol's guards (1-SE, consistency, one-pass test, random
ablation) did exactly their job — they prevented reading this as signal.

## 8. Limitations

* After-calibration positives = 5 (declared binding limitation in
  advance); family selection over 6 signals on 5 positives.
* `bm25_top1` uncalibrated across queries (scale varies with query
  length/IDF); the normalized gap partly addresses this.
* Recall@5 coarse (four distinct values); MRR secondary; nDCG excluded
  (corrupt section labels, as before).
* Latency means carry provider outliers; call rate is the cost finding.
* Dense-trace outliers (rate-limited embedding calls) enter escalated
  queries' clocks asymmetrically — charged as measured, per protocol.

## 9. What Phase 14 establishes and does NOT establish

Establishes: the cheap-first oracle has 4–5× the positives of the old
track (16/21, both arms); a frozen, pre-cost, BM25-side rule improves
over BM25-only (+0.05–0.08) while avoiding ~78% of Dense calls, with TP >
0 and a random-ablation win on one arm — and still cannot be claimed as a
router (mixed replication, wide CIs, dense-only far above). The
methodology for testing cheap-first routing now exists and has survived
contact with data without drifting.

Does NOT establish: that BM25 insufficiency is predictable (the test
record is consistent with a weak, unconfirmed hint); that any
quality–cost trade-off beats dense-only; anything about Dense→Hybrid
(the closed track is untouched); anything beyond this corpus/rung.

Implication: the new track remains open but unproven — the honest
sequel is a powered-up cheap-first test (the C4 benchmark-expansion
fallback Phase 13 named), not a threshold re-sweep on this data.

## 10. Artifacts (all under `experiments/phase14/`, gitignored)

`oracle_phase8_{after,before}.jsonl` (107 rows each: signals, both
outcomes, deltas, clocks, oracle label), `oracle_provenance.json`,
`signal_analysis_after.json` + `signal_analysis_before_calib.json`,
`frozen_policy.json` (`bm25_slope ≥ −0.915`, high, non-degenerate),
`policy_eval_{after,before}.json` + `policy_rows_{arm}.jsonl`,
`policy_evaluation.json` (mechanical verdict inputs), `figures/` (4 PNGs
+ `figures_manifest.json` with sha256).
Code: `src/adaptive_rag/evaluation/cheapfirst.py`
(`NEED_ORACLE_VERSION = "phase14_cheapfirst_v1"`),
`scripts/phase14_{build_oracle,signal_analysis,evaluate_policy,figures}.py`,
`tests/test_cheapfirst_oracle.py` (20 tests). No change to any Phase
8–13 artifact, retrieval, routing, or runner code.
