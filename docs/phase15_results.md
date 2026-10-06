# Phase 15 Results — Powered Cheap-First Confirmation (BM25 → Dense)

Verdict: **INSUFFICIENT EVIDENCE** (frozen mapping applied mechanically:
4 of 5 success bars met, including the powered primary; the dense-margin
bar (a) missed by 0.004 with no failure bar met).
Plan of record: [`docs/phases/phase-15.md`](phases/phase-15.md) (frozen
before any powered retrieval ran; power simulations used Phase 14 rows as
a population model and selected nothing).

## A. Dataset

Powered set `phase15_eval_v1`: **110 freshly curated queries**, all
`split: "test"` (nothing fitted, so no calibration split exists). Zero
reused queries: the 32 v1-excluded part-records and all of `dense_eval_v1`
were ineligible by protocol. Category quotas exact (factual 27,
conceptual 30, terminology 21, fine_grained 17, comparative 8,
multi_document 7); all 14 paper minimums met (lewis 19, robertson 18,
retro 14, rest at/above minimum). Duplicate screen (normalized
token-Jaccard < 0.6) against 266 existing queries: **0 flags**. Dataset
gate green with G7 column-contiguity enabled (0 contaminated, 125
indeterminate, 183 verified of 308 quotes); gate artifact
`experiments/phase15/dataset_gate.json` records `single_split_test_only`.
Curation → gate → freeze → retrieval order held; no BM25 run, embedding
call, or signal computation touched a new query before the freeze.

Oracle positives (frozen `phase14_cheapfirst_v1`, Δ ≥ 0.01):
**14 after (12.7%) / 11 before (10.0%)** — at the low edge of the
pre-registered expectation (~20/arm, P5 ≈ 14). Per §15.16 this shortfall
is reported, not repaired: no top-up curation post-retrieval.

Corpus note: `pyserini_lin_2021` chunks contain a biomedical
knowledge-graph survey, not the Pyserini toolkit paper (0/94 chunks
mention Pyserini; manifest title is a misnomer). Its 4 queries were
curated against the actual content. All other papers match their titles.

## B. Frozen policy

Option A, exact Phase 14 rule from
`experiments/phase14/frozen_policy.json` (sha256 recorded in the powered
provenance): **escalate iff `bm25_slope ≥ −0.9152305000000001`**
(direction high; non-degenerate). Applied unchanged to 110+110 powered
rows. Missing signals → STOP (none missing).

## C. Oracle

14/11 positives (§A). 96/99 queries need no Dense call. The powered set
has a lower positive rate than Phase 14's test (18.3%) — a property of
the fresh distribution, not a finding about the rule.

## D. Policy selection

| Arm | Rate (CI95) | TP | FP | FN | TN | Catch rate |
| --- | --- | --- | --- | --- | --- | --- |
| after | 0.282 [0.200, 0.373] | 9 | 22 | 5 | 74 | 9/14 |
| before | 0.318 [0.236, 0.409] | 9 | 26 | 2 | 73 | 9/11 |

The rule escalates more here than on Phase 14 test (0.28/0.32 vs 0.217)
and catches more (9/14, 9/11 vs 5/11) at more false positives. Precision
is low (29%/26%); recall of positives is high (64%/82%).

## E. Retrieval quality

| Arm | BM25 R@5 | Adaptive R@5 | Dense R@5 | Δ adapt−BM25 (CI95) | Δ adapt−dense (CI95) |
| --- | --- | --- | --- | --- | --- |
| after | 0.8545 | 0.9129 | 0.9432 | +0.058 [0.016, 0.107] | −0.030 [−0.067, 0.000] |
| before | 0.8795 | 0.9242 | 0.9424 | +0.045 [0.004, 0.092] | −0.018 [−0.046, 0.000] |

MRR likewise between arms (after +0.071/−0.036, before +0.098/−0.042).
Holm-adjusted recall@5 comparisons: none significant on either arm
(sign-test p 0.146/0.219 after, 0.064/0.500 before; 98–101 ties of 110 —
recall@5 is coarse). MRR adaptive-vs-BM25 is significant on both arms;
MRR adaptive-vs-dense is significantly negative on before (−0.042).

## F. Efficiency

Dense calls avoided: **71.8% after / 68.2% before** (rates above).
Adaptive mean latency 140.9/165.5 ms vs BM25 1.6/1.5 ms — descriptive
(dense-trace provider outliers); the call rate, not the mean, is the cost
finding. Decision cost recorded as 0. No USD (null on retrieval-only
rows, as ever).

## G. Adaptive vs Random (the critical result)

Per-arm ablation (1000 draws, seed 20250101): **P = 0.008 after /
0.019 before** — adaptive beats random-at-rate on BOTH arms, resolving
Phase 14's mixed replication (0.029/0.210). Combined randomization test
(pre-registered primary): **P_comb = 0.001** (observed 0.9186 vs null
mean 0.8895, null p95 0.9045). The frozen rule spends Dense calls
decisively better than chance. This is the experiment's positive finding.

## H. Cross-arm replication

Benefit direction replicates (+0.058/+0.045 R@5, +0.071/+0.098 MRR);
ablation win replicates (0.008/0.019); catch pattern replicates (9/14,
9/11). No arm contradicts any other. The combined cluster-bootstrap CI
on adaptive−BM25 is +0.0515 [0.0110, 0.0977] — excludes 0.

## I. Statistical uncertainty

Reported throughout §§D–H: bootstrap CIs on differences and rates
(seed 20250115), Holm within each C-vs-A/C-vs-B pair per arm per metric,
1000-draw random nulls per arm plus the combined test. Coarse recall@5
(≈90% ties) is why paired p-values stay non-significant while bootstrap
CIs exclude 0 — both reported, neither hidden.

## J. Verdict

Frozen mapping (§§15.13–15.15), checked mechanically
(`policy_evaluation.json` → `verdict_mapping`):

* SUCCESS needs all of (a)–(e): (a) combined adaptive ≥ combined dense −
  0.02 → **fails** (0.9186 vs 0.9428: gap −0.0242, miss by 0.004);
  (b) rate ≤ 50% holds (0.28/0.32); (c) TP ≥ 1 holds (9/9);
  (d) P_comb < 0.05 with neither per-arm P ≥ 0.5 holds (0.001; 0.008/0.019);
  (e) combined CI excludes 0 with per-arm benefit ≥ 0 holds. → not SUCCESS.
* FAILURE needs any of (i)–(iv): TP = 0 (no), adaptive < BM25 combined
  (no: +0.051), P_comb ≥ 0.5 (no: 0.001), both per-arm P ≥ 0.5 (no). →
  not FAILURE.
* **INSUFFICIENT EVIDENCE**, 4 of 5 success bars met.

## K. Scientific conclusion

Establishes: the Phase 14 mechanism **replicates its selection claim**
on 220 fresh query-instances — the frozen BM25-slope rule catches 18/25
oracle positives at ~30% spend and beats random spending decisively on
both arms individually (P = 0.008/0.019) and combined (P = 0.001),
improving over BM25-only by +0.05–0.06 R@5 (+0.07–0.10 MRR) while
avoiding ~70% of Dense calls. The exact open question from Phase 14
(mixed random ablation) is closed in the affirmative.

Does NOT establish: that the tradeoff is *worth it* under the frozen
margin — adaptive trails dense-only by 0.018–0.030 R@5 (0.036–0.042
MRR), missing the pre-registered 0.02 bar by 0.004 on the combined mean.
The rule is a real but lossy selector: nearly half its escalations are
false positives, and 7/25 positives are still missed. Nothing about
other rungs, corpora, or learned routers.

Implication (per the §29 decision tree): two consecutive
INSUFFICIENTs with the same signature — genuine selection signal,
dense-margin shortfall. The remaining uncertainty is no longer
statistical power over the ablation (P_comb = 0.001 settles that) but
the *claim*: whether near-dense quality at ~70% fewer embedding calls
counts as success is a margin judgment, not a measurement gap. A further
experiment on this benchmark needs a genuinely new information source
(new rung, new decision point, or a cost model that prices the 0.024
gap); re-sweeping thresholds or features on these queries is prohibited
by the same discipline that made this confirmation credible.

## Artifacts

`experiments/phase15/`: powered dataset gate artifact, 6 E1 runs
(110/110 ok each), `oracle_powered_{after,before}.jsonl` +
`oracle_provenance.json`, `policy_eval_{after,before}.json` +
`policy_rows_{arm}.jsonl` + `policy_eval_combined.json` +
`policy_evaluation.json` (mechanical mapping), 5 figures + manifest.
Code: `src/adaptive_rag/evaluation/powered.py`,
`scripts/phase15_{build_oracle,evaluate,figures}.py`,
`tests/test_powered_confirmation.py` (20 tests), `--allow-test-only`
flag in `validate_phase7_dataset.py` (default-off, recorded in artifact).
No change to retrieval, routing, runner, evaluation-library, or any
Phase 8–14 artifact.
