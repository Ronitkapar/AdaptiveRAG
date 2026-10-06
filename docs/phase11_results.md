# Phase 11 Results — Missing Signal Behind Useful Retrieval Escalation

Outcome: **B — Insufficient evidence** (plausible mechanism, observable
precursors, far too few examples to establish a signal; no candidate
promoted, no intervention run).
Plan of record: [`docs/phases/phase-11.md`](phases/phase-11.md) (frozen before
any Phase 11 computation). Phases 8–10 unchanged.

## 1. Groups (frozen Phase 10 oracle, 107 queries/arm)

| Arm | helps | harms | ties | states (S2 / S3 / S1) |
| --- | --- | --- | --- | --- |
| after | 4 | 9 | 94 | 4 / 9 / 94 |
| before | 4 | 12 | 91 | 4 / 12 / 91 |

Union: **5 unique helps** (`p7a_014` both arms; `p7g_001` after; `p7e_005`
before; `p7e_011`, `p7e_014` both arms — test). Every harm query satisfies
`dense_recall_at_5 > 0`, so **all harms are State 3** (dense correct, fusion
corrupted it); all helps are State 2; State 1 is ties. The three-state
mapping is exact, not approximate.

## 2. Gate answers (§18 of the brief)

### Q1 — What exactly causes Hybrid to help?

Dense misses relevant evidence (dense recall 0–0.33 on all 5 unique helps)
while BM25 holds or highly ranks it, and RRF keeps it inside the top-5:

* `p7a_014` (0.0→1.0, both arms): dense top-1 wrong (colbert 0.41); BM25 has
  the relevant doc at #1 (17.3); hybrid recovers it. Source: `bm25_only`.
* `p7e_011` (0.33→0.67, both): relevant `crag` enters from `bm25_only`.
* `p7e_014` (0.0→0.33, both): relevant `bm25_robertson` enters (`both`
  pool); two further `bm25_only` docs enter.
* `p7g_001` (0.0→0.33, after): relevant `self_rag` enters from `both`.
* `p7e_005` (0.0→0.5, before): relevant `dpr` enters from `both`.

Common mechanism: **BM25-side rescue of dense-missed evidence**. Dense top-1
scores are low on all five (0.40–0.56); disagreement is high (jaccard 0–0.4).

### Q2 — What exactly causes Hybrid to hurt?

Two sub-mechanisms, both State 3 (dense held relevant evidence):

* **H1 — intruder promotion** (e.g. `p7_003`, `p7b_011`, `p7e_006`,
  `p7f_006`): `bm25_only`/`both` docs BM25 ranks confidently but irrelevantly
  (crag, retro, realm, contriever) displace the relevant doc from the top-5.
* **H2 — re-ranking among shared docs** (`p7_006`, `p7b_007`, `p7e_001`,
  `p7e_002`, `p7e_009` after and several before): **no new document enters**
  (`entered_source` empty) — RRF reorders the shared pool and the relevant
  doc's chunks fall below the cutoff (e.g. `p7b_007`: hybrid top-5 is five
  chunks of a single wrong doc). This is the Phase 4 RRF-flattening effect at
  per-query resolution, and it means even an agreeing BM25 can preside over
  harm — agreement is not safety either.

### Q3 — Can the two cases be distinguished BEFORE paying for Hybrid?

Directionally yes, establishably no. Calibration orderings (helps vs harms
means, both arms): dense top-1 **0.49/0.48 vs 0.63/0.56**; jaccard
**0.13/0.27 vs 0.41/0.40** (helps = weaker dense + higher disagreement);
union reversed identically; gap/slope/DDR overlap without a consistent
direction. But n=2 helps/arm, ranges overlap (`before` top-1: helps max 0.544
> harms min 0.442), and the bounded separation check over the six
pre-declared features separates on **1/6 (after) and 0/6 (before)**
calibration arms — the one hit (`dense_top1_score`, helps-below-harms on
`after`) fails on the replication arm, the signature of fitting 5 points.
The hypothesized interaction (weak-dense × disagreement → State 2;
strong-dense × disagreement → State 3) is the only mechanism-consistent
reading, and it is **not established**.

### Q4 — Enough examples?

No. Five unique helps total, two per split/arm; harms 3–6 per split/arm.
No supervised routing claim can be built on this — stated as the finding,
not worked around. Any rule written against the calibration helps (e.g.
"top-1 < 0.6 and jaccard < 0.3") would be a 2-point fit regardless of its
test record.

### Q5 — Is another intervention justified?

**No.** The gate was pre-registered closed: criterion 12.3 (enough
development examples) fails a priori, and §5F's conditional candidate never
materialized (no calibration-consistent, decision-time-computable rule
survived the replication check). No adaptive experiment was run.

### Q6 — Minimal defensible future design (if the question is revisited)

1. A benchmark with dozens of oracle-positive queries (power analysis
   upfront; AUC claims need positives in the tens, not twos).
2. Pre-registered rule from the State-2/State-3 interaction family
   (dense-confidence × BM25-disagreement), with BM25-at-decision (~2 ms,
   no API call) explicitly admitted as cheap second evidence.
3. Frozen calibration/test split discipline as in Phases 10–11.
4. A second-stage action with positive expected value somewhere — on this
   corpus RRF-hybrid harms 2–3× more queries than it helps, so the rung
   itself needs re-justification before any router is built around it.

### Q7 — What should AdaptiveRAG conclude?

* Dense-only stands as the operating strategy; no adaptive policy is
  licensed by any evidence in hand.
* DDR is a retrieval-dispersion diagnostic, not a routing signal (Phases 9–11
  agree; Phase 11 adds the mechanism: dispersion here is dense succeeding
  where fusion fails).
* Disagreement ≠ insufficiency ≠ escalation value — demonstrated per query,
  not asserted: helps need BM25-held missing evidence (unobservable without
  labels); harms need only a confident dense result meeting RRF.
* The carry-forward is the three-state framing plus the evidence-
  insufficiency hypothesis and the H1/H2 harm taxonomy, all with exact
  per-query records for any future study with adequate positives.

## 3. Required figures (4; 5th omitted per protocol — no candidate)

`experiments/phase11/figures/`: `ddr_vs_delta`, `gap_vs_delta`,
`disagreement_vs_quality` (helps sit low-jaccard/low-dense-recall; harms
span all disagreement levels at high dense recall — the visual form of
"disagreement ≠ value"), `nontie_waterfall` (asymmetric: deeper, more
numerous harms) + sha256 manifest.

## 4. Limitations

* n=5 unique helps: everything comparative is descriptive; no inferential
  claim is made on the help set.
* Source attribution is top-10 membership (candidate-depth 11–20 not
  persisted in traces); `neither` is reported honestly where it occurs.
* Labels characterize mechanism only and enter no signal (builder has no
  label-consuming code path except change attribution).
* Dense absolute scores are not compared across queries (uncalibrated
  cosine magnitudes); only within-query margins/shapes are used.
* Corpus- and rung-specific (RRF hybrid over this 107-query benchmark).

## 5. Artifacts

`src/adaptive_rag/evaluation/mechanism.py` (`phase11_mechanism_v1`),
`scripts/phase11_mechanism.py`, `scripts/phase11_figures.py`,
`tests/test_mechanism.py` (12 tests), `experiments/phase11/`
(`mechanism_table_{after,before}.jsonl` — 107 records each with group,
state, confidence, agreement, and change fields; `mechanism_summary.json`
with counts, orderings, separation check, and all non-tie case blocks;
figures + manifest). No Phase 8–10 artifact, retrieval, routing, or runner
code touched.
