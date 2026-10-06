# Phase 15 — Powered Cheap-First Confirmation Experiment (BM25 → Dense)

Status: **COMPLETE** — verdict `INSUFFICIENT EVIDENCE` (4 of 5 success bars met;
the dense-margin bar was missed by 0.004). Final report: `docs/phase15_results.md`.
The body below remains the frozen pre-registration, written before any powered
retrieval ran (power simulations used Phase 14 rows as a population model and
selected nothing). Phases 8–14 are frozen and unchanged by this phase.

## 15.1 Research question

> When BM25 is used as the cheap first-stage retriever, can a pre-routing
> decision rule reliably predict when paying for Dense retrieval provides
> enough additional retrieval value to justify its cost?

Specifically: **does the Phase 14 cheap-first mechanism replicate when
evaluated with substantially more oracle-positive opportunities?** This is a
confirmation experiment, not signal discovery.

## 15.2 Hypothesis

The exact frozen Phase 14 rule (`bm25_slope ≥ −0.9152305000000001`,
escalate on flat BM25 score decay; §15.6) identifies BM25→Dense
oracle-positive queries out of sample on fresh queries, such that the
adaptive policy (a) improves over BM25-only while avoiding most Dense
calls, (b) selects Dense calls better than random spending at the same
rate, and (c) does so on both corpus arms. Failure of any frozen bar is a
finding about the mechanism, not a trigger to search further on this data.

## 15.3 Motivation

Phase 14 (`docs/phase14_results.md`, verdict INSUFFICIENT EVIDENCE) found
a 4–5× oracle opportunity (16/21 positives) and a BM25-side rule that
beats BM25-only (+0.078/+0.050 recall@5 at 21.7% call rate, TP 5/11 per
arm) but could not establish intelligent selection: 11 positives per test
arm, mixed random-ablation replication (P = 0.029 / 0.210), wide CIs. The
binding limitation, declared in advance, was statistical power (5
after-calibration positives drove selection; 11 test positives per arm
drove evaluation). Phase 13 named the sequel in advance: the C4
benchmark-expansion fallback — a powered-up cheap-first test, not a
threshold re-sweep. This phase is that test.

## 15.4 Frozen oracle

Identical to Phase 14 (`phase14_cheapfirst_v1`, EPSILON = 0.01):

```text
ΔQuality(q) = recall_at_5(dense, q) − recall_at_5(bm25, q)
oracle_escalate(q) = YES  iff  ΔQuality(q) ≥ 0.01, else NO
```

Ties/harm → stop. Primary metric recall@5; MRR secondary (never
selected post-hoc); nDCG@5 excluded (corrupt section labels, as ever). A
contract test asserts the Phase 15 builder emits byte-identical oracle
labels to the Phase 14 artifacts when run over Phase 14 inputs.

## 15.5 Frozen signal family

The six Phase 14 pre-Dense signals, unchanged, with identical definitions
(`src/adaptive_rag/evaluation/cheapfirst.py`: `PRE_DENSE_SIGNALS`,
`signal_value`, `SIGNAL_K = 5`): `ddr_bm25`, `bm25_top1`, `bm25_gap`,
`bm25_slope` (BM25-state, recomputed from BM25 traces with the Phase 9
functions) plus `complexity_score`, `content_term_count` (T0 query
features via `QueryFeatureAnalyzer` on the query string — no retrieval
run needed). Dense/hybrid/reranker-derived quantities, cross-arm
agreement, and labels remain structurally refused. Only `bm25_slope`
enters the decision; the rest serve descriptive mechanism context (§21),
never selection.

## 15.6 Frozen policy (Option A — direct test, no re-estimation)

**Option A.** The exact Phase 14 frozen rule is applied unchanged:

```text
escalate  iff  bm25_slope ≥ −0.9152305000000001   (direction "high")
```

recovered from `experiments/phase14/frozen_policy.json` (signal
`bm25_slope`, threshold `-0.9152305000000001`, non-degenerate). No
document authorizes re-estimation (Phase 14 §8 mandates reuse without
refitting; Phase 13 specifies a frozen-rule test), and the research
question is replication, so re-estimation would answer a different
question. The implementation reads the threshold from the frozen file; a
test pins all three fields. Missing signal → STOP (no imputation, as
Phase 14).

## 15.7 Dataset expansion (C4 fallback, same distribution, blind)

**Powered set: 110 freshly curated queries**, `dataset_version
phase15_eval_v1`, ids `p15_001…`, all `split: "test"`
(calibration/test separation is vacuous under Option A — nothing is
fitted; §15.8). Same corpus (all 14 papers), same task distribution:

* Category quotas (mirror v1 proportions): factual 27, conceptual 30,
  terminology 21, fine_grained 17, comparative 8, multi_document 7 (=110).
* Paper coverage minimums (a multi-doc query counts toward each listed
  paper): rag_lewis_2020 18, bm25_robertson_2009 17, retro_borgeaud_2022
  14, splade_v2_formal_2021 9, rrf_cormack_2009 7, contriever_izacard_2022
  9, colbert_khattab_2020 8, monobert_nogueira_2019 8,
  self_rag_asai_2023 9, crag_yan_2024 6, adaptive_rag_jeong_2024 6,
  dpr_karpukhin_2020 9, realm_guu_2020 9, pyserini_lin_2021 4 (the one
  paper v1 never covers).
* Grounding: one ledger record per query (verbatim quotes from chunks on
  disk, `answer_terms` in quotes, non-empty curator notes);
  `relevant_sections` prefixes must resolve (G6); the dataset gate
  (`scripts/validate_phase7_dataset.py`, incl. G7 column-contiguity where
  decidable) must pass with the single pre-registered accommodation of a
  `--allow-test-only` flag (see §15.17; default gate behavior unchanged).
* Genuinely new questions: normalized token-set Jaccard < 0.6 against
  every query in `phase7_eval_v1`, `dense_eval_v1`, `phase7_part_*`, and
  within the new set; any pair at/above threshold is rewritten or dropped
  before retrieval. The 32 part-records excluded from v1 and all of
  `dense_eval_v1` are ineligible for inclusion (unknown exclusion reasons;
  unverified labels; analyst exposure) — fresh curation only.
* Blindness: curation → gate green → freeze → retrieval. No BM25 run, no
  embedding call, no signal computation on a new query before the freeze
  (the duplicate screen uses query text only). Curator knowledge of the
  frozen rule cannot determine BM25 decay or oracle labels without
  retrieval; residual risk is recorded as a limitation, not controlled
  beyond the quotas, grounding, and blindness above.
* No positive-count targeting: queries are never added, removed, or
  replaced based on retrieval outcomes or oracle labels. Whatever positive
  count the powered set yields is reported as-is.

## 15.8 Calibration/test split

None. Option A fits nothing, so all 110 powered queries are test. The old
47/60 split is untouched; no old query enters Phase 15 in any role (the
Phase 14 test rows were analyst-visible and, for after-calibration,
selection-influencing — reuse would void confirmation).

## 15.9 Baselines

Per corpus arm: **A** BM25-only, **B** dense-only (= always-escalate),
**C** adaptive (frozen rule applied to powered BM25 traces), **D**
random-at-C's-arm-rate (fixed-count subsets, 1000 draws, seed 20250101 —
the Phase 10/14 seed, kept for cross-track comparability). Runs:
bm25 + dense × {phase8_after, phase8_before} via `run_phase7_suite.py`
(`--retrieval-only`, `--pace 1.5`), 110/110 `ok` per arm or re-run; no
partial arms; no adaptive/hybrid/reranker runs needed (C is a blend of A/B
rows by the frozen rule).

## 15.10 Metrics

Primary: recall@5. Secondary: MRR (reported separately). Always with:
TP/FP/FN/TN vs oracle, escalation rate, dense calls / calls avoided
(1 − rate), latency accounting per the Phase 14 convention (BM25 clock +
dense clock iff escalated; decision cost 0; means descriptive, rate is the
finding; no USD — null on retrieval-only rows).

## 15.11 Statistical tests

* Per arm (continuity with Phase 14): paired C-vs-A and C-vs-B on recall@5
  via `evaluation/statistics.py::compare_paired` (Wilcoxon + sign +
  paired effect, seed 20250101), Holm-adjusted within the pair;
  bootstrap 95% CI on mean paired C−A difference and on escalation rate
  (seed 20250115, 10 000 reps). Secondary: same for MRR.
* Random ablation per arm: `cheapfirst.random_escalation` (1000 draws,
  seed 20250101); report full distribution and `P(random ≥ adaptive)`.
* Powered primary for the selection question — combined randomization
  test: M = (adapt_after + adapt_before)/2; null draw i =
  (rand_after_i + rand_before_i)/2 with independent per-arm fixed-count
  draws at the same draw index (seed 20250101, 1000 draws);
  P_comb = mean(null ≥ M). One test over both arms (no multiplicity
  correction inside it); valid under query pairing because it compares
  the observed mean against its own randomization null without assuming
  arm independence.
* Replication support: combined cluster-bootstrap 95% CI (resample
  query ids with replacement, both arms together; seed 20250115, 10 000
  reps) on the combined adaptive−BM25 mean.
* C-vs-D is descriptive per arm (distribution + P, as Phase 10/14); the
  combined P_comb is the inferential statement.

## 15.12 Multiplicity correction

Family 1 (per arm, per metric): the C-vs-A / C-vs-B pair, Holm within the
pair (Phase 14 discipline, unchanged). Family 2: none — P_comb is a single
pre-registered test; the combined CI is estimation. No other comparisons
are tested; mechanism descriptives (§21) carry no p-values.

## 15.13 Success criterion

**SUCCESS** — all of: (a) combined adaptive R@5 ≥ combined dense-only R@5
− 0.02 (the Phase 14 margin, unchanged); (b) escalation rate ≤ 50% on
each arm; (c) TP ≥ 1 on each arm test set; (d) P_comb < 0.05 AND neither
per-arm P ≥ 0.5; (e) combined cluster-bootstrap 95% CI on adaptive−BM25
excludes 0 AND per-arm Δ(adapt−BM25) ≥ 0.

## 15.14 Failure criterion

**FAILURE** — any of: (i) TP = 0 on both arms with rate > 5% (spends
without catching); (ii) combined adaptive R@5 < combined BM25-only R@5
(dominated — the Phase 10 signature); (iii) P_comb ≥ 0.5; (iv) per-arm P ≥
0.5 on BOTH arms (random matches-or-beats everywhere).

## 15.15 Insufficient-evidence criterion

Otherwise: rule applied as frozen, arms complete, but the record is mixed
(e.g. P_comb significant without the dense margin, or benefit without the
ablation win). Includes any integrity failure that forces a protocol
deviation (reported, verdict capped here).

## 15.16 Stopping rule and power

Curate until 110 queries pass the gate (attempts, drops with reason codes,
and replacements logged; replacements fill the same quota slots before
any retrieval). Freeze. Run the four arms. Evaluate once.

A priori power (simulation over pooled Phase 14 test rows as the query
population, Phase-14 effect size, 300-draw ablations): combined test power
0.63 / 0.71 / 0.79 / **0.81 at n = 60 / 80 / 100 / 110**; expected ~20
oracle positives per arm at n = 110 (P5 ≈ 14). Per-arm ablation power at
n = 110 is ~0.63 — the reason the combined test is primary and per-arm
ablations are replication support. If the powered set yields far fewer
positives than expected (P5-style shortfall), that is reported, not
repaired — no top-up curation post-retrieval.

## 15.17 Integrity guards

* Pre-routing: reuse `cheapfirst.signal_value` allowlist (structural
  refusal of dense/hybrid/reranker/cross-arm/label quantities) + the
  Phase 14 guard tests, extended with a Phase 15 oracle-equivalence test
  (Phase 15 builder ≡ Phase 14 artifacts on Phase 14 inputs) and a
  frozen-threshold pin test.
* Split/allocation integrity: the evaluation script has no code path that
  alters the dataset after retrieval starts; curation scripts never call
  retrieval, embeddings, or signal code (asserted by test).
* Dataset-gate accommodation: `--allow-test-only` is added to
  `validate_phase7_dataset.py` with default-off semantics (existing S4
  behavior and its pinning test unchanged): it passes S4 iff calibration
  is empty AND test is non-empty, and records
  `single_split_test_only: true` plus the Phase 15 rationale in the gate
  artifact. Its use here is pre-registered by this section.
* Calibration/test separation is structural: there is no calibration set
  and no fitting code in Phase 15.

## 15.18 Provenance

Every artifact records: powered dataset path + sha256, gate artifact +
ledger hashes, corpus arm + manifests + collection names, oracle version
`phase14_cheapfirst_v1`, frozen-rule source
(`experiments/phase14/frozen_policy.json` + sha256), cost/latency
assumptions, all seeds (ablation 20250101, bootstrap/cluster 20250115),
software version (git commit), timestamps. Plots derive from written
artifacts only.

## Files

New: `data/evaluation/phase15_eval_v1.jsonl` (+ ledger — data, not code);
`src/adaptive_rag/evaluation/powered.py` (combined test, cluster
bootstrap; pure functions); `scripts/phase15_{build_oracle,evaluate,figures}.py`
(reuse `cheapfirst` + suite outputs; no retrieval code); new tests
(`tests/test_powered_confirmation.py` + validator-flag test); artifacts
(gitignored) `experiments/phase15/`; docs (this file,
`docs/phase15_results.md`, `docs/progress.md` entry). Small additive
change: `--allow-test-only` in `validate_phase7_dataset.py`. No changes
under `retrieval/`, `routing/`, `experiments/runner.py`, `evaluation/`
library code, or any Phase 8–14 artifact.
