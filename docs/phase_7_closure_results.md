# Phase 7 — Closure Results

> **Research question.** Does query-aware adaptive routing earn its complexity
> over a fixed retrieval strategy, on this benchmark?
>
> **Answer.** The experiments do not establish a practical adaptive routing
> policy under the evaluated conditions. The evaluation found limited
> query-level retrieval heterogeneity: an oracle can improve over the best fixed
> strategy on a small subset of queries. However, this headroom is not reliably
> identifiable from the currently observable query and retrieval-feedback
> signals in out-of-sample evaluation. Consequently, the experiments do not
> establish a practical adaptive routing policy under the evaluated conditions.

This document is the final Phase 7 result record. It closes the phase by
separating three claims that are easy to conflate, each supported by a different
strength of evidence:

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Oracle-level query-specific headroom **exists** | Established, small | Gate 2 (§2) |
| 2 | That headroom is **detectable** from observable signals | **Not established** at this sample size | Gate 3 (§3) |
| 3 | A router can **exploit** it | Not established, and not attempted | §5, §6 |

Claim 1 being true is not evidence for claims 2 or 3. An oracle is an upper bound
computed with a signal that does not exist; it bounds routing without evidencing
it.

---

## 1. Scope and integrity statement

- **No routing implementation was changed to produce any result in this
  document.** `src/adaptive_rag/routing/` is byte-identical to its Phase 6 state.
  The two gate scripts read the router; they never configure it.
- **No learned router was trained or deployed.** Gate 3 fits a ridge logistic
  regression purely as a *measuring instrument* to ask whether signal exists. No
  decision rule is derived from it and no router is configured from it.
- **The corpus was not rebuilt.** The 42 spliced section paths are unchanged and
  remain outside the primary routing closure experiment.
- **`strategy_cost_ms` was not changed.** It remains unfrozen.
- **ADR-028 was not implemented.** Its status is DEFERRED — see §6.

---

## 2. Gate 2 — the oracle ceiling, over selectable strategies only

**Question.** Does routing by query characteristic have *any* headroom over the
best fixed strategy?

**Method.** Offline join over the clean E1 arms, no index and no API, fully
deterministic. The **selectable** set is the frozen `available_strategies` =
`{bm25, dense, hybrid, hybrid_rerank}`. The `adaptive` arm is deliberately
excluded: it is the shipped router's own output over the benchmark, not a
strategy a router may select, so a ceiling permitted to select it would partly
measure the router against itself.

**Correcting an earlier defect does not overturn the result.** The first version
of this ceiling included `adaptive` in its selectable set. Re-run without it,
recall@5, MRR and both gate verdicts are **unchanged** — `adaptive` was credited
on 6 (`after`) / 1 (`before`) queries but only ever as a *tie* at the recall
maximum, so it never raised the ceiling. The leak was methodological, not an
inflation; removing it **confirms** the published numbers.
| | `after` | `before` |
|---|---|---|
| oracle Recall@5 | 0.9206 | 0.9486 |
| best fixed (`dense`) Recall@5 | 0.8988 | 0.9143 |
| **oracle − best fixed** | **+0.0218** | **+0.0343** |
| paired bootstrap 95% CI on that delta | **[0.0031, 0.0467]** | **[0.0093, 0.0670]** |
| queries carrying the gain | **4 / 107 (3.7%)** | **6 / 107 (5.6%)** |
| frontier(ε=0.01) latency saving | −842.11 ms | −398.21 ms |

**What this establishes.** Oracle-level query-specific headroom is real and its
sign is solid — the CI excludes zero on both arms.

**What it does not establish.** Three qualifications travel with the number:

1. **The advantage is highly concentrated.** It rides on 4 of 107 queries
   (`before`: 6). `recall_at_5` takes only four distinct values because 92 of 107
   queries carry a single relevant document, so the strategies tie at the maximum
   on 98 queries and the `after` delta clears its +0.02 threshold by 0.0018 —
   about two queries.
2. **Tie-breaking materially affects which strategy receives oracle credit.**
   Recall@5 is tie-break-invariant by construction, but the credited arm and its
   latency are not. Oracle MRR is **0.8224** under the `latency` tie-break and
   **0.9420** under `mrr` — a 0.1195 swing that *flips its sign* relative to the
   best fixed strategy (−0.0421 vs +0.0775). The earlier observation that "the
   oracle costs MRR" is therefore a property of the latency tie-break alone and
   **must not be reported as an unconditional finding**.
3. **A secondary efficiency observation, not a routing result.** A rank-aware
   frontier (ε = 0.01, MRR tolerance 0.05) retains **72.7%** of the recall-only
   frontier's latency saving (−611.93 ms of −842.11 ms) while carrying MRR 0.9420
   in place of 0.8224. The two objectives are not in zero-sum conflict at this
   tolerance. This is an efficiency/quality trade-off *about the ceiling*, and is
   **not** evidence that a deployable router works.

**Pre-registered gate outcome:** `routing_has_headroom` on both arms — the
quality branch. The *latency* branch fails: the needed-vs-sufficient split is not
predictable from category or any recorded sufficiency signal.

Artifact: `experiments/phase8/oracle_ceiling_v2.json` (sha256 `8e48e22b…`),
reproducible via `python scripts/oracle_routing_ceiling.py`. 48 tests.

---

## 3. Gate 3 — full observable signal exhaustion

**Question.** Can the information a router can actually observe tell it, out of
sample, which queries are the few where a different retrieval strategy is
actually beneficial?

### 3.1 Methodology

Offline and deterministic; no index, no API. Labels are `needed` / `sufficient`
at ε = 0.01 over the four selectable E1 arms — the *same* definition Gate 2 used,
so the two gates agree. `needed` means nothing else comes within ε of the
best-fixed reference, i.e. the reference is the only acceptable answer.

**Leakage control.** Features are read **only** from the `adaptive` arm's traces
— the analyzer's `QueryFeatures`, the sufficiency checker's scalars, and the
routing decision's own scores and confidence. No per-strategy retrieval outcome
enters the feature side; a test asserts no feature row carries `recall_at_5`,
`mrr`, `hit_at_5` or `ndcg_at_5`. Labels, by contrast, are derived from all
selectable arms, because that is the target and it is only available offline.

**Measuring instrument.** Ridge logistic regression by IRLS (λ = 1,
### 3.2 Feature set — 37 columns, four groups, no new features introduced

| Group | n | Features |
|---|---|---|
| **Query features** | 13 | `query_length_words`, `query_length_chars`, `content_term_count`, `lexical_density`, `entity_indicator_count`, `entity_ratio`, `technical_term_count`, `technical_ratio`, `semantic_indicator_count`, `semantic_ratio`, `concept_count`, `comparison_indicator_count`, `complexity_score` |
| **Retrieval feedback** | 5 | `result_count`, `coverage`, `top1_coverage`, `sufficiency_score`, `initial_result_count` |
| **Routing signals** | 6 | `routing_confidence`, `score_bm25`, `score_dense`, `score_hybrid`, `score_hybrid_rerank`, `evidence_margin` |
| **Initial strategy + categoricals** | 13 | `initial_strategy` (4 levels), `question_type` (7), `multi_concept` (2), one-hot |

Retrieval-feedback features are observable only *after* paying for stage one,
which is how a real router would use them.

### 3.3 Train / calibration / test separation

Fit on the frozen **`calibration`** split (**n = 47**); every reported
out-of-sample number is scored on the frozen **`test`** split (**n = 60**). This
is the benchmark's own split (ADR-027), not one drawn for this study.

### 3.4 Statistical tests, pre-registered before execution

- **Primary:** permutation test of the **whole procedure** — labels permuted
  across all queries, then standardise → fit → score → AUC re-run each time.
  10,000 resamples, seed 20250103. (Permuting after the model is fitted would
  test a weaker null than the claim rests on.)
- **Effect size:** AUC with a stratified percentile bootstrap CI (10,000
  resamples).
- **Screen:** one feature at a time, two-sided permutation on |AUC − 0.5|,
  **Holm-Bonferroni** across a family of **24**.
- **Power:** Hanley-McNeil analytic.

**Pre-registered decision rule.** Passes only if, on **both** corpus arms:
(i) permutation p < 0.05, (ii) test AUC ≥ 0.65, (iii) bootstrap 95% CI lower
bound > 0.50, (iv) at least one feature survives Holm at 0.05.

### 3.5 Results

| | `after` | `before` |
|---|---|---|
| positives (`needed`) / negatives | **8 / 99** (base rate 7.5%) | **8 / 99** |
| positives in calibration / test | **2 / 47** and **6 / 60** | **3 / 47** and **5 / 60** |
| **out-of-sample AUC** | **0.5340** | **0.5345** |
| bootstrap 95% CI | **[0.2469, 0.8086]** | **[0.2000, 0.8691]** |
| permutation p | 0.4016 | 0.4087 |
| features surviving Holm (of 24) | **0** | **0** |
| minimum detectable AUC at 80% power | 0.825 | 0.850 |

**All four pre-registered conditions fail on both arms.** The two arms agree
### 3.6 Four results that matter more than the p-values

**(a) The overfitting signature.** Fit-split AUC is **1.0000** (`after`) and
**0.9621** (`before`) against out-of-sample **0.534** on both. With 2–3 positives
the model separates the fit split perfectly and generalises to nothing. Reading
the fit-split number as the result would have inverted the conclusion entirely —
this is precisely why the out-of-sample protocol is mandatory here.

**(b) Power is the binding constraint, so this is a power statement.** At 6
positives against 54 negatives, a true AUC of 0.60 has 12% power to produce a CI
excluding chance; 0.80 has 76%; only **≥ 0.825** reaches 80%. The scientifically
correct reading is therefore:

> **No reliable out-of-sample routing signal was detected at the available
> sample size.**

This is **not** a demonstration that no signal exists. It is a statement about
what 8 positives can support. The distinction is preserved deliberately and must
not be collapsed into "there is no signal".

**(c) The retrieval-feedback signal is inverted, on both arms.** `top1_coverage`
is the strongest feedback feature and is *inversely* associated with needing the
reference: AUC 0.346 (`after`) / 0.357 (`before`), i.e. oriented 0.654 / 0.643.
Mean `coverage` on the 8 positives is **0.803** versus **0.808** on the 99
negatives. The sufficiency check reports "terms found" on precisely the queries
where the cheap stage found the terms *and the reference was still the only
acceptable answer*. This is the same inversion Gate 2 recorded in the
detectability block, now measured out of sample. `score_dense` behaves similarly
(AUC 0.375 / 0.323). This is a substantive negative about the *existing* signals,
not merely a small-sample artefact.

**(d) The strongest apparent feature is unstable across corpus arms.**
`content_term_count` leads the `after` arm (AUC 0.711, raw p = 0.045 — the only
raw p below 0.05 anywhere in the study) and collapses to AUC 0.562 in `before`.
The two arms disagree on which feature ranks first. A feature that does not
replicate across two corpora built from the same PDFs is not a basis for a
routing policy.

### 3.7 Which queries drive the result

Per-query evidence for all 107 queries is preserved in the artifact (split, label,
all 37 feature values, oracle gain, credited winner, model score). The 8
positives are **heterogeneous**: they span `factual`, `conceptual`, `comparative`,
`multi_document`, `terminology` and `fine_grained`; `query_length_words` ranges
19–66 and `coverage` 0.53–1.00. No single category, length band, or feature
separates them.

Artifact: `experiments/phase8/gate3_signal_exhaustion.json` (sha256 `f1f8736f…`),
reproducible via `python scripts/gate3_signal_exhaustion.py`. 32 tests.

---

## 4. Established

- **Dense is the strongest fixed retrieval strategy** under the evaluated
  benchmark (Recall@5 0.8988 `after` / 0.9143 `before`).
- **Hybrid and adaptive are equivalent on observed shipped routing behaviour**
  because the router effectively did not route: the sufficiency gate never fired,
  so `adaptive` collapsed onto hybrid (identical Recall@5 and Hit@5 on 107/107).
  This is an artifact of a gate that never opens — it is *not* evidence that
  routing is unnecessary.
- **Reranking is not competitive** under the measured conditions: last on every
  quality metric, at ~6–7× hybrid's median latency.
- **The corrected oracle shows a small amount of query-level retrieval
  heterogeneity / headroom** (+0.0218 / +0.0343 recall@5 over the best fixed
  strategy, CIs excluding zero).
- **That headroom is highly concentrated** — 4–6 of 107 queries.
- **The full currently observable feature set does not reliably identify the
  relevant queries out of sample** (AUC ≈ 0.534, 0/24 surviving Holm, on both
  arms).

## 5. Not established

- **That adaptive routing can improve over the best fixed strategy.**
- **That the current router can exploit the oracle headroom.**
- **That a learned router would improve performance.** Gate 3 does not license
  one: the rule-based router's failure is not evidence that a learned router
  would succeed, and a learned router on this benchmark would be fitted on 8
  positives and inherit the fit-split AUC of 1.0 with none of it surviving
  held-out data.
- **That no future feature set could ever identify the useful subset.** A larger
  or more informative feature set is not excluded by this result; it is simply
  untested here.

## 6. ADR-028 — status: DEFERRED for Phase 7

ADR-028 makes two recommendations: remove the rerank rung, and recalibrate the
sufficiency threshold. **Neither is implemented, and neither is promoted on the
basis of this study.** The conservative resolution:

- The existing router **does** have a configuration and ladder problem: the
  `sufficiency_threshold=0.5` gate sits above the observed sufficiency range
  (minimum 0.6364 test / 0.65 calibration), so escalation cannot fire; and the
  ladder terminates on a net-negative rung.
- **However, Gate 3 failed to establish that the observable features contain
  reliable predictive signal.** Therefore changing thresholds, the ladder, or
  reranking **cannot currently be justified as a way to exploit a demonstrated
  routing signal**, because no such signal has been demonstrated.
- Recalibrating the threshold would make the gate *fire*, but firing on a signal
  that does not out-of-sample predict the target would produce escalation without
  benefit — and under ADR-028's rung removal, escalation from hybrid would be a
  no-op by construction.
- **No further routing sweep was run to search for a positive result.** Doing so
  would be a multiple-comparisons exercise over configurations looking for a
  favourable p-value, which is exactly what the pre-registered gates exist to
  prevent.

ADR-028 is therefore recorded as **DEFERRED — not adopted for Phase 7**,
superseding its prior "RECOMMENDED — NOT IMPLEMENTED" status. It is not
rejected on the merits: both findings remain measured facts. It is deferred
because the action it implies is not currently justified by demonstrated
predictive signal.

## 7. Measurement limitations

- **Small number of positive queries.** 8 `needed` of 107 (7.5%), splitting
  2–3 calibration / 5–6 test. This caps everything below.
- **Limited statistical power.** Only a true AUC ≥ 0.825 (after) / 0.850 (before)
  could clear chance at 80% power. The Gate 3 null is a power statement.
- **nDCG@5 and precision@k are blocked** by section-label corruption: 47 of 140
  gold section labels are spliced and 40 no longer resolve on the fixed corpus.
  Reported as *not measured* rather than computed on labels of unknown
  provenance. Recall@5 / MRR / Hit@5 are label-driven and sound.
- **E8 document-level analysis** is valid where already established.
- **Only 22 of the 47 corrupt labels are safely repairable.** 15 of the 37
  high-confidence candidates have no clean reconstruction and 3 resolve only to
  paths that are themselves spliced. Root cause is upstream: the fixed corpus
  still carries **42 spliced section paths**. **No labels were invented or
  rewritten.**
- **Latency claims need a control arm.** `total_latency_ms` alone cannot
  separate provider variance from system cost on this hardware; the `bm25`
  no-API control is required for any future latency comparison.

## 8. Reproducibility

| Item | Value |
|---|---|
| Branch / tag | `phase7-closure` / `phase7-closure-freeze-01` |
| Gate 2 artifact | `experiments/phase8/oracle_ceiling_v2.json` — sha256 `8e48e22bb3b02a8fc58dcb3df4d6cccf27a7e95566606660e85dd428ae971e22` |
| Gate 3 artifact | `experiments/phase8/gate3_signal_exhaustion.json` — sha256 `f1f8736f8e9b5f483c84f15a4494f98974191cc08c0d881a8736b361126bcc53` |
| Command (Gate 2) | `python scripts/oracle_routing_ceiling.py` |
| Command (Gate 3) | `python scripts/gate3_signal_exhaustion.py` |
| Determinism | Both byte-identical across runs apart from `generated_at` |
| Input snapshot | All 24 recorded SHA-256 digests verified; see `docs/phase7-closure-evidence.md` §7 |
| Test suite | 882 passing, 2 deselected |

Both gate scripts reuse the Phase 8 arm loader, so the contamination guard and
the query-set equality assertion that caught the Phase 8 measurement-integrity
incident apply to both gates without being restated. A short, contaminated, or
mismatched arm **aborts** the analysis rather than producing a number.

## 9. What remains open

1. **A larger benchmark.** The binding constraint on every routing question here
   is 8 positives. Any future routing study should be sized for the effect it
   aims to detect before it is run.
2. **Feature sets that are not query-intrinsic.** Gate 3 tested everything the
   router can see about the *query* and about the *first stage's* output. A signal
   in neither place — for example one requiring a cheap probe retrieval, or
   corpus-level statistics about the query's terms — is untested, not excluded.
3. **Unblocking nDCG@5 / precision@k** requires fixing the corpus's 42 spliced
   section paths first; label repair alone cannot make the metric trustworthy.
4. **ADR-028** remains available if a future study demonstrates predictive
   signal, or if the rerank rung is reconsidered on independent grounds.
5. **A latency comparison on this hardware** requires a no-API control arm and a
   stage-level breakdown.

No further routing experiment is scheduled or recommended on the current
evidence.
