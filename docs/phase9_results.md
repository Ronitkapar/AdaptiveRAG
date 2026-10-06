# Phase 9 — Final Report: Strategy Sensitivity (Mechanism, Not Correlation)

**Status:** COMPLETE — frozen final report.
**Official Phase 9 verdict:** `PROMISING_SIGNAL`
**Plan of record:** [`docs/phases/phase-9.md`](phases/phase-9.md) (§2 preregistration frozen; §8 amendments dated 2026-10-04).
**Artifacts:** `experiments/phase9/` (historical outputs; unmodified by this report).
**Scope of this document:** documentation freeze only. No retrieval, provider call,
statistical test, router experiment, or Phase 10 work was run to produce it.

> This report supersedes the earlier truncated draft of this file, which stated
> that Gates 9.4 and 9.5 were "not executed". All five gates (9.1–9.5) are
> executed and recorded below.

---

## 1. Phase 9 objective

**Research question:**

> Which measurable query–corpus properties explain per-query dispersion in retrieval outcome across strategies, and is that structure predictable from information available before paying for the expensive strategy?

Phase 9 investigated:

* **T-disp3** as the primary continuous target (max − min `recall_at_5` over `{bm25, dense, hybrid}`);
* **T-disp-mrr** as secondary where applicable (max − min `mrr` over the same three sound strategies);
* **oracle gain / needed / tie-break controls** where preregistered (`T-gain` qualitative only, never fitted; `T-needed` Gate 3 control; `T-tiebreak` artefact control);
* **BM25, dense, and hybrid retrieval** (the three sound strategies; the reranker is excluded from the target per Finding 3 and ADR-028);
* **frozen calibration/test split** (ADR-027; calibration n = 47, test n = 60);
* **12 final hypotheses** after the documented H5 withdrawal (8 disagreement + 4 score geometry; Holm denominator = 12).

Two separable questions, deliberately not collapsed:

* **mechanism** — what *explains* dispersion (Gates 9.1–9.2, 9.4);
* **operationalisation** — what is *knowable in advance* at a usable cost (Gate 9.3, and only after 9.2 passes).

Phase 10 asks whether a policy can exploit the mechanism. **Phase 9 builds no
router under any verdict** (per §2.10 and ADR-026).

---

## 2. Gate 9.1 — Audit

Gate 9.1 asked whether `T-disp3` is real or a measurement artefact. Summary:

* **Target definition.** `T-disp3` = max − min `recall_at_5` over `{bm25, dense, hybrid}` (primary). `T-disp3-mrr` secondary. `T-disp4` diagnostic only (H6-confounded by the reranker). `adaptive` excluded throughout (router output, not a selectable strategy).
* **Leakage audit.** Ground-truth information (`relevant_documents`, `relevant_sections`, `reference_answer`, and every metric derived from them) is used **only** to construct targets; no signal function receives an object carrying it. Enforced in code via `build_signal_table` signature (generalising the `test_sufficiency_check_uses_no_ground_truth` guard).
* **T-disp3 definition.** Frozen at k = 5 (the `top_k` the E1 arms ran at; §2.5). Sensitivity at k ∈ {3, 10} reported separately, never as headline.
* **Target was not a metric artefact.** `T-disp3 > 0` on 21/107 (`after`/shipping) and 29/107 (`before`/replication); `T-disp4 > 0` on 34/107 and 41/107; tie-break artefacts (H7 control) 73 and 66. The `T-needed` Gate 3 control reproduces AUC ≈ 0.53 (0/24 survive Holm), so the Phase 9 pipeline agrees with Phase 7 rather than contradicting it.
* **Reranker decomposition / ADR-028 handling.** Reranker strictly worst on 26/34 dispersed (`after`) and 30/41 (`before`); strictly best on 0 and 1. Decomposition counts overlap rather than partition (a query can be dispersed among the sound three *and* have the reranker last). This justifies excluding the reranker from the target.
* **Frozen split.** ADR-027 calibration/test respected; no threshold, model, or signal selection tuned on `test`.
* **Final 12-hypothesis family.** 8 disagreement (H1) + 4 score geometry (H4). Per-signal expected directions in §2.4.1 (including the `union_concentration`-positive / `score_decay_slope`-positive documentation correction). Mid-rank Spearman, seeds distinct from Phase 7, Holm within the family, replication per §2.7.1.
* **H5/query-intrinsic withdrawal and dated preregistration amendment.** H5 withdrawn by amendment dated 2026-10-04 (§8.7) because its four supposed representatives could not be identified from provenance (§8.5 repository-wide search: three versions of the phase-9 doc, plans, ADRs, commits — none names the four; the "4" likely conflates the "37 columns, four groups" split in `phase_7_closure_results.md` §3.2). Family 16 → 12; Holm denominator is 12. Phase 6 features remain as control only, reported separately and uncorrected.
* **Missing-data treatment.** `rank_correlation@bm25|dense` defined only where the two arms share ≥ 2 documents in top-5: n = 23 per arm (frozen rule §8.9: pairwise-defined analysis, no imputation, bootstrap/permutation on defined observations, reported with effective n as a limitation). BEFORE/AFTER defined-query sets differ (intersection 13 of 23; §8.10).
* **Rank-correlation n = 23 limitation.** Recorded as limitation (power 0.782 at |ρ| = 0.30 vs 0.994 at n = 107); the signal stays in the family of 12.
* **No new retrieval/API calls.** Gates 9.1–9.2 read already-persisted `rows.jsonl` / `traces.jsonl`; nothing under `src/adaptive_rag/retrieval/` changed.

Do not rewrite history: H5 was withdrawn because its four supposed
representatives could not be identified from provenance.

---

## 3. Gate 9.2 — Mechanism Screening

**Verdict: `mechanism_signal_identified`. Five of twelve signals survive all three
preregistered conditions** (`|ρ| ≥ 0.30`, `p_holm < 0.05` over the family of 12,
and replication on the other arm). Full per-signal table in
`experiments/phase9/gate9_mechanism_screening.json`.

| Signal | Shipping ρ | Shipping Holm p | Replication ρ | Result |
| ------------------------- | ---------: | --------------: | ------------: | ------ |
| jaccard@bm25\|dense | −0.454 | 0.0012 | −0.520 | PASS |
| jaccard@dense\|hybrid | −0.431 | 0.0012 | −0.373 | PASS |
| union_concentration | +0.454 | 0.0012 | +0.520 | PASS |
| distinct_doc_ratio@dense | +0.343 | 0.0012 | +0.406 | PASS |
| top1_agreement@bm25\|dense | −0.409 | 0.0024 | −0.492 | PASS |

Notes (all preregistered, none relaxed):

* **Holm denominator = 12** (amended from 16 by the dated H5 withdrawal, §8.7).
* **Threshold = |ρ| ≥ 0.30** (`MIN_ABS_RHO`), on the shipping arm.
* **Replication requires same sign and |ρ| ≥ 0.30** on the replication arm (§2.7.1); `p_holm` is derived once on shipping, not re-derived on replication.
* **Rank correlation had n = 23 and did not survive** (reported with effective n; reduced power is a limitation, not a re-litigation).
* **Score-geometry signals did not survive** (retained as registered controls; a family containing only signals that work is not a test).
* **No threshold was relaxed after observing results.** The pilot sweep (§7, k-sensitivity, single-arm +0.327 figure) is disclosed as exploratory; Gate 9.2 used the honest three-arm target (single-arm BM25 proxy +0.226 there — the pilot was inflated by reranker-driven dispersion).

**Interpretation, kept narrow.** Replicated association exists between
retrieval-result disagreement/coverage structure and per-query `T-disp3`. No
causal claim is made.

---

## 4. Gate 9.3 — Deployability

**Question.** Can any Gate 9.2 survivor inform a routing decision *before paying
for the expensive arm*? (§2.10 pass condition: a proxy from
{query-only ∪ single-arm ∪ hybrid-internal} holds out-of-sample on `test`;
calibration n = 47, test n = 60; `scripts/gate9_deployability_proxy.py`.)

### A. Pre-routing

**NO.**

No query-only feature reached |ρ| ≥ 0.30 on test. Every deployable query-only
feature reported; none selected, dropped, or promoted (no multiplicity family created).

Best:

`lexical_density = +0.194` (test; calibration +0.115).

Only query-only features (Phase 6 `QueryFeatures`, available at T0) can be
PRE_ROUTING. Those lived in family H5, withdrawn by amendment, so Gate 9.2
contains zero query-only candidates by construction — recorded so it is not
mistaken for a negative result about query features. The out-of-sample
measurement above independently confirms zero pre-routing signals.

The one preregistered hybrid-internal channel (`bm25_candidate_count` /
`dense_candidate_count`, `hybrid.py:71-76`) is constant 20/20 on all 107 traces:
available at zero extra cost but with no per-query variance, so no proxy can be
built from it.

### B. Single-arm

**YES.**

`distinct_doc_ratio@dense`

It can be calculated after dense retrieval returns (T3). §2.4-registered, Gate
9.2 survivor, no fitting: calibration ρ = +0.254, **test ρ = +0.410** (holds
|ρ| ≥ 0.30 out-of-sample). `distinct_doc_ratio@hybrid` (+0.307 / +0.368) is
reported for completeness but was **not** in the registered 12 and carries no
evidential weight for the verdict.

### C. Post-hoc

Four cross-arm signals (explanatory but cannot decide whether to invoke dense in
the first place):

* BM25/Dense Jaccard
* Dense/Hybrid Jaccard
* union concentration (computed from bm25 **and** dense top-5 IDs, `signals.py:299-304` — a two-arm quantity, not a corpus-level one)
* BM25/Dense top-1 agreement

Each requires the result sets of two strategies including the expensive arm, so
the information arrives after the choice is made. They explain *why* strategies
disagree, not *which to run*.

### Actual pipeline (timing reference)

Cost from the frozen Phase 7 cost table and `hybrid.py:71-76`
(the dense branch executes first, so hybrid inherits dense's cost):

| Stage | What happens | Cost (median) | External call? |
|---|---|---|---|
| **T0** | query string only; Phase 6 `QueryFeatures` | ~0 | no |
| **T1** | BM25 retrieval (local index) | ~1.7–2.3 ms | no |
| **T2** | query embedding | — | **yes (provider API)** |
| **T3** | dense retrieval (includes T2) | ~452–621 ms | **yes** |
| **T4** | hybrid = dense ‖ bm25 fused by RRF | ~456–721 ms | **yes** |

Therefore cross-arm disagreement that includes dense arrives **after** the
expensive embedding operation (T2/T3). `distinct_doc_ratio@dense` informs a
**second** decision (post-dense escalation), not whether to run dense.

**Gate 9.3 verdict: no pre-routing signal; one single-arm signal
(`distinct_doc_ratio@dense`) holds out-of-sample.**

---

## 5. Gate 9.4 — Explanation

**Objective.** Determine whether the five Gate 9.2 survivors represent a coherent
underlying phenomenon rather than five unrelated correlations.

> **Status of the computations below.** The inter-signal correlations and the
> rank-residual partial correlations are **exploratory/descriptive**. They were
> not preregistered hypotheses, carry **no p-values**, are **not** entered into
> any Holm family, and were not used to select or drop any signal. They are
> reported as explanation, not as further evidence of significance.

### Named mechanism

**Per-query retrieval disagreement / evidence diffuseness.** On queries whose
relevant evidence is not concentrated in one retrievable region, (a) different
retrievers surface different subsets of it — cross-arm disagreement rises — and
(b) a single retriever returns more redundant, lower-coverage hits. Both would
raise `T-disp3`.

### Mechanism wording (final)

> The surviving signals are consistent with a shared, multidimensional retrieval-disagreement/evidence-diffuseness structure.

That is:

* cross-arm disagreement/coverage quantities are strongly related (descriptive inter-signal Spearman, n = 107, after / before: J(BM25,DNS)–J(DNS,HYB) +0.776 / +0.710; J(BM25,DNS)–TOP1 +0.560 / +0.535; J(DNS,HYB)–TOP1 +0.472 / +0.374; DDR@dns against the cluster −0.49 to −0.55);
* `union_concentration` and `jaccard@bm25|dense` are algebraically dependent (both deterministic functions of |A∩B| and |A∪B| at fixed k; rank-identical up to reversal, Spearman −1.000 on both arms — one quantity measured twice, so the five survivors reduce to at most three distinct readouts);
* the remaining inter-signal correlations are not near-perfect (+0.37 to +0.78 range outside the algebraic pair);
* therefore the evidence supports a shared phenomenon, but not a single latent measurable quantity.

### Discriminating prediction — failed (Gate 9.4 criterion met)

If the mechanism were a **single latent disagreement dimension**, all four
cross-arm readouts would be near-perfectly rank-consistent, as the
UNION/Jaccard pair demonstrably are. They are not (+0.776, +0.560, +0.472 on
`after` leave genuine structure beyond one dimension). Per §2.10 ("a named
mechanism with a discriminating prediction it fails"), the criterion is met:
the phenomenon is real and shared, but **not a single measurable quantity**.

### DDR@dense relationship to the disagreement cluster

`distinct_doc_ratio@dense` takes only five values {0.2, 0.4, 0.6, 0.8, 1.0}
(mean 0.419; full coverage on 2.8% of queries) — a coarse, near-binary coverage
measure. It is moderately related to the disagreement cluster (ρ = −0.49 to
−0.55) without being redundant. Residualising it on the disagreement axis
(descriptive rank-residual partial correlation, **not a significance test**):

| | raw ρ(DDR, T-disp3) | partial ρ given UNION | retained |
|---|---|---|---|
| after | +0.343 | **+0.121** | 35% |
| before | +0.406 | **+0.229** | 57% |

These partial analyses are descriptive/exploratory and not significance tests:
the non-zero residual cannot be called significant, and its existence cannot be
called a refutation.

The data cannot determine whether DDR is:

* a noisy single-arm proxy for the broader disagreement phenomenon, or
* a distinct single-arm coverage quantity.

Both readings (degraded proxy vs distinct quantity) remain consistent with the
evidence.

### Gate 9.4 verdict

* **Supported** — the cross-arm signals are manifestations of a common,
  replicated retrieval-disagreement structure rather than unrelated correlations.
* **Supported (§2.10 criterion met)** — a mechanism can be named and its
  strongest discriminating prediction (single latent dimension) **fails**.
* **Not supported / unresolved** — whether `distinct_doc_ratio@dense` is a proxy
  for that structure or a distinct quantity; and whether it could support
  post-dense escalation (`T-gain` has 4–6 members, qualitative only, and the
  Phase 8 oracle artifact is aggregate-only with no per-query gain, so the link
  cannot be examined on this data).

Gate 9.4 explains the observed association. It does not establish causality,
and it does not demonstrate that any escalation policy would improve end-to-end
retrieval quality. No intervention was run.

---

## 6. Gate 9.5 — Methodological Audit

An audit of Gate 9.5 only. No threshold, Gate 9.2 result, or Gate 9.3 result was
altered. Three defects were found in the original Gate 9.5 write-up. **The
verdict is unchanged**, but one supporting number is withdrawn as evidence.

### Defect 1 — Ridge (WITHDRAWN as evidence)

Ridge was not preregistered.

Reasons:

* feature set assembled inside the gate script (`scripts/gate9_deployability_proxy.py`);
* only 12 signals were registered (§2.4 as amended);
* `distinct_doc_ratio@hybrid` was not registered (yet was drawn into the model);
* 13 query-only features were used despite the stale "eight" wording (§7.5) — the preregistration never froze a query-only feature count;
* λ = 1 was copied from Gate 3's frozen ridge, a constant belonging to a different phase's experiment;
* feature set was not frozen before test evaluation (assembled in the same pass that reported on `test`; §2.8 forbids tuning on `test`);
* therefore Ridge test ρ = +0.369 is WITHDRAWN as evidence.

> Ridge has no role in the official Phase 9 verdict.

The verdict rests on the single unfitted registered signal below, computed
independently of Ridge. Removing Ridge changes nothing. The Ridge number is
retained in `experiments/phase9/gate9_deployability_proxy.json` labelled
exploratory.

### Defect 2 — evaluate_gate() (false documentation claim corrected)

There is NO Phase 9 `evaluate_gate()` function.

The earlier statement that the verdict was mechanically emitted by such a
function was incorrect (`evaluate_gate()` exists only in
`gate3_signal_exhaustion.py` and `oracle_routing_ceiling.py`, not for Phase 9).

> The verdict follows from the preregistered conditions, but no Phase 9 function mechanically emitted it.

### Defect 3 — §2.10 internal tension (recorded, not silently rewritten)

The actual tension:

* §2.10's pass condition admits `{query-only ∪ single-arm ∪ hybrid-internal}`;
* but its question wording asks whether the signal is usable at a "cheap decision point";
* post-dense escalation is deployable under the enumerated class but does not avoid the dense cost (T2/T3 embedding already paid).

Recorded here as an unresolved preregistration wording ambiguity. The
preregistration is not silently rewritten: on the text's own cross-arm vs
single-arm line, `distinct_doc_ratio@dense` passing is a legitimate Gate 9.3
pass — but the "cheap decision point" question cell is not satisfied by a
post-dense signal. Both statements are true and both are recorded.

### Supporting checks (unchanged)

* Power for `MIN_ABS_RHO` = 0.30 (Fisher z, two-sided, α = 0.05): n = 107 →
  0.994 (n = 60 → 0.954; n = 47 → 0.920); `INSUFFICIENT_DATA` not triggered
  except the already-reported n = 23 rank-correlation limitation (power 0.782).
* Falsification conditions (§2.11): `T-needed` control reproduces AUC ≈ 0.53;
  no survivor requires ground truth; no sign reversals (all five met §2.7.1);
  `T-disp3` not a tie-break artefact (Gate 9.1 audit).
* Verdict alternatives are not mutually exclusive (§2.10 specifies no precedence
  rule); each is excluded by direct evaluation of its stated condition.

### Verdict table

| §2.10 verdict | Condition | Met? |
|---|---|---|
| `NO_RELIABLE_SIGNAL` | no signal satisfies §2.7 on both arms | **no** — five do |
| `INSUFFICIENT_DATA` | power < 0.8, or target is an artefact | **no** — power 0.994, target audited |
| `WEAK_SIGNAL` | replicated, no deployable proxy | **no** — proxy holds at +0.410 |
| **`PROMISING_SIGNAL`** | replicated **and** deployable proxy holds out-of-sample | **yes** |

## **`PROMISING_SIGNAL`**

The verdict is supported by the preregistered `distinct_doc_ratio@dense`
single-arm signal:

* Gate 9.2: shipping ρ = +0.343, replication ρ = +0.406, Holm p = 0.0012
* Gate 9.5: calibration ρ = +0.254, test ρ = +0.410, no fitting/model required,
  registered signal, single-arm observable after dense retrieval.

Do NOT use the withdrawn Ridge result as evidence.

---

## 7. Phase 9 final evidence matrix

| Question | Result |
| --------------------------------------------------------- | ------------------------------------ |
| Is strategy dispersion real? | YES |
| Does replicated association exist? | YES |
| Is there one single latent disagreement quantity? | NO / not supported |
| Are cross-arm disagreement signals available pre-routing? | NO |
| Is there a query-only pre-routing signal? | NO |
| Is there a single-arm signal? | YES — DDR@dense |
| Does DDR generalize OOS? | YES — test ρ +0.410 |
| Does DDR prove escalation improves retrieval? | NO |
| Was an intervention/policy evaluated? | NO |
| Is a pre-routing router justified? | NO |
| Is post-dense escalation worth investigating? | YES, as a Phase 10 research question |
| Official Phase 9 verdict | PROMISING_SIGNAL |

---

## 8. Final scientific conclusion

> Phase 9 establishes that per-query retrieval-strategy dispersion is real and reproducible, and that it is associated with a shared but multidimensional retrieval-disagreement/evidence-diffuseness structure. However, the strongest signals are cross-arm quantities that become observable only after the expensive dense path has already been executed. No query-only pre-routing signal met the preregistered effect threshold. One registered single-arm signal, dense distinct-document ratio, generalized out-of-sample and therefore provides evidence for a potential post-dense escalation decision. Whether acting on that signal improves retrieval outcome remains untested because no intervention was run and the available Phase 8 oracle artifact lacks the required per-query gain information.

> Therefore Phase 9 does not justify a cheap pre-routing router. It does justify investigating whether a two-stage retrieval policy can use dense-output structure to decide when additional retrieval effort is worthwhile.

What the verdict does and does not mean:

* **It means:** a replicated association exists (9.2), a named mechanism met its
  discriminating-failure criterion (9.4), and at least one deployable
  (single-arm) proxy generalises out-of-sample (9.3/9.5).
* **It does not mean a router is warranted.** The qualifying proxy is available
  only *after* dense has run and informs a **second** decision. Zero pre-routing
  signals exist (best query-only test |ρ| = 0.194). This is associational
  evidence, not causal, and no intervention was run.

---

## 9. Limitations

* Rank-correlation effective n = 23 (pairwise-defined; §8.9–§8.10).
* BEFORE/AFTER rank-correlation subsets differ substantially (intersection 13 of 23).
* DDR has only five possible values {0.2, 0.4, 0.6, 0.8, 1.0} — coarse, near-binary.
* DDR calibration ρ (+0.254) is below 0.30 even though test (+0.410) exceeds it — a single-split pass, not a stable two-split margin.
* No per-query T-gain oracle artifact (T-gain 4/6 members; Phase 8 oracle aggregate-only; §2.7 forbids fitting on T-gain).
* No intervention — no escalation policy evaluated; no causal claim.
* §2.10 wording ambiguity (enumerated class admits post-dense single-arm; question cell asks for a "cheap decision point") — recorded unresolved.
* §2.10 verdict alternatives not mutually exclusive; no precedence rule; no Phase 9 emitter function.
* Moderate survivor explanations remain unresolved (DDR-as-proxy vs DDR-as-distinct-quantity; Gate 9.4 partials descriptive only).
* `T-disp3` target is continuous but concentrated (21/107 and 29/107 dispersed); `T-gain` too small to characterise or fit.
* All Gate 9.4 inter-signal/partial analyses exploratory/descriptive, uncorrected, no p-values.

---

## 10. Phase 10 boundary

Phase 10 is not designed here. No architecture, policy, threshold, or router is
specified.

> The natural next research question is whether `distinct_doc_ratio@dense` can support a decision-theoretically justified post-dense escalation policy, provided an appropriate per-query gain/cost oracle can first be constructed.

Such a policy's improvement is not claimed. Establishing it requires a
per-query gain/cost oracle Phase 8 did not produce, plus an intervention Phase 9
did not run. No router is built, trained, tuned, or deployed under the Phase 9
verdict.

---

## Freeze note

* Phase 9 = COMPLETE / `PROMISING_SIGNAL` (Gates 9.1–9.5 executed).
* This file is the complete final report; no stale "not executed" wording remains herein.
* Ridge (test ρ = +0.369) is marked withdrawn wherever it appears.
* No Phase 9 `evaluate_gate()` function is claimed.
* §2.10 ambiguity is recorded, not rewritten.
* Gate 9.4 exploratory analyses are marked descriptive.
* Historical experiment artifacts under `experiments/phase9/` are unmodified.
* No new experimental artifact was generated by this documentation freeze.
