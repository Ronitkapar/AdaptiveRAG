# Phase 9 — Strategy Sensitivity: Mechanism, Not Correlation

**Status:** COMPLETE — Gates 9.1–9.5 executed; verdict `PROMISING_SIGNAL`. Final report: `docs/phase9_results.md`. (§2 below remains the frozen preregistration as written before any Phase 9 measurement existed.)
**Phase:** 9
**Purpose:** Determine *why* a small set of queries disperse in retrieval outcome
across strategies, and whether any of that structure is obtainable at a point
where acting on it is still cheap.
**Pre-registration:** §2. Written before the Phase 9 modules existed and before
any Phase 9 number was computed. Nothing in §2 may be edited after results
exist; a change to a pre-registration is a new, later document.

---

## 0. Why this phase exists (three findings, each from existing evidence)

This section states what motivated the phase. It is **not** a pre-registration:
the thresholds, seeds, targets and decision rules that make the results
interpretable are in §2, and those are the frozen part.

**Finding 1 — Gate 3's target label is disjoint from the queries that carry the
headroom.** Gate 3 labelled a query `needed` when no alternative came within
ε = 0.01 of the reference strategy, and trained on that. A query carries oracle
gain only when some strategy *beats* the reference. Those two predicates are
mutually exclusive. Cross-tabulating the label against `oracle_gain` in the
Gate 3 artifact itself (`experiments/phase8/gate3_signal_exhaustion.json`):

| | `needed` & gain > 0 | `sufficient` & gain > 0 |
|---|---|---|
| `after` | **0** | 4 |
| `before` | **0** | 6 |

All 8 positives have `oracle_gain == 0.0`; every gain-carrying query sits in the
negative class. Gate 3's AUC ≈ 0.53 and its "retrieval-feedback signals run the
wrong way" inversion are therefore **consistent with a mis-specified target**.
Phase 7's conclusion stands as written — it is a statement about the `needed`
label — and Phase 9 does not revise it. It observes that the label does not
answer the question Phase 7 posed.

**Finding 2 — the residual target is larger than 8.** Dispersion of recall@5
across the four selectable strategies exceeds zero on 34/107 queries (`after`)
and 41/107 (`before`). A continuous target of that size supports rank-correlation
testing at n = 107, which an 8-positive binary label never could.

**Finding 3 — most 4-arm dispersion is the reranker dragging.** On 26 of the 34
dispersed queries (`after`) and 30 of 41 (`before`), the reranker is *strictly
worse* than every other strategy; it is strictly best on 0 and 1 respectively.
ADR-028 already adjudicated that rung as net-negative. Measuring sensitivity
across four arms would therefore mostly re-measure a closed finding. Dispersion
over the three sound strategies alone is **21/107 (`after`)** and **29/107
(`before`)** — that is this phase's target.

---

## 1. Objective

**Which measurable properties explain the residual per-query dispersion in
retrieval outcome across the three sound strategies, and is any of that structure
obtainable from information available at a point where acting on it is still
cheap?**

Two separable questions, deliberately not collapsed:

* **mechanism** — what *explains* dispersion (Phase 9, Gates 9.1–9.2, 9.4);
* **operationalisation** — what is *knowable in advance* at a usable cost
  (Gate 9.3, and only after 9.2 passes).

Phase 10 asks whether a policy can exploit the mechanism. Phase 9 builds no
router under any verdict.

---

<!-- ANCHOR_PREREG -->
## 2. Pre-registration

> Written before the Phase 9 modules existed and before any Phase 9 number was
> computed. Nothing in this section may be edited after results exist; a change
> to a pre-registration is a new, later document.

### 2.1 Design

Phase 9 is an **explanatory measurement study**. It reads already-persisted
retrieval outcomes and computes targets and candidate signals from them. It runs
**no retrieval**, builds **no index**, and makes **no provider call** in
Gates 9.1–9.2. Nothing in `src/adaptive_rag/retrieval/` changes in this phase.

This is a deliberate contrast with Phase 8, which required re-extracting a corpus
and re-running 29 arms. Phase 9's inputs already exist: `rows.jsonl` carries
`retrieved_document_ids` and `retrieved_chunk_ids` per arm, and `traces.jsonl`
carries `score`, `rank` and `text` per result, on 107/107 clean traces in all
five arms of both corpus versions.

### 2.2 Data (frozen)

| Input | Path | Role |
|---|---|---|
| Benchmark | `data/evaluation/phase7_eval_v1.jsonl` (n=107) | query IDs, category, split |
| Shipping arm | `experiments/phase8/combined/p8a_e1` | primary |
| Replication arm | `experiments/phase8/combined/p8b_e1` | replication |

Selectable strategies for every target are the **frozen** `available_strategies`
= `{bm25, dense, hybrid, hybrid_rerank}`, as in Gate 2 and Gate 3. `adaptive` is
excluded throughout: it is the shipped router's own output, not a strategy a
router may select.

### 2.3 Targets — registered before measurement

| Target | Definition | Role |
|---|---|---|
| `T-disp3` | max − min `recall_at_5` over `{bm25, dense, hybrid}` | **Primary** |
| `T-disp3-mrr` | max − min `mrr` over `{bm25, dense, hybrid}` | **Secondary** |
| `T-disp4` | max − min `recall_at_5` over all four selectable | Diagnostic only (H6-confounded) |
| `T-tiebreak` | all four within ε of each other | Artefact control (H7) |
| `T-needed` | Gate 3's label, recomputed | Control — must reproduce AUC ≈ 0.53 |
| `T-gain` | `oracle_gain` > 0 (4 / 6 queries) | **Qualitative only — never fitted** |

`T-disp3` is primary because it excludes the reranker per Finding 3.
### 2.4 Candidate signals — the registered family

Every signal below is fixed in advance. A signal not listed here may not enter
the analysis as a candidate; adding one after seeing results requires a new,
later document.

| Family | Signals | Expected direction |
|---|---|---|
| **Disagreement (H1)** | `jaccard_k(bm25,dense)`, `jaccard_k(bm25,hybrid)`, `jaccard_k(dense,hybrid)`, `union_concentration`, `distinct_doc_ratio_k(bm25)`, `distinct_doc_ratio_k(dense)`, `top1_agreement`, `rank_corr_k(bm25,dense)` | **per-signal — see §2.4.1.** Not a single blanket direction. |
| **Score geometry (H4)** | `score_gap_top1_top2(bm25)`, `score_gap_top1_top2(dense)`, `score_decay_slope(bm25)`, `score_decay_slope(dense)` | **per-signal — see §2.4.1.** |
| **Query-intrinsic (H5)** | **UNRESOLVED — see §8 Issue C.** §2.4 originally read "the 37 Phase 6 `QueryFeatures` columns as used by Gate 3" here while the arithmetic below read "4 query-intrinsic representatives". 8 + 4 + 37 ≠ 16. | either |
| **Control** | `T-disp4`, reranker decomposition, `T-tiebreak` distribution | — |

**Declared family size: 16** (8 disagreement + 4 score geometry + 4
query-intrinsic representatives). Recorded on every test. The Phase 6 features are
reused as a *control*, not re-tested as new hypotheses: Gate 3 already found
0/24 surviving Holm, so a positive here would be a red flag for a bug, not a
discovery.

> **AMENDED (§8.7, 2026-10-04): the declared family size is 12.** The "16" above
> is preserved as originally written; it is superseded. H5 is withdrawn entirely
> and the Gate 9.2 hypothesis family is **8 disagreement + 4 score geometry = 12**.

> **The arithmetic above does not reconcile with the table.** 8 + 4 + 37 = 49,
> not 16. The *intent* recorded in the sentence above — that the 37 Phase 6
> columns were not to be re-entered as 37 independent hypotheses, but as a small
> representative set — is not in dispute; the *identity* of the representatives
> is, and the repository does not determine it. The query-intrinsic family is
> therefore **empty and unresolved**, `DECLARED_FAMILY_SIZE = 16` is
> **not yet satisfied**, and Gate 9.2 must not run. See §8 Issue C.

### 2.4.1 Per-signal expected directions (added by amendment, §8.2)

Direction convention: **negative** = higher signal associates with *lower*
dispersion; **positive** = higher signal associates with *higher* dispersion.

| Signal | Direction | Why |
|---|---|---|
| `jaccard_k(bm25,dense)` | negative | higher = more agreement |
| `jaccard_k(bm25,hybrid)` | negative | higher = more agreement |
| `jaccard_k(dense,hybrid)` | negative | higher = more agreement |
| `union_concentration` | **positive** | **higher = MORE different** — the opposite of Jaccard |
| `distinct_doc_ratio_k(bm25)` | either | broader coverage; sign not predicted a priori |
| `distinct_doc_ratio_k(dense)` | either | as above |
| `top1_agreement` | negative | True = same top-1 = more agreement |
| `rank_corr_k(bm25,dense)` | negative | higher = more ordering agreement |
| `score_gap_top1_top2(bm25)` | negative | larger gap = more confident = less sensitive |
| `score_gap_top1_top2(dense)` | negative | as above |
| `score_decay_slope(bm25)` | **positive** | **flat (≈0) = more dispersion**; slope is ≤0, so flat is a *higher* value |
| `score_decay_slope(dense)` | **positive** | as above |

The original §2.4 annotated the whole disagreement row and the whole score-geometry
row as "negative". That was correct for five of the six signals that share those
rows' intuitive reading and **wrong for `union_concentration` and
`score_decay_slope`**, both of which run opposite. Had the blanket annotation been
applied mechanically, those two would have been scored backwards. This is a
**documentation correction**, not a change of hypothesis: no feature definition was
altered, and no result had been observed when it was made.

### 2.5 `k` is frozen at 5

All rank- and set-based signals use **k = 5**, fixed here before execution.

This is not arbitrary and it is not the value that maximises the pilot
correlation. The pilot sweep in §7 shows the association is sharp in `k`
(ρ = −0.41 at k=1, −0.47 at k=3, **−0.45 at k=5**, −0.19 at k=10). Choosing k=5
*after* seeing that sweep is exactly the selection Phase 7 warns against, so the
sweep is disclosed in §7 and k is frozen anyway. k ∈ {3, 10} are reported as
**sensitivity analysis**, never as the headline.

`k=5` is also the `top_k` the E1 arms were run at, so it is the k the benchmark
actually measures.

### 2.6 Seeds and statistics — fixed before execution

| Constant | Value |
|---|---|
| `BOOTSTRAP_SEED` | 20250109 |
| `PERMUTATION_SEED` | 20250109 |
| `N_BOOTSTRAP` | 10 000 |
| `N_PERMUTATIONS` | 10 000 |
| `ALPHA` | 0.05 |
| `MIN_ABS_RHO` | 0.30 |
| `EPSILON` | 0.01 (matches Gate 2 / Gate 3) |
| `K` | 5 (frozen, §2.5) |

Seeds are **distinct from Phase 7's 20250103** so a Phase 9 artifact can never be
mistaken for a Phase 7 one reproduced.

Primary statistic is **Spearman ρ with mid-ranks**. Mid-ranks are mandatory, not
stylistic: `recall_at_5` takes only four distinct values on this benchmark (92 of
107 queries carry one relevant document), so ties are pervasive and an
untie-broken rank correlation would be wrong.

Effect sizes and intervals are primary; p-values secondary.

### 2.7 Multiplicity

Holm–Bonferroni **within the declared family of 16** (§2.4). Benjamini–Hochberg
FDR is reported alongside but does not gate. Both the family size and the
correction method are recorded on every test.

> **AMENDED (§8.7, 2026-10-04): the family size is 12, not 16.** The text above
> is preserved as written. H5 (query-intrinsic) is withdrawn as an unidentifiable
> family, so Holm is applied over the **12** registered hypotheses. Every
> surviving "16" in this document that refers to the Gate 9.2 hypothesis family
> is superseded by this note; the original wording is retained for the record.

A candidate **survives** screening only if, on the shipping arm:

1. \|ρ\| ≥ 0.30;
2. `p_holm` < 0.05 within the family of 12 (**amended** from 16, §8.7);
3. and it **replicates** — defined in §2.7.1.

Condition 3 is not optional decoration. Phase 7's strongest univariate feature
(`content_term_count`, AUC 0.711 on `after`) collapsed to 0.562 on `before`. A
signal that does not hold across two corpus versions has not been shown to be a
property of retrieval.

> **The two arms are not independent samples.** `after` and `before` are two
> versions of the *same* corpus over the *same* 107 queries, sharing query text,
> gold labels and splits. Replication here means the association is stable under
> a corpus perturbation — **not** independent-sample confirmation, and no p-value
> from one arm may be combined with the other's as though they were disjoint data.

### 2.7.1 Replication condition, defined numerically (amendment, §8.3)

Original wording was "same sign and **comparable magnitude**", which is not
operational. It is replaced by the convention already used by Phase 7's Gate 3
(`scripts/gate3_signal_exhaustion.py:evaluate_gate`, whose registered criterion is
*"the same four conditions hold on BOTH corpus arms (replication)"*): **replication
means the same decision holds on both arms**, with no magnitude ratio.

Concretely, a candidate replicates when, on the **replication** arm:

1. the sign of ρ matches the sign on the shipping arm; **and**
2. \|ρ\| ≥ `MIN_ABS_RHO` (0.30) — the same bar conditions 1 and 2 impose there.

`p_holm` is not re-derived on the replication arm; the multiplicity correction is
performed once, on the shipping arm, as §2.7 states. The replication arm is a
**stability check, not a second family**, and is reported separately so it can
never be read as an independent test.

This rule is deterministic, symmetric in the two arms (neither is privileged
beyond being the shipping arm, and the same two conditions are applied to each),
fixed before Gate 9.2, and recovered from project convention rather than invented.
`T-disp4` is retained **only** to report the decomposition that justifies the
choice, never as an analysis target.

`T-gain` has 4 and 6 members. It is characterised qualitatively in Gate 9.4 and
is **never** used to fit, select, or threshold anything. A model fitted on 4
### 2.8 Splits

ADR-027's frozen split is respected and not modified.

* **Exploratory.** Association screening over all 107 queries. Labelled
  exploratory in every artifact, because all 107 queries have now been seen by
  the analyst.
* **Confirmatory.** Fit on `calibration` (47), report **once** on `test` (60),
  pre-registered. Gate 9.3 only.

No threshold, model, or signal selection is tuned on `test`.

### 2.9 Leakage control

Three information classes are kept strictly separate and the separation is
enforced in code, not by convention:

| Class | Contents | May be a feature? |
|---|---|---|
| **Query-only** | Phase 6 `QueryFeatures`, query string | yes |
| **Retrieval-observable** | retrieved IDs, ranks, scores, texts, latencies | yes |
| **Ground-truth** | `relevant_documents`, `relevant_sections`, `reference_answer`, and every metric derived from them | **no — target construction only** |

Ground-truth information is used **only** to construct the targets `T-disp*` and
`T-gain`, and a signal function is never given an object that carries it. An
architecture guard test asserts this structurally: `build_signal_table` accepts
retrieval outcomes and query features, and the label-bearing fields are not
reachable from its signature. This generalises the existing
`test_sufficiency_check_uses_no_ground_truth` guard.

The distinction that matters most here: **a signal may be computed from another
strategy's retrieval outcome and still be non-deployable.** Cross-arm Jaccard
requires having already run both arms, so it is admissible as evidence about
mechanism and **inadmissible** as a pre-routing feature. Gate 9.3 exists
separately to ask whether any *deployable* proxy carries the same information.
Conflating "informative" with "usable" is the failure this phase is built to
avoid.

### 2.10 Gates and decision rule

| Gate | Question | Passes when |
|---|---|---|
| **9.1** Audit & artefact control | Is `T-disp3` real, or a measurement artefact? | `T-tiebreak` distribution reported; `T-needed` control reproduces AUC ≈ 0.53; reranker decomposition reported |
| **9.2** Mechanism | What explains dispersion? | ≥1 signal satisfies all three §2.7 conditions |
| **9.3** Deployability | Is it usable at a cheap decision point? | a proxy from {query-only ∪ single-arm ∪ hybrid-internal} holds out-of-sample on `test` |
| **9.4** Explanation | Mechanism or correlate? | a named mechanism with a discriminating prediction it fails |
| **9.5** Verdict | What next? | exactly one of the four verdicts below |

Gate 9.2 **fails** if no signal satisfies §2.7, and a failing 9.2 does not
proceed to 9.3.

Verdicts, emitted by `evaluate_gate()` and not chosen by hand:

```
PROMISING_SIGNAL   mechanism replicated (9.2) and a deployable proxy held
                   out-of-sample (9.3)
WEAK_SIGNAL        mechanism replicated (9.2), no deployable proxy (9.3)
NO_RELIABLE_SIGNAL no signal satisfies §2.7 on both arms
INSUFFICIENT_DATA  power < 0.8 for MIN_ABS_RHO, or the target is a
                   measurement artefact
```

**No router is built, trained, tuned or deployed under any verdict.** A learned
router on this data would be fitted on a handful of positives; ADR-026 keeps it
deferred and this phase does not revisit that.

### 2.11 Falsification conditions

Recorded now, so an outcome cannot be reinterpreted later.

---

## 3. Gates in execution order

1. **9.1** — `dispersion.py` targets, `T-tiebreak` audit, reranker decomposition,
   `T-needed` reproduction.
2. **9.2** — `agreement.py` + score geometry + Phase 6 control, Holm within the
   family of 12 (**amended** from 16, §8.7), cross-arm replication.
3. **9.3** — deployability, out-of-sample on `test` (executed — see final report).
4. **9.4** — `T-gain` characterisation, mechanism naming (executed — see final report).
5. **9.5** — verdict (executed — see final report).

Gate 9.1's control runs before any association is computed, because a pipeline
that cannot reproduce Phase 7's number has no business contradicting it.

---

## 7. Pilot evidence (exploratory — not a result)

This section discloses the read-only probes that were run **before** §2 was
written, so that the frozen choices in §2 can be audited rather than taken on
trust. These numbers are **exploratory**. They are recorded here because §2.5's
choice of `k = 5` is only defensible if the alternative was visible, and because
a reader is entitled to know which hypotheses were not blind.

All probes read persisted artifacts. None ran retrieval, wrote an artifact, or
called a provider.

### 7.1 Probe — cross-strategy disagreement vs. dispersion

Spearman ρ against 4-arm recall dispersion, n = 107:

| Signal | `after` | `before` |
|---|---|---|
| `jaccard@5 (bm25, dense)` | −0.480 | −0.526 |
| `jaccard@5 (bm25, hybrid)` | −0.405 | −0.397 |
| `jaccard@5 (dense, hybrid)` | −0.423 | — |
| union overlap ratio | −0.435 | — |

Permutation p = 0.0002 on both arms for `jaccard@5 (bm25, dense)`.

Against Gate 3's `needed` label the same feature yields only ρ = −0.212
(`after`) and −0.250 (`before`) — which is what a mis-specified target predicts,
and is the direct evidence for Finding 1.

### 7.2 Probe — score geometry is weak

`score_gap_top1_top2` gave ρ = −0.150 (dense) and −0.010 (bm25). Retained as a
registered **control** in §2.4 precisely because it is expected to be weak: a
family that only ever contains signals that work is not a test.

### 7.3 Probe — `k` sensitivity, and why §2.5 freezes k=5

`jaccard(bm25, dense)` against 3-arm recall dispersion, `after` arm:

| k | ρ | mean Jaccard |
|---|---|---|
| 1 | −0.409 | 0.551 |
| 3 | −0.473 | 0.550 |
| **5** | **−0.454** | 0.517 |
| 10 | −0.187 | 0.471 |

Chunk-level Jaccard at k=5 gives −0.446, so the association is not an artefact of
document-level grouping.

The sweep is reported because **k=5 is not the maximum** (−0.473 at k=3 is
marginally larger). Freezing k=5 despite that is the point: selecting the argmax
of a sweep on the same data one is about to test is selection, and it would make
every downstream number a product of a search. k ∈ {3, 10} are reported in
Gate 9.2 as sensitivity analysis only.

### 7.4 Probe — the honest 3-arm target retains the signal

Finding 3 removes the reranker from the target. The association survives that
removal, which is the reason to believe it was not a reranker artefact:

| Signal | `after` | `before` |
|---|---|---|
| `jaccard@5 (bm25, dense)` vs `T-disp3` | **−0.454** (p=0.0002) | **−0.518** (p=0.0002) |
| `jaccard@5 (bm25, dense)` vs `T-disp3-mrr` | −0.575 | −0.502 |
| `jaccard@5 (dense, hybrid)` vs `T-disp3` | −0.431 | −0.372 |

### 7.5 Probe — deployability preview

Query-only features are weak against dispersion (max \|ρ\| = 0.172 across eight
Phase 6 features), consistent with Gate 3's null. A single-arm proxy is stronger:
`distinct_doc_ratio@5 (bm25)` gives ρ = **+0.327**.

Separately, a structural observation shapes Gate 9.3 rather than Gate 9.2:
`hybrid.py:71-76` already fetches **both** branches at `candidate_k=20` and fuses
them, persisting only `bm25_candidate_count` / `dense_candidate_count` (verified
20/20 on all 107 traces). The disagreement information is computed and discarded.
Whether it is recoverable at zero extra cost is Gate 9.3's question, and
instruments the `hybrid` arm; it is **not** assumed here.
* If the Gate 9.1 control fails to reproduce Gate 3's AUC ≈ 0.53, the Phase 9
  pipeline disagrees with the Phase 7 pipeline and **no Phase 9 association is
  interpretable** until the discrepancy is found.
* If `T-disp3` is shown to be a tie-breaking artefact of near-binary recall, the
  target is wrong, and the verdict is `INSUFFICIENT_DATA` — *not* a search for a
  better-looking target on the same data.
* If a signal survives screening on one arm and reverses sign on the other, it is
  reported as **not replicated** regardless of its p-value.
* If a "surviving" signal turns out to require ground truth to compute, it is
  withdrawn and recorded as withdrawn.

### 2.12 What would count as an answer

| Outcome | Reading |
|---|---|
| A disagreement signal survives §2.7 on both arms | strategy disagreement is a real, reproducible property of these queries |
| Signals survive on `after` but not `before` | the property is corpus-specific; say so and stop |
| Only score geometry survives | dispersion is about evidence diffuseness, not retriever disagreement |
| Nothing survives | no reliable missing signal identified **at this sample size** — not "no signal exists" |
| `T-disp3` is an artefact | the benchmark cannot support this question; enlarge it before any further routing work |

Each row is a finding. None of them licenses building a router on its own.
positives measures the fit, not the signal.
---

## 8. Amendments (2026-10-04, post-audit, pre-Gate-9.2)

> §2 above is the original preregistration and is left intact as the historical
> record. This section records the corrections made after an implementation audit
> and **before any Gate 9.2 measurement**. Nothing here was chosen by looking at a
> Gate 9.2 result; no Gate 9.2 result exists.

### 8.0 Why

An audit of the implementation against this document found three mismatches that
would have made a Gate 9.2 verdict un-attributable to the preregistration: two
registered signals that were never implemented, six unregistered signals present
in the signal table, and a self-contradictory family arithmetic that makes the
Holm denominator unrecoverable. Gate 9.2 was **not** executed pending this
reconciliation.

### 8.1 Issue A — score-geometry signals implemented (completion)

`score_gap_top1_top2` and `score_decay_slope` are registered in §2.4 but did not
exist in code. They are now implemented in `evaluation/agreement.py`, reading
scores from `traces.jsonl` (which carries them) because `rows.jsonl` does not.

**The preregistration names these signals but does not define them**, so the
following closed forms are supplied here and frozen now:

* `score_gap_top1_top2 = (s1 − s2) / |s1|` over the top-`k` scores. Normalising by
  `|s1|` is required: BM25 scores reach ~27 while cosine similarity tops out near
  1.0, so a raw gap would put the arms on incomparable scales.
* `score_decay_slope` = OLS slope of score on rank over the top-`k` results,
  fitted as `score = a + b·rank`.

Both return `None` when undefined (fewer than two finite scores; `s1 = 0`), never
`0.0`. Verified by hand against raw traces: bm25 `p7_001` gap `0.1155666542`,
slope `−0.8668832078`; dense `p7_001` gap `0.0178917186`, slope `−0.0338354397`.

Neither definition was selected by inspecting a correlation with dispersion.

### 8.2 Issue B — candidate family restricted to §2.4 (scope correction)

The implementation computed `overlap_at_k` (all three pairs),
`top1_agreement` for two unregistered pairs, `distinct_doc_ratio@hybrid`, and
`rank_correlation@bm25|hybrid`. §2.4 states a signal not listed there "may not
enter the analysis as a candidate".

`evaluation.signals.REGISTERED_CANDIDATES` is now the single authority on the
family, and `signal_columns()` restricts to it **by default**. The helpers remain
computed on the row for inspection; `registered_only=False` exposes the full row
for debugging only. No helper was deleted.

### 8.3 Direction annotation corrected per signal (documentation)

`union_concentration` and `score_decay_slope` run opposite to the jaccard-style
signals. §2.4.1 replaces the blanket "negative" annotation with a per-signal table.
This corrects a documentation error; **no feature definition was altered**.

### 8.4 Issue C — query-intrinsic family UNRESOLVED (blocking)

The §2.4 table names "the 37 Phase 6 `QueryFeatures` columns"; the sentence below
it computes 8 + 4 + **4**. The 37 figure is the *total* Gate 3 column count
(13 query + 5 feedback + 6 routing + 13 one-hot), so "the 37 Phase 6
QueryFeatures columns" was itself inaccurate — Gate 3's numeric query group is 13.

The repository does not determine which four representatives were intended.
Evidence examined:

| Candidate reading | Count | Why it does not settle it |
|---|---|---|
| Gate 3 `QUERY_FEATURES` | 13 | not 4; contradicts the arithmetic |
| Phase 6 `schemas.config.FEATURE_GROUPS` | 6 | lexical, semantic, entity, complexity, question_type, multi_concept |
| `rule_based._signals()` numeric groups | 5 | `question_type` is excluded by its own docstring as non-numeric |
| First four of `FEATURE_GROUPS` | 4 | defensible, but not stated anywhere; requires inferring that `multi_concept` is excluded as redundant with `complexity_score` (which already embeds `concept_count`) |
| Gate 3's top-4 query features by AUC | 4 | **forbidden** — selects on observed correlation |

No document, plan, constant, or test in the repository names the four. Choosing
among the surviving readings would be a scientific decision taken *after* seeing
`ρ = −0.454` in §7, which is precisely what §2.4 forbids.

**Therefore the query-intrinsic family is left empty and Gate 9.2 must not run.**
The evidence above is reported so the choice can be made on scientific rather
than statistical grounds.

### 8.5 Provenance investigation — the four are NOT recoverable (2026-10-04)

A read-only, repository-wide search was run for any artifact that names the four
query-intrinsic representatives **before** Phase 9 execution. It found none.

**History searched.** Every commit whose tree contains `docs/phases/phase-9.md`:
exactly three — `80e8721` (14:31, the original draft), `efd6bdc` (14:50, the
frozen preregistration), and `5613f46` (the §8 reconciliation). The H5 row is
byte-identical in the first two, so the contradiction was present **at creation**
and was never a later edit. The phrase "4 query-intrinsic representatives" has
never, in any version, been followed by four names.

**Other artifacts searched, all negative:** `.kilo/plans/` (four plans — Phase 4,
Phase 5, Phase 8, oracle-routing-ceiling; no Phase 9 plan exists); `:memory:.ses`
(empty, 1 byte); `docs/decision.md` (no Phase 9 or dispersion entry); all ADRs; the
`.kilo/worktrees/hill-telephone` worktree (predates Phase 9, phases 1–5 only); and
`git log --all -S` on `complexity_score`, `concept_count`, `multi_concept`,
`query_length_words`, `QueryFeatures`, `content_term_count`, `family size`, `H5`.
No forward-looking artifact names any Phase 9 query feature.

**Where the "4" most likely came from.** `docs/phase_7_closure_results.md` §3.2 is
titled *"Feature set — 37 columns, **four groups**"*, and its table splits the 37
columns into **Query features (13) / Retrieval feedback (5) / Routing signals (6)
/ Initial strategy + categoricals (13)**. That is a four-way split of *all* the
columns, of which exactly **one** group is query-intrinsic. It appears the "4" in
§2.4 is a conflation of "four **column groups**" with "four **query-intrinsic
representatives**". If so, the four may never have existed as a design, and the
declared family size of 16 rests on an arithmetic slip rather than on an intended
list.

This is reported as a **conflict in the frozen record**, not resolved here. It
cannot be settled without an explicit scientific decision.

**`QueryFeatures` has no four-way structure.** `schemas/routing.py:33` defines a
flat 16-field record with no grouping; `schemas.config.FEATURE_GROUPS` has six
entries; `rule_based._signals()` exposes five numeric ones.

### 8.6 Candidate sets and their provenance

| Candidate set | n | Provenance | Selectable now? |
|---|---|---|---|
| The 13 Gate 3 `QUERY_FEATURES` | 13 | **C/D** — contradicts the declared size of 16 | no |
| All 37 Gate 3 columns | 37 | **D** — the total column count, misread as a feature set | no |
| Gate 3 top-4 query features by AUC | 4 | **D** — selected on observed results | **forbidden** |
| First four of `FEATURE_GROUPS` (lexical, semantic, entity, complexity) | 4 | **C** — requires inferring `multi_concept` is redundant with `complexity_score` | no |
| Four `*_count` cue features (content/entity/technical/semantic) | 4 | **C** — a plausible symmetric reading, stated nowhere | no |
| The four "column groups" of Phase 7 §3.2 | 4 | **B** for the *groups*, but **not** query-intrinsic — 3 of the 4 are post-retrieval | no |

No set qualifies as **A** (explicitly preregistered) or **B** (explicitly
specified elsewhere before execution). Every surviving set is **C** or **D**.

**Consequence.** The four are **not scientifically recoverable from the frozen
record**. The family remains at 12 registered candidates against a declared size
of 16, the Holm denominator is still unrecoverable, and Gate 9.2 must not run.

**Minimum amendment required** (to be decided before any Gate 9.2 result is
inspected): an explicit, dated addendum to §2.4 that either

* **names** four specific query-intrinsic features, together with the scientific
  rationale for choosing that set and for excluding the other nine; or
* **declares the query-intrinsic family empty** and revises the declared family
  size from 16 to 12, stating that Phase 6 features enter only as the already-
  decided control rather than as four further hypotheses.

The second is the more conservative option and is consistent with §2.4's own
stated purpose ("reused as a *control*, not re-tested as new hypotheses"). It
requires no new scientific choice, only the withdrawal of an arithmetic slip.

### 8.7 AMENDMENT — H5 withdrawn; Gate 9.2 family is 12 hypotheses

**Dated 2026-10-04, made before any Gate 9.2 result was inspected. No Gate 9.2
execution has occurred at any point; there are no Gate 9.2 results to have
inspected.** This is a correction of an internal inconsistency in the frozen
document, not a result-driven hypothesis selection.

**What was wrong.** §2.4 originally declared a family of **16** while naming
only **12** signals (8 disagreement + 4 score geometry) and referring, in the same
breath, both to "the 37 Phase 6 `QueryFeatures` columns" and to "4 query-intrinsic
representatives". The four H5 hypotheses were therefore **never identifiable**.

**Provenance finding (§8.5).** No historical document, commit, plan, ADR or code
comment ever named those four. `docs/phases/phase-9.md` has three versions; the
H5 row is byte-identical in the two pre-reconciliation ones, so the ambiguity was
present at creation. The likely origin of the "4" is `phase_7_closure_results.md`
§3.2, *"37 columns, four groups"* — a four-way split of **all** columns, only one
group of which is query-intrinsic. Every surviving candidate set classifies as
provenance **C** (architectural inference) or **D** (selected on observed
results). None is **A** or **B**.

**Decision.** H5 is **withdrawn as an unidentifiable preregistered hypothesis
family.** The four are not invented, inferred, or selected by AUC or correlation.
§2.4's own stated purpose — Phase 6 features are *"reused as a control, not
re-tested as new hypotheses"* — is the basis for leaving them out rather than
filling the slot.

**Corrected family (12 hypotheses):**

| # | Signal | Family |
|---|---|---|
| 1 | `jaccard@bm25\|dense` | H1 disagreement |
| 2 | `jaccard@bm25\|hybrid` | H1 disagreement |
| 3 | `jaccard@dense\|hybrid` | H1 disagreement |
| 4 | `union_concentration` | H1 disagreement |
| 5 | `distinct_doc_ratio@bm25` | H1 disagreement |
| 6 | `distinct_doc_ratio@dense` | H1 disagreement |
| 7 | `top1_agreement@bm25\|dense` | H1 disagreement |
| 8 | `rank_correlation@bm25\|dense` | H1 disagreement |
| 9 | `score_gap_top1_top2@bm25` | H4 score geometry |
| 10 | `score_gap_top1_top2@dense` | H4 score geometry |
| 11 | `score_decay_slope@bm25` | H4 score geometry |
| 12 | `score_decay_slope@dense` | H4 score geometry |

**The Holm denominator is 12.** Phase 6 query features are **not** 12, 13 or 37
further hypotheses; they remain the control defined in §2.4 and are reported
separately, uncorrected, and outside the family.

**Stated neutrally.** Reducing 16 → 12 changes the multiplicity family. It is
**not** claimed to be a more conservative procedure in the sense of reducing
false positives: a smaller family yields *smaller* Holm-adjusted p-values, so
this weakens the correction relative to the declared 16. It is adopted because
the four withdrawn hypotheses were **never identifiable** and a family size must
be composed of hypotheses that actually exist. Anyone comparing Phase 9 to a
hypothetical 16-hypothesis gate should note the family is smaller.

**Unchanged by this amendment** — verified, not asserted: `T-disp3`; the
bm25/dense/hybrid strategy set; `k = 5`; the BEFORE/AFTER arms; all seeds
(`BOOTSTRAP_SEED`, `PERMUTATION_SEED`); `N_BOOTSTRAP`/`N_PERMUTATIONS` = 10 000;
mid-rank Spearman; the Holm procedure and `ALPHA = 0.05`; the survival rule and
`MIN_ABS_RHO = 0.30` including the §2.7.1 replication condition; the benchmark and
splits; and the leakage policy. `DECLARED_FAMILY_SIZE` in
`evaluation/signals.py` is now **12**.

### 8.8 OPEN — missing-data handling for `rank_correlation@bm25|dense` (unresolved)

Flagged, **not** silently resolved.

`rank_correlation@bm25|dense` is defined only where bm25 and dense share **≥ 2**
documents in their top-5. At `k = 5` over a 14-document corpus they frequently do
not, so the signal is **undefined on 84 of 107 queries**.

* **Effective sample size: n = 23** per arm (identical in BEFORE and AFTER).
* It is **not** converted to 0.0 — a fabricated 0.0 would be indistinguishable
  from a measured absence of relationship.
* It is **not** dropped from the hypothesis family. It stays a member of the 12.

**The gap.** §2.6 fixes the statistic (Spearman ρ, mid-ranks) and §2.7 fixes the
correction (Holm over the family), but **neither states how a candidate with
incomplete observations is handled.** Two consequences are unspecified:

1. **Whether the |ρ| ≥ 0.30 bar and the Holm threshold are applied at n = 23.**
   The two are **separate conditions**: the minimum-effect threshold does not by
   itself imply significance, but sufficiently large |ρ| can be significant at
   n = 23. At |ρ| = 0.30 the two-sided p ≈ 0.164 (p × 12 = 1.00), while
   |ρ| ≈ 0.45 gives p ≈ 0.031 and |ρ| ≈ 0.55 gives p ≈ 0.0066 — so this
   candidate **can** in principle clear `p_holm < 0.05` over a family of 12 at a
   large effect. It is therefore **not** "arithmetically incapable" of
   surviving.

   > **CORRECTION.** An earlier draft of this section asserted that n = 23 made
   > survival arithmetically impossible. **That was wrong** and is withdrawn: the
   > minimum-effect threshold and statistical significance are independent
   > conditions, and the hypothesis is testable, merely less powerful. It is a
   > member of the family, receives a Holm-adjusted p-value, is reported normally
   > with its effective n, and is **not** labelled "untestable".

2. **Whether the bootstrap and permutation resamples are drawn from the defined
   observations or from all 107 rows.** Resolved in §8.9: the defined
   observations, with no imputation of any kind.

**This must be specified in the amendment before Gate 9.2 runs.** A silent choice
here would decide the reported outcome for 1 of the 12 hypotheses. No method has
been chosen on the author's behalf.

The mechanically forced part — Spearman ρ is computed on the 23 queries where
both variables are defined — follows from the statistic itself and is not a
choice. The two items above are the ones requiring an explicit decision.

### 8.9 FROZEN RULE — missing-data handling (closes §8.8)

**Dated 2026-10-04, before any Gate 9.2 result was generated.** Gate 9.2 has not
been executed.

**The structural fact.** Spearman rank correlation between the BM25 and dense
top-5 rankings is defined only where the two arms share **at least two
documents**. At k = 5 over a 14-document corpus that fails often:

| | total queries | defined | undefined |
|---|---|---|---|
| AFTER | 107 | **23** | 84 |
| BEFORE | 107 | **23** | 84 |

**Inferential population (§2.6/§2.7 amended).** Statistical inference for a
signal is performed **over the observations on which that signal is defined**.
Undefined observations are excluded **pairwise, for that signal only**. **No
imputation is performed.** This is directly implied by the signal's mathematical
definition, and is the standard pairwise-defined analysis.

Consequently there is **no forced n = 107 complete-case restriction across the
family**: each signal is analysed on its own defined subset, and a sparse signal
does not shrink any other. `evaluation.signals.defined_observations()` is the
single accessor for that population.

**Explicitly forbidden** (each would corrupt the measurement):

* imputing missing values;
* converting NULL to `0.0` — a fabricated zero is indistinguishable from a
  measured absence of agreement;
* treating missing as zero agreement;
* removing the signal from the registered family;
* changing `k`;
* changing the signal's definition.

**Bootstrap.** Resamples the signal's **defined query-level observations** with
replacement — for `rank_correlation@bm25|dense`, the 23 defined observations.
It does **not** bootstrap all 107 rows and reconstruct missing values.

**Permutation.** The preregistered permutation test runs on the **same defined
observations** — the 23 defined pairs. NULL entries are never permuted and never
imputed.

`defined_observations()` returns the same `(query_ids, values)` tuple to the
correlation, the bootstrap and the permutation, so all three see identical
queries by construction and the effective n is reported rather than inferred.

**Family membership is unchanged.** The Holm family remains **exactly 12**. The
n = 23 signal receives a Holm-adjusted p-value, is reported normally with its
effective n, and its reduced power is reported as a **limitation**. It is not
dropped, down-weighted, or relabelled.

**Thresholds unchanged.** `MIN_ABS_RHO = 0.30` applies unchanged, on both arms,
as do the Holm threshold, `ALPHA = 0.05`, and the §2.7.1 replication condition
(same sign **and** |ρ| ≥ 0.30). The minimum-effect threshold and statistical
significance are **separate conditions**; a signal must satisfy both. Nothing is
re-tuned on the basis of an observed result.

### 8.10 BEFORE/AFTER eligibility are NOT the same queries

Verified directly rather than assumed. The defined-query sets were compared by
identity:

| Quantity | Value |
|---|---|
| n defined, AFTER | 23 |
| n defined, BEFORE | 23 |
| **intersection of query IDs** | **13** |
| defined in AFTER only | 10 |
| defined in BEFORE only | 10 |
| identical sets? | **No — asymmetric** |

The counts coincide at 23, but only **13 queries are defined in both arms**;
the remaining 20 differ. This is a real limitation and is recorded as such:
for this signal alone, replication (§2.7.1) compares an association measured on
23 shipping-arm queries against one measured on a **largely different set of 23**
replication-arm queries. A disagreement between the two arms' ρ for this signal
therefore cannot be read as instability on a common sample.

This does not alter the frozen rule — eligibility is evaluated **independently
per arm**, exactly as specified. It is recorded because an earlier draft of
§8.8 assumed "identical availability in BEFORE and AFTER", meaning identical
*n*. The identity check shows the assumption was wrong at the query level, and
the corrected statement is the table above.
