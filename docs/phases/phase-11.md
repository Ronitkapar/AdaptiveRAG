# Phase 11 — Missing Signal Behind Useful Retrieval Escalation

Status: **COMPLETE** — outcome B (mechanism found, evidence insufficient). Final
report: `docs/phase11_results.md`. The body below remains the frozen
pre-registration, written before any Phase 11 computation. Phases 8–10 are
frozen and unchanged by this phase.

## 1. Question

> What additional information is required to distinguish retrieval
> disagreement that represents useful missing evidence from disagreement that
> represents harmful alternative evidence?

The goal is NOT to rescue DDR. Phase 10 showed DDR tracks dispersion while
dispersion is dominated by dense succeeding where fusion fails. Phase 11 is a
mechanism-analysis phase with a go/no-go gate; a new intervention is NOT
pre-authorized (see §8).

## 2. Frozen groups (from the Phase 10 oracle, unchanged)

`delta = recall@5(hybrid) − recall@5(dense)`, `EPSILON = 0.01` (frozen):

* **helps**: delta ≥ +0.01 (== Phase 10 `oracle_escalate`, by construction)
* **harms**: delta ≤ −0.01
* **ties**: otherwise

Audited membership (union: 5 unique helps; harms 9 after / 12 before):

* helps-calib: `p7a_014` (both arms), `p7g_001` (after), `p7e_005` (before)
* helps-test: `p7e_011`, `p7e_014` (both arms)
* harms-test shared: `p7_003`, `p7_006`, `p7b_007`, `p7g_003` (+ arm-specific)

Two pilot cases already motivate the mechanism (to be systematized, not
concluded from): `p7a_014` (help: dense top-1 wrong at 0.41, BM25 holds the
relevant doc at #1 with 17.3, hybrid recovers it) vs `p7_003` (harm: dense
top-1 correct at 0.60 with margin, BM25 confidently wrong, hybrid corrupts).
Hypothesis to test: **dense confidence × disagreement** separates State 2
(uncertain + helpable → escalate) from State 3 (disagree-but-correct → stop).

## 3. Admissible evidence (all already persisted; no new retrieval)

* Oracle tables (`experiments/phase10/oracle_phase8_{after,before}.jsonl`):
  outcomes, deltas, DDR, latency model, splits.
* Signal tables (`experiments/phase9/`): DDR, `jaccard@` pairs,
  `union_concentration`, `top1_agreement`, `score_gap_top1_top2` and
  `score_decay_slope` for bm25+dense — i.e. the dense-confidence and
  disagreement families already exist; they are re-examined *against the gain
  oracle* (which Phase 9 lacked), not re-mined as a 12-way family.
* E1 traces (dense/bm25/hybrid): ranked ids + raw scores + full top-10 for
  change attribution. Boundary: candidate-depth ranks 11–20 are NOT in
  traces, so source attribution is top-10 membership only (stated limit).
* Dataset + ledger: `relevant_documents` (mechanism analysis ONLY — labels
  may characterize what happened, never enter a candidate signal),
  category, evidence quotes for case narratives.
* BM25 output at decision time is admissible evidence for *analysis* (~2 ms
  local, no API call); whether it may enter a future policy is a design
  question for the report, decided explicitly, not smuggled in.

## 4. Discovery discipline (frozen)

1. Hypothesis generation on **calibration** helps/harms only
   (after: 2 helps / 3 harms; before: 2 / 6).
2. Test helps/harms (2 / 6 per arm) used SOLELY for consistency description
   of an already-stated hypothesis — never for feature/threshold/metric
   selection. No threshold is fitted anywhere in Phase 11.
3. Descriptive statistics only: exact counts, group means, rank orderings.
   No AUC-as-evidence on ≤6 positives; no p-values on the help set; no
   classifier of any kind.
4. The three states (§13 of the brief) are labelled per query: State 2 =
   helps; State 3 = harms where dense was already correct
   (`dense_recall_at_5` at the query max); State 1 = remainder NOs.

## 5. Analyses (all pre-declared)

* A. Group counts and delta distributions per arm/split.
* B. Per-query hybrid-change attribution for every non-tie query: which docs
  entered/left the top-5, relevant-doc movements, source arm (dense-only /
  BM25-only / both at top-10), displacing intruders on harms.
* C. Dense-confidence vs group: top-1 score, top1–top2 gap, decay slope,
  DDR (calibration ordering; test consistency).
* D. Disagreement vs group: jaccard(bm25|dense), union, top-1 agreement
  (calibration ordering; test consistency).
* E. Case narratives for all 5 unique helps + matched harms (what changed,
  observable-before-escalation mechanism, dataset adequacy per case).
* F. Candidate-signal statement ONLY if C/D yield a calibration-consistent,
  label-free, decision-time-computable rule — written down with its
  calibration record, then checked once against test helps/harms
  descriptively. A check that fails kills the candidate; a check that passes
  on n=2 does NOT license an intervention (see §8).

## 6. Figures (meaningful only)

1. DDR vs escalation value (delta scatter, both arms, groups marked).
2. Dense confidence (top1–top2 gap) vs delta, by group.
3. Disagreement (jaccard bm25|dense) vs dense quality, by group.
4. Per-query dense→hybrid change for all non-tie queries (lollipop deltas).
5. Conditional: candidate-signal panel ONLY if §5F produces a candidate;
   otherwise omitted (no filler figures).

## 7. Outcome mapping (frozen)

* **A (Promising)**: not available in Phase 11 (see §8) — recorded only as
  "would require" criteria.
* **B (Insufficient evidence)**: mechanism plausible + observable, but ≤4
  positives cannot establish a reliable signal.
* **C (No observable signal)**: helps/harms indistinguishable at decision
  point with available evidence.
* **D (Wrong intervention)**: evidence shows Hybrid cannot help this
  corpus/task regardless of routing (e.g. helps are RRF accidents with no
  observable precursor).

## 8. Go/no-go gate for another intervention (frozen: CLOSED)

A new adaptive intervention requires ALL of §12.1–12.6 of the brief.
Criterion 12.3 (enough development examples) fails a priori: at most 2–3
helps exist on any single split/arm, so no policy can be frozen without
fitting noise. **Phase 11 therefore cannot authorize an intervention
experiment.** The gate instead answers the seven §18 questions and delivers
either a future-experiment design (what data + signal + protocol would be
needed) or a stop recommendation with reasons. Running an intervention on
2 positives would violate constraints 3, 6, and 10 of the brief.

## 9. Files

New module: `src/adaptive_rag/evaluation/mechanism.py`
(`MECHANISM_VERSION = "phase11_mechanism_v1"`; pure functions over trace/row
dicts; reuses `agreement.*`, `dispersion.EPSILON`, `escalation.THRESHOLDS`
conventions; NO oracle redefinition — helps == Phase 10 oracle YES,
asserted).
New script: `scripts/phase11_mechanism.py` → `experiments/phase11/`
(`mechanism_table_{after,before}.jsonl`, `mechanism_summary.json`,
`figures/` + manifest).
New tests: `tests/test_mechanism.py`.
Docs: this file, `docs/phase11_results.md`, `docs/progress.md` entry.
No changes to Phases 8–10 artifacts, retrieval, routing, or runner code.
