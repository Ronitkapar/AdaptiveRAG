# Phase 12 — Research Boundary of Adaptive Retrieval

Status: **COMPLETE** — recommendation `STOP CURRENT ADAPTIVE-ROUTING TRACK`
(Position C). Final report: `docs/phase12_results.md`. The body below remains
the frozen pre-registration, written before any Phase 12 analysis beyond the
Phase 8–11 audit. Phases 8–11 are frozen and unchanged by this phase.

## 1. Question

> What can AdaptiveRAG now legitimately conclude about adaptive retrieval
> under the tested experimental setting, what hypotheses have been ruled
> out, what limitations explain the negative result, and whether there is a
> scientifically justified next research direction?

This is a **research-boundary analysis, not another attempt to obtain a
positive result**. No router is sought here.

## 2. Evidence base (all already persisted; no new retrieval)

* Phase 8: `docs/phase-8-results.md`, `docs/phase7-closure-evidence.md` §8
  (Gate 2 ceiling: +0.0218/+0.0343, CIs excluding zero, 4–6 carriers;
  Gate 3: AUC 0.5340/0.5345, 0/24 Holm, `no_out_of_sample_signal`),
  `experiments/phase8/combined/p8{a,b}_e1`, `oracle_ceiling_v2.json`,
  `gate3_signal_exhaustion.json`.
* Phase 9: `docs/phase9_results.md` (verdict `PROMISING_SIGNAL`; 5 survivors;
  DDR@dense test ρ +0.410; no pre-routing signal; §9.4 proxy-vs-distinct
  unresolved), `experiments/phase9/signal_table_phase8_{after,before}.jsonl`.
* Phase 10: `docs/phases/phase-10.md` (frozen protocol),
  `docs/phase10_results.md` (verdict `FAILURE`: AUC 0.55, t=0.8 high,
  test Δ −0.050/−0.033, TP 0/0, P(random ≥ adaptive) 1.000/0.954),
  `experiments/phase10/` (oracle tables + provenance, frozen policy,
  one-pass test rows, 5 figures + manifest).
* Phase 11: `docs/phases/phase-11.md`, `docs/phase11_results.md`
  (Outcome B; 5 unique helps, all harms State 3, H1/H2 taxonomy,
  separation 1/6 + 0/6, gate CLOSED),
  `experiments/phase11/` (mechanism tables + summary, 4 figures + manifest).
* Provenance anchors: ADR-026 (learned router deferred), ADR-027 (47/60
  calibration/test split), ADR-028 (rerank rung net-negative; gate
  mis-set; DEFERRED), dataset sha256 prefix `f0695189903c8ef8`.

## 3. Discipline (frozen)

1. Offline analysis only: no retrieval runs, routing experiments,
   threshold sweeps, network/embedding/reranker calls, or test-set
   optimization. Read-only artifact verification is permitted.
2. The Phase 10 oracle and Phase 11 mechanism labels are not changed.
3. `not demonstrated` is never rewritten as `impossible` without evidence.
4. The corpus is not called inadequate unless the evidence supports it —
   measured headroom properties are reported as properties, not defects.
5. All negative results are preserved; no failure is reinterpreted.

## 4. Analyses (all pre-declared)

* A. Evidence matrix, one row per research question across Phases 8–12,
  with exact repository evidence (no invented claims).
* B. Four-way classification: established / not established (inconclusive) /
  falsified-under-setting / still-plausible-but-untested.
* C. Boundary formulation derived from the evidence (not assumed wording).
* D. Limiting-setting hypotheses A–G: each judged supported / mixed /
  speculative against existing evidence.
* E. Positions A–D adjudicated from the evidence.
* F. Directions 1–5 assessed (motivation, supporting/counter-evidence,
  required data/experiments, value, repeat-failure risk) as hypotheses,
  not implementation proposals.
* G. Twelve-question decision framework answered directly.
* H. Contribution analysis (engineering / empirical / negative /
  mechanistic / boundary / uncertainty) for the eventual final report.

## 5. Phase 13 decision gate (frozen)

A new adaptive-routing experiment requires ALL of:

1. A clearly defined unresolved research question.
2. A plausible observable signal or mechanism.
3. Sufficient positive examples, or a justified way to obtain them.
4. A meaningful intervention with measurable upside.
5. A clean evaluation protocol.
6. A reason the new experiment can answer something Phases 9–11 could not.

Otherwise the recommendation is `STOP CURRENT ADAPTIVE-ROUTING TRACK`
(or `CHANGE RESEARCH QUESTION` with the new question stated and gated).

## 6. Outcome mapping (frozen)

* `CONTINUE` — only via the §5 gate, naming the new question, signal,
  positives source, rung, and protocol.
* `STOP CURRENT ADAPTIVE-ROUTING TRACK` — with the boundary claim, what is
  preserved, and explicit re-entry conditions.
* `CHANGE RESEARCH QUESTION` — with the replacement question and why it is
  not a relabelled routing attempt (Directions 1–5 evaluated, not assumed).

## 7. Files

Docs: this file, `docs/phase12_results.md`, `docs/progress.md` entry.
No new source module is anticipated; tests are added only if reusable
analysis code is introduced. No changes to Phases 8–11 artifacts,
retrieval, routing, or runner code.
