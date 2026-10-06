# Phase 13 — Next Research Direction After the Adaptive-Routing Track

Status: **COMPLETE** — selection recorded: cheap-first adaptive retrieval
(BM25 → Dense need prediction). Final report: `docs/phase13_results.md`. The
body below is the record of the selection (the only computation was read-only
counts over frozen rows — no retrieval, no experiment, no fitting). Phases
8–12 are frozen and unchanged by this phase.

## 1. Objective

Select the strongest scientifically defensible next step now that
retrieval-state-only post-dense routing is closed (Position C, Phase 12):
a new research track, a project reframe, or a research stop — from evidence,
not from a desire to keep the project alive.

## 2. Closed track (boundary condition, not to be revived)

```text
retrieval-state signal → predict escalation value → conditional Dense → Hybrid
```

Prohibited without a fundamentally new assumption, explicitly treated as a
new question: another DDR threshold/classifier, recombinations of the
24/12/6 feature families, black-box routers over the same features,
same-evidence Hybrid escalation, or hyperparameter search to rescue the
result. The Phase 10 oracle and Phase 11 labels are not revisited.

## 3. Discipline (held)

1. Offline design/selection only: no retrieval, embeddings, API calls,
   router, threshold optimization, classifier, or Phase 8–12 code change.
2. Read-only frozen-row counts are permitted as audit (they determine
   feasibility; they fit nothing).
3. `not demonstrated` ≠ `impossible`; the corpus is not called defective;
   negatives from Phases 9–12 are preserved verbatim (§15 of the brief).
4. A new track must name a new assumption for each closed-track element it
   touches (decision point, rung, information, currency).

## 4. Analyses performed (pre-declared)

* A. Project-state model: implemented / validated / failed / ruled out /
  plausible / unexplored, from the repository.
* B. New feasibility evidence: insufficient-class size and cheap-first
  (BM25→Dense) oracle-positive counts per arm and split, counted read-only
  from `experiments/phase8/combined/p8{a,b}_e1` frozen rows.
* C. Sufficiency operationalization audit (§7): candidate definitions
  checked for decision-time availability and circularity.
* D. Stopping-vs-escalation adjudication (§8): genuinely different or same
  problem renamed.
* E. Intervention inventory (§9) and benchmark adequacy (§10) from measured
  behavior, not adjectives.
* F. 3–5 candidate problems with full dossiers (§5) scored on A–J (§6);
  qualitative decision matrix (§12).
* G. Single selection (§13) and, if a track, the Phase 14 research design
  (§14) — design only, no implementation.

## 5. Selection rule (held)

`NEW RESEARCH TRACK` only with: new question, EV-positive rung, adequate
positives (counted, not hoped), decision-time-available information,
offline-evaluable oracle, falsifiable success/failure criteria. Else
`PROJECT REFRAME` (only with evidence for the new frame) or
`RESEARCH STOP` (preferred over another weak experiment).

## 6. Files

Docs: this file, `docs/phase13_results.md`, `docs/progress.md` entry.
No source, test, experiment, retrieval, routing, or runner change.
