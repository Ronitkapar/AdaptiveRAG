# AdaptiveRAG Documentation

This directory contains the project's research record, architecture and
reproduction instructions.

## Start Here

For someone new to the project:

1. [`../README.md`](../README.md) — 2–3 minute overview
2. [`ADAPTIVERAG_REPORT.md`](ADAPTIVERAG_REPORT.md) — the central research
   report (the single document that answers: what was asked, what was tested,
   what happened, what was learned)
3. [`research/`](research/) — the research record, split by topic
4. [`history/experiment_timeline.md`](history/experiment_timeline.md) — phase-by-
   phase chronology

For coding-agent work:

1. [`../AGENTS.md`](../AGENTS.md)
2. [`progress.md`](progress.md)
3. [`phases/<phase-N>.md`](phases/)

The research investigation is complete (Phases 1–15). Do not start new
experiments or tune the routing policy unless explicitly requested.

---

## Documentation Map

### Central report

* **`ADAPTIVERAG_REPORT.md`** — the compact research paper: abstract,
  introduction, research question, experimental setup, baselines, the
  adaptive-routing investigation (Phases 6–15), findings, conclusions,
  limitations, what is *not* claimed, final status, future work, and an index
  of evidence. **Every other document is subordinate to this one.**

### Research record (`research/`)

| File | Answers |
| --- | --- |
| `research_question.md` | What was actually asked, and what was deliberately not asked? |
| `methodology.md` | How was it set up and measured (corpus, splits, metrics, statistics, integrity controls)? |
| `experiments.md` | What was run, experiment by experiment? |
| `findings.md` | What was discovered — confirmed / promising / negative / unresolved? |
| `negative_results.md` | What was rejected, on what evidence, and what was learned? |
| `conclusions.md` | What does it all add up to? |
| `limitations.md` | What are the boundaries of the evidence? |

### Architecture

* **`architecture.md`** — detailed layer-by-layer reference (ingestion →
  chunking → embedding → retrieval → generation → routing → evaluation).
* **`architecture/system_overview.md`** — the final architecture *as
  implemented*, distinguishing router / retriever / evaluation / experiment
  infrastructure.
* **`decision.md`** — the locked architectural decisions (ADR-001 … ADR-028).

### Reproducibility

* **`reproducibility/reproduction.md`** — commands for the corpus pipeline,
  baselines, and every phase 9–15 analysis, with the caveats that matter
  (arm identity, corpus rebuilds, latency control arms).
* **`phase7-closure-evidence.md`** — SHA-256 provenance for the inputs and
  artifacts the closure gates rely on.
* **`../experiments/README.md`** — run-artifact layout.

### History

* **`history/experiment_timeline.md`** — Phase 1 → 15 with objective,
  hypothesis, experiment, result and decision.
* **`history/plans/`** — pre-execution planning notes (Phases 4, 5, 8 and the
  oracle-routing study), archived from the agent workspace.

### Per-phase records

* **`phases/phase-1.md` … `phases/phase-15.md`** — plan of record for each
  phase. Phases 10–15 are frozen pre-registrations; their completion status and
  final reports are recorded in the status header and in the result files.
* **`phase_7_results.md`**, **`phase_7_closure_results.md`**,
  **`phase-8-results.md`**, **`phase9_results.md` … `phase15_results.md`** —
  the result record per phase.
* **`corpus.md`**, **`dense_baseline.md`**, **`evaluation.md`** — corpus,
  baseline and evaluation design.
* **`strategy-cost-freeze-decision.md`** — the measured basis for the (not
  executed) cost-freeze recommendation.

---

## Context Principle

Keep active context small. The repository and its documentation are the source
of truth; do not rely on old conversation context when the repository says
otherwise.
