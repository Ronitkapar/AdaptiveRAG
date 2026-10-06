# AdaptiveRAG — Agent Instructions

## Project

AdaptiveRAG is a query-aware RAG system that evaluates retrieval strategies and
investigates whether retrieval effort can be adapted per query.

Planned progression:

Dense → BM25 → Hybrid → Reranking → Adaptive Routing

The research investigation is complete (Phases 1–15) and experimentation has
been deliberately stopped; see `docs/progress.md` for the final status and
`docs/ADAPTIVERAG_REPORT.md` for the final research report.

---

## Core Rules

* Inspect existing code before changing it.
* Prefer the smallest change that satisfies the task.
* Preserve existing interfaces and architecture unless a change is genuinely
  required.
* Keep components modular and independently testable.
* Keep evaluation independent of retrieval strategy.
* Preserve provenance and structured data contracts.
* Prefer deterministic, reproducible implementations.
* Do not introduce unnecessary dependencies or frameworks.
* Do not use LangChain/LlamaIndex as core architecture unless explicitly
  requested.
* Do not implement future-phase functionality early.

---

## Context Loading

Do not load all documentation by default.

For non-trivial tasks:

1. Read `docs/progress.md`.
2. Read the current `docs/phases/phase-N.md`.
3. Read `docs/decision.md` when making or modifying architectural decisions.
4. Read `docs/architecture.md` only when system-wide architecture details are
   needed.

Use the smallest amount of context necessary to complete the task.

The repository and current documentation are the source of truth; do not rely
on old conversation context when the repository says otherwise.

---

## Phase Boundaries

The research investigation is complete. Do not start new experiments, tune the
routing policy, or add features beyond maintenance and documentation work
unless explicitly requested.

Final progression status:

* Phase 1 — Corpus Foundation: COMPLETE
* Phase 2 — Dense RAG Baseline: COMPLETE
* Phase 3 — BM25: COMPLETE
* Phase 4 — Hybrid: COMPLETE
* Phase 5 — Reranking: COMPLETE
* Phase 6 — Adaptive Routing: COMPLETE
* Phase 7 — Evaluation & Ablations: COMPLETE (closed by two pre-registered gates)
* Phase 8 — Corpus Fix Study: COMPLETE (quality study; cost measured, not frozen)
* Phase 9 — Strategy Sensitivity: COMPLETE (`PROMISING_SIGNAL`)
* Phase 10 — Post-Dense Escalation: COMPLETE (`FAILURE` — honest negative)
* Phase 11 — Missing-Signal Analysis: COMPLETE (outcome B: mechanism found,
  evidence insufficient)
* Phase 12 — Research Boundary: COMPLETE (recommendation: STOP current
  adaptive-routing track)
* Phase 13 — Next Direction: COMPLETE (cheap-first track selected)
* Phase 14 — Cheap-First Experiment: COMPLETE (`INSUFFICIENT EVIDENCE`)
* Phase 15 — Powered Confirmation: COMPLETE (`INSUFFICIENT EVIDENCE`, 4/5 bars)

Status: research investigation completed; adaptive-routing experimentation
paused; no final optimal routing policy claimed. See
`docs/ADAPTIVERAG_REPORT.md`.

---

## Testing

* Add or update tests for meaningful behavior changes.
* Prefer offline and deterministic tests.
* Do not require live API credentials for unit tests.
* Do not weaken existing tests just to make code pass.
* Run relevant tests after changes and verify existing functionality remains
  intact.

---

## Documentation

Documentation is part of the project's source of truth.

After completing a task, update documentation if the project's documented
state changed:

* `docs/progress.md` → status, completed work, validation, next steps.
* `docs/phases/phase-N.md` → current phase scope, decisions, validation, DoD.
* `docs/decision.md` → important architectural decisions.
* `docs/architecture.md` → actual system architecture changes.

Do not update documentation for every minor code change.

Do not create duplicate documentation or unnecessary documentation files.

Documentation should describe the current/final state, not every step of the
coding process.

---

## Completion

Before considering a task complete:

1. Implement the requested change.
2. Run appropriate tests/validation.
3. Check for unintended changes.
4. Update relevant documentation if necessary.
5. Clearly report what changed and what was validated.
