# AdaptiveRAG — Agent Instructions

## Project

AdaptiveRAG is a query-aware RAG system that evaluates retrieval strategies and
eventually learns to route queries to an appropriate strategy.

Planned progression:

Dense → BM25 → Hybrid → Reranking → Adaptive Routing

The current phase is defined in `docs/progress.md`.

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

Only implement functionality belonging to the current phase.

Current progression:

* Phase 1 — Corpus Foundation: COMPLETE
* Phase 2 — Dense RAG Baseline: COMPLETE
* Phase 3 — BM25: COMPLETE
* Phase 4 — Hybrid: NEXT
* Phase 5 — Reranking: FUTURE
* Phase 6 — Adaptive Routing: FUTURE

Do not implement future phases unless explicitly requested.

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
