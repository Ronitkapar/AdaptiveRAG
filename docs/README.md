# AdaptiveRAG Documentation

This directory contains the project's current architecture, decisions, progress,
and phase-specific documentation.

## Start Here

For coding-agent work, normally read:

1. `../AGENTS.md`
2. `progress.md`
3. `phases/<current-phase>.md`

Read additional documents only when the task requires them.

---

## Documentation Files

### `progress.md`

Current project state.

Answers:

> Where is the project right now?

Contains:

* completed phases
* current phase
* current implementation status
* validation status
* next steps

---

### `decision.md`

Current accepted architectural decisions.

Answers:

> What decisions are currently locked?

This is not a conversation history.

---

### `architecture.md`

Detailed system architecture.

Answers:

> How is the system designed?

This is the deeper architectural reference and does not need to be read for
every coding task.

---

### `corpus.md`

Phase 1 corpus documentation.

Describes:

* corpus composition
* metadata
* provenance
* validation
* reproducibility

---

### `phases/`

Phase-specific implementation documentation.

Completed phases remain available as reference.

The active phase should normally be the primary phase document read by the
coding agent.

---

## Context Principle

Keep active context small.

The coding agent should not load every documentation file for every task.

Prefer:

```text
AGENTS.md
    +
progress.md
    +
current phase
```

and retrieve deeper documentation only when necessary.
