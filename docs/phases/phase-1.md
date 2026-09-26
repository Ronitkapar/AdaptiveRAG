# Phase 1 — Corpus Foundation

Status: COMPLETE

---

## Objective

Create a fixed, reproducible corpus that can be used for all later
retrieval experiments.

---

## Completed

* Selected 14 RAG / Information Retrieval research papers.
* Created canonical metadata.
* Implemented deterministic downloading.
* Implemented PDF validation.
* Implemented SHA-256 verification.
* Added corpus tests.
* Added corpus documentation.
* Configured Git to ignore raw PDFs.

---

## Corpus Categories

```text
Foundational RAG
Dense Retrieval
Sparse Retrieval
Hybrid Retrieval
Reranking
Advanced RAG
Adaptive Retrieval
```

---

## Reproducibility

The corpus can be validated against the metadata manifest.

The manifest contains file identity information including SHA-256 hashes.

---

## Deliberate Non-Goals

Phase 1 did not implement:

* PDF text extraction
* document hierarchy
* chunking
* embeddings
* vector indexing
* retrieval
* generation
* evaluation
* adaptive routing

These were intentionally deferred to Phase 2 and later phases.

---

## Definition of Done

Phase 1 is complete when:

* the 14-paper corpus exists
* metadata is available
* downloads are deterministic
* PDFs validate
* hashes can be checked
* tests pass
* raw PDFs are excluded from Git
* corpus documentation exists

Phase 1 is complete.
