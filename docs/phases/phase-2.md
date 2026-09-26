# Phase 2 — Dense RAG Baseline

**Status:** COMPLETE
**Phase:** 2
**Purpose:** Build the first end-to-end Dense RAG baseline on the frozen Phase 1 corpus.

---

## 1. Objective

Implement a reproducible Dense RAG pipeline that:

1. Ingests the Phase 1 research-paper corpus.
2. Produces structure-aware chunks.
3. Generates dense embeddings.
4. Stores embeddings in a local vector index.
5. Retrieves relevant chunks using dense similarity.
6. Generates answers from retrieved context.
7. Provides a deterministic evaluation framework for retrieval, generation, and efficiency.

This phase establishes the baseline against which later retrieval strategies are compared.

---

## 2. Scope

### Included

* PDF ingestion
* Structure-aware chunking
* Dense embeddings
* Embedding cache
* Local Qdrant index
* Fixed top-k dense retrieval
* Context construction
* LLM generation
* Evaluation dataset
* Retrieval metrics
* Generation metrics
* Efficiency metrics
* Experiment runner
* Offline/deterministic testing

### Excluded

The following were intentionally not implemented:

* BM25
* Hybrid retrieval
* Reranking
* Adaptive routing
* Query classification
* Retrieval-strategy selection

These belong to later phases.

---

## 3. Implementation

The Phase 2 pipeline is organized into modular components:

```text
PDFs
  ↓
Ingestion
  ↓
Document
  ↓
Structure-aware Chunking
  ↓
Chunks
  ↓
Embedding + Cache
  ↓
Qdrant Index
  ↓
Dense Retrieval
  ↓
Context Builder
  ↓
Generation
  ↓
Evaluation
```

Major implementation areas:

* `ingestion/` — PDF → structured `Document`
* `chunking/` — structure-aware chunk generation
* `embeddings/` — embedding provider and cache
* `indexing/` — local Qdrant vector index
* `retrieval/` — dense top-k retrieval
* `generation/` — context construction and LLM generation
* `evaluation/` — experiments and metrics
* `rag.py` — runtime composition

---

## 4. Frozen Chunk Corpus

The final Phase 2 corpus contains:

* **713 chunks**
* Mean size: **408 tokens**
* Median size: **447 tokens**
* **106 atomic chunks**
* Maximum size: **891 tokens**

The chunking configuration uses structure-aware boundaries with:

* Target: 500 tokens
* Maximum: 800 tokens
* Minimum: 150 tokens
* Overlap: 80 tokens

Some larger chunks are intentionally retained when splitting would damage table structure.

The corpus was frozen after iteration because the initial configuration produced excessive fragmentation.

---

## 5. Evaluation

A curated evaluation set was created:

```text
data/evaluation/dense_eval_v1.jsonl
```

It contains **20 evaluation examples** validated against the Phase 1 paper manifest.

The experiment framework records:

### Retrieval

* Recall
* Precision
* Hit Rate
* MRR
* nDCG

### Generation

* Lexical F1
* ROUGE-L
* Cached LLM judge results

### Efficiency

* Latency
* Token usage
* Estimated cost

Experiment runs support resume behavior and per-stage failure capture.

---

## 6. Validation

Offline validation was completed successfully.

Validation included:

* Unit/integration tests
* Offline end-to-end smoke test
* Fake providers
* In-memory Qdrant
* Configuration validation
* Experiment artifact generation
* Provider-isolation checks
* Architecture guards preventing future retrieval strategies from entering Phase 2

Current test status:

* **42 pytest tests passing**
* **5 Phase 1 unittest tests passing**

The implementation remains deterministic and does not require live API credentials for tests.

---

## 7. Live Validation

The implementation is complete, but live provider execution remains a user-side validation step requiring configured credentials.

Expected live sequence:

```text
check_providers.py
        ↓
embed_chunks.py
        ↓
build_index.py
        ↓
run_experiment.py --name dense_baseline_v1
        ↓
ask.py
```

Live validation should confirm that the complete Dense RAG pipeline works with the configured embedding and generation providers.

---

## 8. Phase 2 Completion Criteria

| Requirement                          | Status                      |
| ------------------------------------ | --------------------------- |
| PDF ingestion                        | Complete                    |
| Structure-aware chunking             | Complete                    |
| Embedding pipeline                   | Complete                    |
| Embedding cache                      | Complete                    |
| Vector index                         | Complete                    |
| Dense retrieval                      | Complete                    |
| Context construction                 | Complete                    |
| Generation                           | Complete                    |
| Evaluation dataset                   | Complete                    |
| Evaluation runner                    | Complete                    |
| Retrieval metrics                    | Complete                    |
| Generation metrics                   | Complete                    |
| Efficiency metrics                   | Complete                    |
| Offline validation                   | Complete                    |
| Live provider validation             | Pending user-side execution |
| BM25 / Hybrid / Reranking / Adaptive | Not part of Phase 2         |

---

## 9. Freeze

Phase 2 is considered **implementation-complete and frozen**.

The Dense RAG implementation provides the baseline for subsequent retrieval strategies.

Future phases must preserve the evaluation framework so that new strategies can be compared against this baseline under consistent conditions.

Detailed architectural decisions are documented in:

* `docs/architecture.md`
* `docs/decisions.md`

Current project status and next steps are tracked in:

* `docs/progress.md`
