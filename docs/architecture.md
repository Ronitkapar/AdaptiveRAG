# AdaptiveRAG — System Architecture

## 1. Purpose

AdaptiveRAG is a retrieval research/engineering system designed to study
whether retrieval strategies can be selected according to query
characteristics while balancing:

* retrieval quality
* answer quality
* latency
* token usage
* estimated inference cost

The system is developed progressively so that each retrieval strategy can be
measured independently before adaptive routing is introduced.

---

# 2. Retrieval Strategy Progression

The eventual architecture supports:

```text
Dense
   ↓
BM25
   ↓
Hybrid
   ↓
Reranking
   ↓
Adaptive Routing
```

The current implementation has established the Dense-RAG baseline (Phase 2),
the BM25 lexical baseline (Phase 3), and the hybrid rank-fusion strategy
(Phase 4). All three operate over the same canonical chunk corpus and share the
same retrieval/evaluation contracts.

Future retrieval strategies should reuse the existing retrieval and
evaluation contracts wherever possible.

---

# 3a. Phase 3 BM25 Branch

Phase 3 adds an independent lexical branch alongside the dense pipeline:

```text
Structured Chunks (canonical corpus)
        ↓
BM25Index (inverted index, JSON at storage/bm25/)
        ↓
BM25Retriever (native Okapi BM25 scores)
        ↓
Retrieved Context (common RetrievalResponse)
        ↓
Context Builder / Evaluation
```

Guarantees:

* Corpus parity: BM25 reads the same canonical chunks as dense, without
  modifying them or assuming a fixed count.
* Zero coupling: dense and BM25 never import or fall back to each other.
* Common protocol: both implement `Retriever` and emit `RetrievalResponse`
  with `retrieval_method ∈ {"dense", "bm25"}`, so the evaluation pipeline
  stays strategy-agnostic.
* Scores are native BM25 values (unnormalized); provenance and metadata are
  preserved end-to-end.

---

# 3b. Phase 4 Hybrid Branch

Phase 4 adds a composition layer above the two existing retrievers. It owns no
index, no embedding model, and no tokenizer:

```text
        Canonical Chunk Corpus
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 DenseRetriever       BM25Retriever
 (Qdrant, cosine)     (BM25Index, JSON)
       │                   │
       └─────────┬─────────┘
                 ▼
      reciprocal_rank_fusion(ranked_lists, rrf_k=60)
                 ▼
        truncate → top_k, renumber ranks 1..n
                 ▼
     RetrievalResponse(retrieval_method="hybrid")
                 ▼
      ExperimentRunner + evaluators (unchanged)
```

Guarantees:

* **Composition, not reimplementation.** `hybrid.py` orchestrates; it contains
  no tokenization, embedding, index access, or alternative fusion logic, and no
  `try`/`except` (architecture-guard enforced on the AST).
* **Rank-based fusion.** `score(chunk) = Σ_lists 1 / (rrf_k + rank)`. Cosine
  similarity and BM25 magnitudes are never mixed, and no score normalization is
  introduced.
* **Deterministic output.** Deduplication is by `chunk_id`; ordering is
  `(-rrf_score, best_single_source_rank, chunk_id)`, so nothing depends on
  input or dict iteration order.
* **Candidate depth.** Both branches are queried at `candidate_k` (default 20)
  and truncation to `top_k` happens only after fusion.
* **Fail closed.** Constituent exceptions propagate unchanged, so a broken
  branch yields `status="retrieval_failed"` with the original error type —
  never a silent dense-only or BM25-only answer.
* **No third index.** Hybrid reuses the existing Qdrant collection and the
  persisted BM25 index; there is no `storage/hybrid/` and no build script.
  Because the dense branch embeds every query, a hybrid run requires
  `AICREDITS_API_KEY` even with `--no-judge --no-generation`.

`retrieval_method ∈ {"dense", "bm25", "hybrid"}` across configuration, traces,
and manifests, with `retriever_version` aligned (`hybrid ⇔ hybrid_v1`).

---

# 3. Phase 2 Pipeline

The locked Phase 2 pipeline is:

```text
14 Research Papers
        ↓
PDF Ingestion
        ↓
Hierarchical Documents
        ↓
Structure-Aware Chunking
        ↓
Structured Chunks
        ↓
Embedding Model
        ↓
Embedding Cache
        ↓
Qdrant
        ↓
Dense Retriever
        ↓
Retrieved Context
        ↓
Context Builder
        ↓
Groq Generator
        ↓
Answer + Sources
        ↓
Experiment Trace
        ↓
Evaluation
```

---

# 4. Ingestion Layer

The ingestion layer converts PDFs into structured, provenance-preserving
Documents.

Responsibilities:

* text extraction
* heading extraction
* table preservation where reliably extractable
* equation preservation where reliably extractable
* figure/caption representation
* page information
* logical section structure
* source order
* provenance
* extraction failure reporting

Ingestion does not perform:

* chunking
* embedding
* indexing
* retrieval
* generation
* routing

---

# 5. Document Model

Documents preserve both physical and logical structure.

Conceptually:

```text
Document
├── document_id
├── metadata
├── pages[]
├── sections[]
└── provenance
```

Sections contain ordered elements and may contain subsections.

---

# 6. Element Model

Supported element types include:

```text
Paragraph
Heading
Figure
Table
Equation
Caption
List
```

Elements have deterministic identity and source order.

Provenance includes at least:

```text
document_id
page
```

Bounding boxes are included only when reliably available.

---

# 7. Chunking Layer

Phase 2 uses structure-aware chunking.

The conceptual flow is:

```text
Document
   ↓
Sections
   ↓
Subsections
   ↓
Elements
   ↓
Coherent Element Groups
   ↓
Chunks
```

Chunking preserves:

* source order
* section path
* headings
* element IDs
* page range
* provenance

Chunks are derived artifacts; the Document remains canonical.

---

# 8. Chunk Contract

A Chunk conceptually contains:

```text
chunk_id
document_id
text
section_path
headings
element_ids
page_start
page_end
provenance
chunking_metadata
```

Embeddings are deliberately not stored in the canonical Chunk model.

The frozen Phase 2 corpus contains:

```text
713 chunks
Mean: approximately 408 tokens
Median: approximately 447 tokens
106 atomic chunks
Maximum: approximately 891 tokens
```

The maximum chunk is intentionally an unsplit table.

---

# 9. Embedding Layer

Phase 2 uses:

```text
AICredits
    ↓
text-embedding-3-large
```

The embedding implementation is hidden behind an embedding interface.

The embedding cache is content/configuration aware so incompatible embeddings
are not accidentally reused.

---

# 10. Vector Index

Phase 2 uses:

```text
Qdrant
```

locally.

The vector store is hidden behind a vector-store abstraction.

Index metadata records information such as:

* corpus version
* chunking version/configuration
* embedding model/configuration
* vector dimension
* similarity configuration
* index version

---

# 11. Dense Retrieval

The DenseRetriever:

```text
Query
 ↓
Query Embedding
 ↓
Qdrant Search
 ↓
Ranked Retrieval Results
```

The Phase 2 baseline uses a fixed `top_k`.

Retrieval results preserve:

* chunk ID
* text
* score
* rank
* metadata
* provenance

Dense retrieval does not:

* rerank
* generate
* perform BM25
* perform hybrid retrieval or rank fusion
* perform adaptive routing

---

# 12. Generation Layer

Generation is independent from embedding.

```text
Generator
    ↓
Groq API
    ↓
Selected Generation Model
```

The generator receives a structured GenerationRequest rather than accessing
the corpus directly.

---

# 13. Context Builder

The Context Builder sits between retrieval and generation:

```text
RetrievalResponse
        ↓
Context Builder
        ↓
GenerationRequest
```

It:

* preserves retrieval order
* formats retrieved chunks
* includes source metadata
* respects context/token limits
* prepares generator input

---

# 14. Generation Result

Generation produces a structured result containing information such as:

```text
answer
source/chunk IDs
model
token usage
latency
metadata
```

Prompt configuration is versioned.

---

# 15. Evaluation Architecture

Evaluation is an independent observer.

```text
System
   ↓
Raw Experiment Trace
   ↓
Evaluation
   ↓
Metrics
```

Evaluation is not embedded inside retrieval or generation.

It is intended to remain reusable for all future retrieval strategies.

---

# 16. Evaluation Dimensions

### Retrieval

Potential/current metrics include:

* Recall@k
* Precision@k
* Hit@k
* MRR
* nDCG

### Generation

Metrics include:

* lexical F1
* ROUGE-L
* cached model-based judging where configured
* correctness/relevance/groundedness-oriented evaluation

### Efficiency

Measure:

* retrieval latency
* generation latency
* total latency
* input tokens
* output tokens
* total tokens
* estimated cost

---

# 17. Experiment Architecture

The ExperimentRunner is retrieval-strategy agnostic.

Conceptually:

```text
Evaluation Dataset
        ↓
ExperimentRunner
        ↓
Retriever
        ↓
Context Builder
        ↓
Generator
        ↓
Raw Traces
        ↓
Evaluation
        ↓
Metrics + Artifacts
```

The same architecture should support future retrieval strategies.

---

# 18. Raw Experiment Traces

Per-query traces retain information needed for analysis and debugging:

```text
query
retrieval method
retrieved chunks
scores
ranks
reference/relevance information
answer
source IDs
latency
token usage
estimated cost
errors
configuration/version
```

Aggregate metrics should be traceable back to raw results.

---

# 19. Provider Boundaries

Provider-specific implementations are isolated.

```text
Embedding
    ↓
AICredits
    ↓
text-embedding-3-large
```

and:

```text
Generation
    ↓
Groq
    ↓
Generation Model
```

The rest of the application should use the project interfaces rather than
provider-specific APIs directly.

---

# 20. Runtime vs Experiment Path

Runtime path:

```text
User Query
    ↓
Retriever
    ↓
Context Builder
    ↓
Generator
    ↓
Answer + Sources
```

Experiment path:

```text
Evaluation Dataset
    ↓
Retriever
    ↓
Generator
    ↓
Evaluation
    ↓
Metrics
    ↓
Traces + Artifacts
```

These paths share core components but have different responsibilities.

---

# 21. Architectural Principle

The project follows this principle:

> Build fixed retrieval systems first, evaluate them independently, and only
> then build the adaptive system that chooses between them.

This prevents the adaptive system from being evaluated against poorly defined
or incomparable retrieval baselines.
