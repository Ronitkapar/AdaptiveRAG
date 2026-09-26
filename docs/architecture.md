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

The current implementation has established the Dense-RAG baseline.

Future retrieval strategies should reuse the existing retrieval and
evaluation contracts wherever possible.

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
* perform hybrid retrieval
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
