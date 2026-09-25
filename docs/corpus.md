# AdaptiveRAG Research Corpus Specification

## 1. Overview & Research Motivation

AdaptiveRAG explores query-aware retrieval optimization: determining *when* a query actually warrants expensive multi-hop, dense, or cross-encoder reranking versus when lightweight BM25 or single-stage retrieval suffices.

To rigorously evaluate lexical, dense, hybrid, reranked, and adaptive retrieval strategies in subsequent phases, we establish an experimental research-paper corpus of **14 seminal papers** in Information Retrieval (IR) and Natural Language Processing (NLP).

This corpus serves as our canonical reference dataset for indexing, chunking, retrieval benchmark queries, and routing evaluation.

---

## 2. Corpus Taxonomy & Paper Rationale

The corpus spans 7 core retrieval categories. Each paper was selected for both its historical significance and its explicit technical contrasts with others in the corpus:

### Category 1: Foundational RAG
* **`rag_lewis_2020`** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., NeurIPS 2020)
  * *Significance*: Defines the modern parametric + non-parametric memory formulation. Distinguishes RAG-Sequence and RAG-Token generation models.
* **`realm_guu_2020`** — *REALM: Retrieval-Augmented Language Model Pre-Training* (Guu et al., ICML 2020)
  * *Significance*: Demonstrates end-to-end differentiable retrieval pre-training within masked language models, utilizing asynchronous MIPS index refreshing.
* **`retro_borgeaud_2022`** — *Improving Language Models by Retrieving from Trillions of Tokens* (Borgeaud et al., ICML 2022)
  * *Significance*: Scales retrieval integration to autoregressive language models trained over trillion-token corpora using chunked cross-attention.

### Category 2: Dense Retrieval
* **`dpr_karpukhin_2020`** — *Dense Passage Retrieval for Open-Domain Question Answering* (Karpukhin et al., EMNLP 2020)
  * *Significance*: Establishes dual-encoder embeddings trained with in-batch negatives and BM25 hard negatives as superior to classical BM25 on open-domain QA.
* **`contriever_izacard_2022`** — *Unsupervised Dense Information Retrieval with Contrastive Learning* (Izacard et al., TMLR 2022)
  * *Significance*: Proves dense bi-encoders can be trained without supervision using inverse cloze tasks (ICT) and contrastive data augmentations, showing strong out-of-domain transfer on BEIR.

### Category 3: Sparse / Lexical Retrieval
* **`bm25_robertson_2009`** — *The Probabilistic Relevance Framework: BM25 and Beyond* (Robertson & Zaragoza, FnTIR 2009)
  * *Significance*: The definitive mathematical treatise on Okapi BM25, term frequency saturation ($k_1$), and document length normalization ($b$).
* **`splade_v2_formal_2021`** — *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval* (Formal et al., 2021)
  * *Significance*: Bridges sparse and neural paradigms via learned vocabulary-space activations and sparse regularization (L1 / FLOPS), resolving vocabulary mismatch while keeping inverted-index efficiency.

### Category 4: Hybrid Retrieval
* **`rrf_cormack_2009`** — *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (Cormack et al., SIGIR 2009)
  * *Significance*: Proves reciprocal rank fusion ($1 / (k + r)$) combines disparate retriever rankings robustly without requiring normalized score calibration.
* **`pyserini_lin_2021`** — *Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations* (Lin et al., SIGIR 2021)
  * *Significance*: Unifies sparse (Lucene BM25) and dense (Faiss HNSW) search under a reproducible hybrid linear fusion framework ($\alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$).

### Category 5: Reranking
* **`monobert_nogueira_2019`** — *Passage Re-ranking with BERT* (Nogueira & Cho, 2019)
  * *Significance*: Foundational cross-encoder re-ranker. Feeds concatenated query-passage pairs into full self-attention layers to produce relevance scores, illustrating the top-tier accuracy vs. latency trade-off.
* **`colbert_khattab_2020`** — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (Khattab & Zaharia, SIGIR 2020)
  * *Significance*: Introduces token-level late interaction with the MaxSim operator, approaching cross-encoder precision while maintaining sub-100ms retrieval speed.

### Category 6: Advanced RAG
* **`self_rag_asai_2023`** — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (Asai et al., ICLR 2024)
  * *Significance*: Introduces self-reflective generation with specialized critique tokens (`[Retrieve]`, `[IsREL]`, `[IsSUP]`, `[IsUSE]`) to retrieve passages selectively and critique factual grounding.
* **`crag_yan_2024`** — *Corrective Retrieval Augmented Generation* (Yan et al., 2024)
  * *Significance*: Introduces an external retrieval evaluator with three confidence confidence tiers (Correct, Ambiguous, Incorrect) to guide document strip decomposition or web search fallback.

### Category 7: Adaptive Retrieval
* **`adaptive_rag_jeong_2024`** — *Adaptive-RAG: Determining When and How to Retrieve for Knowledge-Intensive Tasks* (Jeong et al., NAACL 2024)
  * *Significance*: Directly addresses query complexity routing. Uses a trained classifier to dynamically route queries across No-Retrieval, Single-Hop Retrieval, and Multi-Hop Retrieval, optimizing accuracy vs. latency.

---

## 3. Query Types & Evaluation Suitability

The selected papers intentionally share vocabulary, architectures, and experimental datasets (e.g., Natural Questions, TriviaQA, MS MARCO, BEIR), providing rich ground for multiple question classes in later evaluation:

| Question Type | Description | Example Target Query in AdaptiveRAG |
| :--- | :--- | :--- |
| **Factual / Definitional** | Exact parameter values, mathematical formulas, token names | *"What default smoothing parameter $k$ is recommended in Cormack et al. for Reciprocal Rank Fusion?"* |
| **Terminology-Heavy** | High lexical specificity suited for sparse / BM25 search | *"What are the names of the four reflection tokens introduced in Self-RAG?"* |
| **Conceptual / Semantic** | Paraphrased or thematic concepts with vocabulary mismatch | *"How do learned sparse retrieval models avoid out-of-vocabulary mismatches without using dense vector indices?"* |
| **Comparative** | Trade-offs between multiple methods or architectures | *"Compare the computational complexity and index storage requirements of ColBERT late interaction versus DPR bi-encoders."* |
| **Multi-Document / Multi-Hop** | Synthesizing knowledge across two or more separate works | *"How does Adaptive-RAG's query complexity classification differ from Self-RAG's internal reflection token retrieval decision?"* |
| **Fine-Grained Detail** | Ablation study findings, negative sampling specifics, metric nuances | *"How did DPR construct hard negatives during training, and how did in-batch negatives compare with random negatives?"* |

---

## 4. Manifest Schema (`data/metadata/papers.json`)

The machine-readable manifest is stored as a formatted JSON array of document records. Every entry conforms to the schema below:

```json
{
  "document_id": "rag_lewis_2020",
  "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
  "authors": ["Patrick Lewis", "..."],
  "year": 2020,
  "category": "foundational_rag",
  "venue": "NeurIPS 2020",
  "source_url": "https://arxiv.org/abs/2005.11401",
  "download_url": "https://arxiv.org/pdf/2005.11401.pdf",
  "local_path": "data/raw/rag_lewis_2020.pdf",
  "description": "...",
  "concepts": ["parametric memory", "non-parametric memory", "..."],
  "sha256": "23e3249e9a1e7541b370605953e6b21852df85e4905d6e2467d1db0922e96457",
  "size_bytes": 885323
}
```

---

## 5. Reproducibility & Integrity

The corpus is fully reproducible and validated via two core utility scripts:

* **Download Corpus**:
  ```bash
  python3 scripts/download_corpus.py
  ```
  * Idempotent: checks for existing PDFs with valid magic bytes (`%PDF-`) and matching SHA-256 digests.
  * Fault-tolerant: retries network interruptions using exponential backoff.
  * `--force` flag allows full re-downloading if necessary.

* **Validate Corpus**:
  ```bash
  python3 scripts/validate_corpus.py
  ```
  * Checks JSON schema, field existence, types, non-empty lists, URL formatting, and unique IDs.
  * Verifies local PDF existence, size bounds (> 50KB), magic byte header, and SHA-256 integrity.
  * Supports `--manifest-only` mode to test the metadata schema without local files.

* **Automated Unit Tests**:
  ```bash
  python3 -m unittest discover -s tests -p "test_*.py" -v
  ```

---

## 6. Known Corpus Limitations

1. **Domain Uniformity**: All 14 papers belong to the computer science / NLP / IR literature. While ideal for evaluating technical RAG queries, it does not assess behavior on non-technical genres (e.g., creative writing or clinical trials).
2. **Document Length & Density**: Academic papers feature dense mathematical notation, multi-column layouts, tables, and dense citations. Subsequent text extraction (Phase 2) will require sensible handling of formulas and citations.
3. **Absence of Performance Claims**: At this phase, this corpus is strictly an experimental dataset. No comparative claims about retriever performance or adaptive routing efficacy are made yet.
