# AdaptiveRAG: Query-Aware Retrieval Optimization for Efficient RAG

AdaptiveRAG investigates query-aware retrieval optimization: dynamically evaluating whether an incoming query warrants expensive dense retrieval, multi-hop search, and cross-encoder reranking, or whether simpler lexical retrieval (BM25) or single-stage retrieval suffices.

---

## Phase 1: Research Paper Corpus

Phase 1 establishes the canonical research-paper corpus serving as the dataset for all downstream ingestion, indexing, and retrieval evaluations.

### Corpus Summary
* **Total Documents**: 14 peer-reviewed / preprint foundational papers in IR and NLP.
* **Format**: Original authoritative open-access PDFs (arXiv, university repositories).
* **Metadata Manifest**: `data/metadata/papers.json` with document IDs, full bibliographic metadata, key concept taxonomies, verified SHA-256 digests, and file sizes.

### Paper Categories Covered
1. **Foundational RAG**: RAG (Lewis et al.), REALM (Guu et al.), RETRO (Borgeaud et al.)
2. **Dense Retrieval**: DPR (Karpukhin et al.), Contriever (Izacard et al.)
3. **Sparse / Lexical Retrieval**: BM25 (Robertson & Zaragoza), SPLADE v2 (Formal et al.)
4. **Hybrid Retrieval**: Reciprocal Rank Fusion (Cormack et al.), Pyserini (Lin et al.)
5. **Reranking**: monoBERT (Nogueira & Cho), ColBERT (Khattab & Zaharia)
6. **Advanced RAG**: Self-RAG (Asai et al.), Corrective RAG (CRAG) (Yan et al.)
7. **Adaptive Retrieval**: Adaptive-RAG (Jeong et al.)

Detailed descriptions, rationales, and concept mappings are documented in [`docs/corpus.md`](docs/corpus.md).

---

## Repository Structure

```
AdaptiveRAG/
├── data/
│   ├── raw/                  # Downloaded raw PDF papers (gitignored)
│   ├── processed/            # Placeholder for extracted/processed text in Phase 2
│   └── metadata/
│       └── papers.json       # Canonical corpus metadata manifest
├── docs/
│   └── corpus.md             # Corpus documentation and taxonomy rationale
├── scripts/
│   ├── download_corpus.py    # Deterministic downloader with retries and hash validation
│   └── validate_corpus.py    # Manifest and file integrity validator
├── tests/
│   └── test_corpus.py        # Automated test suite
├── pyproject.toml            # Project configuration
└── README.md
```

---

## Reproducing the Corpus

### 1. Download Corpus Documents
The corpus can be fully reproduced and validated using pure Python standard library:

```bash
python3 scripts/download_corpus.py
```

This script:
* Reads `data/metadata/papers.json`
* Downloads missing documents from authoritative sources (e.g. arXiv)
* Validates PDF magic bytes (`%PDF-`) and SHA-256 integrity
* Skips existing valid files (use `--force` to force re-download)

### 2. Validate the Manifest & Files
Verify the metadata schema, URLs, unique IDs, category completeness, and PDF file status:

```bash
python3 scripts/validate_corpus.py
```

To validate only the JSON schema without checking local PDFs:
```bash
python3 scripts/validate_corpus.py --manifest-only
```

### 3. Run Automated Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
