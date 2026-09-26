# AdaptiveRAG — Corpus

## Purpose

The corpus provides the fixed research-paper collection used throughout
AdaptiveRAG experiments.

The corpus is intentionally focused on RAG and Information Retrieval so that
the project can construct queries that expose differences between retrieval
strategies.

---

# Corpus Size

Total documents:

```text
14 research papers
```

---

# Papers

## Foundational RAG

1. `rag_lewis_2020`
2. `realm_guu_2020`
3. `retro_borgeaud_2022`

## Dense Retrieval

4. `dpr_karpukhin_2020`
5. `contriever_izacard_2022`

## Sparse Retrieval

6. `bm25_robertson_2009`
7. `splade_v2_formal_2021`

## Hybrid Retrieval

8. `rrf_cormack_2009`
9. `pyserini_lin_2021`

## Reranking

10. `monobert_nogueira_2019`
11. `colbert_khattab_2020`

## Advanced RAG

12. `self_rag_asai_2023`
13. `crag_yan_2024`

## Adaptive Retrieval

14. `adaptive_rag_jeong_2024`

---

# Metadata

Each paper has structured metadata including:

```text
document_id
title
authors
year
category
venue
source_url
download_url
local_path
description
concepts
sha256
size_bytes
```

The metadata manifest is maintained separately from the raw PDFs.

---

# Reproducibility

The corpus uses deterministic downloading and validation.

The corpus pipeline supports:

* repeatable downloads
* file validation
* SHA-256 verification
* metadata-based identification

The manifest hash allows downloaded files to be checked against the expected
corpus version.

---

# Validation

PDF validation checks basic file integrity and expected PDF structure.

The corpus was validated during Phase 1.

Phase 1 tests also verify corpus-related behavior.

---

# Git Policy

Raw PDFs are intentionally excluded from Git.

The repository tracks the metadata and scripts required to reproduce/validate
the corpus rather than committing the binary PDF collection.

---

# Phase 1 Boundary

Phase 1 establishes the corpus only.

It does not define:

* chunking
* embeddings
* vector indexing
* retrieval
* generation

Those decisions belong to later phases.
