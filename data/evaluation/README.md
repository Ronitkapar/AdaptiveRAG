# Evaluation Dataset — `dense_eval_v1`

20 curated question/answer pairs over the 14-paper Phase 1 corpus, stored as one
`EvaluationExample` per JSONL line.

## Schema

- `example_id` — stable id (`eval_001` … `eval_020`)
- `query` — the benchmark question
- `reference_answer` — curator-written ground truth from the paper text
- `relevant_documents` — primary relevance labels (`document_id`s from `papers.json`)
- `relevant_sections` — optional section-path prefixes for finer granularity
- `relevant_chunks` — intentionally empty in v1 (chunk ids depend on chunking config)
- `category` — one of `factual | terminology | conceptual | comparative | multi_document | fine_grained`
- `requires_multi_hop` — true when the answer synthesizes multiple papers
- `notes` — section pointer used during curation
- `dataset_version` — `dense_eval_v1`

## Coverage

- 15 single-paper questions (≥1 per corpus paper, incl. all 7 taxonomy categories)
- 5 cross-paper comparative / multi-hop questions

## Validation rules (enforced by `evaluation.dataset`)

1. `example_id`s are unique and non-empty `relevant_documents` is required.
2. Every `relevant_documents` id must exist in `data/metadata/papers.json`.
3. Chunk-level relevance is **derived at evaluation time** from document (+ optional
   section-prefix) labels with a documented rule, so labels never depend on
   whichever retriever happens to retrieve them.

To extend the benchmark, append validated lines and bump `dataset_version`.

---

## `phase7_eval_pilot` — curation ledger (Phase 7.0c)

`dense_eval_v1` is not verified: its 20 records carry document-level labels with no
quotation, so nothing can distinguish a curated label from an invented one. Phase 7
expands the benchmark to ~100 queries, and at that volume "write a plausible answer
and a plausible-sounding quote" is the realistic failure mode. The cure is a second
file that is checked against the corpus, not against itself.

### Files

- `phase7_eval_pilot.jsonl` — benchmark records, one `EvaluationExample` per line,
  `dataset_version: "phase7_eval_pilot"`. Deliberately minimal: same shape as
  `dense_eval_v1` plus the `split` field.
- `phase7_eval_pilot.ledger.jsonl` — curation provenance, one record per query.

Evidence lives in the ledger rather than the benchmark because the benchmark is a
metric input: `EvaluationExample` is `extra="forbid"`, and provenance is not a
metric. A companion file can also be *enforced* — a ledger record with no benchmark
record is a gate failure, whereas evidence smuggled into the benchmark can be
deleted without anything noticing.

### Ledger record format

```json
{"example_id":"p7_001","split":"calibration","relevant_documents":["rag_lewis_2020"],
 "evidence":[{"document_id":"rag_lewis_2020",
              "chunk_id":"rag_lewis_2020::structure_aware_v1::c00005",
              "quote":"<verbatim substring of that chunk's text>"}],
 "answer_terms":["RAG-Sequence","RAG-Token","top K","latent"],
 "curator_notes":"why this document is relevant"}
```

- `example_id` / `split` / `relevant_documents` are assertions about the benchmark
  record; the gate requires exact equality with it, order included.
- `evidence` — at least one item; each must name a real chunk and quote it verbatim.
- `answer_terms` — the content terms of `reference_answer`; each must occur in the
  concatenated quotes. This is what makes an invented answer unrepresentable.
- `curator_notes` — required non-empty, not otherwise checked.

### Verifying a dataset

```bash
python scripts/validate_phase7_dataset.py \
    --dataset data/evaluation/phase7_eval_pilot.jsonl \
    --ledger  data/evaluation/phase7_eval_pilot.ledger.jsonl
```

Writes `experiments/phase7/dataset_gate.json` and exits non-zero on any failure.
See the script's docstring for the full check list (S1–S5 structural, G1–G6
grounding) and for why an absent chunk corpus is a failure rather than a skip.

Note the pilot's category minimum: 12 queries over 4 categories cannot satisfy the
Phase 7 target of 5 per category (4x5 = 20 > 12), so the pilot is validated with
`--min-per-category 3`. The artifact records `min_category_relaxed: true` and the
target value, so a pilot pass is never mistaken for a Phase-7-final pass. The
pilot also carries no `comparative` or `multi_document` query: these two papers
support no honest cross-paper label, and faking one would defeat the point.

---

## `phase7_eval_v1` — frozen Phase 7 benchmark

The single dataset Phase 7 ablations, routing thresholds and the final report are
measured against. It is a merge of the six curation parts (pilot, parts a–e) into one
file, so that a Phase 7 number is always reproducible from one artifact with one gate
run.

### Files

- `phase7_eval_v1.jsonl` — 88 `EvaluationExample` records, one per line,
  `dataset_version: "phase7_eval_v1"` on every line.
  sha256 `9d7885e9b9d13493ba2eab6ff85dbec19541bfd866923efc730f7e5cf753d00d`
- `phase7_eval_v1.ledger.jsonl` — 88 curation-provenance records, 1:1 with the dataset.
  sha256 `435ffea04befe4310cfb088b8e5a9aa44da26deff9adb77f6db0fbe7ee8bde9e`

`phase7_eval_v1` **supersedes `dense_eval_v1` for Phase 7**. `dense_eval_v1.jsonl` is
left byte-for-byte untouched and remains the benchmark of record for Phases 2–6, so the
already-reported baseline numbers stay reproducible against the artifact they were
computed from.

### Counts

| split | n |  | category | n | calibration | test |
|---|---|---|---|---|---|---|
| calibration | 36 | | comparative | 8 | 4 | 4 |
| test | 52 | | conceptual | 23 | 6 | 17 |
| **total** | **88** | | factual | 20 | 10 | 10 |
|  |  |  | fine_grained | 14 | 5 | 9 |
|  |  |  | multi_document | 8 | 2 | 6 |
|  |  |  | terminology | 15 | 9 | 6 |

Per-paper coverage — 13 of the 14 corpus papers:

| paper | n | paper | n |
|---|---|---|---|
| `bm25_robertson_2009` | 18 | `colbert_khattab_2020` | 10 |
| `rag_lewis_2020` | 9 | `contriever_izacard_2022` | 10 |
| `adaptive_rag_jeong_2024` | 8 | `monobert_nogueira_2019` | 10 |
| `crag_yan_2024` | 8 | `splade_v2_formal_2021` | 10 |
| `rrf_cormack_2009` | 8 | `retro_borgeaud_2022` | 7 |
| `self_rag_asai_2023` | 8 | `dpr_karpukhin_2020` | 4 |
|  |  | `realm_guu_2020` | 2 |

`pyserini_lin_2021` has **0** records — see the corpus defect below.
`realm_guu_2020` (2) and `dpr_karpukhin_2020` (4) are thin because their part-A records
failed grounding and were dropped; per-category counts, not per-paper counts, are what
the gate enforces, and all six categories clear the minimum of 5.

### `example_id` prefix scheme

Ids are **not** renumbered. The prefix records which curation part produced the record,
so any label can be traced back to the part's curation provenance and to that part's
ledger line:

| prefix | source part | ids in `phase7_eval_v1` | n |
|---|---|---|---|
| `p7_` | `phase7_eval_pilot.jsonl` | `p7_001`…`p7_012` | 12 |
| `p7a_` | `phase7_part_a.jsonl` | `p7a_005`…`p7a_018` (gaps — see below) | 6 |
| `p7b_` | `phase7_part_b.jsonl` | `p7b_001`…`p7b_018` | 18 |
| `p7c_` | `phase7_part_c.jsonl` | `p7c_001`…`p7c_018` | 18 |
| `p7d_` | `phase7_part_d.jsonl` | `p7d_001`…`p7d_018` | 18 |
| `p7e_` | `phase7_part_e.jsonl` | `p7e_001`…`p7e_016` | 16 |

The input parts totalled 112 records; 88 survive. The two id gaps are meaningful, not
an accident, and every dropped id is listed below so a reader can tell "absent" from
"never curated".

### Ledger format and re-verifying grounding

The format is unchanged from `phase7_eval_pilot` (above): one JSON object per line with
`example_id`, `split`, `relevant_documents`, `evidence[]` (`document_id`, `chunk_id`,
`quote`), `answer_terms` and `curator_notes`.

Grounding is not a claim in the documentation — it is re-derived mechanically from the
chunks on disk every time the gate runs. Re-verify with:

```bash
python scripts/validate_phase7_dataset.py \
    --dataset data/evaluation/phase7_eval_v1.jsonl \
    --ledger  data/evaluation/phase7_eval_v1.ledger.jsonl \
    --out experiments/phase7/dataset_gate.json
```

This is the real gate, not a relaxed one: no `--min-per-category` override, so
`min_per_category` is the Phase 7 target of 5 and the artifact records
`min_category_relaxed: false`. It writes `experiments/phase7/dataset_gate.json` and
exits 0 only when every check passes. The recorded run is
`gate_open: true`, `grounding_verified: true`, `records_pass 88/88`, 713 chunks read
from `data/processed/chunks`.

Checks are S1–S5 (structure, uniqueness, split separation, non-empty splits,
per-category minimum) and G1–G6 (ledger↔dataset 1:1 both directions, exact agreement on
`split`/`relevant_documents`, evidence chunk exists and belongs to a relevant document,
**quote is a verbatim whitespace-normalised substring of that chunk and ≥20 chars**, every
`answer_term` occurs in the verified quotes, every `section_path_prefix` matches at least
one chunk of a relevant document). G4/G5 are what make an invented answer
unrepresentable: an answer cannot be grounded in source text it does not share, and a
quote that "sounds like the paper" but appears in no chunk is rejected mechanically.

### Records dropped, and why

Two independent reasons, both applied before the gate was re-run. No label, quote,
`chunk_id` or section path was edited, and the gate was not relaxed.

**1. `pyserini_lin_2021` corpus defect — 6 records dropped (`p7b_019`…`p7b_024`).**
`data/metadata/papers.json` declares `pyserini_lin_2021` as *"Pyserini: A Python Toolkit
for Reproducible Information Retrieval Research with Sparse and Dense Representations"*
(arXiv 2102.10062), but the downloaded PDF and all 91 of its chunks are **"A Review of
Biomedical Datasets Relating to Drug Discovery: A Knowledge Graph Perspective"** (Bonner
et al., AstraZeneca). Independently re-verified on the chunk corpus: 0 of 91 chunks
mention Pyserini, BM25, Lucene, Faiss, HNSW or "inverted index"; the chunk frontmatter
carries the Bonner et al. title and author list; 59 chunks mention drug-discovery
/knowledge-graph terms. Checksum validation passed vacuously because the manifest sha256
was recorded *after* download, and the declared title is stamped into every chunk's
metadata, so the mismatch is invisible downstream.

A query whose gold evidence is chunks of a drug-discovery knowledge-graph review is a
*correct* answer to a *different* question, and the gate cannot catch this: the quotes
really are verbatim in those chunks. So these are dropped, not repaired.

`p7b_019` … `p7b_024` were dropped. A scan of **all six parts** — dataset
`relevant_documents`, `relevant_sections`, ledger `relevant_documents`, every evidence
`document_id` and `chunk_id`, and every `answer_terms` — found `pyserini_lin_2021`
referenced **only** by those 6 records; no surviving record refers to it in any field.

**Remediation is deliberately deferred.** The correct fix is
re-download → re-ingest → re-chunk → re-embed → re-index, which changes
`corpus_version`, and that would invalidate the already-reported Phase 2–6 baselines and
the frozen Phase 7.0b strategy-cost table. Re-chunking also renumbers every `chunk_id`,
which would invalidate the grounded evidence of all 88 surviving records. The defect is
therefore contained by exclusion and documented here, to be fixed as its own corpus
version bump outside Phase 7.

**2. Grounding failures — 18 records dropped (all `p7a_*`).**
Part A's ledger contains **30 empty `quote` strings across 60 evidence items in 16
records** (the other five parts have zero). An empty quote cannot satisfy G4, and because
G5 checks `answer_terms` against the *concatenated verified quotes*, an empty quote also
cascades into G5 failures. `p7a_001` and `p7a_004` fail G5 for a different reason — their
`answer_terms` (`'Curated Trec'`, `'fouroptions'`) are spelled with spaces while the paper
extracts with no inter-word spaces (`CuratedTrec(CT)`, `fouroptions`). Fixing either would
mean editing evidence or a label, so the records were dropped.

Dropped: `p7a_001`, `p7a_002`, `p7a_003`, `p7a_004`, `p7a_006`, `p7a_007`, `p7a_008`,
`p7a_009`, `p7a_010`, `p7a_011`, `p7a_012`, `p7a_017`, `p7a_019`, `p7a_020`, `p7a_021`,
`p7a_022`, `p7a_023`, `p7a_024`. Retained from part A: `p7a_005`, `p7a_013`, `p7a_014`,
`p7a_015`, `p7a_016`, `p7a_018`.

### Known interpretation caveats — the report must repeat these

1. **`Precision@k` is document-derived.** `relevant_chunk_ids`
   (`src/adaptive_rag/evaluation/dataset.py`) treats every chunk of a relevant document
   as relevant unless a `section_path_prefix` matches. So `Precision@k` partly measures
   how many chunks of a relevant document a retriever happened to return, not whether it
   found the right passage. Every surviving record does carry a section prefix (0 records
   are document-level only), and prefix granularity is good — of 124 section labels, 61
   match a single chunk, the median is 2, the maximum is 13, and none is 0 or above 20 —
   so the caveat is a statement about the metric's construction, not a live data defect.
2. **`section_path` values are noisy.** They are copied verbatim from chunk metadata and
   contain running heads, author affiliations and page numbers, e.g.
   `"3 Experiments" -> "3.1 Open-domainQuestionAnswering"`. Do not read them as a clean
   section taxonomy.
3. **Orthography varies per paper.** Some PDFs extract with no inter-word spaces, so
   `answer_terms` follow each paper's own orthography (`vicious cycle` and
   `cold-startproblem` in the same dataset). Answer-term matching is whitespace-normalised
   but not de-hyphenated or ligature-repaired, which is why a term can fail to match and
   the record is dropped rather than silently normalised.
