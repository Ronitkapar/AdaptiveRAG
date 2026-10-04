# Phase 8 — Corpus Re-extraction and Re-baseline

**Status:** IN PROGRESS — §2 pre-registered before any Phase 8 number exists
**Phase:** 8
**Purpose:** Determine whether a diagnosed corpus defect explains two Phase 7
results — the cross-encoder reranker's collapse, and "dense dominates hybrid" —
by re-extracting the corpus with two-column reading order handled correctly and
re-running the full E-series against the clean corpus.
**Pre-registration:** §2. Written before the new corpus existed.
**Plan of record:** `.kilo/plans/1790943474579-eager-cactus.md`

---

## 1. Objective

Phase 7 executed E1–E8 and recorded two findings that both remain open:

1. **The cross-encoder reranker is net-negative.** `hybrid_rerank` is last on
   every E1 quality metric and 7.5x hybrid's median latency. Phase 5 had already
   attributed this to domain mismatch, but a competing explanation was never
   excluded: the corpus text itself may be defective.
2. **Adaptive collapses onto hybrid**, and the sufficiency gate never fires
   because observed sufficiency (0.6364–0.65) sits above the shipped 0.5 bar.

The corpus defect in question is diagnosed and recorded: two-column pages are
extracted by merging characters whose vertical positions are within 3.0 pt,
which on a two-column page merges *both columns* into one line, and that merged
line is then sorted left-to-right by `x0`. The resulting text interleaves the
columns mid-sentence and produces glued tokens at a measured 1.4% rate, far off
MS-MARCO's distribution for a cross-encoder trained on web passages.

Both findings could therefore be corpus artifacts rather than findings about
the systems. Phase 8 tests that directly.

**No retrieval method is removed.** All five arms are re-run and reported side
by side. A null result — clean text does not rescue the reranker — is a
legitimate outcome, recorded in advance so it cannot later be re-framed.

---

## 2. Pre-registration

> Written before the new corpus existed, before the fixed normalizer was
> implemented, and before any Phase 8 number was computed. Nothing in this
> section may be edited after results exist; a change to a pre-registration is a
> new, later document.

### 2.1 Design

This is a **before/after study**. The new corpus **coexists** with the old one:

| Artifact | Old (Phase 7) | New (Phase 8) |
|---|---|---|
| `corpus_version` | `corpus_6c416f423920385d` | new value, recorded in §8 |
| BM25 index | `storage/bm25/bm25_index.json` | new directory |
| Qdrant collection | existing | new collection |
| `strategy_cost_ms.json` | frozen Phase 7 values | new file for the new version |

Nothing is overwritten. Phase 7's results remain valid *of the old corpus* and
become the **before** arm of the comparison. `strategy_cost_ms.json` keeps one
frozen value **per corpus version** rather than being silently redefined.

### 2.2 The comparison family (Step 0 — pre-registered)

* **Systems:** the five E1 arms — `bm25`, `dense`, `hybrid`, `hybrid_rerank`,
  `adaptive`.
* **Pairs:** every unordered pair, enumerated by `itertools.combinations`,
  **10 pairs** (C(5,2)). No baseline-anchored subset. No pair is dropped after
  the fact.
* **Metrics:** `recall_at_5`, `mrr`, `ndcg_at_5`, `total_latency_ms`.
* **Correction:** Holm-Bonferroni **within each metric across all 10 pairs**.
  `family_size = 10` is recorded on every comparison unless some pairs carry no
  evaluable value for that metric, in which case the actual family size is
  recorded and the omission is reported.
* **Implementation:** `adaptive_rag.evaluation.statistics.compare_all_pairs`,
  which reuses `compare_metric` unchanged. Only pair enumeration and the
  correction family differ from `compare_systems` / `compare_variants`.
  Offline: it reads existing `rows.jsonl` and makes no network calls.

### 2.3 Disclosure — status of these tests

**Phase 7's E1 point estimates were observed before this family was specified.**
E1 compared four arms against `bm25` only, and the resulting ordering — dense
first, `hybrid_rerank` last — is what motivated this phase in the first place.

These 10-pair tests are therefore **post-hoc and exploratory within this
program**. Pre-registered, but not originally planned. They are stated here
rather than buried so that no reader mistakes a pre-registered test for an
unanticipated one. Two consequences follow and are honoured:

* the Phase 7 → Phase 8 direction of change is itself exploratory, and a
  before/after flip in any pair is reported as exploratory regardless of its
  adjusted p-value; and
* the primary Phase 8 questions remain the two substantive ones — did the corpus
  fix change the quality ordering, and does reranking now help — for which the
  before/after comparison, not a within-corpus p-value, is the evidence.

### 2.4 What would count as an answer

Recorded now, so the outcome cannot be reinterpreted later:

| Outcome | Reading |
|---|---|
| `hybrid_rerank` improves materially on the new corpus and overtakes `hybrid` | corpus defect was a major cause; ADR-028's recommendation is withdrawn |
| `hybrid_rerank` improves but still ranks last | corpus was a real contributor, not the whole cause; ADR-028's recommendation stands on narrowed grounds |
| `hybrid_rerank` unchanged | corpus defect was not the mechanism; the finding stands as a model/domain-mismatch result |
| quality ordering otherwise unchanged | "dense dominates hybrid" was not a corpus artifact |

### 2.5 Falsification conditions

* If residual interleaving does **not** fall to near zero on two-column pages, the
  defect was not fixed and no Phase 8 result is interpretable as a corpus
  result. Re-ingestion is blocked at that gate (§3).
* If single-column extraction is not byte-identical to the old behaviour, the fix
  is not a fix — it is a different extractor — and is reverted.
* If the chunk-level gold remap in §6 cannot be verified, `nDCG@5` and E8 are
  reported as **not measured on the new corpus** rather than as measured on
  remapped labels whose provenance is unknown.

---

## 3. Step 1 — the two-column extraction fix

`ingestion/normalizer.py:92-116` (`_cluster_chars_into_lines`) merges every
character within 3.0 pt of the running line's `top`. On a two-column page both
columns share the same `top` values, so they merge into a single line; that line
is then sorted by `x0` and joined left-to-right, which is exactly the
glued-token symptom. `RawChar` already carries `x0`/`x1`/`top`/`bottom`, so
**the extractor is untouched**.

### 3.1 Column detection

New `_detect_columns(chars, page_width)`:

* project character coverage onto the x-axis into a 1 pt occupancy histogram
* find **interior** gaps — runs of zero coverage — and require a candidate to be
  **wide** (>= 6.0 pt), **near the page centre** (within 25% of `page_width`),
  and **vertically persistent** (a large fraction of the page's occupied rows
  are split by it)
* return the resulting column ranges, or `None` for single-column pages

Occupancy, not char counts, so a single stray glyph or a hyphen at a column edge
cannot fabricate or break a gutter.

### 3.2 Column-aware line clustering

`_cluster_lines_with_breaks(chars, column_ranges)` returns the lines plus the
indices at which a paragraph must be broken:

* when `column_ranges is None`, **exactly** the pre-Phase-8 code path, byte for byte
  (the old algorithm is retained verbatim as `_cluster_lines_flat` and pinned to the
  new path by a unit test)
* otherwise, a visual line is split **only if it spans a gutter** — it carries
  characters in two columns *and* leaves a gutter-sized horizontal gap between them.
  Each half is then re-clustered with the unchanged 3.0 pt rule
* reading order is emitted as left column top→bottom, then right column
  top→bottom
* a line that spans no gutter — a centred title, a wide figure, a table row, or a
  line confined to one column — is emitted **intact**. Splitting a title at the
  gutter would break a heading in half and lose a section boundary, and there is
  nothing to gain: a full-width line read left to right is already in the order
  the column streams emit
* **the paragraph is broken at every column transition.** Splitting the lines is
  necessary but not sufficient: the last paragraph of the left column and the first
  of the right are unrelated text, and joining them into one paragraph reinstates
  the very splice the split removes

**Three design decisions were made against the obvious approach**, and each was
forced by a page in this corpus:

* **Emptiness is counted per row, not on a union projection.** A projection built
  by OR-ing rows together fails on the first page of every two-column paper: the
  title and author block span the full width, and one such row erases the interior
  gap that every body row below it has. Counting, per bin, how many rows leave it
  empty recovers those pages — and takes detection from 38 pages to 77.
* **The split test is the gutter gap, not the column edges.** Comparing a line's
  extent to the column boundaries looks equivalent and is not: a margin glyph puts
  the left column's geometric edge outside the text block, so every legitimate body
  line fails the test and is emitted intact — which is the corruption being fixed.
  Deciding by font size fails too; one page mixes 8 pt footnotes with 10 pt body on
  a shared baseline.
* **A seam recurring inside a column disqualifies the page.** A figure rule that
  intrudes into the gutter on a minority of rows leaves the band *beside* it empty
  on all of them, so that wider band passes every individual check and the
  boundary lands mid-column. `_has_nested_gutter` re-probes each detected column
  and declines when one is really two.

### 3.3 Validation gates — measured, not asserted

`scripts/validate_column_extraction.py` measures all five from the raw PDFs and
writes `experiments/phase8/column_validation.json`. Nothing in the corpus is
touched and no index is built; the script answers "did the fix work?" and leaves
re-ingestion to a human.

| Gate | Criterion | Measured | Status |
|---|---|---|---|
| Residual interleaving | ≤ 10 output lines still show a gutter-sized gap between column parts | **0** | **pass** |
| Spot check | the sentence occurs in the column-aware text and not the flat text | as specified | **pass** |
| Single-column regression | a document with no detected two-column page re-ingests byte-identical to its stored Phase 7 artifact | 14/14 compared, 0 unexpected | **pass** |
| Reading order | first elements of 3 papers hand-verified against the PDF | verified | **pass** |
| Chunk delta | number of chunk texts that change, reported | 618 → 613 chunks; 603 shared ids, **220 changed**; 15 ids only before, 10 only after | reported |

**Corpus-wide:** 77 of 285 pages detected as two-column, 208 declined. **7009**
adjacent word pairs existed only in the interleaved reading and are gone from it.

Two numbers in this table are deliberately *not* what §2 anticipated, and the
reason is recorded rather than smoothed over:

* **The 1.4% baseline is not used as a threshold.** Phase 7 recorded "1.4% of
  tokens glued" (`docs/phase_7_results.md` §11) without retaining the measurement
  that produced it. It cannot be reproduced, and no number here is presented as
  confirming or refuting it. The gate instead measures residual interleaving on
  the column-aware output, which is the defect itself, on its own scale.
* **Chunk ids shift.** The chunker merges elements under section headers and splits
  on token budgets, so a changed element moves boundaries and renumbers ordinals.
  15 chunk ids exist only in the old corpus and 10 only in the new. This is
  expected, and it is the reason §6 re-validates chunk-level gold rather than
  assuming `relevant_chunk_ids` survive.

**Known residual artifact, not introduced here:** page 1 of the arXiv papers still
shows interleaved digits (`6 0`, `9 4`). Those are the arXiv margin stamp, which
is *rotated* text; `_detect_columns` sees it as a narrow left-margin band and it
predates this change. It is cosmetic and affects a margin stamp, not body prose.
Recorded rather than absorbed — see §10.

**No re-ingestion until the single-column regression gate passes.** A naive split
that scrambles reading order would make everything worse; that was the main
engineering risk in this phase, and the gate is what ruled it out.

---

## 4. Step 2 — Re-ingestion into two arms

All 14 PDFs are retained in `data/raw/`; nothing is re-downloaded.

**Both arms are built from the same PDFs and the same chunking config, differing
only in whether `_detect_columns` runs.** That isolation is the entire point of
the study, and it forced a correction to this section as originally written:

| | Path | Chunks | BM25 index | Qdrant collection |
|---|---|---|---|---|
| before (reading order broken) | `data/processed_phase8_before/` | 618 | `bm25_index_phase8_before.json` | `adaptiverag_dense_v2_before` |
| after (column fix) | `data/processed_phase8_after/` | 613 | `bm25_index_phase8_after.json` | `adaptiverag_dense_v2_after` |
| Phase 7 (historical, not an arm) | `data/processed/` | 713 indexed | `bm25_index.json` | `adaptiverag_dense_v1` |

### 4.1 Why the Phase 7 artifacts are not the "before" arm

The plan assumed the Phase 7 corpus *was* the before arm. It is not, and using it
would have voided the comparison.

`storage/bm25/bm25_index.json` and `adaptiverag_dense_v1` hold a **713-chunk**
corpus, and that is genuinely what Phase 7 evaluated — all 422 chunk ids Phase 7
retrieved exist in that index with matching page provenance. But chunking the
current documents yields **618**, and the two sets share only ~1.01M characters of
text between them: same content, different split. So a Phase 8 before/after run
against those indexes would have measured the column fix *and* an unrelated
chunk-boundary change together, and any quality difference could have been
attributed to either. The docs are inconsistent on this too, recording both "713
chunks" and "target 500 / max 800", which under the current chunker give 618.

The Phase 7 artifacts are therefore left **exactly as they are** — they are the
historical record of that study, not an arm of this one — and the before arm is
rebuilt to differ from the after arm by exactly one thing.

### 4.2 The arms are told apart explicitly, not by fingerprint

The two arms share a `corpus_version`: same PDFs, same pipeline version, so the
existing fingerprint cannot separate them. A Qdrant collection carries no version
at all, and `ensure_collection` silently reuses an existing collection of the same
name. So a run pointed at the wrong arm would retrieve real-looking results from
the wrong text and report them as a finding.

Three mechanisms close that:

* `IndexConfig.corpus_arm` **derives** the collection and the lexical index, so the
  two cannot disagree — it is behaviour, not a label.
* `BM25Index` records the arm it holds, and `load()` rejects a mismatch with
  `IndexConfigMismatchError`. The corpus-version guard cannot catch this; the arm
  check is the only thing that can.
* `scripts/run_phase7_suite.py --corpus-arm` selects the arm for a whole run.

`docs/phases/phase-8.md` §4 also supersedes §2.1: the coexistence claim there was
written before this was measured, and holds for the Phase 7 artifacts but not for
the arms.

## 5. Step 3 — Re-embedding

Re-embed **both** arms via AICredits, batched and paced. The two live hazards
already hit in Phase 7 are live again: SDK implicit retries (`max_retries=0` is
fixed) and 45 s timeouts on a stalling server. Pacing sleep sits **outside** the
timed region, so raising `--pace` cannot contaminate measured latency.

## 6. Step 4 — Gold label re-validation

**The premise of this step was wrong, and finding that out was the work.** Two of
its three assumptions did not survive contact with the data.

### 6.1 There is no chunk-level gold to remap

All 107 records have an **empty** `relevant_chunks`. Chunk relevance is not stored
at all — it is derived at evaluation time by `relevant_chunk_ids`
(`evaluation/dataset.py:158`), which marks a retrieved chunk relevant when its
document is labelled and its `section_path` matches a labelled prefix. So the remap
this section specified cannot be performed because there is nothing to remap, and
the schedule risk it named does not exist.

### 6.2 The section labels are themselves corrupt

They were derived by reading the corpus as the broken extractor produced it, so a
label carries a real heading followed by two columns of prose spliced onto it:

```
'1. RECIPROCAL RANK FUSION shown in table 1, indicated that k = 60 was near-optimal,'
'2 BM25)thatrelyoninvertedindexes.WhileBOWmodelsremain'
```

Those labels happen to match the *garbled* section paths. Fixing reading order
breaks the match: **132 of 140 section references resolve before the fix, 95 after.
40 regress.**

`scripts/audit_phase8_gold_labels.py` quantifies this and proposes repairs
(`experiments/phase8/gold_label_audit.json`):

* **47 of 140** labels (34%) are demonstrably damaged.
* Corruption is detected by evidence, not by length: a space-free run too long to
  be a word (threshold 20, chosen from the data — valid labels top out at 18,
  spliced ones start at 26, with no false positives among the 90 valid labels), or
  a label that resolves nowhere. An earlier length-and-word-count heuristic
  flagged 57 labels of which most were perfectly valid.
* **37** are recoverable mechanically with high confidence, by taking the heading
  the spliced label literally starts with; **10** admit only a guess and are
  withheld or marked `low`.
* The frozen benchmark is **not** rewritten. A label repaired by a heuristic has to
  be read by a person before it is allowed to change a number.

### 6.3 What this permits Phase 8 to claim

The metrics split cleanly, and the split decides the study:

| Metrics | Key on | Status |
|---|---|---|
| `recall_at_5`, `mrr`, `hit_at_5` | `relevant_documents` only (`evaluation/retrieval.py:134,151`) | **valid on both arms** — the headline before/after rests here |
| `ndcg_at_5`, `precision_at_k` | section labels, via `relevant_chunk_ids` | **not comparable until labels are repaired** |

Per §2.5, `nDCG@5` and E8 are reported as **not measured on the new corpus** —
carrying forward a number derived from labels known to be corrupt would be worse
than reporting nothing. Repairing the 37 high-confidence labels is mechanical; the
remaining 10 need a human read of the paper, and that is a judgement about what a
query is actually asking, not a string operation.

## 7. Step 5 — Re-running E1–E5

Same commands, same frozen dataset file, new corpus. The ablation arms depend on
the router config staying untouched; **no ladder or threshold change in this
phase.**

| Study | Arms | Queries |
|---|---|---|
| E1 | 5 | 107 (`--split all`) |
| E2 + E3 | 10 | 47 calibration |
| E4 + E5 | 14 | 47 calibration |

### 7.1 13 of 29 arms are quarantined

All six suites reported `ok` and a full trace count. **They were not.** 13 arms
carry failed traces, every one of them
`EmbeddingAPIError: Connection error` at the retrieval stage, and they fail
*per query*, so the shortfall is a fraction of an arm rather than all of it:

| Suite | Arm | ok / total |
|---|---|---|
| `p8b_e1` | `hybrid_rerank` | 59 / 107 |
| `p8b_e2e3` | `B`, `C`, `full`, `without_{lexical,semantic,entity,complexity,question_type,multi_concept}` | 0–5 / 47 |
| `p8a_e2e3` | `A`, `B` | 0 / 47 |
| `p8a_e2e3` | `C` | 33 / 47 |

The mechanism is structural rather than incidental. The reranker embeds every
candidate, so one dropped embedding call fails the whole trace; and
`rerank_fallback` is opt-in and was **off**, so no trace degraded into a partial
result — it either scored or died. Every arm that routes through the terminal rung
dies with it, which is why the escalation and adaptive arms collapsed while the
fixed `bm25`/`dense`/`hybrid` arms are untouched.

**Nothing downstream reported this.** `metrics_retrieval.json` states the arm's
recall as `value=0.6384, n=59` beside `trace_count=107`, with no error field, so
the surviving queries read as a complete arm. That number was on its way to
becoming the headline finding of this phase: set against the after arm it reads as
a **+0.067 improvement**, whereas the arm's true mean over all 107 queries is
**0.352** and the clean re-run gives **0.7072** — a **regression** of −0.355. The
contaminated figure would have landed on §2.4's second answer row ("corpus was a
real contributor, ADR-028 stands on narrowed grounds"), which is the opposite of
what the measurement shows.

`compare_phase8_arms.py` now refuses a contaminated arm outright rather than
averaging over whatever survived, and `experiments/phase8/run_integrity.json`
records the full audit.

### 7.2 What survives

Re-run at `--pace 1.5` after the failures above, all 14 arms came back clean
(107/107 and 47/47), which confirms the cause was rate limiting rather than a
defect in the pipeline. Final state:

* **E4 + E5 — all 28 runs, both arms.** Complete, and these are the sweeps that
  were least at risk, since the router rarely escalates far enough to reach the
  reranker.
* **E1 — all 5 arms, both arms** (`hybrid_rerank` before arm from the re-run).
* **E2 + E3 — both arms**, from the re-runs.

The contamination was asymmetric between arms, which was its own hazard: the two
arms did not even cover the same queries, so comparing them would have paired 107
against 59 and inverted the sign of the reranker's before/after delta (§7.1).

## 8. Step 6 — Freeze, all-pairs, analyse

* re-freeze `strategy_cost_ms.json` **for the new corpus version only**
* run the pre-registered all-pairs statistics from §2
* re-run the E6/E7/E8 analyses; regenerate tables and figures
* report the before/after matrix: 5 arms x {old corpus, new corpus}

### 8.1 Two tools had to be opened before any of this could run

`measure_strategy_cost.py` took no `--corpus-arm`, and `build_experiment_config`
defaults to `IndexConfig()`, whose `corpus_arm` is `phase8_after`. A cost freeze
run in that state would have measured one arm and recorded it as *the* corpus —
the same shape of error as §4.2, so the flag now exists and the arm is written
into the artifact's provenance next to the corpus version it cannot be
distinguished by.

`compare_all_pairs` was implemented, unit-tested, and unreachable: nothing outside
`tests/` called it, so the §2.2 family had no way to be executed.
`scripts/compare_phase8_arms.py` runs it twice per arm — the literal 10-pair
family, and the before-vs-after comparison §2.3 nominates as the actual evidence,
which `compare_all_pairs` cannot express (it enumerates pairs *among* systems, and
the before/after question pairs each system with itself on the other corpus).
It lives in its own script rather than as a flag on `run_phase7_analysis.py` so a
Phase 8 family is never added to the artifact that documents Phase 7.

### 8.2 The result, on the arms that stand

The pre-registered family over all five E1 systems, both arms, Holm-corrected
within each metric, n=107 per arm. (`hybrid_rerank` in the before arm is the
`--pace 1.5` re-run; the contaminated 59-trace run is excluded.)

**Before vs after — each system against itself, Holm across the five:**

| Metric | bm25 | dense | hybrid | hybrid_rerank | adaptive |
|---|---|---|---|---|---|
| recall@5 | +0.0109 (p=1.00) | −0.0156 (p=1.00) | +0.0093 (p=1.00) | **−0.0016 (p=1.00)** | +0.0093 (p=1.00) |
| mrr | +0.0283 (p=0.43) | −0.0069 (p=1.00) | +0.0085 (p=1.00) | **+0.0004 (p=1.00)** | +0.0130 (p=1.00) |
| latency_ms | +0.01 (p=1.00) | **+411 (p<0.001)** | +98.8 (p=0.16) | **+112 (p<0.001)** | +12.8 (p=0.35) |

**No quality metric moved significantly on any of the five arms.** Every recall
and MRR delta is inside the noise at n=107, in both directions. The two
significant results are both latency, and **neither measures the corpus**: they sit
in the query-embedding API call, which embeds the query rather than the index. The
`bm25` control arm, which makes no API call, moved 1.83 → 1.85 ms across the fix,
and the reranker's local ONNX stage moved −2.3 ms. See
`docs/phase-8-results.md` §6.1 — those two latency results are withdrawn as corpus
findings.

### 8.3 The reranker did not improve, and that is the finding

This is the arm the whole phase was convened to examine, and the answer is
negative and unusually clean — `hybrid_rerank` moved by **−0.0016 recall@5**
(4 wins, 4 losses, 99 ties out of 107) and **+0.0004 MRR** (19 wins, 21 losses,
67 ties). The corpus defect was real, was fixed to a measured standard (§3.3), and
**did not touch this arm's quality**.

Its standing against `hybrid` is significant and near-identical on both corpora:

| | `hybrid` | `hybrid_rerank` | delta | p_adj |
|---|---|---|---|---|
| before recall@5 | 0.8489 | 0.7072 | −0.1417 | 0.0139 |
| after recall@5 | 0.8583 | 0.7056 | −0.1526 | 0.0007 |
| before mrr | 0.8593 | 0.6016 | −0.2577 | 0.0000 |
| after mrr | 0.8678 | 0.6021 | −0.2657 | 0.0000 |

The penalty did not shrink; it grew very slightly. So the reranker is not being
rescued by clean text, and the corrupted reading order was **not** the mechanism
behind it.

`adaptive` remains statistically indistinguishable from `hybrid` on every quality
metric (recall delta exactly 0.0000, p=1.00) while costing ~132 ms less, so
routing buys nothing over fixed hybrid on this benchmark.

### 8.4 Reading against §2.4

| Outcome | Reading | Status |
|---|---|---|
| `hybrid_rerank` improves materially and overtakes `hybrid` | corpus defect was a major cause; ADR-028 withdrawn | **ruled out** |
| `hybrid_rerank` improves but still ranks last | corpus was a real contributor, not the whole cause; ADR-028 stands narrowed | **ruled out** |
| `hybrid_rerank` unchanged | corpus defect was not the mechanism; the finding stands as a model/domain-mismatch result | **confirmed** |
| quality ordering otherwise unchanged | "dense dominates hybrid" was not a corpus artifact | **confirmed** |

Both pre-registered readings that the fix would *rescue* reranking are excluded by
measurement. §2.3's exploratory caveat applies to every comparison here, since
Phase 7's ordering was observed before this family was specified — so this is
evidence that **the corpus fix does not explain Phase 7's rerank result**, not a
replication of that result under a clean corpus. What survives is a
model/domain-mismatch reading: the cross-encoder is being asked to rank chunks out
of a 14-paper IR corpus it was never trained to score, and it returns orderings
that are worse than the fusion that produced its candidates, on clean text and
with the same magnitude as on garbled text.

`nDCG@5` is excluded throughout, and §2.5's falsification condition is what
excluded it: the Step 4 audit found the gold section labels spliced, so an nDCG
before/after number would compare two corrupt label sets rather than two corpora.
The decision was taken from the label audit before any after-arm quality number
existed.

## 9. Step 7 — Report and documentation

* `docs/phase-8-results.md` — the before/after study, leading with whether the
  corpus fix changed the quality ordering and whether reranking now helps
* revise **ADR-028** on the evidence; its "remove the rerank rung"
  recommendation stands or falls on this result
* `docs/progress.md`, `docs/architecture.md`
* new corpus and results versioned; the "before" results retained and labelled

---

## 10. Risks, stated plainly

1. **A null result is a real possibility.** Clean text may not rescue the
   cross-encoder. Pre-registering makes that a legitimate finding rather than a
   disappointment — recorded before the effort, not after.
2. **Reading-order regression** was the main engineering hazard (§3). Three
   detection designs were wrong before the fourth was right, and the gate that
   caught them is the reason this is safe to proceed on.
3. **Chunk-id remap may fail.** §3.3 already shows ids shifting in both
   directions, so this is a live risk, not a theoretical one — it may force manual
   re-annotation. Biggest schedule risk; lives in §6.
4. **`corpus_version` change breaks Phase 2–6 comparability** — mitigated by
   coexistence, not eliminated. Those results remain true *of the old corpus*.
5. **Some judgments were made on bad text** and may need revision regardless of
   remap success.
6. **The arXiv margin stamp is still interleaved** on page 1 of several papers.
   It is rotated text that reads as a narrow left-margin band; it predates this
   phase and affects the stamp, not body prose. Carried as a known artifact rather
   than fixed here, because a rotated-text fix is a separate concern from column
   detection and should not be smuggled into a result that is already in flight.

## 12. Progress

| Step | Status |
|---|---|
| 0 — pre-registration + `compare_all_pairs` | **complete** |
| 1 — column-aware extraction, all gates measured | **complete** (§3.3) |
| 2 — both arms ingested, chunked, indexed | **complete** (§4) |
| 3 — embeddings + Qdrant collections for both arms | **complete** |
| 4 — gold label audit | **complete** (§6); nDCG@5 blocked on label repair |
| 5 — re-run E1–E5 | **complete, 13/29 arms quarantined** (§7.1) |
| 6 — all-pairs + before/after statistics | **complete on all five clean E1 arms** (§8.2) |
| 6 — re-freeze costs per arm | **measured both arms, neither frozen** (`docs/phase-8-results.md` §6) |
| 7 — report + docs | **complete** (`docs/phase-8-results.md`); ADR-028 revised |

## 11. Explicit non-goals

No retrieval method is removed. No router, ladder, or threshold change — that
stays out until this evidence exists. No generation evaluation. No composite
score. Nothing under `experiments/phase7/` or the old corpus is modified.