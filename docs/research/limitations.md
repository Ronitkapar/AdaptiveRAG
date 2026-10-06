# Limitations

Boundaries of the evidence, all of them measured or encountered during the
investigation rather than assumed. Full version:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md) §8.

## Corpus

* **One corpus, one domain** — 14 foundational IR/NLP papers. No second
  collection, no non-academic domain, no multilingual or long-document setting.
* **Residual defects** — the `pyserini_lin_2021` content/title mismatch
  (contained by excluding 6 records) and **42 spliced section paths** remaining in
  the shipping corpus after the Phase 8 fix.
* **Ambiguous versioning** — both Phase 8 arms report the same
  `corpus_version` string; only the namespace directory distinguishes them.

## Evaluation data

* **Positives, not queries, were the binding constraint.** 107 queries, but only
  4–8 oracle positives for the post-dense question; 5 calibration / 11 test for
  Phase 14; 14 / 11 for Phase 15. AUC claims need positives in the tens.
* **Phase 15 positives landed at the low edge** of the pre-registered expectation
  (~20/arm) — reported, not repaired by top-up curation.
* **Recall@5 is coarse** (~90% ties), which is why paired tests and bootstrap CIs
  disagree; both are reported rather than selected between.
* **Thin category cells** — `comparative` (8) and `multi_document` (7) fall below
  `min_cell = 5` per split.

## Metrics

* **`nDCG@5` is excluded entirely** — the IDCG denominator is trace-relative, and
  47 of 140 section labels are corrupt. Reported as *not measured*.
* **`Precision@k` is document-derived** — it partly measures how many chunks of a
  right document were returned.
* **No USD cost figures** — all runs are retrieval-only (`estimated_cost_usd`
  `n_pairs = 0`); cost is latency and call-rate only.
* **No answer-quality claims** — generation and the LLM judge were not part of
  any routing evaluation.

## Routing policy

* **The shipped router never routes** at the frozen configuration (the gate does
  not open), so adaptive ≈ hybrid by construction.
* **No learned router** — all signals are hand-specified rules; a learned router
  was deferred (ADR-026) and never built.
* **The best rule is lossy** — catches ~30% of positives at ~30% spend with
  26–29% precision; nearly half its escalations are false positives.
* **Latency is provider-dependent** — a ~2.4 s rate-limited embedding outlier
  sits in the Dense stage; `total_latency_ms` alone cannot separate provider
  variance from system cost.

## Retrieval systems

* **One of everything** — one embedding model (`text-embedding-3-large`), one
  reranker (general-web MS-MARCO), one fusion method (RRF).
* **The reranker's failure diagnosis is a hypothesis** (model/domain mismatch),
  not a confirmed cause; the corpus explanation was tested and withdrawn.
* **Retrieval-only** — fusion behaviour is studied; end-to-end generation is not.

## Statistics

* **Holm-corrected paired tests on a coarse metric** are often non-significant
  where bootstrap CIs exclude zero; the two answer different questions and both
  are shown.
* **Pairwise-defined effective n** (Phase 9: n = 23, before/after overlap only
  13), so those correlations are replicated associations, not effect estimates.
* **DDR has five possible values** {0.2 … 1.0} — near-binary; its calibration ρ
  (+0.254) sits below the 0.30 threshold while its test ρ (+0.410) exceeds it,
  i.e. a single-split pass, not a stable two-split margin.
* **Mechanism attribution uses top-10 membership** — candidate depth 11–20 is
  not persisted in traces.

## Reproducibility

* **Artifacts are gitignored** — `experiments/` and `storage/` hold traces,
  figures and tables, recoverable only from the working tree; SHA-256 provenance
  is pinned in `docs/phase7-closure-evidence.md`.
* **The working-tree `data/processed/` corpus does not text-match the BM25 index
  used in Phases 2–7** (713 chunks, manifest `max_tokens: 600`, rebuilt
  2026-10-03, vs the indexed distribution of mean 408 / median 447 / max 891
  tokens). Historical results are therefore authoritative in the storage
  indexes plus the documented figures. This is recorded rather than repaired —
  rebuilding would change `corpus_version` and invalidate frozen results.
* **Live-provider numbers are hardware-specific** (CPU reranker); they do not
  transfer.
* **The test suite is offline by design** — 1056 deterministic tests; the 2
  integration tests require live artifacts and are deselected by default.

---

*Limitations. Conclusions: [`conclusions.md`](conclusions.md). Full report:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md).*