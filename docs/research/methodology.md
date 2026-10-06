# Methodology

How the project was set up and measured. Full narrative in
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md) §3; results in
[`findings.md`](findings.md) and the per-phase result documents.

## 1. Design

A **controlled, offline-first experimental program** over one corpus, built
incrementally:

1. Establish a canonical corpus and a reproducible ingestion/indexing pipeline
   (Phases 1–2).
2. Add retrieval strategies one at a time, each measured against the same
   benchmark and evaluator (Phases 3–5: BM25 → Hybrid RRF → cross-encoder).
3. Add the adaptive routing layer as pure orchestration over the unchanged
   strategies (Phase 6).
4. Evaluate, ablate and calibrate under pre-registration (Phase 7).
5. Characterize the corpus and close the phase with two decisive gates
   (Phase 8).
6. Run seven hypothesis-driven research phases (9–15), each pre-registered,
   split-separated, and run on two corpus arms.

The project's discipline is the experimental design, not the code: every claim
of significance carries a pre-registered family, a correction, and a split.

## 2. Units of analysis and their splits

| Artifact | Content | Use |
| --- | --- | --- |
| `data/evaluation/phase7_eval_v1.jsonl` | 107 queries, **47 calibration / 60 test**, sha256 `f0695189903c8ef8…` | Phases 7–14 |
| `data/evaluation/phase15_eval_v1.jsonl` | 110 fresh queries, **all `split: "test"`**, duplicate-screened vs 266 prior | Phase 15 only |

The calibration/test split is frozen (ADR-027). Rule *selection* happens on
calibration; the test split is read exactly once. Phase 15, whose policy was
frozen before the powered run, has **no** calibration split by construction —
a stronger separation, not a weaker one.

## 3. Systems compared

Four fixed retrieval strategies under one `Retriever` interface (ADR-007),
evaluated by one strategy-agnostic evaluation pipeline (ADR-008):

* **`bm25`** — Okapi BM25 (k1=1.2, b=0.75) over a deterministic tokenizer;
  local inverted index; no embedding, no API call.
* **`dense`** — `text-embedding-3-large` (3072-d, normalized) + Qdrant cosine;
  costs one remote embedding call per query.
* **`hybrid`** — Reciprocal Rank Fusion of dense@20 and BM25@20 → top-10
  (rrf_k=60). A composition of existing retrievers (ADR-019), not a new
  retrieval algorithm.
* **`hybrid_rerank`** — ONNX cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`)
  reordering the top *N* candidates (ADR-020).

Plus **`adaptive`**, a rule-based router + sufficiency checker + bounded
escalation ladder (Phase 6, ADR-021/023/024/025), which orchestrates the four
above without modifying them.

## 4. Metrics and cost model

* **Primary**: Recall@5 (document-level, from `relevant_documents`).
* **Secondary**: MRR; Hit@5 reported in E-series tables.
* **Excluded**: nDCG@5 — its IDCG denominator derives from chunks a trace
  happened to retrieve, *and* 47 of 140 gold section labels are corrupt
  (40 unresolvable). Reported as not measured, never estimated.
* **Cost**: median retrieval latency (ms) and, for the cheap-first track, the
  **Dense/embedding call rate** (the primary currency). No USD figures exist
  for retrieval-only runs, so none are quoted.

## 5. Oracle-based evaluation

Policies are evaluated against an **oracle** that labels each query by whether
escalation would have helped:

* **Post-dense oracle** (`phase10_escalation_v1`): escalate iff
  `recall@5(hybrid) − recall@5(dense) ≥ 0.01`.
* **Cheap-first oracle** (`phase14_cheapfirst_v1`): escalate iff
  `recall@5(dense) − recall@5(bm25) ≥ 0.01`.

Ties and harms are labeled NO. **Labels score a rule; they never enter it** —
every signal function is label-free and, in Phases 14–15, restricted to an
allowlist of *pre-Dense* single-arm scalars enforced by contract tests. The
oracle is an upper bound, not an implementable policy; its value bounds the
problem.

## 6. Statistical protocol

* **Multiple-testing correction.** Holm–Bonferroni within pre-registered
  families (e.g. 16 comparisons in Phase 7 E1; 12 hypotheses in Phase 9).
* **Uncertainty.** Cluster-bootstrap CIs (seed 20250115) on differences and
  rates; 1000-draw random-at-equal-rate nulls per arm plus a combined test.
* **Replication.** Two corpus arms (`after` shipping, `before` pre-fix); a claim
  must hold in direction on both.
* **Random-selection ablation.** Each policy is compared to random spending at
  the same rate — the control separating intelligent selection from merely
  spending some budget.
* **Coarse-metric honesty.** Recall@5 ties on ~90% of queries, so paired
  sign-tests are frequently non-significant while bootstrap CIs exclude zero.
  Both are reported; neither is hidden, and neither is used selectively.

## 7. Data-hygiene and integrity controls

* **Corpus versioning.** Corpora carry a `corpus_version` fingerprint of
  `papers.json` + raw-PDF SHA-256s; Phase 8 adds namespace directories and
  per-arm manifests so the two arms are distinguishable beyond the shared
  fingerprint.
* **Section-label integrity.** The `G7` column-contiguity check and the Phase 8
  gold-label audit found spliced/unresolvable labels; affected records were
  dropped or the metric excluded — never silently patched.
* **Trace-status guard.** Phase 8's first sweep produced 13 of 29 arms in which
  the reranker arm failed on 48/107 queries yet the suite reported `ok`. The
  lesson, now enforced: *trace status — not trace count, and not a metric's own
  `n` — is what establishes that a measurement covers its query set.*
* **Latency requires a control arm.** `total_latency_ms` alone cannot separate
  provider variance from system cost; a no-API arm (bm25) is the control, and two
  Phase 8 latency findings were withdrawn for lacking one.
* **Provenance by hash.** `docs/phase7-closure-evidence.md` pins SHA-256 for the
  inputs and artifacts the closure relies on.

## 8. What the methodology deliberately does *not* do

* No learned router is trained or deployed (deferred, ADR-026); a ridge
  regression in Phase 7's Gate 3 is a *measuring instrument* only.
* No end-to-end generation-quality routing claim (retrieval-only).
* No re-tuning of a frozen policy on its test set.
* No metric substitution after seeing results (nDCG excluded, not swapped).

---

*Method and protocol. Narrative: [`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md).
Commands: [`../reproducibility/reproduction.md`](../reproducibility/reproduction.md).*