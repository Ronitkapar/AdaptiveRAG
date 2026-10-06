# AdaptiveRAG — Final Research Report

**Investigating query-aware adaptive retrieval: when does a query need expensive
retrieval, and can retrieval effort be adapted without sacrificing too much
retrieval quality?**

Phases 1–15 complete · Research investigation completed · No final optimal
routing policy claimed

This is the central research document for the project. It summarises what was
investigated, what was measured, what failed, what was learned, and why
experimentation was stopped. Every number quoted here comes from a frozen
experiment artifact recorded in `docs/phases/` and `docs/phase*_results.md`; the
provenance of each claim is stated in §12.

---

## Abstract

Retrieval-augmented generation systems commonly pay for the most expensive
retrieval configuration on every query. AdaptiveRAG investigated whether that is
necessary: whether a cheap, decision-time-available signal can predict which
queries genuinely benefit from expensive retrieval, so effort can be spent
adaptively without an unacceptable quality loss.

The project built and measured four fixed retrieval strategies (BM25, Dense,
Hybrid RRF, Hybrid + cross-encoder rerank) plus a rule-based adaptive router on a
frozen 107-query academic-RAG benchmark, then executed seven research phases on
top of that foundation (Phases 9–15) under pre-registration, frozen
calibration/test splits, dual corpus arms, bootstrap uncertainty, and
random-selection ablation.

**Main finding.** Identifying *when additional retrieval is valuable* is
substantially harder than detecting *that retrieval strategies disagree*. A
post-dense disagreement signal (Dense distinct-document ratio) replicated as a
---

## 1. Introduction

### 1.1 The problem

The retrieval component of a RAG system has a quality–cost frontier. On the
corpus studied here (Phase 7 E1, 107 paired queries):

* BM25 retrieves in **1.88 ms** median with Recall@5 **0.7788**;
* Dense retrieval costs **472.84 ms** for Recall@5 **0.9283**;
* Hybrid (RRF over both) sits at **465.65 ms** / **0.8723**;
* adding a cross-encoder reranker costs **3491.69 ms** and *lowers* Recall@5 to
  **0.7165**.

So the expensive stages buy quality — but every query pays for them, including
the many queries whose answers a cheap retriever already retrieves. On an
embedding-API provider, the Dense arm's cost is dominated by one remote call per
query (~374 ms of its ~472 ms).

### 1.2 Why investigate adaptive retrieval

The hypothesis motivating the project is that the *marginal* value of an
expensive retrieval stage is query-dependent: some queries are easy for BM25
and some genuinely need Dense semantics. If the query-specific value of the
expensive stage could be predicted **before paying for it**, the expensive call
could be skipped when it adds nothing. This is the classic quality/latency/cost
trade-off, and it is why adaptive retrieval is worth investigating at all.

### 1.3 What AdaptiveRAG investigated — and what it is not

This distinction matters for reading everything that follows:

| AdaptiveRAG **investigated** | AdaptiveRAG did **not** build |
| --- | --- |
| Whether query-level dispersion across retrieval strategies exists and is measurable | A learned, production-ready router |
| Which decision-time signals predict *useful* escalation | A policy proven optimal over strategies or corpora |
| Whether a cheap-first BM25 → Dense rule can select Dense calls better than chance | A replacement for full Dense retrieval |
| Where the evidence fails, and why | Generalization beyond one academic-RAG corpus |

The report therefore describes an **investigation that reached a negative-to-
inconclusive verdict**, with its negative results preserved as the project's
main scientific output.

---

## 2. Research Question

### 2.1 Primary question

> **When does a query actually need expensive retrieval, and can retrieval effort
> be adapted automatically without sacrificing too much retrieval quality?**

This question was pursued through two tracks, each with a defined decision
point relative to the cost:

* **Post-dense track (Phases 6–12).** A router chooses an initial strategy,
  judges whether the retrieved evidence is sufficient, and escalates to a
  stronger strategy when it is not. The escalation ladder is
  `BM25 → Dense → Hybrid → Hybrid + Rerank`, a total order from configuration
  (`EscalationPolicy`, ADR-025).
* **Cheap-first track (Phases 13–15).** BM25 runs first; a pre-cost decision rule
  decides whether to pay for Dense at all. The decision point sits *before* the
  embedding API call, so the savings currency is API calls avoided.

### 2.2 Secondary questions actually investigated

1. Does per-query retrieval dispersion across strategies exist, and can it be
   explained mechanistically rather than by correlation alone? (Phase 9)
2. Do query-intrinsic or first-stage-retrieval-feedback signals predict *which*
   strategy will help, out of sample? (Phases 7, 9, 14)
3. Why does escalation help on some queries and hurt on others — and can the two
   cases be distinguished before paying for the escalated stage? (Phase 11)
4. Is the escalation target itself (Dense → RRF hybrid) net-positive on this
   corpus? (Phases 10–12)
5. Can cheap first-stage information select *among cheap-first escalations*
   better than random spending at the same rate? (Phases 14–15)

These are the questions the project actually asked. No research question is
introduced here that was not investigated.
real dispersion measure but failed entirely as a routing signal (Phase 10,
`FAILURE`): its adaptive policy was dominated by dense-only and even by random
escalation at the same spend. Mechanism analysis (Phase 11) explained why —
disagreement, insufficiency, and escalation value are three different things. A
cheap-first track (BM25 → Dense) then produced the project's strongest positive
result (Phase 15): a frozen BM25-side rule replicated its selection signal on
220 fresh query-instances, catching 18 of 25 oracle positives at ~30% spend and
beating random spending (p = 0.001 combined) while avoiding ~70% of Dense calls
— yet it still trailed full Dense retrieval by 0.018–0.030 Recall@5, missing
the pre-registered margin bar by 0.004.

**Conclusion.** Adaptive retrieval is plausible and cheap first-stage
information does contain usable selection signal, but on this corpus and with
this evidence the quality gap to full Dense retrieval remained too large to
claim a successful replacement policy. The adaptive-routing research track was
deliberately stopped rather than continued through indefinite feature
engineering. AdaptiveRAG did **not** solve adaptive retrieval.

---

## 3. Experimental Setup

### 3.1 Corpus

14 foundational IR/NLP papers collected in Phase 1 (`docs/corpus.md`,
`data/metadata/papers.json`), each a canonical open-access PDF with a recorded
SHA-256 digest. Two corpus namespaces are used as experimental arms:

| Corpus | Namespace | Chunks | Role |
| --- | --- | --- | --- |
| Phase 7 (`before`) | `data/processed/` | 713 | historical, known-defect corpus |
| Phase 8 (`before`) | `data/processed_phase8_before/` | 618 | replication arm, pre-extraction-fix |
| Phase 8 (`after`, shipping) | `data/processed_phase8_after/` | 613 | replication arm, two-column extraction fixed |

**Corpus-integrity findings carried through the project** (Phase 8, and the
label audits):

* A two-column PDF extraction defect interleaved the columns of multi-column
  papers, producing glued tokens. It was diagnosed, fixed, and re-chunked; this
  is the `before`/`after` study of Phase 8.
* `pyserini_lin_2021` does not contain the Pyserini toolkit paper but a
  biomedical knowledge-graph survey. The defect is contained by excluding the
  six affected benchmark records rather than by rewriting evidence
  (`data/evaluation/README.md`).
* 47 of 140 gold section labels are spliced and 40 no longer resolve on the
  fixed corpus; only 22 are safely repairable. Consequence: **`nDCG@5` is
  excluded from all reported quality claims** (`docs/phase-8-results.md` §0). The
  reported metrics — Recall@5, MRR, Hit@5 — key on `relevant_documents` only and
  are unaffected.

### 3.2 Retrieval systems

All four strategies conform to one `Retriever` protocol (ADR-007) and are
evaluated by the same strategy-agnostic evaluators (ADR-008):

| Strategy | Mechanism | Notes |
| --- | --- | --- |
| `bm25` | Okapi BM25 (k1=1.2, b=0.75) over a deterministic tokenizer | no embedding, no API call |
| `dense` | bi-encoder `text-embedding-3-large` (dim 3072, normalized) + Qdrant cosine | costs one embedding API call per query |
| `hybrid` | RRF(dense@20, BM25@20) → top-10, rrf_k=60 | fusion of existing retrievers (ADR-019); no new retrieval algorithm |
| `hybrid_rerank` | cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` on ONNX Runtime | second stage that reorders what retrieval found (ADR-020) |

Generation and LLM-judge layers exist in the codebase but were **not used** for
any retrieval-quality claim in Phases 7–15; all reported numbers are
retrieval-only.

### 3.3 Evaluation benchmark and protocol

* `phase7_eval_v1` — 107 curated queries, 47 calibration / 60 test (ADR-027
  frozen split), sha256 prefix `f0695189903c8ef8`, gate checks G1–G7 pass,
  `gate_open=true`, `grounding_verified=true`.
* `phase15_eval_v1` — 110 freshly curated queries, all `split: "test"`,
  duplicate-screened against all 266 pre-existing queries (0 flags), used only
  for the Phase 15 powered confirmation.
* Dual-arm execution: every claim in Phases 9–15 is computed on both the
  `after` (shipping) and `before` corpus arms. A claim counts as replicated only
  if it holds in direction on both arms.

**Oracle definitions** (labels score a rule; they never enter it):

| Oracle | Label | Used in |
| --- | --- | --- |
| `escalation_v1` (Dense→Hybrid) | escalate iff `recall@5(hybrid) − recall@5(dense) ≥ 0.01` | Phases 10–11 |
| `cheapfirst_v1` (BM25→Dense) | escalate iff `recall@5(dense) − recall@5(bm25) ≥ 0.01` | Phases 14–15 |

### 3.4 Metrics

* **Recall@5** — primary metric in all policy experiments; document-level
  relevance from `relevant_documents`.
* **MRR** — secondary, never selected post-hoc.
* **Hit@5**, **Precision@k**, **nDCG@5** — nDCG excluded (see §3.1); Precision@k
  is document-derived and carries a documented construction caveat.
* **Cost** — median retrieval latency (ms) and **Dense/embedding call rate**
  (the primary cost currency for the cheap-first track). No USD figure exists for
  the retrieval-only runs, so none is quoted.

### 3.5 Reproducibility controls

Every policy experiment ran under the following controls:

1. **Pre-registration before execution.** Each of Phases 10–15 froze its
   question, oracle, signal family, threshold-selection rule, and verdict mapping
   in `docs/phases/` before the corresponding number existed.
2. **Frozen calibration/test split.** Rules were selected on calibration only and
   evaluated on test in a single pass (Phases 10, 11, 14); Phase 15 has no
   calibration split at all because the rule was frozen beforehand and applied
   unchanged.
3. **Random-selection ablation.** Each adaptive policy was compared with random
   spending at the same rate — the control that separates intelligent selection
   from merely spending some fraction of the budget.
4. **Bootstrap uncertainty.** Cluster-bootstrap confidence intervals (seed
   20250115) on differences and rates; 1000-draw random nulls per arm.
5. **Multiplicity correction.** Holm–Bonferroni within each comparison family
   (16 comparisons in E1, 12 hypotheses in Phase 9, per-pair correction in
   Phases 14–15).
6. **Dual-arm replication.** Both corpus arms; the second arm is a replication,
   never a selection pool.
7. **Leakage guards in code.** Signal-building functions cannot receive
   ground-truth objects (enforced by the `build_signal_table` signature and
   contract tests); the Phase 14/15 cheap-first signal code serves only an
   allowlist of pre-Dense scalars, so cross-arm quantities are excluded by
   construction.
8. **Provenance by hash.** `docs/phase7-closure-evidence.md` records SHA-256 for
   every input and artifact the closure relies on, so any later run can be shown
   to have used the same evidence.

---

## 4. Baselines

Numbers below are **never mixed across incompatible corpus versions or
experimental conditions**. Each table states its source. The Phase 8 table is the
shipping-corpus record and the one to quote; the Phase 7 table is the historical
record on the pre-fix corpus, retained because Phase 7's closure gates were
computed on it.

### 4.1 Phase 7 E1 — five-arm comparison (107 queries, pre-fix 713-chunk corpus)

Source: `docs/phase_7_results.md` §2. Retrieval-only, paired queries.

| system | Recall@5 | MRR | Hit@5 | median ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.7788 | 0.7264 | 0.8318 | 1.88 | 3.45 |
| dense | 0.9283 | 0.8604 | 0.9720 | 472.84 | 691.20 |
| hybrid | 0.8723 | 0.8576 | 0.9346 | 465.65 | 733.62 |
| hybrid_rerank | 0.7165 | 0.5851 | 0.7757 | 3491.69 | 3958.55 |
| adaptive | 0.8723 | 0.8564 | 0.9346 | 524.85 | 655.41 |

Three facts from this table carry the rest of the report:

1. **Dense is the quality leader** (0.9283) and is *not* what the adaptive
   router escalates toward.
2. **The reranker is the worst arm on every quality metric and the most
   expensive** (7.5× hybrid's median latency). It is a net-negative rung at the
   top of the ladder.
3. **Adaptive ≡ hybrid on quality** — identical Recall@5 and Hit@5 on 107/107
   queries — while costing ~59 ms more (§5.2).

### 4.2 Phase 8 E1 — five arms on both corpus arms (107 queries, fixed corpus)

Source: `docs/phase-8-results.md` §2. Quality is unchanged by the corpus fix: no
quality delta on any arm survives Holm correction.

| arm | recall@5 before → after | MRR before → after | Hit@5 before → after |
| --- | --- | --- | --- |
| `bm25` | 0.7757 → 0.7866 | 0.7030 → 0.7313 | 0.8224 → 0.8505 |
| `dense` | 0.9143 → 0.8988 | 0.8714 → 0.8645 | 0.9533 → 0.9346 |
| `hybrid` | 0.8489 → 0.8583 | 0.8593 → 0.8678 | 0.9159 → 0.9252 |
| `hybrid_rerank` | 0.7072 → 0.7056 | 0.6016 → 0.6021 | 0.7664 → 0.7664 |
| `adaptive` | 0.8489 → 0.8583 | 0.8535 → 0.8664 | 0.9159 → 0.9252 |

Three findings from this study:

* **The corpus defect was real, was fixed to a measured standard, and did not
  change retrieval quality** — and did not rescue the reranker (−0.0016
  Recall@5). Both outcomes that would have *rescued* the reranker are excluded
  by measurement, which is a negative result.
* The two significant before/after deltas are both **latency**, and neither
  measures the corpus: they sit inside the query-embedding API call, which embeds
  the query rather than the index. The no-API `bm25` control arm moved
  1.83 → 1.85 ms. Those latency findings were withdrawn, and the methodological
  lesson recorded: *trace status — not trace count, and not a metric's own `n` —
  is what establishes that a measurement covers its query set* (13 of 29 arms in
  the first sweep were silently invalid and had to be re-run).
* `adaptive` remains numerically identical to `hybrid` on Recall@5 and Hit@5 in
  both arms.

### 4.3 Dense-only as the quality reference

Because Dense is the best fixed strategy on this corpus in every condition
measured, **full Dense retrieval is the bar an adaptive policy must approach**.
All headline adaptive results are therefore reported as a delta against Dense
(R@5 0.8988 on the shipping corpus; 0.9432/0.9424 on the fresh Phase 15 query
set), not against the weaker hybrid.

---

## 5. The Adaptive-Routing Investigation

### 5.1 Phase 6 — the initial router (implementation)

Phase 6 added an orchestration layer over the unchanged Phase 2–5 retrievers
(ADR-021). Its properties, all of which held in the experiments that followed:

* **Query-aware routing.** `QueryFeatureAnalyzer` computes deterministic,
  model-free features from the query alone (lexical, semantic, entity,
  complexity, multi-concept, question-type groups).
* **Structured decision.** `RuleBasedRouter` returns a `RoutingDecision`
  carrying strategy, confidence, per-strategy evidence, `candidate_k`,
  `final_top_k`, reranking requirement and versions — not a bare strategy name
  (ADR-022).
* **Confidence is not a probability.** It is a normalized score margin and never
  gates sufficiency on its own (ADR-023).
* **Sufficiency is judged from retrieved evidence alone**, label-free, at query
  time (ADR-024).
* **Bounded by construction.** Escalation follows a total-order ladder drawn from
  configuration with `max_escalation_steps` defaulting to 1, so `BM25 → Dense →
  Hybrid → BM25` is impossible by construction (ADR-025).
* **Router / retriever separation.** `AdaptiveRetriever` orchestrates only and
  conforms to the same `Retriever` protocol as the fixed strategies; an escalated
  query returns the stronger stage's results and the two rankings are never
  merged, because merging them would be a new fusion algorithm.

The learned router was explicitly deferred behind a `Router` interface
(ADR-026); no learned router was trained or deployed anywhere in the project.

**Limitation discovered immediately.** At the shipped defaults
(`sufficiency_threshold=0.5`, `cost_weight=0.25`, `max_escalation_steps=1`) the
sufficiency gate **never fires** on this benchmark. The router therefore
collapsed to its initial pick — Hybrid, for effectively the whole benchmark — and
routed nothing. Under the observed routing behaviour, Adaptive ≈ Hybrid.

### 5.2 Phase 7 — evaluation of the router, and the closure gates

**E1 (five arms, 107 queries).** Adaptive matched hybrid on quality and paid
~59 ms median overhead for the equivalence. Escalation bought latency, not
quality.

**E2 (escalation ablation A/B/C, 47 calibration).** Quality was tied end to end
across no-sufficiency / sufficiency / sufficiency+bounded-escalation (p_adj 1.0
on all quality metrics). The single significant result is B-vs-C latency:
**+58.91 ms median, p = 0.00309**. Escalation cost time and returned nothing.

**E3 (leave-one-out over the six feature groups, 47 calibration).**
**0 of 24 comparisons significant.** No feature group carried the result.

**E4 (sufficiency-threshold sweep).** Thresholds 0.3–0.6 are identical
(Recall@5 0.8830, MRR 0.8547, **0.0% escalation**); 0.7 escalates 3 of 47
queries (6.4%) and *loses* MRR (0.8377).

**E5 (cost-weight sweep 0.0–1.0).** Identical Recall@5 across all five values —
the cost term cannot matter when the gate never opens.

**Closure gates (pre-registered, both answered).**

* **Gate 2 — is there any headroom at all?** Over the *selectable* strategies
  only, an oracle over per-query choices beats the best fixed strategy by
  **+0.0218** (after) / **+0.0343** (before) Recall@5, CIs excluding zero — but
  the gain is carried by only **4/107 and 6/107 queries**, with 98 ties at max.
* **Gate 3 — is that headroom identifiable out of sample?** No. Fitting a ridge
  logistic regression purely as a measuring instrument reaches AUC 1.000 on the
  fit split and **0.534 out of sample, with 0 of 24 features surviving Holm**
  correction, on both arms. The fit-split/out-of-sample gap is the signature of
  overfitting to 8 positives.

The Phase 7 conclusion, preserved verbatim in spirit: *oracle-level
query-specific headroom exists; it is not reliably detectable from observable
signals at this sample size; and a router exploiting it is therefore not
established.* Claim 1 being true is not evidence for claims 2 or 3.

**Implication.** On this benchmark the only available routing answer was "use
Hybrid", which the router already produced — so adaptive routing earned nothing
over a fixed strategy while adding a decision layer and ~59 ms.

### 5.3 Phase 9 — signal discovery (verdict `PROMISING_SIGNAL`)

**Question.** Which measurable query–corpus properties explain per-query
dispersion in retrieval outcome across strategies, and is that structure
predictable from information available *before* paying for the expensive
strategy?

**Target.** `T-disp3` = max − min Recall@5 over `{bm25, dense, hybrid}` — a
continuous dispersion measure, frozen at k = 5, evaluated on both corpus arms.
The reranker is excluded from the target after it was shown strictly worst on
26/34 (after) and 30/41 (before) dispersed queries while never strictly best.

**Finding 1 — dispersion is real and replicated.** `T-disp3 > 0` on 21/107
queries (after) and 29/107 (before). Not a tie-break artefact.

**Finding 2 — the mechanism is evidence diffuseness, not a single latent
quantity.** A 12-hypothesis family (8 disagreement + 4 score geometry, Holm
denominator 12) was screened against the target. Five hypotheses survived
mechanism screening. The survivor that generalized was **Dense distinct-document
ratio (DDR)** — how many distinct documents Dense's top-k actually covers:

| quantity | relationship to dispersion | evidence |
| --- | --- | --- |
| `DDR@dense` | Spearman ρ +0.343 (after) / +0.406 (before), Holm p = 0.0012 | replicated association |
| `DDR@dense`, held-out test split | ρ +0.410, no fitting | generalizes out of sample |
| pairwise Jaccard between strategies | ρ −0.454 / −0.520 | replicated |
| best query-only pre-routing signal | abs(ρ) = 0.194 | **no query-only signal met the effect threshold** |

DDR's own correlation was slightly higher than Jaccard's, and its direction is
the intuitive one: when retrieval returns few distinct documents, the
strategies disagree more. That is the **retrieval-disagreement /
evidence-diffuseness structure** the project set out to find, and it is a real,
replicated, mechanistic finding.

**Finding 3 — the limits, stated as limits.** DDR is a dispersion *diagnostic*,
not a routing signal. It is available only **after** Dense has already run, so
it can inform only a *second* decision — never a pre-cost decision. It is
associational, not causal, and it has no intervention behind it. The verdict was
therefore `PROMISING_SIGNAL` with a follow-on boundary note, not a mandate to
build a router.

### 5.4 Phase 10 — Dense → Hybrid escalation (verdict `FAILURE`)

**Hypothesis.** After paying for Dense retrieval, `distinct_doc_ratio@dense` can
decide whether escalating to Hybrid is worthwhile, so that the adaptive policy
improves the quality–latency–cost trade-off over fixed strategies.

**Setup.** Fully offline, no new retrieval. Escalation rung is Dense → Hybrid
only (ADR-028 excludes the reranker rung). The frozen oracle escalates iff
`recall@5(hybrid) − recall@5(dense) ≥ 0.01`. Incremental escalation cost is
measured at **31.0 ms** per query (hybrid's BM25 + fusion + vector stages;
the embedding call is already paid, so the API-call count is flat under every
policy).

**Headroom first.** Escalation helps on 4 queries per arm but **harms 9 (after)
and 12 (before)**. Always-escalate is therefore net-negative *by construction* —
a useful policy must **select**, not merely spend. The selectable headroom is 4
queries, the power limitation declared in advance.

**Result — the policy fails, cleanly:**

| observation | after | before |
| --- | --- | --- |
| calibration ROC-AUC | **0.550** (chance) | 0.861 (small-sample noise; its own Spearman runs the wrong way) |
| calibration Spearman(DDR, Δrecall) | −0.129 | −0.164 |
| test Δ(adaptive − dense-only), Recall@5 | **−0.050** [−0.117, 0] | −0.033 |
| oracle positives caught (TP) | **0 of 2** | **0 of 2** |
| P(random escalation ≥ adaptive) at equal spend | **1.000** | **0.954** |

The two shipping-arm positives sit at DDR 0.2 and 0.8 — split across the whole
range — and the frozen threshold selected a non-degenerate rule (rate 0.191,
tp/fp/fn/tn = 1/8/1/37). So this is not a degenerate-threshold artefact: the
signal simply does not identify the queries where escalation helps.

**Verdict: `FAILURE`.** The frozen DDR policy is **dominated** — by dense-only
(higher quality, lower cost) *and* by random escalation at the same spend — on
both arms.

**The central conceptual lesson.** Phase 9's correlation stands: DDR tracks
cross-strategy dispersion. But dispersion on this benchmark is dominated by
*Dense succeeding where RRF fusion degrades the ranking*. DDR therefore marks
disagreement **without** marking escalation value:

> **retrieval disagreement ≠ retrieval insufficiency ≠ escalation value.**

The two phenomena Phase 9 could not separate resolve here against usefulness.
Correlation with dispersion did not predict gain — and the honest reporting of
that mismatch is the single most valuable result in the project's history.

### 5.5 Phase 11 — mechanism analysis (outcome B)

**Question.** What exactly causes Hybrid to help on some queries and hurt on
others — and can the two cases be distinguished *before* paying for Hybrid?

**Groups.** Escalation helps on only **5 unique queries** across both arms (4 per
arm) and harms on 9–12. Per-query inspection shows every harm query satisfies
`dense_recall_at_5 > 0` — i.e. **all harms are State 3** (Dense was already
correct and fusion corrupted it) and **all helps are State 2** (Dense genuinely
missed the evidence). The three-state mapping is exact, not approximate.

**Q1 — why Hybrid helps: BM25-side rescue of Dense-missed evidence.** On all
five helps, Dense's Recall@5 is 0–0.33 while BM25 holds or highly ranks the
relevant document and RRF keeps it in the top-5. The relevant document enters
from the `bm25_only` pool in 4 of 5 cases. In every case Dense's top-1 score is
low (0.40–0.56) and pairwise Jaccard overlap is low (0–0.4). This is the
**useful** case: the cheap retriever supplied the evidence Dense missed.

**Q2 — why Hybrid hurts: two sub-mechanisms, both State 3.**

* **H1 — intruder promotion.** `bm25_only` documents that BM25 ranks confidently
  but irrelevantly (crag, retro, realm, contriever) displace the relevant
  document from the top-5.
* **H2 — re-ranking among already-shared documents.** For several harm queries
  **no new document enters at all**: RRF reorders the shared pool and the
  relevant document's chunks fall below the cutoff (e.g. one query's Hybrid
  top-5 is five chunks of a single wrong document). This is the Phase 4
  RRF-flattening effect at per-query resolution — and it means **even an
  agreeing BM25 can preside over harm**. Agreement is not safety either.

**Q3 — can the two cases be distinguished beforehand?** Directionally yes,
establishably no. On calibration, helps show weaker Dense and higher
disagreement than harms (Dense top-1 0.49/0.48 vs 0.63/0.56; Jaccard
0.13/0.27 vs 0.41/0.40), but the ranges overlap and the bounded separation check
over the six pre-declared features separates **1/6 (after) and 0/6 (before)** —
and the single hit fails on the replication arm, the signature of fitting five
points.

**Outcome B — insufficient evidence.** The mechanism is plausible and the
per-query records are exact, but n = 5 unique positives cannot establish a
signal; no candidate was promoted and no intervention was run. The pre-registered
gate that "enough development examples exist" failed *a priori*, and the
project honoured it by stopping rather than writing a two-point rule.

### 5.6 Phase 12 — why the post-dense track was stopped

Position **C — STOP CURRENT ADAPTIVE-ROUTING TRACK** was adopted, with the
following evidence converging:

1. **Gate 3** (Phases 7/8): no observable query or first-stage signal identifies
   the headroom out of sample (AUC 0.534, 0/24 Holm).
2. **Gate 9.3** (Phase 9): no query-only pre-routing signal met the effect
   threshold (best abs(ρ) = 0.194).
3. **Phase 10**: the one surviving single-arm signal (DDR) fails as a routing
   signal at chance AUC, and its policy is dominated by dense-only *and* by
   random spending.
4. **Phase 11**: the mechanism is understood per query, but the helpful and
   harmful cases cannot be distinguished before paying (1/6, 0/6 separation), and
   there are only 5 positives.
5. **The escalation rung itself is net-negative** on this corpus (helps 4, harms
   9–12), so the first rung of any ladder built on it would need
   re-justification before any router were worth building around it.

This is not a failure to hide. It is a **rigorous stopping point**: continuing
blind feature engineering on 4–8 positives would have risked an endless
optimization loop in which any apparent gain is fitting noise. The re-entry
conditions Phase 12 states (a benchmark with dozens of oracle positives plus an
upfront power analysis; a rung that beats Dense somewhere substantial; a
pre-registered confidence×disagreement rule; the frozen split discipline; a gate
that can say "no" again) are recorded in `docs/phase12_results.md` §9.

### 5.7 Phases 13–15 — the cheap-first track (BM25 → Dense)

The Phase 12 stop applied to the *post-dense* track. Phase 13 then asked whether
a **different decision point** — before any cost is paid — could work, and chose
the cheap-first track on feasibility evidence rather than optimism:

**Why BM25 first.** Three measured facts (Phase 13 §2, read-only counts over
frozen rows):

1. **The rung is the only EV-positive one in the project.** Dense beats BM25 by
   **+0.11 to +0.14 Recall@5** on both arms, so escalating BM25 → Dense has real
   upside (unlike Dense → Hybrid, which is net-negative).
2. **There is far more headroom.** Cheap-first positives (Dense − BM25 ≥ 0.01)
   number **16 (after) / 21 (before)** — 4–5× any earlier framing. Conversely,
   **91 / 86** of 107 queries need no Dense call at all.
3. **The savings currency is real.** The decision sits *before* the embedding
   API call: ~2 ms of local BM25 versus ~450 ms of provider call. The currency is
   **API calls avoided**, not 31 ms of local fusion.

#### Phase 14 — first cheap-first experiment (`INSUFFICIENT EVIDENCE`)

Fully offline from frozen rows. Frozen oracle: escalate iff
`recall@5(dense) − recall@5(bm25) ≥ 0.01`. Six pre-Dense signals were allowlisted
(query + BM25-output state only; cross-arm quantities excluded by construction,
since Dense has not run).

On calibration (n = 47, only **5 positives** — the binding limitation declared in
advance) all six signals ran the mechanism way, with `bm25_slope` strongest
(flat BM25 score decay ⇒ Dense likely needed; AUC 0.757, descriptive only). The
frozen rule selected was **`bm25_slope ≥ −0.915`**, non-degenerate (rate 0.191).

On test (one pass, untouched during selection):

| observation | after | before |
| --- | --- | --- |
| Dense-call rate | 21.7% | 21.7% |
| oracle positives caught | 5 / 11 | 5 / 11 |
| Δ vs BM25-only, Recall@5 | **+0.078** | +0.050 |
| Δ vs dense-only, Recall@5 | **−0.081** | −0.086 |
| P(random ≥ adaptive) at equal rate | **0.029** (wins) | 0.21 (does not) |

The rule improved over BM25-only on both arms and beat random spending on one,
but **missed the success criterion (within 0.02 of dense-only) outright**, with
TP = 5 against FN = 6 on *both* arms — it found as many positives as it missed.
The Phase 14 protocol's guards (1-SE selection, cross-arm consistency, one-pass
test, random ablation) did exactly their job: they prevented this flat,
5-positive landscape from being read as signal. Verdict: `INSUFFICIENT
EVIDENCE` — the one open question left was whether the mixed random-ablation
result was power or noise.

#### Phase 15 — powered confirmation on 110 fresh queries (`INSUFFICIENT EVIDENCE`)

Phase 14's binding limitation was statistical power, so Phase 15 was a
**confirmation**, not a new search: the frozen Phase 14 rule was applied
**unchanged** (`bm25_slope ≥ −0.915`, sha256-pinned in the provenance) to
**110 freshly curated queries per arm**, all `split: "test"`, duplicate-screened
against all 266 prior queries (0 flags). Because the rule was frozen in advance,
there is **no calibration split** — the policy is evaluated exactly once on data
it never saw.

| observation | after | before |
| --- | --- | --- |
| oracle positives | 14 | 11 |
| Dense-call rate (CI95) | 0.282 [0.200, 0.373] | 0.318 [0.236, 0.409] |
| positives caught | **9 / 14** | **9 / 11** |
| false positives | 22 | 26 |
| Recall@5 — BM25 / adaptive / Dense | 0.8545 / 0.9129 / 0.9432 | 0.8795 / 0.9242 / 0.9424 |
| Δ(adaptive − BM25) | **+0.058** [0.016, 0.107] | **+0.045** [0.004, 0.092] |
| Δ(adaptive − dense) | **−0.030** [−0.067, 0.000] | **−0.018** [−0.046, 0.000] |
| P(random ≥ adaptive) | **0.008** | **0.019** |

**What replicated (the project's strongest positive result).** The Phase 14
selection claim **replicated its selection claim** on 220 fresh
query-instances: the frozen rule catches **18 of 25** oracle positives at ~30%
spend — avoiding **~70% of Dense calls** — and beats random spending decisively
on both arms individually (P = 0.008 / 0.019) and combined (**P = 0.001**). The
combined cluster-bootstrap CI on adaptive − BM25 is **+0.0515 [0.0110, 0.0977]**,
excluding zero. The Phase 14 open question (mixed random ablation) is closed in
the affirmative.

**What did not.** Adaptive still trails full Dense by **0.018–0.030 Recall@5**
(0.036–0.042 MRR). Against the pre-registered margin bar (a) — combined adaptive
≥ combined dense − 0.02 — the combined figures are 0.9186 vs 0.9428, a gap of
**−0.0242, missing the bar by 0.004**. No failure bar was met either, so the
frozen mapping yields `INSUFFICIENT EVIDENCE`: **4 of 5 success bars met,
including the powered primary, with the dense-margin bar missed by 0.004.**

**Why it stopped here.** Two consecutive `INSUFFICIENT`s with the same
signature — genuine selection signal, dense-margin shortfall — mean the
remaining uncertainty is no longer statistical power but the *claim* itself:
whether near-Dense quality at ~70% fewer embedding calls is a success is a
**margin judgment, not a measurement gap**. A further experiment on this
benchmark would need a genuinely new information source (a new rung, a new
decision point, or a cost model that prices the 0.024 gap). Re-sweeping
thresholds or features on these queries is prohibited by the same discipline
that made the confirmation credible — and doing so anyway would have manufactured
a stronger conclusion than the evidence supports.

---

## 6. Findings

Separated by evidential strength, because conflating these is how negative
results get lost.

### 6.1 Confirmed findings

Supported by repeated, replicated measurement on two corpus arms, with the
methodology able to detect the effect if it existed.

1. **Per-query retrieval dispersion across strategies is real.** `T-disp3 > 0`
   on 21/107 and 29/107 queries (after/before); it is not a tie-break artefact.
2. **Dense distinct-document ratio tracks that dispersion, and the association
   generalizes out of sample.** ρ +0.343 / +0.406 (Holm p = 0.0012) on
   calibration; ρ +0.410 on held-out test with no fitting. Disagreement is
   measurable, real, and mechanically tied to evidence diffuseness.
3. **The reranker is a net-negative rung on this corpus.** Worst arm on every
   quality metric (Recall@5 0.7165 pre-fix, 0.7056 shipping) at 6.2–7.5× hybrid's
   median latency, and the corpus fix did not rescue it (−0.0016). It was
   removed from escalation consideration (ADR-028).
4. **Dense retrieval is the best fixed strategy in every condition measured**,
   and adaptive routing collapsed onto the second-best one at the shipped
   configuration, paying ~59 ms for the equivalence.
5. **Disagreement does not imply escalation value.** The DDR policy failed at
   chance AUC and was dominated by both dense-only and random spending (Phase
   10) — the association of §6.1.2 did not survive intervention.
6. **The helpful and harmful escalation cases have different mechanisms.**
   Helps are BM25-side rescue of Dense-missed evidence (State 2); harms are
   fusion corrupting an already-correct Dense result (State 3), via intruder
   promotion or pure re-ranking among shared documents (Phase 11, exact
   per-query records).
7. **Cheap first-stage information can select Dense calls better than random.**
   On 220 fresh query-instances the frozen `bm25_slope` rule caught 18 of 25
   oracle positives at ~30% spend, beating random spending on both arms
   individually (P = 0.008 / 0.019) and combined (P = 0.001), with the combined
   cluster-bootstrap CI on adaptive − BM25 excluding zero
   (+0.0515 [0.0110, 0.0977]).
8. **Trace status — not trace count, and not a metric's own `n` — is what
   establishes that a measurement covers its query set.** 13 of 29 Phase 8 arms
   were silently invalid while reporting `ok`, and adopting a conclusion from
   them would have been right by accident.

### 6.2 Promising findings

Real signal that is not yet sufficient for a production-quality adaptive policy.

1. **Flat BM25 score decay predicts Dense's marginal value.** `bm25_slope`
   (AUC 0.757 on calibration) beat the Phase 9 protagonist `DDR@bm25`
   (0.586) and won the frozen family selection. It is BM25-side and
   pre-cost, so it is decision-time usable — and it is the strongest
   pre-cost candidate the project found. But its calibration rested on **5
   positives**, and its test record was TP 5 / FN 6 on *both* arms.
2. **The cheap-first decision point is where the headroom is.** 16/21
   cheap-first positives and 91/86 queries needing no Dense call, with an
   EV-positive rung (+0.11–0.14 R@5) — structurally unlike the post-dense track,
   whose rung was net-negative. Feasibility is established; the policy is not.
3. **Escalation is affordable when it happens.** The incremental cost of the
   one extra local stage is ~31 ms, against ~450 ms for the provider call a
   cheap-first policy avoids. If a *net-positive* rung and a well-calibrated
   selector were ever found, the cost arithmetic is favourable.

### 6.3 Negative findings

Tested and rejected, each preserved as part of the research record.

1. **Query-aware rule-based routing does not beat a fixed strategy on this
   benchmark** (Phases 6–7): the sufficiency gate never opens, adaptive ≡ hybrid,
   and 0 of 24 feature-ablation comparisons were significant.
2. **The reranker rung does not pay for itself** (Phase 5, Phase 8): every head
   metric fell on all three first-stages; the depth ablation shows R@5 falling
   and rerank latency climbing monotonically with candidate depth.
3. **The corpus extraction defect was not the cause of the reranker's collapse**
   (Phase 8): fixing it changed reranker Recall@5 by −0.0016 on 99 of 107 ties.
4. **Post-dense escalation on `distinct_doc_ratio@dense` fails** (Phase 10):
   AUC 0.55, TP 0 of 2, dominated by dense-only and by random.
5. **Gate 3's observable-signal search is exhausted for this setting**
   (Phase 8): AUC 0.534 out of sample, 0 of 24 features surviving Holm.
6. **The RRF-hybrid rung harms more queries than it helps on this corpus**
   (Phases 10–11): 4 helps against 9–12 harms — so always-escalate is
   net-negative by construction, and any ladder built on it needs
   re-justification before it is worth routing to.
7. **The Phase 14 rule did not reproduce on the Phase 15 positive rate** — not a
   failure of the mechanism but evidence that a 5-positive selection is
   noise-exposed, exactly as the 1-SE guard predicted.

### 6.4 Unresolved questions

Genuinely open; none is answered by this project's evidence.

1. **Would a sufficiently powered cheap-first study clear the Dense margin?**
   Phase 15 removed the power objection but not the trade-off: ~70% fewer Dense
   calls for −0.018 to −0.030 Recall@5 is a *value judgment* the project
   deliberately did not make on the user's behalf.
2. **Can a learned (non-rule) router exploit the State-2/State-3 interaction?**
   The mechanism-consistent reading — weak-Dense × disagreement → State 2;
   strong-Dense × disagreement → State 3 — is the only hypothesis consistent
   with the data, and it is *not established* (n = 5, separation 1/6 and 0/6).
   The `Router` interface leaves room for it (ADR-026); no learned router was
   built.
3. **Does the adaptive policy generalize beyond this corpus?** Every claim here
   is from one 14-paper academic-RAG collection. No second domain was tested.
4. **Is a second-stage action with positive expected value available at all?**
   On this corpus RRF-hybrid hurts 2–3× more queries than it helps, so the rung
   itself — not the router — may be the binding constraint for the post-dense
   track.
5. **Can `nDCG@5` and Precision@k be trusted at all?** Blocked upstream by 42
   spliced section paths in the corpus; only 22 of 47 corrupt labels are safely
   repairable, and label repair alone cannot fix the metric.

---

## 7. Main Research Conclusions

Derived from the findings above rather than asserted:

**Adaptive retrieval is plausible, but identifying when additional retrieval is
valuable is substantially harder than detecting retrieval disagreement.** The
project measured both halves of that sentence. Dispersion is real, measurable,
mechanistic, and replicates (Phase 9). Turning it into a routing decision failed
at every level tried — no query-only signal exists, the one surviving
single-arm signal is chance-level, and the resulting policy is beaten by random
(Phases 7, 9, 10). The mechanism analysis explains exactly why: disagreement,
insufficiency, and escalation value are three different things, and on this
corpus the first is common while the third is rare and hard to predict (Phase
11).

**Cheap first-stage information does contain useful signal about whether
expensive Dense retrieval will help.** This is the project's most important
positive result, and it survived the strictest test available: a rule frozen on
one split, applied unchanged to 110 fresh queries per arm with no calibration
data, caught 18 of 25 oracle positives, avoided ~70% of Dense calls, improved
over BM25-only by +0.045 to +0.058 Recall@5 with a bootstrap CI excluding zero,
and beat random spending decisively (P = 0.001 combined).

**But the quality gap to full Dense retrieval remained large enough that the
evidence does not support a replacement policy.** Adaptive trailed Dense by
0.018–0.030 Recall@5 and missed the pre-registered margin bar by 0.004 on the
combined mean. Under the frozen verdict mapping that is `INSUFFICIENT
EVIDENCE` — not success, not failure, and explicitly not a licence to keep
searching on the same data.

**Therefore the adaptive-routing research track was deliberately stopped.** Two
consecutive inconclusive phases with the same signature, a net-negative
escalation rung, and a benchmark whose positives could not support a supervised
claim, made further feature engineering on this benchmark an optimisation loop
rather than an experiment. Stopping — and writing down why, and on what evidence
one could re-enter — is the scientifically correct outcome here, and it is the
one the project took.

---

## 8. Limitations

Only limitations this project actually measured or encountered are listed.

### 8.1 Corpus limitations

* **Single corpus, single domain.** 14 foundational IR/NLP papers. No second
  collection, no non-academic domain, no multilingual or long-document setting.
* **Known corpus defects.** The `pyserini_lin_2021` content/title mismatch
  (contained by excluding 6 records); two-column extraction interleaving (fixed
  in Phase 8); **42 spliced section paths remain in the shipping corpus**.
* **Corpus versioning is not self-identifying.** Both Phase 8 arms report the
  same `corpus_version` string, so the namespace directory — not the version
  string — is what distinguishes them.

### 8.2 Evaluation-set limitations

* **The binding constraint was positives, not queries.** 107 queries on the
  Phase 7 benchmark, but only 4–8 oracle positives for the post-dense question
  and 5 calibration / 11 test positives for Phase 14. AUC claims need positives
  in the tens; these studies had units or low tens.
* **Phase 15 reached 14 / 11 positives** — near the low edge of the
  pre-registered expectation (~20/arm), which the protocol required to be
  reported rather than repaired by top-up curation.
* **Recall@5 is coarse** (~90% of queries tie), which is why paired sign-test
  p-values stay non-significant while bootstrap CIs exclude zero. Both are
  reported; neither is hidden.
* **Category cells are underpowered**: `comparative` (8) and `multi_document`
  (7) fall below the `min_cell = 5` threshold per split individually.

### 8.3 Metric limitations

* **`nDCG@5` is not reported at all** — its IDCG denominator derives from the
  chunks a trace happened to retrieve, and the underlying section labels are
  corrupt (47 of 140). Reported as *not measured*, never estimated.
* **`Precision@k` is document-derived**: every chunk of a relevant document
  counts as relevant unless a section prefix matches, so it partly measures how
  many chunks of a right document were returned.
* **No USD figures.** All runs are retrieval-only, so `estimated_cost_usd` has
  `n_pairs = 0`. Cost is reported as latency and call-rate only.
* **No end-to-end answer quality.** Generation and the LLM judge exist in the
  codebase but were not used for any routing claim, so nothing here says whether
  retrieval differences change generated answers.

### 8.4 Routing-policy limitations

* **The shipped router never routes.** At the frozen configuration its
  sufficiency gate does not open, so the adaptive arm is hybrid in practice.
* **The only deployed signals are rule-based and hand-specified.** No learned
  model was trained (deliberately deferred, ADR-026).
* **The best pre-cost rule misses half the positives** (9 of 25 caught at ~30%
  spend; precision 29% / 26%) — it is a real but lossy selector.
* **Latency is provider-dependent.** Dense-stage timings carry a ~2.4 s
  rate-limited embedding outlier; `total_latency_ms` alone cannot separate
  provider variance from system cost without a no-API control arm.

### 8.5 Retrieval-system limitations

* **One embedding model, one reranker, one fusion method.** `text-embedding-3-large`,
  a general-web MS-MARCO cross-encoder, and RRF. The reranker's failure is
  attributed to model/domain mismatch, not to the composition-layer design — a
  hypothesis, not a diagnosis.
* **No cross-encoder fine-tuning** was attempted (torch is deliberately absent;
  ONNX Runtime only).
* **Retrieval-only evaluation**: fusion behaviour is studied, generation is not.

### 8.6 Statistical limitations

* **Holm-corrected paired tests on coarse Recall@5** are frequently
  non-significant even where bootstrap CIs exclude zero; the two answer different
  questions and are reported side by side.
* **Rank-correlation effective n is pairwise-defined** (Phase 9: n = 23), and the
  before/after subsets overlap on only 13 of 23 — which is why those
  correlations are reported as replicated associations, never as estimates of
  effect size.
* **DDR takes only five values** {0.2, 0.4, 0.6, 0.8, 1.0} — near-binary, and
  its calibration ρ (+0.254) is below the 0.30 threshold even though its test ρ
  (+0.410) exceeds it, i.e. a single-split pass rather than a stable two-split
  margin.
* **Mechanism attributions are top-10 membership**, since candidate depth 11–20
  is not persisted in traces; `neither` is reported honestly where it occurs.

### 8.7 Reproducibility limitations

* **Experiment artifacts are gitignored.** Traces, figures and tables live under
  `experiments/` and `storage/` and are recoverable only from the working tree;
  `docs/phase7-closure-evidence.md` pins the SHA-256 of the inputs and artifacts
  that matter, but the artifacts themselves are not versioned.
* **The working-tree `data/processed/` corpus (713 chunks, manifest
  `max_tokens: 600`, rebuilt 2026-10-03) does not text-match the BM25 index
  used in Phases 2–7** (mean 408 / median 447 / max 891 tokens, matching the
  documented distribution). The authoritative record for historical results is
  therefore the storage indexes plus the documented figures, not the current
  derived corpus. This discrepancy is recorded rather than silently repaired:
  rebuilding would change `corpus_version` and invalidate the frozen results.
* **Live-provider results are hardware- and provider-dependent** (the host ran the
  ONNX reranker on CPU; latency numbers do not transfer).

---

## 9. What This Project Does NOT Claim

Stated explicitly, because the gap between "signal exists" and "system works" is
where adaptive-retrieval work most often overreaches.

AdaptiveRAG does **not** claim:

* **An optimal adaptive retrieval algorithm.** No policy was shown optimal over
  the strategy space; the only optimality figure in the project is an oracle
  ceiling, which bounds the problem rather than solving it.
* **A production-ready routing policy.** The best policy missed its pre-registered
  quality bar, and the shipped router does not route at all.
* **Universal generalization.** Every result is one corpus, one embedding model,
  one fusion method, one benchmark family. Nothing was tested on a second
  domain.
* **Causal explanations for the observed correlations.** Phase 9 established
  associations (and a mechanism narrative), not causation; the one intervention
  attempted (Phase 10) falsified the causal reading rather than confirming it.
* **That adaptive retrieval outperforms Dense retrieval.** It does not: every
  cheap-first configuration trailed full Dense by 0.018–0.030 Recall@5.
* **That the final policy is ready to replace full Dense retrieval.** The frozen
  verdict is `INSUFFICIENT EVIDENCE`, which by construction is not a pass.
* **That Phase 7's rerank failure was caused by the corpus defect.** That
  attribution was explicitly withdrawn; the cause is unidentified.
* **Any answer-quality conclusion.** No routing claim was ever validated
  end-to-end through generation.
* **That its own code is validated beyond its offline test suite.** Live-provider
  arms were executed as recorded, but the test suite is offline and
  deterministic by design (1056 tests, 2 integration tests deselected).

---

## 10. Final Status

```text
Status:
Research investigation completed.
Adaptive-routing experimentation paused.
No final optimal routing policy claimed.
Project consolidated for documentation, presentation, and future research.
```

Phase status at close: **Phases 1–15 COMPLETE.** Phase verdicts were
`PROMISING_SIGNAL` (9), `FAILURE` (10), outcome B (11), `STOP CURRENT
ADAPTIVE-ROUTING TRACK` (12), and `INSUFFICIENT EVIDENCE` (14 and 15). The
re-entry conditions for the stopped track are preserved in
`docs/phase12_results.md` §9; the open questions are in §6.4 above.

---

## 11. Future Work

Possible directions, **not** completed work and **not** authorized by this
project's evidence. Listed so the stopping decision is not mistaken for a claim
that nothing further is worth trying.

* **Learned routing.** A calibrated classifier over the State-2/State-3
  interaction family, trained on a benchmark with enough positives to support
  it (ADR-026 leaves the interface open).
* **Larger or multiple corpora.** The single most valuable change: dozens of
  oracle positives per framing, with an upfront power analysis, on a second
  domain.
* **Cost-aware objectives that price the gap.** Whether −0.024 Recall@5 is worth
  ~70% fewer embedding calls is a deployment decision that depends on real
  latency and cost, not on Recall@5 alone.
* **Better calibrated utility estimation.** Rather than classifying "does Dense
  help", estimate the *expected utility* of escalating under an explicit cost
  model.
* **Alternative retrieval architectures.** Different rungs (not RRF-hybrid, which
  is net-negative here), late-interaction scoring, or domain-matched rerankers.
* **Answerability-grounded routing.** The generation-side judge exists in the
  codebase and was never used for routing; it is an untested information source.

---

## 12. Evidence Index

Where each class of claim in this report is recorded.

| Claim class | Authoritative source |
| --- | --- |
| Corpus, metadata, digests | `docs/corpus.md`, `data/metadata/papers.json` |
| Dense baseline (Phase 2) | `docs/phases/phase-2.md`, `docs/dense_baseline.md` |
| BM25 (Phase 3) | `docs/phases/phase-3.md` |
| Hybrid / RRF (Phase 4) | `docs/phases/phase-4.md`, ADR-019 |
| Reranking (Phase 5) | `docs/phases/phase-5.md`, ADR-020 |
| Router design (Phase 6) | `docs/phases/phase-6.md`, ADR-021…026 |
| E-series results (Phase 7) | `docs/phase_7_results.md` |
| Phase 7 closure gates | `docs/phase_7_closure_results.md` |
| Phase 8 corpus study | `docs/phase-8-results.md` |
| Cost-freeze recommendation | `docs/strategy-cost-freeze-decision.md` |
| Provenance hashes | `docs/phase7-closure-evidence.md` |
| Signal discovery (Phase 9) | `docs/phase9_results.md` |
| Post-dense failure (Phase 10) | `docs/phase10_results.md` |
| Mechanism analysis (Phase 11) | `docs/phase11_results.md` |
| Stopping decision (Phase 12) | `docs/phase12_results.md` |
| Cheap-first selection (Phase 13) | `docs/phase13_results.md` |
| Cheap-first experiment (Phase 14) | `docs/phase14_results.md` |
| Powered confirmation (Phase 15) | `docs/phase15_results.md` |
| Architectural decisions | `docs/decision.md` (ADR-001 … ADR-028) |
| Detailed architecture | `docs/architecture.md`, `docs/architecture/system_overview.md` |
| Reproduction commands | `docs/reproducibility/reproduction.md` |
| Phase-by-phase narrative | `docs/history/experiment_timeline.md` |
| Pre-execution planning notes | `docs/history/plans/` |

---

*This report describes an investigation that reached a negative-to-inconclusive
verdict and stopped deliberately. Its negative results are the project's primary
contribution and are preserved in full.*