# Experiment Timeline

Phase 1 → 15, each with its objective, hypothesis, experiment, result and
decision — enough to reconstruct how the investigation moved from building
retrieval strategies to a deliberate stopping point. Full narrative:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md).

```text
Phase 1  Corpus Foundation
   ↓
Phase 2  Dense RAG Baseline
   ↓
Phase 3  BM25 Lexical Baseline
   ↓
Phase 4  Hybrid Retrieval (RRF)
   ↓
Phase 5  Reranking (cross-encoder)
   ↓
Phase 6  Adaptive Routing (orchestration layer)
   ↓
Phase 7  Evaluation & Ablations  ── closed by two gates
   ↓
Phase 8  Corpus Fix Study (before/after)
   ↓
Phase 9  Signal Discovery ── PROMISING_SIGNAL
   ↓
Phase 10 Post-Dense Escalation ── FAILURE
   ↓
Phase 11 Mechanism Analysis ── outcome B
   ↓
Phase 12 Research Boundary ── STOP the post-dense track
   ↓
Phase 13 Next Direction ── cheap-first track selected
   ↓
Phase 14 Cheap-First Experiment ── INSUFFICIENT EVIDENCE
   ↓
Phase 15 Powered Confirmation ── INSUFFICIENT EVIDENCE (4/5 bars)
   ↓
      STOPPED — research investigation complete
```

---

## Phases 1–5: building the fixed strategies

### Phase 1 — Corpus Foundation

* **Objective** — a canonical, verifiable research corpus.
* **Hypothesis** — 14 foundational IR/NLP papers suffice to compare retrieval
  strategies meaningfully.
* **Experiment** — deterministic download, PDF magic-byte and SHA-256
  validation, metadata manifest.
* **Result** — 14 papers with verified digests (`data/metadata/papers.json`).
* **Decision** — proceed; raw PDFs kept out of git, reproducible via script.

### Phase 2 — Dense RAG Baseline

* **Objective** — the fixed Dense-RAG baseline everything else is measured
  against.
* **Hypothesis** — a bi-encoder over structure-aware chunks gives a strong
  single-stage baseline.
* **Experiment** — ingestion → structure-aware chunking → embedding → Qdrant →
  retrieval → evaluation on the 20-query set.
* **Result** — Dense Recall@5 0.8417, MRR 0.95, ~750 ms (dominated by the
  embedding API call).
* **Decision** — adopt as the quality reference and the expensive rung.

### Phase 3 — BM25 Lexical Baseline

* **Objective** — an independent cheap strategy over the same corpus.
* **Hypothesis** — lexical retrieval is dramatically cheaper and modestly worse.
* **Experiment** — inverted index, same evaluation pipeline.
* **Result** — BM25 Recall@5 0.7167, MRR 0.6771, **1.48 ms** (~500× cheaper than
  Dense).
* **Decision** — keep as the cheap rung and the no-API latency control arm.

### Phase 4 — Hybrid Retrieval

* **Objective** — does rank fusion of the two beat either?
* **Hypothesis** — RRF(dense, BM25) recovers evidence each misses.
* **Experiment** — RRF fusion, k=20 per arm, top-10 out.
* **Result** — Hybrid Recall@5 0.8583 — better than BM25, *below* Dense
  (0.8417 vs 0.8583 on the 20-query set; 0.9283 Dense vs 0.8723 hybrid on the
  107-query set). Fusion flattens ranked scores.
* **Decision** — keep as a selectable strategy; note that Dense alone leads.

### Phase 5 — Reranking

* **Objective** — does a cross-encoder second stage pay for itself?
* **Hypothesis** — reranking the top-N lifts precision at acceptable cost.
* **Experiment** — ONNX cross-encoder over all three first-stages + candidate-depth
  ablation.
* **Result** — **every head metric fell** (MRR −0.083 / −0.061 / −0.127); depth
  ablation makes R@5 fall as latency climbs; ~155 ms per candidate pair, two
  orders of magnitude over the estimate.
* **Decision** — record as observed; the rung is not adopted as a default.

---

## Phases 6–8: the adaptive layer and its evaluation

### Phase 6 — Adaptive Routing (implementation)

* **Objective** — a query-aware decision layer over the unchanged strategies.
* **Hypothesis** — deterministic query features plus a label-free sufficiency
  check can pick a strategy and escalate when warranted.
* **Experiment** — offline implementation + BM25-only arm validation.
* **Result** — implemented and validated; **the sufficiency gate never fires**
  at the shipped defaults, so adaptive collapses onto its initial pick.
* **Decision** — proceed to full evaluation (Phase 7).

### Phase 7 — Evaluation & Ablations

* **Objective** — does adaptive routing earn its complexity?
* **Hypothesis** — adaptive beats fixed strategies on the quality/cost frontier.
* **Experiment** — E1 five-arm comparison (107 queries), E2 escalation ablation,
  E3 feature ablation, E4 threshold sweep, E5 cost-weight sweep; then two
  pre-registered closure gates (oracle ceiling; observable-signal search).
* **Result** — **adaptive ≡ hybrid** on quality at ~59 ms overhead; E2 shows
  escalation buys latency not quality; **0/24** feature hits; the gate never
  opens at 0.3–0.6. Gate 2: oracle headroom +0.0218/+0.0343 but on only 4–6/107
  queries. Gate 3: OOS AUC **0.534, 0/24 Holm**.
* **Decision** — **close Phase 7** with a bounded negative answer: headroom
  exists, is not identifiable out of sample, and a router is not established.

### Phase 8 — Corpus Fix Study

* **Objective** — was the corpus defect responsible for the reranker collapse and
  for "dense dominates hybrid"?
* **Hypothesis** — a diagnosed two-column extraction defect explains both.
* **Experiment** — fix extraction, re-chunk/re-embed, re-run all five arms on a
  before **and** after corpus; plus a gold-label audit and a run-integrity
  re-sweep.
* **Result** — quality unchanged on every arm; reranker −0.0016 (99/107 ties);
  both "rescue" outcomes excluded; 47 of 140 section labels corrupt (22
  repairable); 13 of 29 arms in the first sweep were silently invalid.
* **Decision** — **withdraw the corpus attribution** (the recommendation to drop
  the rung stands, its rationale does not); exclude nDCG@5; keep the cost table
  unfrozen pending a user call (recommendation: do not freeze).

---

## Phases 9–12: the post-dense research track

### Phase 9 — Signal Discovery (verdict `PROMISING_SIGNAL`)

* **Objective** — which measurable properties explain per-query dispersion, and
  is any of them known in advance?
* **Hypothesis** — dispersion across strategies is real and partly predictable.
* **Experiment** — 12-hypothesis family (Holm denominator 12) screened against
  `T-disp3` on both arms; five gates (audit, mechanism, deployability,
  mechanism-explanation, proxy).
* **Result** — dispersion is real (21/29 of 107); **Dense distinct-document
  ratio** tracks it (ρ +0.343/+0.406, OOS +0.410) and generalizes; **no
  query-only pre-routing signal exists** (best abs(ρ) = 0.194).
* **Decision** — build **no** router under any verdict; justify testing a
  two-stage post-dense decision next (Phase 10).

### Phase 10 — Post-Dense Escalation (verdict `FAILURE`)

* **Objective** — can DDR decide whether Dense → Hybrid escalation is worth it?
* **Hypothesis** — the dispersion signal predicts escalation gain.
* **Experiment** — frozen oracle and threshold, one-pass test on both arms, with
  a random-at-equal-rate ablation.
* **Result** — AUC **0.550** (chance); test Δ **−0.050 / −0.033** vs dense-only;
  **TP 0/2** per arm; **random beats the policy** (P = 1.000 / 0.954). Escalation
  helps 4 queries and harms 9–12.
* **Decision** — **`FAILURE`**; the policy is dominated. The result reframes the
  whole track: disagreement ≠ escalation value.

### Phase 11 — Mechanism Analysis (outcome B)

* **Objective** — why does escalation help or hurt, and can the cases be told
  apart before paying?
* **Hypothesis** — helps come from BM25-held evidence Dense missed; harms from
  fusion corrupting correct Dense results.
* **Experiment** — per-query decomposition of all helps and harms into
  three states, with a bounded separation check over six pre-declared features.
* **Result** — the hypothesis is **confirmed as a mechanism** (helps = State 2
  BM25-rescue; harms = State 3, via intruder promotion or re-ranking among shared
  docs) but **not separable in advance** (1/6 and 0/6); only 5 unique positives.
* **Decision** — outcome **B**; no candidate promoted, no intervention run; the
  "enough development examples" gate failed a priori and was honoured.

### Phase 12 — Research Boundary

* **Objective** — what can AdaptiveRAG legitimately conclude, and should it keep
  going?
* **Hypothesis** — n/a (a boundary assessment, not an experiment).
* **Experiment** — audit of Gates 2/3/9 and Phases 10–11 against a
  go/no-go framework.
* **Result** — five independent failures agree (no pre-routing signal, chance
  DDR, no separation, few positives, net-negative rung).
* **Decision** — **STOP CURRENT ADAPTIVE-ROUTING TRACK** (Position C), with
  re-entry conditions written down (dozens of positives + power analysis, an
  EV-positive rung, a pre-registered rule, split discipline, a gate that can say
  "no").

---

## Phases 13–15: the cheap-first track

### Phase 13 — Next Direction (selection recorded)

* **Objective** — which alternative track, if any, is worth opening?
* **Hypothesis** — a decision point *before* the embedding call could work where
  the post-dense one did not.
* **Experiment** — read-only counts over frozen rows (no retrieval, no fitting).
* **Result** — the BM25 → Dense rung is **EV-positive** (+0.11–0.14 R@5) with
  **16/21** cheap-first positives and 91/86 queries needing no Dense call —
  4–5× the headroom of any earlier framing, at a genuinely avoidable cost.
* **Decision** — open the **cheap-first track**.

### Phase 14 — Cheap-First Experiment (verdict `INSUFFICIENT EVIDENCE`)

* **Objective** — can decision-time pre-Dense information predict when Dense is
  worth buying?
* **Hypothesis** — a pre-registered low-dimensional rule over query + BM25 state
  matches dense-only quality at far fewer calls.
* **Experiment** — six allowlisted pre-Dense signals; frozen oracle; calibration
  rule freeze; one-pass test with random ablation.
* **Result** — the rule `bm25_slope ≥ −0.915` improves over BM25-only (+0.078 /
  +0.050) and beats random on one arm (P = 0.029) but not the other; it trails
  dense-only by 0.081/0.086 and catches only 5/11 positives; the success
  criterion fails outright.
* **Decision** — `INSUFFICIENT EVIDENCE`; the open question (power vs noise) is
  named, and the sequel is a powered test, not a re-sweep.

### Phase 15 — Powered Confirmation (verdict `INSUFFICIENT EVIDENCE`)

* **Objective** — does the Phase 14 selection claim replicate with far more
  positives?
* **Hypothesis** — the frozen rule selects Dense calls better than random and
  approaches dense-only quality.
* **Experiment** — the rule applied **unchanged** to **110 fresh queries per
  arm** (duplicate-screened, all test split), with bootstrap CIs, Holm within
  each comparison pair, and random-at-rate nulls.
* **Result** — **18/25** positives caught at ~30% spend; **beats random**
  (P = 0.008 / 0.019, combined **0.001**); +0.058/+0.045 over BM25 with CIs
  excluding zero; but **−0.030 / −0.018 vs dense-only**, missing the 0.02 bar by
  **0.004**. Verdict: 4 of 5 success bars, no failure bar → `INSUFFICIENT
  EVIDENCE`.
* **Decision** — **stop**. Two consecutive inconclusive phases with the same
  signature; the remaining question is a margin judgment, not a measurement gap;
  further work on this benchmark would need a genuinely new information source.

---

## After Phase 15

Experimentation is **complete**. The project was consolidated for
documentation, presentation and future research. Nothing after Phase 15 changed
any measured result; the only code change was a stale path in
`scripts/report_chunk_stats.py` (it read a pre-Phase-8 stats location that no
longer holds current data) and the addition of this documentation set.

*Pre-execution planning notes for Phases 4, 5, 8 and the oracle-routing study are
archived in [`plans/`](plans/).*