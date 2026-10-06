# Negative Results

The approaches that were tested and rejected, with the evidence that rejected
them and what was learned. These are part of the project's research record, not
failures to be hidden: an adaptive-retrieval investigation that reported only
its positive signal would misrepresent what was measured.

---

## N1. Query-aware rule-based routing does not beat a fixed strategy (Phases 6–7)

**What was tried.** A rule-based router over deterministic query features, a
label-free sufficiency check on retrieved evidence, and a bounded escalation
ladder, evaluated against fixed BM25/Dense/Hybrid arms on 107 queries.

**Evidence.** The sufficiency gate **never opens** at the shipped defaults
(threshold 0.5, cost_weight 0.25, max_steps 1), so the router picks Hybrid for
essentially the whole benchmark. Adaptive ≡ hybrid on Recall@5 and Hit@5
(107/107) at ~59 ms median overhead. Feature-group ablation: **0 of 24**
comparisons significant. Threshold sweep 0.3–0.6 identical with 0% escalation.

**What was learned.** A decision layer that never decides is pure overhead. The
problem was not the routing logic but that the *decision variable* (evidence
sufficiency) did not separate queries on this benchmark. Every later phase
treated that as the open question rather than tuning the router harder.

**Source.** `docs/phase_7_results.md`, `docs/phases/phase-7.md`.

---

## N2. Cross-encoder reranking does not pay for itself (Phase 5)

**What was tried.** An ONNX cross-encoder second stage over Dense, BM25 and
Hybrid first-stages, plus a candidate-depth ablation (k = 10/20/40).

**Evidence.** Every head metric **fell** on all three strategies (MRR −0.083 /
−0.061 / −0.127). R@5 degrades 0.7917 → 0.7583 → 0.7333 as candidate depth
rises, while rerank latency climbs 3.00 s → 3.49 s → 7.67 s. Measured cost is
~155 ms per candidate pair — two orders of magnitude above the pre-registered
estimate.

**What was learned.** A general-web MS-MARCO cross-encoder is a poor fit for dense
academic RAG prose. The composition-layer design was sound; the *model choice*
was not. Phase 8 later confirmed the corpus was not the cause (N3), leaving the
diagnosis open.

**Source.** `docs/phases/phase-5.md`, `docs/phase-8-results.md` §3.

---

## N3. The corpus extraction defect was not the reranker's cause (Phase 8)

**What was tried.** Fix two-column PDF extraction (which interleaved columns of
multi-column papers), re-chunk, re-embed, re-run all five arms on both a
`before` and an `after` corpus.

**Evidence.** Reranker Recall@5 moved **−0.0016** on the fix, with 4 wins,
4 losses and **99 ties** out of 107. No quality delta on any arm survives Holm
correction. Both outcomes that would have *rescued* the reranker (it improves
materially, or improves but stays last) are excluded by measurement.

**What was learned.** The original attribution — the reranker was suppressed by
garbled text — was **withdrawn**. The cause is unidentified; "replace the model"
remains a hypothesis rather than a diagnosis. ADR-028's *recommendation*
(drop the rung) stands; its *rationale* does not.

**Source.** `docs/phase-8-results.md`, `docs/decision.md` (ADR-028 Phase 8
revision).

---

## N4. Post-dense escalation on `distinct_doc_ratio@dense` fails (Phase 10)

**What was tried.** The Phase 9 survivor signal, used in a real intervention:
escalate Dense → Hybrid when DDR indicates dispersion. Frozen oracle, frozen
threshold (DDR ≥ 0.8, rate 0.191, non-degenerate), one-pass test on both arms.

**Evidence.** Calibration AUC **0.550** (chance) on the shipping arm; the two
positives sit at DDR 0.2 and 0.8, split across the range. Test Recall@5 **−0.050
/ −0.033** vs dense-only. **TP = 0 of 2** per arm. Random escalation at the same
rate beats the policy (P = 1.000 / 0.954). Incremental cost +4–6 ms.

**What was learned.** The decisive conceptual result of the project. DDR marks
*disagreement*; on this benchmark dispersion is dominated by Dense succeeding
where fusion degrades the ranking, so the signal fires on queries where
escalating *hurts*. Correlation with dispersion does not predict gain:

> **retrieval disagreement ≠ retrieval insufficiency ≠ escalation value**

**Source.** `docs/phase10_results.md`.

---

## N5. Gate 3's observable-signal search is exhausted for this setting (Phase 8)

**What was tried.** Everything the router can see about the query (37 columns:
13 query features, 5 retrieval-feedback, 6 routing-signal, …) and about the first
stage's output, scored by a ridge logistic regression used purely as a
measuring instrument.

**Evidence.** AUC **1.000 on the fit split** and **0.534 out of sample**, with
**0 of 24** features surviving Holm correction on either corpus arm. The
fit/OOS collapse is the signature of overfitting to 8 positives.

**What was learned.** With this positive class size, no signal of this family is
detectable; the 1.0 fit AUC is an artefact of memorization, not evidence. This
closed the "just add features" avenue with evidence rather than assertion.

**Source.** `docs/phase_7_closure_results.md` §3.

---

## N6. The RRF-hybrid rung harms more queries than it helps (Phases 10–11)

**What was measured.** The escalation target itself: on how many of the 107
queries would Dense → Hybrid actually help?

**Evidence.** Helps on **4** queries per arm; harms on **9 (after) / 12
(before)**. Always-escalate is therefore **net-negative by construction** — a
policy must *select* 4 queries out of 107, not merely spend. All harms are
"State 3" (Dense already correct, fusion corrupted it); all helps are State 2
(Dense genuinely missed the evidence BM25 held).

**What was learned.** The binding constraint for the post-dense track was never
the router — it was that the rung itself has negative expected value on this
corpus. No amount of selection quality repairs a rung that hurts 2–3× more
queries than it helps.

**Source.** `docs/phase10_results.md` §3, `docs/phase11_results.md`.

---

## N7. The Phase 14 rule did not reproduce its catch rate on Phase 15 (Phases 14→15)

**What was observed.** The frozen rule caught 5/11 positives per arm in Phase
14 (21.7% spend) and 9/14, 9/11 in Phase 15 (≈30% spend) — a *lower* catch rate
on a *larger* positive set, with precision of only 29% / 26%.

**What was learned.** The Phase 14 selection was made on **5 calibration
positives**, where 35 rules sat within 1 SE of the family maximum — a flat
landscape in which selection is noise-exposed by construction. The 1-SE guard
had already predicted this; the powered run confirmed it. The rule remains a
*real but lossy selector*: it beats random spending decisively while missing
roughly half the positives.

**Source.** `docs/phase14_results.md` §7, `docs/phase15_results.md`.

---

## Not tested (distinct from rejected)

Recorded so the boundary of the evidence is explicit:

* **Learned routers** — deferred behind the `Router` interface (ADR-026); never
  built, so not refuted.
* **A second corpus / domain** — never run, so generalization is *untested*,
  not disproven.
* **Generation-grounded answerability as a routing signal** — the judge exists
  in the codebase and was never used for routing; untested.
* **Fine-tuned or domain-matched rerankers** — out of scope by design (torch is
  banned; ONNX only), so N2's diagnosis remains open.

---

*What was rejected and why. Surviving findings:
[`findings.md`](findings.md). Full narrative:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md).*