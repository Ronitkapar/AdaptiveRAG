# Findings

What the project discovered, separated by how well the evidence supports each
claim. This separation is the point of the document: conflating "measured and
replicated" with "plausible but unproven" is exactly how a negative result gets
re-read as a success. Narrative version:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md) §6.

---

## 1. Confirmed findings

Repeatedly measured, replicated across both corpus arms, by a methodology able
to detect the effect if it were there.

### C1. Per-query retrieval dispersion across strategies is real

`T-disp3` (max − min Recall@5 over bm25/dense/hybrid) exceeds zero on 21/107
queries (after) and 29/107 (before). Not a tie-break artefact — the tie-break
control fires on 73/66 queries, i.e. far more often.

### C2. Dense distinct-document ratio tracks that dispersion, and it generalizes

| measure | value | reading |
| --- | --- | --- |
| `DDR@dense` vs `T-disp3`, calibration | ρ +0.343 / +0.406, Holm p = 0.0012 | replicated association |
| `DDR@dense` vs `T-disp3`, held-out test | ρ +0.410, no fitting | generalizes out of sample |
| pairwise Jaccard vs `T-disp3` | ρ −0.454 / −0.520 | replicated, same direction |

Interpretation: when retrieval returns few distinct documents, the strategies
disagree more. Disagreement is *measurable, real, and mechanistic* — the
evidence-diffuseness structure the project set out to find.

### C3. The reranker is a net-negative rung on this corpus

Worst arm on every quality metric (Recall@5 0.7165 pre-fix / 0.7056 shipping) at
6.2–7.5× hybrid's median latency, and the corpus fix did not rescue it (−0.0016).
Removed from escalation consideration (ADR-028).

### C4. Dense is the best fixed strategy in every condition measured

R@5 0.9283 (Phase 7 pre-fix) / 0.8988 (shipping). Adaptive routing collapsed
onto the *second*-best (hybrid) at the shipped configuration, paying ~59 ms for
the equivalence.

### C5. Disagreement does not imply escalation value

The DDR policy (Phase 10) failed at chance AUC (0.550), caught 0 of 2 positives
per arm, and was beaten by random escalation at the same spend
(P(random ≥ adaptive) = 1.000 / 0.954). The C2 association did **not** survive
intervention. The falsification is the point:

> **retrieval disagreement ≠ retrieval insufficiency ≠ escalation value**

### C6. Helpful and harmful escalations have different mechanisms (Phase 11)

* **Helps** (5 unique queries): BM25 holds or highly ranks evidence Dense missed
  (Dense Recall@5 0–0.33), and RRF keeps it top-5 — BM25-side rescue.
* **Harms** (9–12 per arm): Dense was *already correct* and fusion broke it
  (all harms are "State 3"), via intruder promotion or re-ranking among already
  shared documents with **no new document entering**. Even an agreeing BM25 can
  preside over harm.

### C7. Cheap first-stage information can select Dense calls better than random

On 220 fresh query-instances, the Phase 14 rule applied unchanged: **18 of 25**
oracle positives caught at ~30% Dense spend (~70% of calls avoided), beating
random spending on both arms (P = 0.008 / 0.019) and combined (**P = 0.001**),
with the combined cluster-bootstrap CI on adaptive − BM25 excluding zero
(+0.0515 [0.0110, 0.0977]). The project's strongest positive result.

### C8. Measurement-integrity lesson (Phase 8)

Trace status — not trace count, and not a metric's own `n` — is what establishes
that a measurement covers its query set. 13 of 29 arms were silently invalid
while reporting `ok`; the conclusion drawn from them would have been right by
accident.

---

## 2. Promising findings

Real signal, not yet sufficient for a policy.

### P1. Flat BM25 score decay predicts Dense's marginal value

`bm25_slope` won the frozen Phase 14 family selection (calibration AUC 0.757,
beating `DDR@bm25` at 0.586). It is BM25-side and pre-cost, so it is usable at
decision time — the strongest pre-cost candidate found. **But**: selection rested
on 5 positives, and its test record was TP 5 / FN 6 on *both* arms.

### P2. The cheap-first decision point is where the headroom is

16/21 cheap-first positives (4–5× any earlier framing), 91/86 queries needing no
Dense call, and an EV-positive rung (+0.11–0.14 R@5). Feasibility is
established; the policy is not.

### P3. Escalation is cheap when it happens

Incremental cost of the one extra local stage is ~31 ms, against ~450 ms for the
provider call a cheap-first policy avoids. If a net-positive rung and a
well-calibrated selector were found, the arithmetic favours escalation.

---

## 3. Negative findings

Tested and rejected; preserved as part of the record. Detail and reasoning in
[`negative_results.md`](negative_results.md).

| # | Finding | Evidence |
| --- | --- | --- |
| N1 | Query-aware rule routing does not beat a fixed strategy here | gate never fires; 0/24 feature-ablation hits (Phases 6–7) |
| N2 | Reranking does not pay for itself | all head metrics fell; depth ablation R@5 ↓ as latency ↑ (Phase 5) |
| N3 | The corpus defect did **not** cause the reranker collapse | −0.0016 Recall@5, 99/107 ties (Phase 8) |
| N4 | Post-dense escalation on DDR fails | AUC 0.55, TP 0/2, dominated (Phase 10) |
| N5 | Gate 3 observable-signal search is exhausted | OOS AUC 0.534, 0/24 Holm (Phase 8) |
| N6 | The RRF-hybrid rung harms more than it helps | 4 helps vs 9–12 harms → always-escalate net-negative (Phases 10–11) |
| N7 | The Phase 14 rule did not reproduce on the Phase 15 positive rate | a 5-positive selection is noise-exposed, as the 1-SE guard predicted |

---

## 4. Unresolved questions

Open; not answered here.

* **Would a powered cheap-first study clear the Dense margin?** Power is no
  longer the objection — whether −0.018 to −0.030 Recall@5 is worth ~70% fewer
  embedding calls is a *value judgment* this project deliberately left open.
* **Can a learned router exploit the State-2/State-3 interaction?** The only
  mechanism-consistent hypothesis, and it is not established (n = 5; separation
  1/6 and 0/6). No learned router was built (ADR-026 defers it).
* **Does any of this generalize beyond this corpus?** One 14-paper academic-RAG
  collection only.
* **Is a net-positive second-stage action available at all?** The rung, not the
  router, may be the binding constraint for the post-dense track.
* **Can nDCG@5 / Precision@k be trusted?** Blocked upstream by 42 spliced section
  paths; label repair alone cannot fix the metric.

---

*Findings by evidential strength. Conclusions:
[`conclusions.md`](conclusions.md). What was rejected:
[`negative_results.md`](negative_results.md).*