# Conclusions

What the investigation concluded, derived from the evidence in
[`findings.md`](findings.md). Full narrative:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md) §7.

---

## 1. The central conclusion

**Adaptive retrieval is plausible, but identifying when additional retrieval is
valuable is substantially harder than detecting retrieval disagreement.**

The project measured both halves of that claim rather than assuming the
relationship between them:

* **Detection worked.** Retrieval disagreement across strategies is real,
  measurable, mechanistically explained (evidence diffuseness), and replicated
  across two corpus arms and an out-of-sample split (Phase 9).
* **Decision-making failed.** Every attempt to convert that detection into a
  routing decision failed: no query-only signal exists (best abs(ρ) = 0.194);
  the one surviving single-arm signal was chance-level as a router (AUC 0.550);
  and the resulting policy was beaten by *random* escalation at the same spend
  (Phases 7, 9, 10).

The mechanism analysis explains the gap rather than papering over it: on this
corpus, disagreement is common while escalation value is rare, and the two
point at different queries. **Disagreement ≠ insufficiency ≠ escalation
value** is the project's most transferable conceptual result.

## 2. The one durable positive result

**Cheap first-stage information contains real, replicated signal about whether
expensive retrieval will help.**

The strongest evidence in the project is the Phase 15 confirmation, because it
was designed to be hard to pass: a rule frozen on one split, applied unchanged
to 110 fresh queries per arm, with no calibration data available and a
random-spending ablation. It caught **18 of 25** oracle positives at ~30% Dense
spend, avoided ~70% of Dense calls, beat random spending on both arms
individually and combined (**P = 0.001**), and improved over BM25-only with a
bootstrap CI excluding zero. The signal is also *mechanistically sensible*:
flat BM25 score decay indicates a weak, low-margin lexical ranking, which is
when semantic retrieval has something to add.

## 3. Why that is still not a solution

**The quality gap to full Dense retrieval remained too large to claim a
replacement policy.** Adaptive trailed Dense by 0.018–0.030 Recall@5 and missed
the pre-registered margin bar by 0.004 on the combined mean.

The frozen verdict mapping has three outcomes, and this landed in the middle one:
not SUCCESS (all five bars), not FAILURE (no failure bar). `INSUFFICIENT
EVIDENCE` is the honest reading — it is explicitly **not** a near-pass to be
rounded up, and it is not a licence to keep searching on the same data.

## 4. Why experimentation stopped

Stopping was a decision, not an absence of one. The evidence for it:

1. **Two consecutive inconclusive phases** (14, 15) with the *same signature*:
   genuine selection signal, Dense-margin shortfall. Phase 15 removed the power
   objection, so the remaining uncertainty is the claim, not the measurement.
2. **A net-negative escalation rung** on the post-dense track — the target was
   the problem before the selector ever was.
3. **A benchmark that cannot support a supervised claim**: 4–8 positives for the
   original framing, 5 for calibration in Phase 14. Continuing feature
   engineering here is an optimization loop, not an experiment.
4. **Project rule**: a rigorous stopping point is preferred over another weak
   experiment.

What would justify re-opening the question is written down in
`docs/phase12_results.md` §9 — a benchmark with dozens of positives and an
upfront power analysis, a rung that beats Dense somewhere substantial, a
pre-registered rule, the frozen split discipline, and a gate that can say "no"
again.

## 5. What the project claims, and does not

**Claimed:** that cheap pre-cost information selects Dense calls better than
random; that retrieval disagreement is real and mechanically tied to evidence
diffuseness; that the shipped router does not earn its complexity on this
benchmark; and that the escalation rung is net-negative here.

**Not claimed:** an optimal adaptive algorithm, a production-ready policy,
universal generalization, causal explanations for the observed correlations, or
that adaptive retrieval outperforms Dense retrieval. See
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md) §9.

## 6. Final status

```text
Status:
Research investigation completed.
Adaptive-routing experimentation paused.
No final optimal routing policy claimed.
Project consolidated for documentation, presentation, and future research.
```

---

*Conclusions. Evidence: [`findings.md`](findings.md). Boundaries:
[`limitations.md`](limitations.md).*