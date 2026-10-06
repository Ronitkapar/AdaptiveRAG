# Research Question

The central question, its secondary forms, and the boundary of each. Every
question here was actually investigated; none is aspirational. Full answers are
in [`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md) §2 and §5.

## Primary question

> **When does a query actually need expensive retrieval, and can retrieval effort
> be adapted automatically without sacrificing too much retrieval quality?**

This splits into two operational halves, which the project kept separate
throughout:

* **Detection** — when do retrieval strategies disagree about a query?
* **Decision** — when is additional retrieval actually *worth paying for*?

The investigation's central empirical result is that these two halves come apart:
detection was measurable and replicable; the decision was not, until a
cheap-first decision point was introduced in Phase 13.

## The two decision points investigated

| Track | Decision point | Phases | Outcome |
| --- | --- | --- | --- |
| Post-dense | *after* paying for Dense: should we escalate to Hybrid? | 6–12 | stopped (`FAILURE`, then mechanism analysis, then explicit STOP) |
| Cheap-first | *before* paying for Dense: after cheap BM25, is Dense worth buying? | 13–15 | `INSUFFICIENT EVIDENCE` on a powered confirmation |

Moving the decision point earlier in the cost sequence — from "after the
expensive call" to "before it" — is the single design change that produced the
project's only replicated selection signal.

## Secondary questions actually investigated

1. **Does per-query retrieval dispersion across strategies exist, and what
   explains it mechanistically?** (Phase 9) — Yes; explained by evidence
   diffuseness, measured by Dense distinct-document ratio.
2. **Do query-intrinsic or first-stage-feedback signals predict which strategy
   will help, out of sample?** (Phases 7, 9, 14) — No for query-only signals
   (best abs(ρ) = 0.194); partially, at chance-to-moderate strength, for
   BM25-side signals on a small positive class.
3. **Why does escalation help on some queries and hurt on others, and can the
   two be told apart before paying?** (Phase 11) — The mechanisms are distinct
   and documented per query; they could not be separated ahead of time (1/6 and
   0/6 on pre-declared features).
4. **Is the escalation target itself net-positive on this corpus?** (Phases
   10–12) — No. RRF-hybrid helps 4 queries and harms 9–12, so always-escalate
   is net-negative by construction.
5. **Can cheap first-stage information select Dense calls better than random
   spending at the same rate?** (Phases 14–15) — Yes: 18 of 25 positives caught
   at ~30% spend, P = 0.001 versus random on 220 fresh query-instances.

## Questions deliberately not asked

* *Which router architecture is best?* — deferred behind a `Router` interface
  (ADR-026); no learned router was built or trained.
* *What is the optimal policy over all strategies?* — only an oracle ceiling was
  computed, which bounds the achievable gain and cannot be implemented.
* *Does adaptive retrieval improve generated answer quality?* — no routing claim
  was validated end-to-end through generation; all findings are retrieval-only.

## The bar a policy had to clear

Stated before the experiments and never moved afterwards:

> On frozen test, in one pass, an adaptive policy must match full Dense
> retrieval within a pre-registered Recall@5 margin while spending materially
> fewer expensive calls — with true positives > 0, a random-at-equal-rate
> ablation that does not win, and the result replicated in direction on the
> second corpus arm.

The final policy cleared the selection bar and missed the quality bar by 0.004.
That is recorded as `INSUFFICIENT EVIDENCE`, not rounded up to success.

---

*Questions, decision points and the passing bar. Method detail:
[`methodology.md`](methodology.md). Full narrative:
[`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md).*