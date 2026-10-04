# Should `strategy_cost_ms` be frozen?

Status: **recommendation, not an executed change.** The freeze is a user call
(`AGENTS.md` — architectural decisions are recorded, not assumed), so nothing has
been frozen and `RoutingConfig.strategy_cost_ms` is untouched. This document
lays out the evidence and a recommendation so the call can be made on numbers.

Recommendation: **do not freeze, and do not re-measure yet.** Both arms are
measured; neither is a good freeze candidate, and the reason is narrower than
"the numbers disagree".

Artifacts: `experiments/phase8/strategy_cost_ms_phase8_{before,after}.json`.
Router consumption: `src/adaptive_rag/routing/rule_based.py:148-157`.

## The question

Phase 8 measured per-strategy retrieval latency on both corpus arms and left the
freeze open, because the two arms propose different tables and Phase 8's own
rule (§2.1 of `docs/phase-8-results.md`) is one frozen value **per corpus
version** — and both arms share `corpus_c5996129918d7e44`. That looked like an
ambiguity to resolve. It is not the real issue.

## What the two arms disagree about

| strategy | before (ms) | after (ms) | delta | vs after |
|---|---|---|---|---|
| bm25 | 2.48 | 2.12 | −0.36 | 17.0% |
| dense | 374.04 | 498.74 | **+124.70** | **25.0%** |
| hybrid | 389.34 | 491.55 | +102.21 | 20.8% |
| hybrid_rerank | 3471.81 | 3545.63 | +73.82 | 2.1% |

Read naively, `dense` moved 125 ms and the freeze is impossible. But the corpus
fix cannot move `dense` by 125 ms: §6 of the results doc already withdrew two
latency findings for exactly this reason — `dense` and `hybrid_rerank` sit
inside the **query-embedding API call**, which embeds the query rather than the
corpus, and the `bm25` control arm (no API call) moved 2.48 → 2.12 ms. The
125 ms is provider variance on a stage the corpus does not touch.

So the two tables do not disagree about the *system*; they disagree about the
*provider on the day*.

## What the router actually consumes

`rule_based.py:148-157` normalises every cost by the maximum in the table before
subtracting `cost_weight × normalised_cost`. Absolute milliseconds are therefore
**not the decision-relevant quantity** — the ratios are.

| ratio | before | after | current config | spread |
|---|---|---|---|---|
| dense / hybrid | 0.9607 | 1.0146 | 0.9908 | **5.3%** |
| hybrid / hybrid_rerank | 0.1121 | 0.1386 | 0.1183 | 19.1% |
| bm25 / dense | 0.0066 | 0.0043 | 0.0050 | 35.9% |

The 25% absolute swing on `dense` becomes a **5.3% swing on `dense/hybrid`**,
because both arms moved together — they share the same query-embedding call.
Converted into the score the router computes, at the shipped `cost_weight=0.25`:

| | bm25 | dense | hybrid | hybrid_rerank |
|---|---|---|---|---|
| penalty, before | 0.0002 | 0.0269 | 0.0280 | 0.25 |
| penalty, after | 0.0001 | 0.0352 | 0.0347 | 0.25 |

The largest penalty disagreement on any arm that is not the reranker is
**0.008**, against rule weights that run 0.2–1.5. It is two orders of magnitude
below anything that could flip a decision.

## Recommendation

**Do not freeze.**

1. **The disagreement is provider noise, not a system property.** Freezing would
   launder a day's provider latency into a permanent routing constant, and the
   `bm25` control arm proves the corpus is not the variable.
2. **Freezing the wrong quantity.** `strategy_cost_ms` stores absolute
   milliseconds; the router consumes their ratios. A freeze pins the unstable
   representation and leaves the stable one unpinned. If anything is ever
   frozen, it should be the normalised table, with the absolutes kept as
   provenance.
3. **It is inert today and would matter at exactly the wrong moment.** E5 shows
   all five `cost_weight` values give identical recall@5 (0.8830) because the
   sufficiency gate never opens, so a 0.008 penalty shift cannot change a
   decision in either direction. It would start to matter the moment ADR-028
   Finding 2 is acted on — which is also the moment the table should be
   re-measured, per the open item.

## To act on this

If you agree, the smallest change is to record the decision in `docs/decision.md`
as an ADR amendment and close open item 1 in `docs/phase-8-results.md` §9. No
code change is needed — and notably, *no* change to `strategy_cost_ms` should be
made, since the recommendation is to leave it alone.

## What would change this

A freeze becomes defensible when the measurement stops being dominated by the
provider:

* measure the **local** stages only (`bm25`, the ONNX reranker) and treat the
  embedding call as a separately-tracked constant; or
* run the **no-API control arm** §6 already prescribes, so provider variance can
  be subtracted rather than argued about; or
* freeze the **normalised** table, documenting that absolute milliseconds are
  hardware- and load-dependent and not portable (the artifact's own `notes`).

If a freeze is wanted regardless, the after arm is the only defensible
candidate — it is the shipping corpus — but it should be frozen as the ratio
table with the absolute values retained as provenance, and re-measured on the
107-query benchmark rather than the legacy 20-record `dense_eval_v1.jsonl` the
sweep inherited from Phase 7.0b.
