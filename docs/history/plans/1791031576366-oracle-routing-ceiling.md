# Decisive routing study: does query-characteristic routing have headroom?

## Goal

Settle, with the cheapest decisive evidence first, whether routing by query
characteristic can beat the **best fixed strategy** on the frozen 107-query
benchmark — on quality, on latency, or both. Phase 7/8 showed `adaptive ≡
hybrid`, but that is ambiguous: it could mean routing is worthless on this
corpus, **or** that the router never actually routes (initial pick = hybrid on
104/107; sufficiency gate never fires at 0.5; escalation-from-hybrid is a
no-op because hybrid is terminal once the rerank rung is removed). This study
separates those two explanations before any API-bound re-run is spent.

## Why the oracle comes first

The literal ADR-028 re-run (remove `hybrid_rerank`, raise `sufficiency_threshold`
to ~0.7) is expected to be near-null, for two structural reasons:

1. Opening the gate only escalates the ~3 bm25-pick queries (and only if their
   sufficiency < 0.7). Escalation from the 104 hybrid-pick queries is a
   **recorded no-op** (`adaptive.py:209-215`: `next_strategy("hybrid")` is
   `None` once the ladder terminates on hybrid).
2. The ladder orders `bm25 < dense < hybrid` (`schemas/config.py:380`), so
   escalation can never reach `dense` — the strategy that actually has the best
   recall@5 (0.8988 vs hybrid's 0.8583). Even a "working" escalation climbs
   away from the best strategy.

So before spending an API-bound re-run on the escalation mechanism, measure the
**ceiling**: what is the best any query-characteristic router built from these 5
strategies could do? That is fully offline — the 5 clean E1 arms already contain
every strategy's per-query result.

## Stage 1 — the decisive test (offline, orchestration agent implements + runs)

New script `scripts/oracle_routing_ceiling.py`. No network, no index, no API.

**Inputs** (per corpus arm, after = primary/shipping, before = replication):
`experiments/phase8/combined/{p8a_e1,p8b_e1}/p8*_e1__E1_baseline_comparison__{bm25,dense,hybrid,hybrid_rerank,adaptive}/rows.jsonl`
— each row already carries `query_id`, `recall_at_5`, `mrr`, `hit_at_5`,
`total_latency_ms`, `initial_strategy`, `final_strategy`, `sufficiency_score`,
`category`, `split`. Join the 5 arms on `query_id` (assert the 5 query_id sets
are identical; abort if not).

**Computation** (per query q, per strategy s: `recall(q,s)`, `mrr(q,s)`,
`lat(q,s)`):

1. **Oracle quality ceiling** — per-query `argmax_s recall(q,s)`, tie-break
   lowest `lat`. Aggregate mean recall@5 / MRR / Hit@5 / latency.
2. **Best fixed** — the single strategy with the highest mean recall@5 (expect
   `dense`). Report its mean recall@5 / MRR / Hit@5 / latency. This is the bar
   routing must clear, **not** hybrid.
3. **Quality/latency frontier** — for ε ∈ {0.005, 0.01, 0.02}: per-query, among
   `{s : recall(q,s) ≥ best_recall(q) − ε}` pick lowest `lat`. Aggregate mean
   recall@5 and mean latency. This bounds the latency a router could save at
   ≤ε quality loss — the realistic routing prize (default to a cheap strategy
   when it suffices, pay for `dense` only when it does not).
4. **Detectability** — for the queries where cheap strategies are NOT within ε
   of `dense` (i.e. `dense` is strictly needed), is the hybrid
   `sufficiency_score` / coverage lower than on queries where hybrid suffices?
   Cross-tabulate `argmax` strategy by `category` and by the router's signal
   groups (from `traces.jsonl` `routing.metadata.signals`). This tests whether
   the existing sufficiency mechanism could actually detect "escalate to dense".

**Outputs**: `experiments/phase8/oracle_ceiling.json` (all tables, per-corpus)
and a short report section appended to `docs/phase-8-results.md` (§9).

**Pre-registered decision gate** (recorded in the artifact before running):

- **Routing has headroom** if EITHER:
  - oracle recall@5 − best-fixed recall@5 > **+0.02** (per-query selection beats
    the best single strategy on quality), OR
  - frontier(ε=0.01) mean latency − best-fixed mean latency < **−100 ms**
    (routing to cheap strategies saves meaningful latency at ≤1pt quality loss)
    **AND** the needed-vs-sufficient split is predictable from
    `category`/signals (step 4 shows a real association).
- **Routing is exhausted on this benchmark** if oracle − best-fixed ≤ **+0.01**
  AND frontier(0.01) saves < **50 ms** — then the honest recommendation is to
  ship the best fixed strategy and stop routing.

## Stage 2 — confirmatory ADR-028 re-run (API-bound, USER-RUN, gated on Stage 1)

Only if Stage 1 shows headroom **and** the headroom is in the escalation
direction. This is the literal Phase 7 recommendation ("then re-run E1 and E8"):
remove `hybrid_rerank` from `available_strategies` + `escalation_ladder`, raise
`sufficiency_threshold` to 0.7, re-run the `adaptive` arm on both corpora,
paired-test against fixed `hybrid`. **Expected near-null** (reasons 1–2 above);
its value is to measure the escalation mechanism directly rather than infer it.
Run at `--pace 1.5` (the rate-limit fix from the Phase 8 contamination incident).

Commands (from repo root; user has egress):
```
.venv/bin/python scripts/run_phase7_suite.py --dataset data/evaluation/phase7_eval_v1.jsonl \
  --experiments E1_baseline_comparison --arms adaptive \
  --split all --retrieval-only --pace 1.5 --strict \
  --run-prefix p8c_gate --corpus-arm phase8_after
```
(with a `RoutingConfig` override: `available_strategies=[bm25,dense,hybrid]`,
`escalation_ladder=[bm25,dense,hybrid]`, `sufficiency_threshold=0.7`). Note the
config override mechanism must be checked in `build_experiment_config` first —
if it cannot express "remove a rung", that is a prerequisite code change the
orchestration agent should flag, not silently make.

## Out of scope (separate plans, gated on Stage 1)

- **Ladder reordering** (`hybrid` above `dense` contradicts measured quality).
  A Phase 6 router change; only worth doing if Stage 1 shows escalation toward
  `dense` would pay.
- **Rule-weight / `QUESTION_TYPE_SUPPORT` fitting.** Risks overfitting the
  107-query benchmark; needs a held-out split (ADR-027) before any change.
- **Learned router.** Phase 7 explicitly deferred it; only relevant if Stage 1
  shows headroom that the current features cannot capture.

## Constraints

- Stage 1 is offline and deterministic; the orchestration agent can implement
  and run it with no credentials.
- Stage 2 needs network egress — the user runs it (egress is split off in this
  environment).
- No router, ladder, or threshold change in Stage 1; it measures, it does not
  retune.
- `nDCG@5` stays excluded (gold section labels corrupt — Phase 8 Step 4); the
  oracle uses `recall_at_5` / `mrr` / `hit_at_5` only.

## Validation

- Unit test for the oracle join + argmax + frontier logic (deterministic, no
  API), alongside the existing `tests/test_phase8_arm_comparison.py`.
- Assert the 5 arms' `query_id` sets match exactly; abort otherwise (the
  Phase 8 contamination incident was exactly a query-set mismatch).
- Re-run on the `before` corpus as a replication; report both.
- Cross-check: oracle recall@5 must be ≥ best-fixed recall@5 by construction;
  if not, the join is wrong.

## Open questions

- Does `build_experiment_config` support expressing a reduced strategy set for
  Stage 2, or is that a prerequisite change? (Orchestration agent to check.)
- Is `query_id` in `rows.jsonl` stable and identical across all 5 arms on both
  corpora? (Asserted in Stage 1; if it fails, the oracle is blocked the same way
  nDCG is.)
