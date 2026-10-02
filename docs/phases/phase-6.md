# Phase 6 — Adaptive Routing

**Status:** IMPLEMENTATION COMPLETE / OFFLINE VALIDATED — live multi-strategy arms pending user-side validation
**Phase:** 6
**Purpose:** Decide *which* existing retrieval strategy a query needs, and whether
the retrieved evidence justifies spending more, without changing any retrieval
algorithm from Phases 2–5.

---

## 1. Objective

Implement an adaptive decision layer that:

1. Analyzes a query with a cheap, deterministic analyzer (no LLM, no embedding).
2. Selects an initial retrieval strategy from evidence.
3. Executes retrieval using the **existing** Phase 2–5 retrievers.
4. Judges, from the retrieved evidence itself, whether that retrieval was enough.
5. If not, escalates **once** to a strictly stronger strategy.
6. Emits a complete, inspectable trace of all of it.

The governing principle:

> Retrieve as cheaply as possible when the query allows it, but spend additional
> retrieval computation when the evidence indicates the initial strategy was
> insufficient.

---

## 2. Scope

### Included

* `routing/base.py` — `QueryAnalyzer` and `Router` protocols (the swappable seam)
* `routing/analyzer.py` — `QueryFeatureAnalyzer`, deterministic and interpretable
* `routing/rule_based.py` — `RuleBasedRouter`, weighted transparent scorer
* `routing/sufficiency.py` — `SufficiencyChecker`, label-free runtime evidence
* `routing/escalation.py` — `EscalationPolicy`, bounded total-order ladder
* `retrieval/adaptive.py` — `AdaptiveRetriever`, the orchestrator
* `schemas/routing.py` — `QueryFeatures`, `RoutingDecision`, `SufficiencyDecision`,
  `EscalationDecision`, `RoutingTrace`, `StrategyEvidence`
* `RoutingConfig` — weights, cost table, thresholds, ladder, ablation switches
* `RetrievalMetadata` adaptive diagnostics and `ExperimentTrace.routing`
* `evaluation/routing.py` — `RoutingEvaluator` (routing metrics only)
* `run_experiment.py --retriever adaptive` plus routing flags
* 83 offline deterministic tests, including 14 architecture guards

### Excluded (Phase 7)

* Adaptive-vs-fixed comparison harness and the ablation framework
* Learned router training and routing-dataset generation
* Failure-mode taxonomy across runs
* Threshold calibration against benchmark labels

### Not touched

`retrieval/dense.py`, `retrieval/bm25.py`, `retrieval/hybrid.py`,
`retrieval/fusion.py`, `retrieval/reranked.py`, `indexing/*`, `embeddings/*`,
`reranking/*`, `evaluation/retrieval.py`, `evaluation/generation.py`, `rag.py`,
`scripts/compare_retrievers.py`, `scripts/compare_reranking.py`. Phase 2–5
retrieval behaviour is unchanged and guard-enforced.

---

## 3. Architecture

```text
Query
  │
  ▼
QueryFeatureAnalyzer          cheap, deterministic, no model
  │  QueryFeatures
  ▼
RuleBasedRouter               weighted evidence → RoutingDecision
  │  strategy + confidence + evidence
  ▼
Initial retrieval             existing Phase 2–5 Retriever
  │  RetrievalResponse
  ▼
SufficiencyChecker            label-free runtime evidence
  │
  ├── sufficient ───────────────────────────► Final results
  │
  └── insufficient
        │
        ▼
    EscalationPolicy            one bounded step, strictly stronger
        │
        ▼
    Stronger strategy
        │
        ▼
    Final results
        │
        ▼
    RoutingTrace → RetrievalMetadata → ExperimentTrace → RoutingEvaluator
```

### Component boundaries

| Component | Decides | Never does |
|---|---|---|
| `QueryFeatureAnalyzer` | what the query *looks like* | touch an index or a model |
| `RuleBasedRouter` | *which* strategy should run | retrieve anything |
| `SufficiencyChecker` | whether evidence is enough | read labels or metrics |
| `EscalationPolicy` | which stronger strategy may follow | run anything itself |
| `AdaptiveRetriever` | orchestration only | implement ranking |

`AdaptiveRetriever` holds **zero** `try`/`except` blocks (AST-guard enforced): a
failing strategy propagates unchanged and surfaces as
`status="retrieval_failed"`, exactly as a failing hybrid branch does in Phase 4.

---

## 4. Implementation notes

### 4.1 Router interface

`Router.route(features, *, top_k) -> RoutingDecision` consumes `QueryFeatures`,
not the raw string. That is what makes a learned router substitutable later
without touching retrieval, sufficiency, or escalation.

### 4.2 Routing confidence

`RoutingDecision.confidence` is the normalized margin between the winner and the
runner-up, in `[0, 1]`; a single available strategy yields `0.0` because there was
no evidence to discriminate on. It is **not** a calibrated probability, and it
deliberately does **not** gate sufficiency — sufficiency is a separate mechanism
that looks at retrieved evidence.

### 4.3 Cost model

`RoutingConfig.strategy_cost_ms` is seeded from Phase 5's *measured* retrieval
latencies (`phase-5.md` §8): bm25 1.71, dense 621.35, hybrid 721.00,
hybrid_rerank 4447.68 ms. `cost_weight` (default 0.25) subtracts
`cost_weight × cost/max_cost` from each score. These are measured defaults to be
re-measured, not theoretical constants — Phase 5 recorded a documented estimate
that was wrong by two orders of magnitude.

### 4.4 Sufficiency signals

`result_count`, `lexical_coverage`, `top1_coverage`, and an optional per-strategy
`score_floor` (default `None`). There is **no global score floor** by default:
BM25 magnitudes, cosine similarity, RRF scores, and cross-encoder logits are not
comparable, so one threshold would be an invented number.

### 4.5 Escalation is structurally bounded

The ladder is a **total order** drawn from configuration, so
`BM25 → Dense → Hybrid → BM25` is impossible by construction rather than merely
untested. `max_escalation_steps` defaults to `1`.

### 4.6 Escalated results replace, not merge

An escalated query returns the stronger stage's results. Merging two rankings is
a new fusion algorithm and belongs to Phase 4's concerns.

---

## 5. Configuration

```text
--retriever adaptive
--routing-strategies bm25,dense,hybrid,hybrid_rerank
--routing-ladder bm25,dense,hybrid,hybrid_rerank
--no-escalation            ablation: route once, never escalate
--no-sufficiency           ablation: skip the sufficiency check
--routing-cost-weight 0.0  pure-evidence routing
--sufficiency-threshold 0.5
--disable-feature-group lexical   signal ablation (repeatable)
--max-escalation-steps 1
```

`--retriever adaptive --rerank` is **rejected**: second-stage scoring is one of
the strategies the router selects, so the flag would silently override it.

---

## 6. Trace

Every adaptive execution records: query features, the routing decision and its
per-strategy evidence, confidence, `candidate_k`, `final_top_k`, the initial
strategy and its latency and result count, the sufficiency decision with each
signal, the escalation decision with its reason, the final strategy, per-stage
latencies, and the total. It rides on `RetrievalMetadata.routing` and is copied
into `ExperimentTrace.routing` by one line in the runner.

---

## 7. Architectural boundary

Phase 6 is an **adaptive decision and orchestration layer**. It does not:

* rewrite dense, BM25, hybrid/RRF, or reranking
* introduce a new vector database or retrieval algorithm
* introduce a general-purpose agent framework or unbounded retrieval loops
* read ground-truth labels at query time

Each is enforced by an architecture guard, not merely by convention.

---

## 8. Validation

**221 offline deterministic tests pass** (140 pre-existing + 83 new), 2
integration-marked deselected, 14/14 architecture guards green.

### Real run against the persisted index

```bash
.venv/bin/python scripts/run_experiment.py --retriever adaptive \
  --routing-strategies bm25 --no-judge --no-generation --name adaptive_bm25_v1
```

Recorded **as observed**: 20/20 traces, `retrieval_failed: 0`, every trace
carrying a full routing payload, on the canonical 713-document BM25 index with no
credentials and no network.

| Metric | Value |
|---|---|
| Recall@5 | 0.7167 (identical to the fixed BM25 baseline) |
| Retrieval latency (mean) | 1.48 ms |
| Routing overhead (mean) | 0.34 ms |
| Escalation rate | 0.0 (20/20 judged sufficient) |
| Mean routing confidence | 0.0 (single-strategy run: no runner-up to contrast) |
| Strategy distribution | bm25 20, dense 0, hybrid 0, hybrid_rerank 0 |

The escalation path was verified separately against the same real index: a
`bm25 → dense` escalation fires and records both stage latencies
(`[1.61 ms, 1.39 ms]`, total 3.61 ms) when the threshold judges real partial
coverage insufficient, and correctly does **not** fire at the default threshold.

### Backward compatibility

All **220** pre-Phase-6 traces across the recorded Phase 1–5 runs still validate
against the extended schema with `routing is None`, and the recorded
`metrics.json` files contain no routing section — the fixed baselines are
untouched.

---

## 9. Limitations

* **Thresholds are uncalibrated.** `sufficiency_threshold=0.5` and
  `cost_weight=0.25` are reasoned defaults, not tuned values. Calibrating them
  against labels is Phase 7 work (Ablation 3).
* **Dense/hybrid arms are unvalidated here.** This environment has no network,
  and those arms need `AICREDITS_API_KEY` for query embeddings, as in Phases
  4–5. Only the BM25-only arm was executed.
* **The rule table is hand-designed**, not learned. A learned router is designed
  for (§4.1) but not implemented.
* **The cost table is Phase 5's numbers on this corpus**, not a general model.
* Adaptive builds the dense store whenever dense/hybrid are available, so it
  pays index load at startup even when routing to BM25 — but not dense's ~621 ms
  per query.

---

## 10. Definition of Done

- [x] `routing/` package with analyzer, router, sufficiency, escalation
- [x] `Router` abstraction making a learned router substitutable
- [x] Deterministic, dependency-free query analyzer
- [x] Weighted rule-based router returning a structured `RoutingDecision`
- [x] Routing confidence recorded and documented as uncalibrated
- [x] Existing BM25/Dense/Hybrid/Reranker components reused unmodified
- [x] Sufficiency check that never reads ground truth
- [x] Explicit, bounded escalation policy with at most one step
- [x] `AdaptiveRetriever` conforming to the `Retriever` protocol
- [x] Routing metadata and trace preserved end-to-end
- [x] Routing metrics that stay silent for fixed-strategy runs
- [x] Experiment config and CLI support `retrieval_method="adaptive"`
- [x] 83 routing tests; 221 total; all guards pass
- [x] Real offline BM25 adaptive run against the persisted index
- [ ] Dense/hybrid/rerank adaptive arms (needs `AICREDITS_API_KEY`)
- [ ] Ablations and adaptive-vs-fixed comparison (Phase 7)
