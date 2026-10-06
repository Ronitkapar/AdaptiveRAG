# System Overview

The architecture **as implemented**, not as planned. Detailed layer-by-layer
reference: [`../architecture.md`](../architecture.md). Decisions:
[`../decision.md`](../decision.md).

## Runtime path

```text
Query
  ↓
Query / Retrieval Analysis          QueryFeatureAnalyzer — deterministic,
  │                                 model-free query features (6 groups)
  ↓
Routing / Decision Logic            RuleBasedRouter → RoutingDecision
  │                                 (strategy, confidence, evidence, versions)
  ↓
Retriever                          ┌── BM25            (local, ~2 ms)
  │                                ├── Dense          (embedding API + Qdrant)
  ├── adaptive ────────────────────┤
  │                                ├── Hybrid         (RRF of dense + BM25)
  │                                └── Hybrid+Rerank  (ONNX cross-encoder)
  ↓
Retrieved Evidence
  ↓
Evaluation / Trace                 ExperimentRunner → ExperimentTrace
                                   → evaluators (retrieval / routing / efficiency)
```

For the `adaptive` path, between the decision logic and the retriever sit two
components that never run for the fixed strategies:

```text
Retrieve (initial pick)
  ↓
SufficiencyChecker — label-free judgement of the retrieved evidence
  ├── sufficient ──────────────────────────────→ Final results
  └── insufficient → EscalationPolicy (total-order ladder, ≤ N steps)
                        ↓
                  Stronger strategy → Final results
```

The two rankings are **never merged**: an escalated query returns the stronger
stage's results, because merging them would be a new fusion algorithm
(ADR-022, ADR-025).

## The four layers, and why they are separate

| Layer | Responsibility | Does **not** do |
| --- | --- | --- |
| **Router** (`routing/`) | choose a strategy; judge sufficiency; bound escalation | retrieve anything itself; the router has no index access |
| **Retriever** (`retrieval/`, `reranking/`, `indexing/`) | produce ranked results from an index | make routing decisions |
| **Evaluation** (`evaluation/`) | compute metrics from persisted traces | depend on any retrieval strategy (ADR-008) |
| **Experiment infrastructure** (`experiments/`, `scripts/`) | run configurations, register arms, export rows | participate in retrieval or routing decisions |

The separations are enforced by tests, not merely by convention:
`tests/test_architecture_guards.py` AST-scans for forbidden coupling (e.g. the
reranker must not reach into an index; the router must not embed queries), and
`tests/test_schemas.py` pins the structured data contracts.

## Contracts

* **`Retriever`** — every strategy, including `AdaptiveRetriever`, implements the
  same protocol, so evaluation cannot tell them apart (ADR-007).
* **`RoutingDecision`** — a structured record, not a strategy name (ADR-022).
  Confidence is a normalized score margin, **not** a calibrated probability, and
  never gates sufficiency on its own (ADR-023).
* **`ExperimentTrace`** — the single source of truth per query. Metrics are
  recomputable from `traces.jsonl` at any time; aggregate numbers are never
  authoritative on their own.
* **Sufficiency** — judged at query time from retrieved evidence alone, with no
  ground truth in scope (ADR-024).
* **Escalation ladder** — a total order drawn from configuration, so
  `BM25 → Dense → Hybrid → BM25` is impossible by construction, not merely
  untested (ADR-025).

## Indexes and provider boundaries

| Index | Backed by | Notes |
| --- | --- | --- |
| `storage/qdrant/` | local embedded Qdrant, cosine | one collection per corpus arm; `ensure_collection` reuses by name, so arms must be told apart by namespace, not by fingerprint |
| `storage/bm25/` | local inverted index (JSON) | one file per corpus arm, beside the Phase 7 one |
| `storage/reranker/` | ONNX cross-encoder | downloaded on first use, reused afterwards |

Provider calls (embeddings, generation, judge) are isolated behind provider
modules (ADR-004); the BM25 path makes **no** network call at all, which is what
makes it the control arm for latency comparisons.

## Generation and judging (present, not part of any routing claim)

A generation layer (`ContextBuilder`, Groq `gpt_v1`) and an LLM judge
(`judge_v1`, cached in `storage/eval_cache.sqlite3`) exist and are exercised by
the baseline experiments. **No Phase 7–15 routing result depends on them** — all
reported quality numbers are retrieval-only, and `--no-judge --no-generation`
keeps runs fully offline and deterministic.

## What exists but is deliberately inert

* **`adaptive` at the shipped configuration** — the sufficiency gate does not
  open on this benchmark, so the adaptive arm behaves as hybrid. This is a
  measured property of the shipped defaults, not a code path that is absent.
* **The rerank rung** — implemented and measured, but excluded from escalation
  consideration (ADR-028) because it is net-negative on this corpus.
* **The learned router** — deferred behind the `Router` interface (ADR-026).
  Never trained, never deployed.

---

*What is implemented. Findings: [`../research/findings.md`](../research/findings.md).
Full report: [`../ADAPTIVERAG_REPORT.md`](../ADAPTIVERAG_REPORT.md).*