# Phase 7 — Evaluation & Ablations

**Status:** IN PROGRESS — foundation (7.0a gate, 7.0b cost freeze) complete; the
evaluation and ablation program has not started
**Phase:** 7
**Purpose:** Establish whether query-aware routing actually earns its complexity,
and by how much, measured against fixed strategies on the same corpus.

---

## 1. Objective

Phase 6 built the adaptive mechanism and left its value unproven. Phase 7 answers:

1. Is the environment even capable of producing trustworthy measurements?
2. Are the router's own assumptions — above all its cost table — grounded in
   measured reality?
3. Does adaptive routing beat fixed strategies on quality, latency, and tokens?
4. Which parts of the routing machinery actually carry that benefit?

Step 3 is unstarted. Steps 1 and 2 are complete and recorded below.

---

## 2. Scope

### Completed in 7.0a — environment gate

* `scripts/validate_environment.py` → `experiments/phase7/gate.json`
* Probes all five arms for response-format compliance and determinism
* Refuses to open the gate when any arm is blocked or fails

### Completed in 7.0b — strategy-cost re-measurement

* `src/adaptive_rag/evaluation/measurement.py` — the protocol and its aggregates
* `scripts/measure_strategy_cost.py` — the sweep
* `experiments/phase7/strategy_cost_ms.json` — the artifact
* `RoutingConfig.strategy_cost_ms` — frozen from the artifact's medians

### Available but unexercised (7.1+)

* `evaluation/stats.py` — significance testing, symmetry assessment
* `evaluation/ablation.py` — ablation variant construction
* `experiments/arms.py` — the arm matrix and shared-condition verification

These were built as the foundation for E-series work and are covered by tests,
but no ablation has been run against them yet.

### Excluded

Nothing from Phases 2–6 is modified by this phase beyond the two defects
recorded in §5, both of which were blockers to measuring at all.

---

## 3. The environment gate (7.0a)

A measurement is only worth as much as the environment that produced it. The gate
answers one question per arm: *does this arm return well-formed, deterministic
results against the canonical corpus right now?*

Result on this machine at commit `9db5054`: **5/5 PASS, gate open.** Every arm —
`bm25`, `dense`, `hybrid`, `hybrid_rerank`, `adaptive` — returned non-empty
results, finite non-negative clocks, matching `top_k`, populated provenance
fields, and identical rankings across two identical calls.

Design decisions worth preserving:

* **A missing latency clock is a failure, not `0.0`.** A `0.0` would silently
  flatter whichever arm produced it in every downstream percentile.
* **A crashing arm becomes a record, not an exception.** The gate reports what
  failed and why; it does not abort on the first bad arm.
* **Determinism is asserted, not assumed.** The repetition design underneath the
  cost table only means something if repeated calls are stable.

This gate also retired a limitation carried since Phase 4: the dense, hybrid, and
reranked arms had never been executed live here for want of network. They now are.

---

## 4. The strategy-cost re-measurement (7.0b)

### 4.1 Why the seed had to be replaced

`RoutingConfig.strategy_cost_ms` is the router's price list. `RuleBasedRouter`
normalises each strategy's cost by the maximum across available strategies, so
**what the router consumes is the ratio between strategies, not the absolute
milliseconds.** An arm whose measurement is noisy contributes a noisy ratio, and
the cost term then moves with machine noise rather than with strategy choice.

The Phase 5 seed came from per-query means over 20 examples in a single pass — no
warm-up, no repetitions. The gate then observed dense retrieval swinging between
roughly 0.85 s and 4 s on this hardware, a spread wider than the entire
dense/hybrid gap those means encoded. The table was not merely imprecise; its
*ordering* was unreliable.

### 4.2 Protocol

Defined in `evaluation/measurement.py`, run by `scripts/measure_strategy_cost.py`:

1. Build every component **outside** any clock.
2. Warm up each arm and **discard** those samples, paying the ONNX session's
   first-call cost.
3. Run **5 repetitions** of all 20 queries per arm, rotating both arm order and
   query order between repetitions.
4. Aggregate with `statistics.median`; report p95, mean, stdev, min, max, the
   per-stage median of every clock, and the raw samples.

Two estimator choices are recorded in the artifact because the names mislead
otherwise:

* The median is `statistics.median`, **not** `evaluation.base.percentile`. That
  helper is nearest-rank and returns the lower of the two middle values at even
  n — correct for a percentile, misnamed as a median.
* p95 **is** `evaluation.base.percentile`, which does not interpolate. At n=100 it
  is literally the 95th of 100 observed samples, not a smoothed curve value.

### 4.3 Why rotation is load-bearing

An arm that always runs first pays for the machine warming up; one that always
runs last pays for whatever else the machine was doing. Both are systematic
biases, and both would land in the cost table as real differences between
strategies. With 5 arms and 5 repetitions every arm occupies every position
exactly once.

### 4.4 Results

n=100 per arm, no rerank fallbacks, artifact `complete: true`.

| Strategy | p50 (ms) | p95 (ms) | mean | stdev | p95/p50 |
| --- | --- | --- | --- | --- | --- |
| `bm25` | 2.25 | 5.14 | 2.57 | 1.60 | 2.28 |
| `dense` | 451.78 | 744.64 | 496.28 | 129.13 | 1.65 |
| `hybrid` | 455.97 | 923.42 | 727.97 | 1373.66 | 2.03 |
| `hybrid_rerank` | 3854.41 | 4389.73 | 4023.80 | 1068.51 | 1.14 |

The adaptive arm is reported at p50 433.78 / p95 932.53 but is **excluded from
the table**: its cost is a per-query mixture of the others decided at runtime,
and the router chooses between strategies, not between the adaptive system.

### 4.5 What the numbers say

* **The frozen table prices a network round-trip, not a retrieval algorithm.**
  The per-stage medians show dense's 451.78 ms is ~426 ms of live
  `text-embedding-3-large` call and only ~24 ms of vector search. Hybrid has the
  same shape (~450 ms dense branch, ~29 ms BM25, 0.5 ms fusion). Comparing dense
  against hybrid on this table is mostly comparing two API calls.
* **Reranking dominates:** `hybrid_rerank` costs ~8.5x the fused arm, with
  ~3.37 s of its 3.85 s median inside the cross-encoder.
* **The median earned its place.** `hybrid` carries a 12.0 s outlier — one
  rate-limited embedding call — that inflates stdev to 1373.66 while leaving the
  median at 455.97. The old mean-based seed would have absorbed that outlier
  directly into the price list.
* **dense and hybrid are now nearly identical in cost**, the opposite of what
  the seed claimed (621 vs 721 ms, a meaningful gap). Fusing BM25 into dense adds
  ~3 ms of lexical work to a retrieval whose cost is dominated by an API call —
  so on cost alone the router has no reason to prefer one.

### 4.6 Freeze

The four medians were reviewed and written into `RoutingConfig.strategy_cost_ms`,
`adaptive` excluded. Every arm moved against the seed: dense −27%, hybrid −37%,
`hybrid_rerank` −13%, `bm25` +32%.

Freezing these values changes the router's cost term, so it is a real behavioural
change to Phase 6 — recorded here rather than made silently.

---

## 5. Defects found and fixed

Both were discovered by trying to measure, not by inspection.

### 5.1 Injected Qdrant client ignored (blocker)

`instantiate_components` accepted a `client` parameter but the adaptive branch
called `_shared_qdrant_client()` unconditionally, opening a second embedded
client. Local Qdrant takes an exclusive lock on the storage folder, so any caller
building several arms at once failed with `AlreadyLocked` — which is exactly what
the cost sweep does.

This was invisible until Phase 7 because the only prior adaptive run configured
`available_strategies=['bm25']`, which opens no vector store at all, and the gate
builds one arm at a time.

Fixed by honouring the injected client. Two regression tests cover both
directions; the reuse test fails without the fix, verified by reverting it.

### 5.2 Provider rate limit (blocker)

The first sweep attempt completed 4 of 5 repetitions, then died on HTTP 429. Three
of the four costed strategies embed every query remotely, so the protocol issues
roughly `3 × 20 × 5` live calls in a burst, and the adapter's `2 ** attempt`
backoff over `max_retries=4` (~14 s total) cannot ride out a sustained limit.

Fixed by pacing queries on provider-calling arms. The sleep sits **between** timed
retrievals, never inside one, so it cannot enter any `latency_ms`; the measured
quantity, sample count, rotation, and estimator are unchanged. The pacing value is
recorded in the artifact so a reader knows the sweep was paced.

BM25 is deliberately not paced — it is local, and pacing it would waste sweep
wall-clock. `adaptive` *is* paced, because whether it calls the provider depends on
the runtime routing decision and is not knowable in advance.

---

## 6. Limitations

* **Not portable.** The table describes one machine against one provider over one
  afternoon. `experiments/phase7/` is gitignored, so refreshing it requires a
  re-run, not a commit.
* **Network-dominated.** Because the embedding call dominates, the ratios are a
  property of the provider's current latency as much as of the retrievers.
* **The median understates the tail.** `hybrid`'s p95 is 2x its p50 and its worst
  sample was 12 s. A router optimising against the median will not avoid that.
* **Still uncalibrated.** `cost_weight=0.25` and `sufficiency_threshold=0.5`
  remain reasoned defaults. Re-measuring the cost table does not calibrate them.
* **No adaptive-vs-fixed claim is made yet.** Nothing here demonstrates that
  routing beats a fixed strategy; only that the router's price list is now real.

---

## 7. Definition of Done

- [x] Five-arm environment gate passes with provenance
- [x] Measurement protocol defined, warm-up and rotation included
- [x] Full sweep completed: n=100 per arm, complete artifact, no fallbacks
- [x] Artifact reviewed before freezing; medians written to `RoutingConfig`
- [x] `adaptive` excluded from the cost table by construction and by guard
- [x] Qdrant client defect fixed and regression-tested
- [x] Rate-limit pacing added and recorded
- [ ] Adaptive-vs-fixed comparison on quality, latency, and tokens
- [ ] Ablation program executed against the built harness
- [ ] Threshold calibration against labels

---

## 8. Next steps

> Phase 7 E-series — adaptive-vs-fixed evaluation and ablations.

The measurement foundation is complete and honest. What remains is the actual
research question: whether routing earns its cost, and which parts of it do the
earning.
