# Phase 7 — Evaluation & Ablations

**Status:** E-SERIES EXECUTED / DEFINITION OF DONE PARTIAL — foundation
(7.0a gate, 7.0b cost freeze, 7.0c harness) complete and E1–E8 run; several DoD
items are only partially met (§10)
**Phase:** 7
**Purpose:** Establish whether query-aware routing actually earns its complexity,
and by how much, measured against fixed strategies on the same corpus.
**Results report:** [`docs/phase_7_results.md`](../phase_7_results.md)

---

## 1. Objective

Phase 6 built the adaptive mechanism and left its value unproven. Phase 7 answers:

1. Is the environment even capable of producing trustworthy measurements?
2. Are the router's own assumptions — above all its cost table — grounded in
   measured reality?
3. Does adaptive routing beat fixed strategies on quality, latency, and tokens?
4. Which parts of the routing machinery actually carry that benefit?

All four are now answered. Steps 1 and 2 are recorded in §3 and §4; steps 3 and
4 are recorded in §7. The answer to step 3 at the shipped settings is **no**,
and §7 states precisely why.

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

### Exercised in 7.1–7.3 — the E-series

* `evaluation/stats.py` — significance testing, symmetry assessment
* `evaluation/ablation.py` — ablation variant construction
* `experiments/arms.py` — the arm matrix and shared-condition verification
* `evaluation/analysis.py`, `evaluation/tables.py`, `evaluation/plots.py`,
  `experiments/figures.py` — the analysis, table, figure, and provenance stage

All eight studies (E1–E8) were executed through `scripts/run_phase7_suite.py`
against the frozen `phase7_eval_v1` benchmark. Results: §7, and the full report
in `docs/phase_7_results.md`.

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
  remain reasoned defaults. Re-measuring the cost table does not calibrate them,
  and the E4/E5 sweeps (§7) were independent one-axis-at-a-time sweeps, so the
  joint optimum is still unmeasured.
* **The adaptive-vs-fixed claim is now made, and it is negative.** §7 supersedes
  the "no claim yet" position this document held at 7.0b: routing does not beat
  a fixed strategy at the shipped settings, and the reason is a gate that never
  opens rather than evidence that routing cannot help.

---

## 7. The E-series (7.1–7.3)

### 7.1 What was run

Benchmark: `data/evaluation/phase7_eval_v1.jsonl`, frozen, 107 records —
47 calibration, 60 test. Dataset sha256 prefix `f0695189903c8ef8`; sanity gates
S1–S5 and dataset gates G1–G7 pass, `gate_open=true`, `grounding_verified=true`;
0 empty and 0 contaminated evidence quotes across 293 quote extractions. Every
run was **retrieval-only**.

| Study | Scope | Headline |
| --- | --- | --- |
| E1 | five arms, 107 paired queries | adaptive Recall@5 0.8723 / MRR 0.8564 vs hybrid 0.8723 / 0.8576 |
| E2 | escalation ablation A/B/C, 47 calibration | quality tied; only significant result is B-vs-C latency +58.91 ms (p_adj 0.00618) |
| E3 | leave-one-out over six feature groups, 47 calibration | 0 of 24 evaluable comparisons significant |
| E4 | `sufficiency_threshold` 0.3–0.7, 47 calibration | 0.3/0.4/0.5/0.6 identical; 0.7 escalates 6.4% (3 of 47) |
| E5 | `cost_weight` 0.0–1.0, 47 calibration | 0.0/0.25 → Recall@5 0.8830; 0.5/0.75/1.0 → 0.8617; escalation 0.0% throughout |
| E6 | routing overhead, n=107 | decision latency median 0.81 ms; four stage clocks n=0 |
| E7 | per-category, adaptive, n=107 pooled | escalation 0.0% in every category |
| E8 | escalation transitions, n=107 | 0 escalations; sufficiency never below 0.6364 |

### 7.2 The headline negative result

At the shipped Phase 6 defaults adaptive routing **earns nothing**. The
sufficiency gate never fires, so `adaptive` collapses onto a single strategy and
routes nothing: Recall@5 and Hit@5 are identical to hybrid on 107/107,
retrieved chunk-ids are identical on 104/107, and MRR differs on exactly one
query. That equivalence costs **~59 ms** of median latency (524.85 vs 465.65),
of which only 0.81 ms is the routing decision — the rest is the hybrid stage the
router chose.

Against bm25 with Holm step-down within each metric, **15 of the 16 evaluable
comparisons are significant**. The single non-significant result is
`recall_at_5` × `hybrid_rerank` (adj 0.1267); the other four requested
comparisons are the `estimated_cost_usd` family, which has `n_pairs = 0`
because no run was generation-bearing. `hit_at_5` is published as a quality
column but was **not** among the tested metrics, so no hit-at-5 test exists.

No composite quality/cost score is computed anywhere, and none may be derived.
The result is a trade-off to be reasoned about, not an ordering.

### 7.3 The two findings underneath it

**The cross-encoder reranker is net-negative on this corpus, and it is the
terminal rung of the escalation ladder.** `hybrid_rerank` is last on every E1
quality metric (Recall@5 0.7165, MRR 0.5851, Hit@5 0.7757) at 7.5x hybrid's
median latency, and it is significantly *worse* than bm25 on MRR (adj 0.00037)
and nDCG@5 (adj 0.000117) — significant differences in the wrong direction.
Because the ladder terminates on it, the escalation gate can only ever escalate
*into* a net-negative stage. The defect isolation behind this (seven classic
reranker defects ruled out, including an exact replay of the stored logits at
max |Δ| = 0.0000) is in `docs/phase_7_results.md` §11.

**The sufficiency threshold sits above the observed score range, so the
escalation path is unreachable.** Observed sufficiency never drops below 0.6364
on the 60 test records or 0.65 on the 47 calibration records, so a 0.5 bar
cannot open. Escalation *was* observed — 3 times, all in the 0.7 calibration
arm — so the defensible claim is that the shipped gate is too tight on this
dataset, not that escalation never happens.

Both are recorded as recommendations in ADR-028. **Neither has been acted on**
— see §8.

---

## 8. Decisions taken in Phase 7

**The frozen configuration is the shipped Phase 6 default, unchanged.**
`sufficiency_threshold=0.5`, `cost_weight=0.25`, `max_escalation_steps=1`. It was
chosen on calibration only and applied unchanged to the 60 test records. The
sweeps did not produce a defensible alternative: no arm dominated on quality and
cost simultaneously, so the shipped default is preferred precisely because it is
the only setting actually exercised end to end. A per-axis optimised point would
have been a configuration this phase never ran.

**The three calibration sweeps were independent, not a joint grid.** E4's
`sufficiency_threshold`, E4's `max_escalation_steps`, and E5's `cost_weight` were
each swept one axis at a time. The per-axis best was therefore never validated in
combination, and the joint optimum — which is the only thing that would justify
changing the shipped default — is **not measured**. This is a design limitation
of the executed protocol, not a result, and it is why the recommendation to
recalibrate the gate explicitly requires a joint grid over
`sufficiency_threshold` × `cost_weight` before anything is changed.

**No router behaviour was changed.** Removing the rerank rung or moving the
threshold would modify Phase 6's router, its default ladder, and its shipped
defaults. Phase 7 measures; it does not retune. The two findings are therefore
recorded as ADR-028, marked as a recommendation, with the implementation left to
an explicit Phase 6 follow-up.

**No composite score was constructed**, at any point. Quality and cost are
reported on separate axes throughout, because the Phase 7 question is a
trade-off and a single number would hide the trade-off it collapses.

---

## 9. Validation

* **729 offline deterministic tests pass, 2 deselected**
  (`.venv/bin/python -m pytest -q`), with one matplotlib `Axes3D` import
  warning. No credentials, no network, no live index.
* **14/14 architecture guards green** (`tests/test_architecture_guards.py`).
* Every Phase 7 run registered in `experiments/phase7/registry.json` and
  cross-checked against its own `manifest.json` / `config.json`, so an entry
  cannot disagree with the run it describes. All five E1 arms completed
  107/107 with zero retrieval failures.
* Figures: four rendered, two skipped with recorded reasons
  (`quality_vs_cost` — no arm carries both `recall_at_5` and
  `estimated_cost_usd`; `escalation_transitions` — no transition observed, so
  there is nothing to plot).
* No change under `src/`, `tests/`, or `experiments/` was required to execute
  the E-series or to write the results report.
* `experiments/phase7/` is gitignored, so traces, figures, and tables are not
  recoverable from version control. The dataset and its ledger are the only
  persisted provenance.

---

## 10. Definition of Done

Foundation:

- [x] Five-arm environment gate passes with provenance
- [x] Measurement protocol defined, warm-up and rotation included
- [x] Full sweep completed: n=100 per arm, complete artifact, no fallbacks
- [x] Artifact reviewed before freezing; medians written to `RoutingConfig`
- [x] `adaptive` excluded from the cost table by construction and by guard
- [x] Qdrant client defect fixed and regression-tested
- [x] Rate-limit pacing added and recorded
- [x] Suite driver, registry, row export, figures, and CLI in place and tested
      offline

E-series:

- [x] Adaptive-vs-fixed comparison executed — E1, 107 paired queries, Holm
      corrected against bm25
- [x] Ablation program executed — E2 (escalation A/B/C) and E3 (six feature
      groups)
- [x] Threshold calibration executed — E4 and E5 sweeps
- [x] E6, E7, E8 analyses produced from the E1–E3 runs
- [x] Results report written with per-study provenance and limitations

**Partially met — these must not be read as complete:**

- [ ] **Partially met — E8 measured escalation but did not characterise it.** Zero transitions
  were observed at the frozen threshold, so what escalation buys *per transition*
  is unmeasured on this run. The escalation-transitions table and figure are
  empty by observation, not by omission — the manifest records the skip and its
  reason.
- [ ] **Partially met — E4's `max_escalation_steps` sweep is unanalysable, not null.** Its 188
  rows were excluded with the limitation flagged in the artifacts: the 0.5 gate
  never fires, and `allows_escalation()` is called exactly once with a hardcoded
  `steps_taken=0` inside a loop-free `retrieve()`, so steps 1/2/3 are
  structurally identical. A flat row of equal numbers would be an artifact of
  that structure.
- [ ] **Partially met — E6 measures the decision layer only.** `reranking_latency_ms`,
  `candidate_generation`, `generation`, and `query_embedding` all have n = 0
  because no query escalated.
- [ ] **Partially met — the escalation-transitions figure was skipped**, with the reason
  recorded in `figures_manifest.json`.
- [ ] **No cost dollar figures.** `estimated_cost_usd` has no data
  (`n_pairs = 0`) in every study; all runs were retrieval-only.
- [ ] **Calibration was swept one axis at a time, not on a joint grid**, so the
  joint optimum is unmeasured and no threshold change is justified by this
  phase's evidence.
- [ ] **E7's `comparative` and `multi_document` cells are below `min_cell=5`** on
  each split individually; only the pooled cell is reported, and only
  descriptively.
- [ ] **No adaptive-vs-hybrid p-value exists.** E1 tests every arm against
  bm25, so the equivalence claim rests on identical Recall@5/Hit@5 on 107/107,
  not on a paired test.

---

## 11. Next steps

> Not a new phase. Two Phase 6 changes are recommended and neither is in scope
> here.

1. **Decide the rerank rung.** ADR-028 recommends removing `hybrid_rerank` from
   the router's strategy set so the ladder terminates on hybrid. That is a
   Phase 6 router change.
2. **Recalibrate the sufficiency gate** inside the observed score range
   (0.6–0.65), validated on a **joint grid with `cost_weight`** rather than the
   independent sweeps used here, and reported on the held-out test split.
3. **Then re-run E1 and E8.** Until the gate can open and the ladder terminates
   on a quality-positive rung, "adaptive ≡ hybrid" is an artifact of a gate that
   never fires — not evidence that routing is unnecessary.

Full numbers, provenance, and per-study limitations:
[`docs/phase_7_results.md`](../phase_7_results.md).
