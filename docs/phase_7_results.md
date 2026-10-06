# Phase 7 — Evaluation & Ablations: Results Report

> **Closure note.** This report documents the E-series (7.1–7.3). Phase 7 was
> subsequently closed by two pre-registered gates, recorded in
> [`docs/phase_7_closure_results.md`](phase_7_closure_results.md). That document
> is the **final** Phase 7 result record and supersedes this one on the routing
> question: oracle headroom exists but is concentrated in 4–6 of 107 queries,
> and it is **not reliably identifiable out of sample** from the currently
> observable signals (AUC ≈ 0.534, 0/24 features surviving Holm, both corpus
> arms). ADR-028 is consequently **deferred, not adopted**. Nothing below is
> withdrawn by that; §13's recommendations are superseded by the closure
> report's §6.

> Research question: does query-aware **adaptive routing earn its complexity** over a fixed hybrid baseline, in quality and in cost?
>
> **Answer (shipped settings): no.** At `sufficiency_threshold=0.5`, `cost_weight=0.25`, `max_escalation_steps=1`, the sufficiency gate never fires, so adaptive routing collapses onto a single strategy and routes nothing. It is quality-equivalent to hybrid (Recall@5 and Hit@5 identical on 107/107) and pays ~59 ms median overhead for that equivalence — overhead that yields no quality gain because no decision is ever taken. Genuine headroom exists (BM25 is ~251× cheaper than dense, at a 0.15 Recall@5 cost), but the gate never admits it. Do **not** invent a composite winner score; reason about the quality/cost trade-off explicitly.

---

## 1. Dataset

- 107 records: 47 calibration, 60 test.
- Categories (n): factual 26, conceptual 29, terminology 20, fine_grained 17, comparative 8, multi_document 7.
- Sanity gates S1–S5 and G1–G7 all pass; `gate_open=true`, `grounding_verified=true`.
- Evidence: 0 empty quotes, 0 contaminated quotes across 293 evidence-quote extractions.
- Dataset sha256 prefix `f0695189903c8ef8`; `data/evaluation/phase7_eval_v1.jsonl` (frozen, fixed split).

**Disclosure — G7 partial:** of the 293 evidence quotes, G7 column-contiguity verification left **159 `indeterminate`** — i.e. not verified-clean. The 134 verified-clean quotes are sound; the indeterminate set is the corpus-extraction quality signal that motivated the reranker finding below.

---

## 2. E1 — Five-arm comparison (107 paired queries)

| system | R@5 | MRR | Hit@5 | median ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.7788 | 0.7264 | 0.8318 | 1.88 | 3.45 |
| dense | 0.9283 | 0.8604 | 0.9720 | 472.84 | 691.20 |
| hybrid | 0.8723 | 0.8576 | 0.9346 | 465.65 | 733.62 |
| hybrid_rerank | 0.7165 | 0.5851 | 0.7757 | 3491.69 | 3958.55 |
| adaptive | 0.8723 | 0.8564 | 0.9346 | 524.85 | 655.41 |

Spot-checked against `experiments/phase7/analysis/e1_e6_e7_e8/tables/e1_main_comparison.md`: values match.

### Headline
- **adaptive ≡ hybrid on quality.** Recall@5 and Hit@5 are identical on all 107 queries; MRR differs on exactly 1. Retrieved chunk-ids are identical on 104/107.
- **Cost of equivalence: ~59 ms median** (524.85 vs 465.65) — routing overhead alone (median 0.81 ms, see E6). On the 60 test records the two arms differ on exactly 1 query (`p7_006`, bm25-routed); there adaptive is 25 ms slower.
- **Dense beats adaptive on quality** on the test split (R@5 0.9278 vs 0.8639) at ~92% of the median cost (dense 472.84 ms vs adaptive 524.85 ms) — dense is the quality leader but is not what adaptive escalates *toward*.
- **hybrid_rerank is the worst arm on every quality metric and the most expensive** (7.5× hybrid median latency: 3491.69 / 465.65). It is the net-negative rung at the top of the escalation ladder (see §8).

### Significance vs bm25 (Holm step-down, family = 4 arms)
- `recall_at_5`: dense p=0.00086 / adj 0.0034 **sig**; hybrid p=0.0074 / adj 0.0222 **sig**; adaptive p=0.0074 / adj 0.0222 **sig**; hybrid_rerank p=0.1267 / adj 0.1267 ns.
- `mrr`: dense p=0.0037 / adj 0.0037 **sig**; hybrid p=1.9e-08 / adj 0.0 **sig**; adaptive p=3.7e-08 / adj 0.0 **sig**; hybrid_rerank p=0.00019 / adj 0.00037 **sig**.
- `ndcg_at_5` and `total_latency_ms`: significant for all four arms.
- **15 of the 16 evaluable comparisons are significant**, and the **one** non-significant result is `recall_at_5` × `hybrid_rerank` (raw p = 0.1267, adj 0.1267) — it is also the only arm below bm25 on quality.
- 20 comparisons were requested (5 metrics × 4 arms), but the 4 `estimated_cost_usd` comparisons have `n_pairs = 0` because every run was retrieval-only (see §14), so they are not evaluable. Enumerated from `experiments/phase7/analysis/e1_e6_e7_e8/statistics.json` → `reports.E1.by_metric`.
- **`hit_at_5` was not among the tested metrics.** It is a published quality column in the table above but carries no paired test in this study, so no hit-at-5 significance claim exists in either direction.

**Note on the figure:** the E1 quality-vs-latency figure plots **mean** total latency (2.1 / 524.2 / 542.5 / 3549.1 / 540.9 ms) — i.e. the "Total latency" column, which is the *mean*, not the median. Tables quote medians. Mean is used so the right tail of hybrid_rerank is visible rather than hidden behind the median.

---

## 3. E2 — Escalation ablation (47 calibration)

Chained A-vs-B then B-vs-C. *(This chaining was a bug fix: a prior statistics pass had computed the paired stats against arm C while the table note claimed A-vs-B / B-vs-C; the table now matches the claimed chains.)*

- **A-vs-B:** quality fully tied, p_adj 1.0 across metrics; latency median −27.86 ms, p_adj 0.1439 — ns.
- **B-vs-C:** quality fully tied, p_adj 1.0; latency median **+58.91 ms**, p=0.00309, **p_adj 0.00618 significant** — the only significant E2 result.

**Finding:** escalation buys latency, not quality.

---

## 4. E3 — Feature ablation (47 calibration)

6 `without_*` arms vs `full`:

- **0 of 30 comparisons significant.**
- Recall@5 dips only for `without_complexity` and `without_question_type` (0.8617 vs full 0.8830); all adj p = 1.0.

**Finding:** no feature group demonstrably carries the routing decision at the frozen threshold.

---

## 5. E4 — Threshold calibration (47 calibration)

Sufficiency threshold sweep: **0.3 / 0.4 / 0.5 / 0.6 are identical** (R@5 0.8830, MRR 0.8547, 0.0% escalation — the gate never fires). **0.7 escalates 6.4%**, MRR drops to 0.8377, median latency 556.71 ms.

### Design limitation (disclosed in artifacts)
The `max_escalation_steps = 0..3` sweep is **NOT ANALYSED** — this is a design limitation, not a null result:
- the 0.5 gate never fires (min observed sufficiency is 0.65, above the 0.5 bar), and
- `allows_escalation()` is called exactly once with a **hardcoded `steps_taken=0`** inside a loop-free `retrieve()`, so steps=1/2/3 are structurally identical.

188 rows from this sub-sweep were excluded from the analysis set with the limitation flagged in the artifacts.

---

## 6. E5 — Cost weight (47 calibration)

- `cost_weight` 0.0 / 0.25 → R@5 0.8830; 0.5 / 0.75 / 1.0 → 0.8617, nDCG@5 0.6486→0.6330, escalation 0.0% throughout.
- Raising cost weight **loses quality with no median-latency benefit**, because the router still commits to hybrid's ~450 ms embedding call on ≥94% of queries regardless of weight.

E4 and E5 receive sweep tables only, no paired statistics (they are one-at-a-time sweeps, not paired comparisons).

---

## 7. Frozen configuration

- `sufficiency_threshold=0.5`, `cost_weight=0.25`, `max_escalation_steps=1` — the shipped Phase 6 defaults.
- Chosen on calibration **only** and applied unchanged to the 60 test records.
- No arm dominated on both quality and cost simultaneously; the minimum adjusted p across E2/E3 was 0.00618 (E2 B-vs-C on latency); all quality comparisons p_adj = 1.0 (ns). The value 0.056 does not correspond to any adjusted p-value in the persisted statistics artifacts; the raw Wilcoxon p = 0.0568 for E3 `without_complexity` on nDCG@5 adjusts to 0.375, and the E4/E5 calibration sweeps received no paired statistics.

**Caveat:** the three sweeps (E4 `sufficiency_threshold`, E4 `max_escalation_steps`, and E5 `cost_weight`) were **independent one-at-a-time sweeps, not a joint grid** — so the per-axis best was never validated in combination. This is why the shipped default, rather than a per-axis optimised point, is the defensible configuration: it is the only setting actually exercised end-to-end.

---

## 8. E6 — Routing decision overhead (n=107)

- `routing_latency_ms`: mean 0.85 ms, median 0.81 ms — **0.71% of total latency per query (mean), 0.16% (median/aggregate)**.
- `retrieval_latency_ms`: mean 540.94 ms, median 524.85 ms (n=107).
- `reranking_latency_ms`, `candidate_generation`, `generation`, `query_embedding`: **n = 0 samples** each.

**Finding:** the escalation path never executed at the frozen threshold, so E6 measures the **decision layer only**. The four zero-sample clocks (reranking, candidate_generation, generation, query_embedding) are a direct consequence of `n_escalated = 0` (see E8), not missing instrumentation.

Spot-checked against `experiments/phase7/analysis/e1_e6_e7_e8/tables/e6_routing_overhead.md`: matches, including the n=0 reranking row.

---

## 9. E7 — Per-category (adaptive, n=107, pooled)

| category | R@5 | MRR | n |
|---|---|---|---|
| terminology | 0.9500 | 0.8000 | 20 |
| factual | 0.9231 | 0.8880 | 26 |
| conceptual | 0.9310 | 0.8905 | 29 |
| fine_grained | 0.8824 | 0.8319 | 17 |
| comparative | 0.6250 | 0.9375 | 8 |
| multi_document | 0.4762 | 0.7262 | 7 |

Escalation 0.0% in every category.

**Finding:** neither the calibration nor the test split **individually** supports generalisation claims for `comparative` or `multi_document`: on calibration alone both are n=4 (multi_document is n=3 on test). Only the pooled cell (this table) clears the `min_cell=5` flag, so it is the only cell reported. The pooled result is descriptive, not inferential.

---

## 10. E8 — Escalation analysis

- `n_escalated = 0`, `n_transitions_observed = 0` of **107 routed rows** — that is the **full dataset**, not the 60-record test split. The E8 selection records `split: null` (`provenance.json` → `row_selection`), and the empty-table banner in `e8_escalation_transitions.md` reads "0 of 107 routed rows escalated".
- Sufficiency minimum is **0.6364** on the 60 test records and **0.65** on the 47 calibration records — both above the 0.5 bar, so the full-dataset minimum is also 0.6364.
- `escalated=false`, `to_strategy=null`, `stage_count=1` on all 107; `initial_chunk_ids is None` on all 107 (every query settled on its first strategy and was never escalated).

**Report this as a finding about the frozen threshold, not a missing result.** Precisely scoped: across the full 107, escalation never fires at `sufficiency_threshold=0.5` because observed sufficiency never drops below 0.6364. Escalation **was** observed — 3 escalations, all in the calibration arm at threshold 0.7 (see E4). Therefore do **not** claim escalation never happens at any threshold; claim only that the shipped gate is too tight to ever open on this dataset.

---

## 11. Cross-encoder reranking — a genuine negative finding (not a bug)

Seven classic defects were ruled out before accepting the result, including replaying the project's own ONNX scoring path and reproducing all 1070 stored logits with max |Δ| = 0.0000 (i.e. deterministic reproducibility confirmed).

On the reranker's own 10-item output set (107 queries × 1488 gold-vs-non-gold pairs):

- **Cross-encoder ordering: 0.4476 gold-above-non-gold — below chance** (chance ≈ 0.50). The RRF ordering feeding it scores **0.7621**.
- Gold-at-rank-1 fell from 81/107 → 47/107.
- Best-gold position improved in **0 of 107** queries (21 worse / 3 better / 83 equal).
- Every gold chunk scores in [−8.5, −6.6] — deep in MS-MARCO "irrelevant" territory — because the corpus text is **column-interleaved garbled PDF extraction** (1.4% of tokens glued), far off the cross-encoder's training distribution.
- Reproduces the Phase 5 §8.1 finding on the identical corpus at n=107.

**Consequence:** `hybrid_rerank` is the top rung of the escalation ladder, and the escalation gate only ever selects the rung that *can* fire. When escalation was forced at threshold 0.7, the 3 escalated queries scored R@5 0.8333 / MRR 0.7333 at 4259 ms median — versus the same arm's 44 non-escalated queries at R@5 0.8864 / MRR 0.8448 at 541 ms (from `experiments/phase7/cal047__E4_threshold_calibration__sufficiency_threshold=0.7/rows.jsonl`). Forcing escalation trades both quality and latency for the reranker's wrong-headed ordering.

**Action:** remove the rerank rung from the router's strategy set; the escalation ladder should step from BM25 → dense → hybrid, not terminate on a net-negative reranker.

---

## 12. Artifact provenance

- This report is the single persisted Phase 7 result artifact. Traces, figures, and tables live under **`experiments/phase7/analysis/`** (gitignored). The E1/E6/E7/E8 tables spot-checked for this report live in `experiments/phase7/analysis/e1_e6_e7_e8/tables/` as `e1_main_comparison.{md,csv,json}`, `e6_routing_overhead.{md,csv,json}`, `e7_query_type.{md,csv,json}`, `e8_escalation_transitions.{md,csv,json}`.
- `experiments/phase7/` is **gitignored** — traces, figures, and tables are untracked and **not recoverable from version control**. The dataset (`data/evaluation/phase7_eval_v1.jsonl`) and its ledger are the only persisted provenance.
- All Phase 7 work is uncommitted on top of `0fafe2eb1229480955d41c44f1fb24b94bbcb49d`.

---

## 13. Recommendations

1. **Remove the rerank rung** (`hybrid_rerank`) from the router's strategy set — it is net-negative on quality and 7.5× the median latency of hybrid (3491.69 ms vs 465.65 ms). The escalation ladder should be BM25 → dense → hybrid.
2. **Recalibrate the sufficiency gate** so it can fire within the observed score range. The 0.5 threshold is above the observed sufficiency minimum (0.6364 on test, 0.65 on calibration), so it never opens. A threshold in the 0.6–0.65 band — validated in a joint grid with cost_weight, not the independent sweeps used here — is the first thing to try.
3. **Re-evaluate** after (1) and (2): the current "adaptive ≡ hybrid" equivalence is an artifact of the gate never firing, not evidence that routing is unnecessary. Escalation may only pay off once the net-negative reranker is removed and the gate is permitted to escalate to a quality-positive rung.

---

## 14. Limitations & Disclosures

- **No adaptive-vs-hybrid p-value exists.** E1 compares each arm against bm25 (Holm, family=4); adaptive is not directly tested against hybrid. The "adaptive ≡ hybrid" claim rests on identical Recall@5/Hit@5 on 107/107 (identical chunk-ids on 104/107), not on a paired test.
- **nDCG@5 is not textbook nDCG.** Its IDCG denominator comes from `relevant_chunk_ids(example, trace)` (`evaluation/retrieval.py:163`, `evaluation/dataset.py:158`), which marks relevance only over the chunks *this trace* retrieved from the labelled documents/sections — so the denominator grows with the number of relevant chunks retrieved rather than being the gold-standard ideal ranking. That makes it self-normalising and not comparable to a textbook nDCG. **No directional penalty follows from that alone, and none is observed here**: across the five E1 arms nDCG@5 tracks the other quality metrics (bm25 0.4679, hybrid 0.6155, adaptive 0.6180, dense 0.6396, hybrid_rerank 0.2631). Read it as a within-trace ranking measure. Recall@5 / MRR / Hit@5 are label-driven and sound.
- **No cost dollar figures.** `estimated_cost_usd` has no data (n_pairs = 0); all runs were retrieval-only (`--retrieval-only`). No dollar figures appear or may be derived.
- **E6 zero-sample clocks.** `reranking_latency_ms`, `candidate_generation`, `generation`, and `query_embedding` all have n=0 because the escalation path never executed (see E8); E6 measures the decision layer only.
- **E4 unanalysable step sweep.** `max_escalation_steps=0..3` is structurally identical (hardcoded `steps_taken=0`, loop-free `retrieve()`); 188 rows excluded and flagged in the artifacts.
- **E7 underpowered cells.** `comparative` and `multi_document` are below `min_cell=5` on each split individually; only the pooled cell is reported, and only descriptively.
- **Calibration sweeps were independent, not a joint grid.** E4's `sufficiency_threshold`, E4's `max_escalation_steps`, and E5's `cost_weight` were each swept one axis at a time; the per-axis best was never validated in combination — which is why the shipped default is used rather than a per-axis optimised point.
- **G7 partial (see §1):** 159/293 evidence quotes are `indeterminate` under column-contiguity verification — not verified-clean.
- **Generation out of scope.** No end-to-end answer-quality claim is made anywhere in this report.
- **Corpus defect.** `pyserini_lin_2021` contains a biomedical knowledge-graph survey that was excluded and re-ingestion was **deliberately deferred** — re-ingestion would change `corpus_version` and invalidate the Phase 2–6 results and the frozen cost baselines. The current corpus is a known-defect superset used only for comparability with Phases 2–6.
- **E1 figure statistic.** The quality-vs-latency figure uses **mean** latency (2.1 / 524.2 / 542.5 / 3549.1 / 540.9 ms); tables quote **medians**. The figure's statistic is stated explicitly so the two are not conflated.
- **Escalation-transitions figure skip is scoped to its rows, not to the router.** `experiments/phase7/figures/figures_manifest.json` records `PlotDataError`: "no escalation transition was observed in the 107 row(s) supplied, so there is nothing to plot. That is a statement about this row set and not about the router: these rows carry one adaptive configuration, and another configuration of the same router may well escalate. An all-zero transition chart would instead claim the router declined to escalate, which these rows cannot establish either way." The figure is handed only the 107 frozen-configuration adaptive rows (`scripts/make_phase7_plots.py`, `routed_rows`), so the scope is **zero transitions in those rows** at the frozen 0.5 threshold (`n_transitions_observed = 0` of 107) — not "escalation never happened". Three escalations do exist in the 0.7 calibration arm (3 of 47, 6.4%), which is why the skip is a scope statement rather than a claim about the ladder.
- **Artifact recoverability.** `experiments/phase7/` is gitignored; traces, figures, and tables are untracked and not recoverable from version control. The dataset and its ledger are the only persisted provenance (see §12).
