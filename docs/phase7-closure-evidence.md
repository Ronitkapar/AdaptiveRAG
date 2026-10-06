# Phase 7 Closure — Evidence Snapshot

This file exists to make the Phase 7/8 evidence **auditable and reproducible**.
Most of it is deliberately *not* in git:

* `experiments/*/` and `storage/` are gitignored, so every Phase 8 artifact —
  the oracle, the gold-label audit/repair, the arm comparison, and the per-arm
  `rows.jsonl` inputs — is unversioned.
* `data/processed_phase8_{before,after}/` are derived corpora (reproducible from
  the raw PDFs by the ingestion pipeline) and are now ignored like `data/processed/*`.

The snapshot therefore records the **source commit** plus a **SHA-256 of every
input and artifact** the Phase 7 closure will rely on. Any later run can be shown
to have used the same evidence, or shown to have drifted.

## 1. Provenance

| Field | Value |
|---|---|
| Branch (freeze) | `phase7-closure` |
| Tag (freeze) | `phase7-closure-freeze-01` |
| Parent commit (Phase 7 close) | `ae64d676b8fcf4f1c6d51d7ca3d631a63b769936` |
| Environment | Python 3.12.14 (`.python-version` = 3.12), numpy 2.5.3 |

The freeze commit captures the 20 modified tracked files plus the new Phase 8
scripts/tests/docs that were uncommitted on `phase-5-reranking`. Code and docs,
not bulk data, are the versioned part; the data/artifacts are pinned by hash below.

## 2. Corpora

Both Phase 8 arms report the **same** `corpus_version` string
(`corpus_c5996129918d7e44`), which does *not* distinguish the column-extraction
before/after pair. The distinguishing key is the **namespace directory** and the
manifest hashes below. Do not conflate the two corpora by `corpus_version` alone.

| Corpus | Namespace | Ingestion | Chunking | Chunks / Docs | Embedding |
|---|---|---|---|---|---|
| shipping (after) | `data/processed_phase8_after` | ingestion_v3 | structure_aware_v1 | 613 / 14 | text-embedding-3-large, dim 3072, normalized |
| pre-fix (before) | `data/processed_phase8_before` | ingestion_v3 | structure_aware_v1 | see manifest | text-embedding-3-large, dim 3072, normalized |

Manifest SHA-256:

| File | after | before |
|---|---|---|
| `chunks_manifest.json` | `bf0b8b2f7836c689ef79177bf5f2b3d01599fa3095a7321ee6bc436ddbfe8800` | `23f7b453ac9cefe7ca3bd29d87cd5634eb5790d6064dbabd767f8905fcfa51f0` |
| `documents_manifest.json` | `3f8b4a6b255c8fb7f3547b89f9d60f34642483db621f01643a3a4cb0e58cbcfc` | `4959b1daa175188282d266b4ce81308f7bf96e77d747833d5550e7ee11b68dab` |
| `embeddings_manifest.json` | `c413a3039211dac5c1c9d3988851edf5500b1f95588284f7585fd891bd806aa2` | `76341bc411af2d7700df85a3190113b51db48fe7eb40f8950fd6b7553c7349fe` |
| `chunk_stats.json` | `bf2c696225f96527cf4f722ed9f9ffb545bcfaaa1dbf59c2b12377e55a113c9a` | `c647df1c3d792d4eacab164eba82291af2949872c6c108d5f455e34ac36cdcbc` |

## 3. Frozen evaluation dataset

| File | SHA-256 |
|---|---|
| `data/evaluation/phase7_eval_v1.jsonl` | `f0695189903c8ef8ea7185ac96a78b80f4f96d836c55214bf025176363e295cb` |
| `data/evaluation/phase7_eval_v1.ledger.jsonl` | `b6219e40df37d0a77aa0484fb05df6e91f66db33d9a6d65eddb9c9b0b49b0edb` |
| `data/evaluation/README.md` | `9bf23bc26de48cc8d24d5738f840e6410760bc610f3c3c4a0d7c6646a15006e9` |

## 4. Frozen routing configuration (all arms)

From `.../E1_baseline_comparison__adaptive/config.json`:

```
available_strategies    = [bm25, dense, hybrid, hybrid_rerank]   <-- NOTE: "adaptive" is NOT selectable
escalation_ladder       = [bm25, dense, hybrid, hybrid_rerank]
max_escalation_steps    = 1
sufficiency_threshold   = 0.5
coverage_threshold      = 0.5
top1_coverage_threshold = 0.3
cost_weight             = 0.25
router_version          = rule_based_v1
strategy_cost_ms        = {bm25: 2.25, dense: 451.78, hybrid: 455.97, hybrid_rerank: 3854.41}
```

**Why this matters for Gate 2:** the shipped router can only ever choose among
`[bm25, dense, hybrid, hybrid_rerank]`. The arm named `adaptive` is the router's
*output* over 107 queries, not a strategy a router could select. The original
`oracle_ceiling.json` included `adaptive` in its selectable set; the corrected
`oracle_ceiling_v2.json` excludes it (§8).

Per-arm `config_hash` (p8a = after, p8b = before):

| Strategy | p8a hash | p8b hash |
|---|---|---|
| bm25 | `a264056d19049edd` | `42bda3317b56a047` |
| dense | `31bad5d6f496eff5` | `884e57e21efa6de4` |
| hybrid | `92d8bb4920011209` | `e1b3a7df7b9c978d` |
| hybrid_rerank | `4825cc94f4c1b6d0` | `aca7277eaa10363f` |
| adaptive | `ba9d8bb6f867edba` | `773b9759cf0926bb` |

## 5. Arm input SHA-256 (`experiments/phase8/combined/**/rows.jsonl`)

| Strategy | p8a (after) | p8b (before) |
|---|---|---|
| bm25 | `c3b66d36c793499ff562c4c2c113b800dda2c7c8b825f32ee17676b7a6cb9d01` | `53b506dc005202c0b60edd1d423b28d86f053dd501bad50bdfe207bc187d46a1` |
| dense | `4e7ac1aa3714408877cc2a05d01215b722cca68d43e7124aa356714f6829d4b1` | `73427ed9cc74b7e8b41f2ba3240cccb5b34f7d5cc1627c531d88177dd820a131` |
| hybrid | `1554a69ffc4ed4896c5f2083727989cbf35e78c7b735e65d702fcfa200a9a81b` | `4ada977598f96aa8520d2102e6a9a54329149326287775e9fdbe96724cd525d4` |
| hybrid_rerank | `a8fe1470f78d3fe4b7ab51250621ce7db952a53ca707c25cdee727d9a70b0614` | `9ff6a3e23d7e36fda69aaea8879d353aad342e517afccc00ebb968e719c40ae2` |
| adaptive | `c24295ba36e050c8348407cb641c8a06e7368937027374f1a3c06aa7e7925890` | `7cd9e390181475e2905f7bea94d02cfad65719c10d85d08cc84c570062ac0faa` |

## 6. Artifact SHA-256 (`experiments/phase8/*.json`)

| Artifact | SHA-256 |
|---|---|
| `oracle_ceiling.json` | `bb155994a07a975266666bd64e5ece752ce76a392fdaa2f81c06af3d7e31ef00` |
| `oracle_ceiling_v2.json` (Gate 2, selectable-only) | `8e48e22bb3b02a8fc58dcb3df4d6cccf27a7e95566606660e85dd428ae971e22` |
| `gate3_signal_exhaustion.json` (Gate 3) | `f1f8736f8e9b5f483c84f15a4494f98974191cc08c0d881a8736b361126bcc53` |
| `gold_label_audit.json` | `56a7608ac9019089fc331091a86c6ab05f98d9709774e42671ed0a70a9c34fd7` |
| `gold_label_repair.json` | `2c452940a841f2db22a0f9cad94b96e3f8f7f74fe2dea47c05543b4a44c9646f` |
| `run_integrity.json` | `a22031d8b076a32e32cf20f58dc2dcb093c9fa6e024c323a5f522578a7d3630d` |
| `arm_comparison.json` | `1a4b679d64207b87211779d6ba61b742174a3317e19f80aae671ad9d8be11157` |
| `arm_comparison_clean4.json` | `71b63af3ceefcc7a573afc923943abcd2a2c240f7274e3abf6cafd6537cb3b68` |
| `arm_comparison_full5.json` | `ff03ab67b942adad9996996bfcb76c7a9b604127479a75804015663a4362851d` |
| `column_validation.json` | `3741ee032d37f0023b51be089fd18f4c014bbfb7c08f17dc8ce110d69804ffab` |
| `strategy_cost_ms_phase8_after.json` | `a1388af19ebb358ec3e6c8ce4161d5a622eea884ee0b91a691ffbb4ee09027ac` |
| `strategy_cost_ms_phase8_before.json` | `fa8b07320fa884945409a8d07434e0d239deaa42bbfba7a1ee2265d2dee35846` |

## 7. Verify the snapshot

```bash
# artifacts + arm inputs + corpora + dataset (from repo root)
sha256sum experiments/phase8/*.json \
  $(find experiments/phase8/combined -name rows.jsonl | sort) \
  data/processed_phase8_after/*manifest.json data/processed_phase8_before/*manifest.json \
  data/evaluation/phase7_eval_v1.jsonl data/evaluation/phase7_eval_v1.ledger.jsonl
```

Regeneration (documented, not run here): the corpora come from
`scripts/build_index.py`/`ingestion` over the raw PDFs; the E1 arms from
`scripts/run_phase7_suite.py`; the oracle from `scripts/oracle_routing_ceiling.py`.

**Verified on branch `phase7-closure` at commit `f27348c`.** All 24 recorded
digests recomputed and matched: 11 artifacts (§6, including `oracle_ceiling_v2.json`
`8e48e22b`), 10 arm `rows.jsonl` (§5), 6 corpus manifests and `chunk_stats.json`
(§2), and 3 dataset files (§3). Nothing in the snapshot has drifted.

Determinism re-checked for the v2 oracle: a second run is byte-identical apart
from `generated_at`.

## 8. Gate 2 — the selectable-set correction (v2)

The closure gate requires the routing ceiling to be computed over the
*selectable* strategies only. The v1 artifact (`oracle_ceiling.json`) included
`adaptive` in its selectable set. `adaptive` is the shipped router's **output**
over the 107 queries, not an arm the router may choose (§4), so a ceiling that
may select it is partly circular — it would measure the router partly against
itself.

`scripts/oracle_routing_ceiling.py` now:

* defaults its selectable set to `SELECTABLE_SYSTEMS = (bm25, dense, hybrid,
  hybrid_rerank)`, i.e. the frozen `available_strategies`;
* refuses `--systems` containing `adaptive` unless `--allow-adaptive-in-panel`
  is passed, so the circular configuration cannot be reproduced by accident;
* reports **both** oracle tie-breaks (`latency`, `mrr`), because recall@5 is
  tie-break-invariant but the credited arm and its latency are not, and reading
  one alone hides that;
* adds a **rank-aware frontier** — the recall-only frontier can trade ranking
  quality for latency without that trade being visible;
* adds a **paired bootstrap CI** for the (selection − best-fixed) delta, so the
  "carried by a handful of queries" caveat has an interval attached rather than
  being a prose warning.

It writes `oracle_ceiling_v2.json` by default, so the v1 artifact is not
overwritten. All new parameters are pre-registered module constants (tolerances,
bootstrap resamples/seed/alpha) and are copied into the artifact before any
number is read.

### 8.1 What changed, and what did not

The correction **does not move the headline numbers**:

| | v1 (with `adaptive`) | v2 (selectable only) |
|---|---|---|
| oracle recall@5, `after` | 0.9206 | 0.9206 |
| oracle recall@5, `before` | 0.9486 | 0.9486 |
| oracle MRR, `after` / `before` | 0.8224 / 0.8338 | 0.8224 / 0.8338 |
| best fixed | `dense` | `dense` |

`adaptive` was *selected* on 6 queries (`after`) and 1 (`before`) in v1, but on
every one of those it **tied** the best recall already reachable from the four
real arms, and the latency tie-break credited it. It never raised the ceiling, so
removing it leaves recall@5 and MRR identical. Verdicts are unchanged
(`routing_has_headroom` on both arms).

The credited-arm latency does move, because a different arm is now credited on
those queries:

| figure | v1 | v2 |
|---|---|---|
| oracle mean ms, `after` | 92.63 | 96.35 |
| oracle mean ms, `before` | 128.96 | 129.05 |
| frontier(0.01) saving, `after` | −845.84 ms | −842.11 ms |
| frontier(0.01) saving, `before` | −398.30 ms | −398.21 ms |

**Conclusion:** the `adaptive` leak was a *methodological* defect, not a
numerical inflation. Removing it is required for correctness, and it *confirms*
rather than overturns the Phase 8 §8 ceiling numbers.

### 8.2 New evidence the v2 run adds

* **Delta uncertainty (paired bootstrap, 10,000 resamples, seed 20250102).**
  recall@5 delta CI `after` **[0.0031, 0.0467]** over 4 nonzero queries;
  `before` **[0.0093, 0.0670]** over 6. Both exclude zero, so the sign is solid
  even though the magnitude rides on 4–6 queries.
* **Tie-break sensitivity.** Oracle credited-arm counts, `after`:
  `latency` → bm25 91, dense 10, hybrid 6; `mrr` → bm25 68, dense 24, hybrid 13,
  hybrid_rerank 2. recall@5 is 0.9206 under both, but the oracle **MRR is 0.8224
  under `latency` and 0.9420 under `mrr`**. Against best-fixed `dense` (0.8645)
  that is −0.0421 or +0.0775: the tie-break alone swings oracle MRR by **0.1195**
  and flips its *sign* relative to the best fixed strategy. So §8.5's "the
  oracle costs MRR" is a property of the latency tie-break only, and is now
  separated from the recall ceiling it was previously bundled with.
* **Rank-aware frontier (ε = 0.01, MRR tolerance 0.05/0.10).** Both tolerances
  select exactly the `mrr`-oracle arms above, at mean **326.54 ms** — a
  **−611.93 ms** saving against best-fixed, i.e. **72.7%** of the recall-only
  frontier's −842.11 ms, while carrying MRR 0.9420 in place of 0.8224. The rank
  constraint costs ~230 ms of the saving and buys back 0.1195 MRR: the two
  objectives are not in zero-sum conflict at this tolerance.

Artifact: `experiments/phase8/oracle_ceiling_v2.json`,
sha256 `8e48e22bb3b02a8fc58dcb3df4d6cccf27a7e95566606660e85dd428ae971e22`.
Deterministic: byte-identical across two runs apart from `generated_at`.
Tests: `tests/test_oracle_routing_ceiling.py` — 48 passing (12 added for this
correction).

## 9. Gate 3 — full observable signal exhaustion

### 9.1 What was asked and what was not

Gate 2 established that routing headroom *exists*. Gate 3 asks the second half,
out of sample:

> Can the information a router can actually observe tell it which queries are
> the few where a different retrieval strategy is actually beneficial?

**Methodology.** Offline, deterministic, no index and no API. Labels come from
per-query `recall@5` across the four **selectable** E1 arms (`needed` if nothing
else comes within ε = 0.01 of the best-fixed reference `dense`, else
`sufficient`) — the same definition Gate 2 used. Features come **only** from the
`adaptive` arm's traces, so no oracle outcome can enter the feature side; a test
asserts that no feature row carries `recall_at_5`, `mrr`, `hit_at_5` or
`ndcg_at_5`. The measuring model is a ridge logistic regression (IRLS, λ = 1,
`class_weight=balanced`, no hyperparameter tuning), used only as an instrument:
no decision rule is derived from it, no router is configured from it, and
`src/adaptive_rag/routing/` is untouched.

**Feature set — 37 columns, 4 groups, no new features introduced.** (a) *Query
features* (13 continuous from `QueryFeatures`, analyzer_v1). (b) *Retrieval
feedback* (5: `result_count`, `coverage`, `top1_coverage`, `sufficiency_score`,
`initial_result_count` — observable only after paying for stage one). (c)
*Routing signals* (6: `routing_confidence`, the four per-strategy scores,
`evidence_margin`). (d) *Initial strategy* + categoricals, one-hot
(`initial_strategy` 4 levels, `question_type` 7, `multi_concept` 2).

**Train/evaluate separation.** Fit on the frozen `calibration` split (n = 47);
every reported out-of-sample number comes from the frozen `test` split (n = 60).
Standardisation and class weights are fitted on the fit rows only.

**Statistical tests.** Primary: permutation test of the **whole procedure** —
labels permuted across all queries, then standardise → fit → score → AUC re-run
each time (10,000 resamples, seed 20250103). Effect size: AUC with a stratified
bootstrap CI. Screen: one feature at a time, two-sided permutation on
|AUC − 0.5|, **Holm-Bonferroni** across a family of 24. Power: Hanley-McNeil
analytic.

**Pre-registered decision rule** (module constants, copied verbatim into the
artifact): passes only if, on **both** arms, (i) permutation p < 0.05, (ii) test
AUC ≥ 0.65, (iii) bootstrap 95% CI lower bound > 0.50, and (iv) at least one
feature survives Holm at 0.05.

### 9.2 Result — `no_out_of_sample_signal`

| | `phase8_after` | `phase8_before` |
|---|---|---|
| positives (`needed`) / negatives | **8 / 99** (base rate 7.5%) | **8 / 99** |
| positives in calibration / test | 2 / 47 and 6 / 60 | 3 / 47 and 5 / 60 |
| **out-of-sample AUC** | **0.5340** | **0.5345** |
| bootstrap 95% CI | **[0.2469, 0.8086]** | **[0.2000, 0.8691]** |
| permutation p | 0.4016 | 0.4087 |
| features surviving Holm (of 24) | **0** | **0** |
| min detectable AUC at 80% power | 0.825 | 0.850 |

All four conditions fail on both arms. The two arms agree closely, which is
replication of a null rather than of a positive.

**The overfitting signature is the clearest evidence.** Fit-split AUC is
**1.0000** (`after`) and **0.9621** (`before`) against out-of-sample **0.534** on
both. With 2–3 positives the model separates the fit split perfectly and
generalises to nothing — precisely the failure the out-of-sample protocol exists
to catch, and the reason a Gate 3 reading of the *fit-split* number would have
been catastrophically wrong.

**Power is the binding constraint, so the null is a power statement.** At 6
positives against 54 negatives, a true AUC of 0.60 has 12% power to produce a CI
excluding chance, 0.80 has 76%, and only ≥0.825 reaches 80%. The honest reading
is therefore "no reliable out-of-sample signal **at this sample size**", not
"no signal exists".

**The retrieval-feedback signal runs the wrong way, on both arms.**
`top1_coverage` is the strongest feedback feature and is *inversely* associated
with needing `dense` (AUC 0.346 / 0.357, i.e. oriented 0.654 / 0.643). Mean
`coverage` on the 8 positives is 0.803 vs 0.808 on the 99 negatives — the
sufficiency check reports "terms found" on precisely the queries where the cheap
stage's terms were found *and the reference was still the only acceptable
answer*. This is the same inversion Gate 2 recorded, now measured out of sample.

**Which queries drive it (per-query evidence preserved in the artifact).** The 8
positives are heterogeneous, spanning `factual`, `conceptual`, `comparative`,
`multi_document`, `terminology` and `fine_grained`, with `query_length_words`
19–66 and `coverage` 0.53–1.00. No single category or feature separates them,
and `content_term_count` — the strongest single feature in `after` (AUC 0.711,
raw p = 0.045) — collapses to AUC 0.562 in `before`. That instability across
arms is itself the finding: the strongest feature on one corpus is not the
strongest on the other.

Artifact: `experiments/phase8/gate3_signal_exhaustion.json`, including a
107-row per-query evidence table (split, label, all 37 feature values, oracle
gain, credited winner, model score). Tests:
`tests/test_gate3_signal_exhaustion.py` — 32 passing.

### 9.3 What the verdict permits

`no_out_of_sample_signal` ⇒ **oracle headroom exists (Gate 2) but is not
identifiable from the currently available observable signals.** Gate 4 and a
learned router are *not* licensed. The rule-based router's failure is not
evidence that a learned router would succeed; on this evidence a learned router
would be fitted on 8 positives and would inherit the fit-split AUC of 1.0 with
none of it surviving contact with held-out data.

The corpus was not rebuilt, `strategy_cost_ms` was not changed, ADR-028 was not
promoted, and the 42 spliced section paths remain outside the primary routing
closure experiment.
