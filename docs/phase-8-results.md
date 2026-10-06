# Phase 8 Results — Corpus Fix Before/After Study

Status: quality study **complete**. Cost freeze **pending** (needs egress).
Scope note and provenance limits are stated in §0 and must travel with any
number quoted from this document.

---

## 0. What this study can and cannot say

Three constraints bound every result here. They are stated first because each one
would otherwise look like an ordinary result.

1. **The comparison family is post-hoc.** Phase 7's E1 ordering was observed
   before this family was specified (`docs/phases/phase-8.md` §2.3). These are
   pre-registered comparisons of a *program that was itself motivated by that
   ordering*, so they are exploratory within the program. What Phase 8
   establishes is that the corpus defect does **not** explain Phase 7's result —
   not a replication of Phase 7's result on a clean corpus.
2. **`nDCG@5` and E8 are not measured.** The Step 4 audit found 47 of 140 gold
   section labels spliced and 40 unresolvable on the fixed corpus
   (`experiments/phase8/gold_label_audit.json`). Every nDCG number here would be a
   comparison between two corrupt label sets rather than between two corpora. The
   exclusion was decided from the label audit, before any after-arm quality number
   existed. §2.5 of the phase doc made it a falsification condition.
3. **13 of 29 arms in the first sweep were silently invalid** and were re-run.
   Their directories are retained as evidence. See §5 — this is the most important
   entry in this document.

The metrics that survive all three constraints are `recall_at_5`, `mrr` and
`hit_at_5`, which key on `relevant_documents` only and are unaffected by the
section-label defect (`evaluation/retrieval.py:134,151`).

---

## 1. Headline

**The corpus defect was real, was fixed to a measured standard, and did not
change retrieval quality — and did not rescue the reranker.**

Every arm's quality metric is unchanged within noise across the fix. The
reranker, the arm this phase was convened to examine, moved by −0.0016
recall@5 while its penalty against `hybrid` stayed exactly where it was.

Reading against the phase doc's §2.4 answer table:

| Pre-registered outcome | Reading | Verdict |
|---|---|---|
| `hybrid_rerank` improves materially, overtakes `hybrid` | corpus was a major cause; ADR-028 withdrawn | **ruled out** |
| `hybrid_rerank` improves, still ranks last | corpus was a real contributor; ADR-028 stands narrowed | **ruled out** |
| `hybrid_rerank` unchanged | corpus defect was not the mechanism; finding stands as model/domain mismatch | **confirmed** |
| quality ordering otherwise unchanged | "dense dominates hybrid" was not a corpus artifact | **confirmed** |

Both outcomes that would have *rescued* the reranker are excluded by measurement.
That is a negative result, and it is the phase's finding.

---

## 2. E1 — the five arms, both corpora

n=107 per arm, retrieval-only, Holm-Bonferroni within each metric across all
C(5,2)=10 pairs. Full output: `experiments/phase8/arm_comparison_full5.json`.

### 2.1 Quality, before → after

| arm | recall@5 | MRR | Hit@5 |
|---|---|---|---|
| `bm25` | 0.7757 → 0.7866 | 0.7030 → 0.7313 | 0.8224 → 0.8505 |
| `dense` | 0.9143 → 0.8988 | 0.8714 → 0.8645 | 0.9533 → 0.9346 |
| `hybrid` | 0.8489 → 0.8583 | 0.8593 → 0.8678 | 0.9159 → 0.9252 |
| `hybrid_rerank` | 0.7072 → 0.7056 | 0.6016 → 0.6021 | 0.7664 → 0.7664 |
| `adaptive` | 0.8489 → 0.8583 | 0.8535 → 0.8664 | 0.9159 → 0.9252 |

### 2.2 Before vs after, paired, Holm across the five arms

| Metric | bm25 | dense | hybrid | hybrid_rerank | adaptive |
|---|---|---|---|---|---|
| recall@5 | +0.0109 (p=1.00) | −0.0156 (p=1.00) | +0.0093 (p=1.00) | **−0.0016 (p=1.00)** | +0.0093 (p=1.00) |
| MRR | +0.0283 (p=0.43) | −0.0069 (p=1.00) | +0.0085 (p=1.00) | **+0.0004 (p=1.00)** | +0.0130 (p=1.00) |
| latency_ms | +0.01 (p=1.00) | **+411 (p<0.001)** | +98.8 (p=0.16) | **+112 (p<0.001)** | +12.8 (p=0.35) |

**No quality delta on any arm survives Holm correction.** Four of five arms nudge
up on recall, `dense` nudges down, and none is distinguishable from zero at
n=107.

The two significant results are both latency, and **neither of them measures the
corpus.** See §8.2: `dense` (+411 ms) and `hybrid_rerank` (+112 ms) are dominated by
the query-embedding API call, which embeds the query rather than the corpus and
therefore cannot be affected by which text is in the index. The `bm25` control arm,
which makes no API call at all, moved **1.83 → 1.85 ms**; the reranker's own local
ONNX stage moved **−2.3 ms**. The corpus fix's measured effect on latency is
approximately zero, and the significance is an artifact of pairing on a
provider-latency signal.

`hybrid_rerank`'s near-perfect tie is worth stating precisely, because "no change"
is doing real work here: on recall@5 it had 4 wins, 4 losses and **99 ties** out of
107 queries; on MRR, 19 wins, 21 losses, 67 ties. The corpus text changed around
it and its ranking behaviour did not respond at all.

### 2.3 Ordering is unchanged, and the reranker stays last

| arm | recall@5 rank, before | recall@5 rank, after |
|---|---|---|
| `dense` | 1 (0.9143) | 1 (0.8988) |
| `hybrid` | 2 (0.8489) | 2 (0.8583) |
| `adaptive` | 2 (0.8489) | 2 (0.8583) |
| `bm25` | 4 (0.7757) | 4 (0.7866) |
| `hybrid_rerank` | 5 (0.7072) | 5 (0.7056) |

`hybrid` and `adaptive` are numerically identical on recall@5 and Hit@5 in both
arms and statistically indistinguishable on every quality metric (recall delta
exactly 0.0000, p=1.00). The router picks `hybrid` for effectively this whole
benchmark, so it buys nothing over fixed `hybrid` here while costing ~132 ms less
than the reranking path it declines to take.

---

## 3. The reranker, in detail

`hybrid_rerank` vs `hybrid`, within each arm, Holm-corrected:

| | metric | `hybrid` | `hybrid_rerank` | delta | p_adj |
|---|---|---|---|---|---|
| before | recall@5 | 0.8489 | 0.7072 | −0.1417 | 0.0139 |
| after | recall@5 | 0.8583 | 0.7056 | −0.1526 | 0.0007 |
| before | MRR | 0.8593 | 0.6016 | −0.2577 | <0.0001 |
| after | MRR | 0.8678 | 0.6021 | −0.2657 | <0.0001 |

It also loses significantly to `dense` on both arms (−0.207 / −0.193 recall@5) and
to `bm25` on MRR (−0.101 / −0.129).

Its latency cost is unchanged and is **entirely local computation**, which is what
makes it the most trustworthy number in the phase: the cross-encoder stage is
6.2x hybrid's median on the fixed corpus (3061.78 → 3059.48 ms of rerank stage,
against hybrid's ~577 ms), and that stage moved by **−2.3 ms** across the fix. No
API call is involved, so unlike every other latency figure here it is not exposed
to provider variance.

The penalty **did not shrink — it grew slightly, on both metrics.** A corpus defect
that was making the reranker look bad would have had to *improve* when the defect
was removed. It did not. So the corrupted reading order was not the mechanism, and
what remains is a model/domain mismatch: a cross-encoder trained to score passages
is being asked to rank chunks out of a 14-paper IR corpus, and it returns
orderings worse than the fusion that produced its candidates, on clean text and by
the same margin as on garbled text.

**ADR-028 is therefore confirmed on its central claim** — the rerank rung earns
none of its cost — but the *justification* recorded in ADR-028 should not be
treated as validated, because Phase 8 cannot reproduce the corpus-defect story
Phase 7 told. The conclusion survives; the reasoning behind it does not.

---

## 4. E2–E5 — routing, features and sweeps

Both arms, 47 calibration queries, all traces clean.

### 4.1 E2 — escalation ablation

Quality is **identical across all three variants** in both arms (recall@5 0.8511
before, 0.8830 after). Ten paired comparisons per arm, **zero significant** in
either. The sufficiency gate and bounded escalation change no quality outcome;
they only move latency (A 477 / B 500 / C 451 ms before). This reproduces Phase
7's E2 conclusion on the new corpus.

### 4.2 E3 — feature ablation

30 paired comparisons before, **zero significant**. No routing feature earns or
costs measurable quality. Also reproduces Phase 7.

### 4.3 E4 / E5 — sweeps

| study | before | after |
|---|---|---|
| `sufficiency_threshold` 0.3–0.6 | recall@5 0.8511 for all four | recall@5 0.8830 for all four |
| `sufficiency_threshold` 0.7 | 0.8511 | **0.8617** (a real drop) |
| `cost_weight` 0.0–1.0 | recall@5 0.8511 for all five | recall@5 0.8830 for all five |

Two things to flag rather than round off:

* **`sufficiency_threshold=0.7` now costs quality** (0.8830 → 0.8617) where on the
  old corpus it was free. The stricter gate escalates more often, and escalation
  ends at the reranker, which §3 shows degrades ordering. This is the first place
  in Phase 8 where the reranker's harm and the router interact. It is a
  single-arm difference with no paired test reported here, so it is a lead, not a
  result.
* **The calibration split as a whole improved** (0.8511 → 0.8830 on E2's identical
  variants). That is a real before/after gap on the 47 calibration queries, and it
  is *not* visible in E1's 107-query numbers, where every delta was noise. The
  difference is the query set: the calibration split is a subset of `--split all`,
  so the same 47 queries moved more than the full 107 did. Reported as an
  observation, not a finding — no paired test covers it, and a split that behaves
  differently from the whole is a prompt to check the split, not a conclusion.

### 4.4 `max_escalation_steps` remains unanalysable

E4's `max_escalation_steps=0..3` variants are absent from `per_variant` in both
arms. Phase 7 recorded this sweep as unanalysable rather than null, for reasons
that the fixed corpus does not address. It stays unanalysable, and Phase 8 does
not claim it otherwise.

---

## 5. Integrity finding — 13 of 29 arms were invalid

**This is the most consequential result in Phase 8, and it is a process finding
rather than a retrieval one.**

The first full sweep reported `ok` and a full trace count for all 29 arms. 13 of
them carried failed traces — every one `EmbeddingAPIError: Connection error` at
the retrieval stage — some at **0 of 47 queries**:

| suite | arms affected | ok / total |
|---|---|---|
| `p8b_e1` | `hybrid_rerank` | 59 / 107 |
| `p8b_e2e3` | `B`, `C`, `full`, 6× `without_*` | 0–5 / 47 |
| `p8a_e2e3` | `A`, `B` | 0 / 47 |
| `p8a_e2e3` | `C` | 33 / 47 |

### 5.1 Why nothing caught it

* The reranker embeds every candidate, so one dropped embedding call fails the
  entire trace. `rerank_fallback` is opt-in and was off, so no trace degraded into
  a partial result — each either scored or died.
* Every arm routing through the terminal rung died with it, which is why the
  adaptive and escalation arms collapsed while fixed `bm25`/`dense`/`hybrid` were
  untouched. The blast radius followed the architecture.
* **`metrics_retrieval.json` reports a contaminated arm as `n=59` beside
  `trace_count=107`, with no error field.** Consumers trusted the aggregate.
* **The contamination was asymmetric between arms**, so the two arms did not even
  cover the same queries: a naive before/after would have paired 107 against 59.

### 5.2 What it would have produced

The before-arm `hybrid_rerank` was reported at **0.6384** recall@5 — the mean over
the 59 queries that survived. Its true mean over all 107 is **0.352**, and the
clean re-run of the same arm on the same corpus gives **0.7072**.

So the contaminated arm did not merely shrink the sample, it **inverted the sign
of the headline**. Set against the after arm's 0.7056, the reported 0.6384 reads
as a **+0.067 improvement**; the true 0.352 against 0.7056 reads as a **−0.355
regression**. A paired test on the 59 survivors would have reported the improvement
as significant (it survives Holm at p_adj < 0.0001), and the regression direction
is the one Phase 8 actually reports.

Either way the number would have landed on a §2.4 answer row — the contaminated one
on *"corpus was a real contributor, not the whole cause; ADR-028 stands on narrowed
grounds"*, the true one on the opposite row. The conclusion would have been
**wrong in its reasoning and right in its recommendation**, for entirely the wrong
reason, published as a measurement.

### 5.3 Response

* All 14 affected arms re-run at `--pace 1.5` (raised from 0.75) — **all returned
  clean at full trace counts**, confirming rate limiting rather than a pipeline
  defect. The pace value is the fix; the failure was self-inflicted by running the
  sweep too fast.
* `scripts/compare_phase8_arms.py` now **refuses a contaminated arm** instead of
  averaging over survivors. `--allow-contaminated` exists to inspect damage, not to
  quote it, and records the shortfall in the artifact.
* `experiments/phase8/run_integrity.json` holds the full audit, with each
  contaminated run mapped to the run that supersedes it.

The generalisable lesson: **trace status, not `trace_count` and not a metric's
own `n`, is what establishes that a measurement covers the query set.** Two of the
three numbers available said "complete".

---

## 6. Cost freeze — and why the latency findings had to be withdrawn

Cost was re-measured per arm (`strategy_cost_ms_phase8_{before,after}.json`, 5
repetitions x 20 queries, 5 warm-up discarded, 100 samples per strategy). This
turned out to matter for the interpretation of §2.2 rather than for the router.

### 6.1 The corpus cannot change these numbers

The cost artifacts break each arm down by stage, and the breakdown settles the
question the paired tests could not:

| stage | API? | before | after | Δ |
|---|---|---|---|---|
| `bm25` search | no | 2.48 | 2.12 | **−0.36** |
| `hybrid` bm25 + search | no | 20.10 | 21.50 | +1.40 |
| `rerank` bm25 + search | no | 20.16 | 20.04 | **−0.12** |
| `rerank` ONNX stage | no | 3057.83 | 3072.79 | +14.96 (**+0.5%**) |
| `dense` query embedding | **yes** | 352.42 | 481.60 | **+129.18** |

Every stage that could respond to the corpus changed by ~1% or less. Every stage
that changed materially makes a **query-embedding API call** — and that call
embeds the *query*, not the index, so the corpus fix cannot reach it.

Two independent controls agree:

* **E1's `bm25` arm** makes no API call and moved **1.83 → 1.85 ms** across the fix.
* **The two arms disagree about the direction of dense's deviation from the
  common Phase 5 seed table** — the before arm says dense is 77.74 ms *faster*
  than the seed, the after arm says 46.96 ms *slower*. A corpus change with a fixed
  direction cannot produce opposite-signed deviations from one reference on one
  machine.

So the +411 ms on `dense` and +112 ms on `hybrid_rerank` in §2.2 are **provider
latency variance, not corpus effect**, and the paired tests detected real signal in
the provider. Those two results are withdrawn as corpus findings. The reranker's
6.2x cost penalty is unaffected — it is local computation (§3).

### 6.2 Freeze decision

**`NOTHING IS FROZEN`, and the recommendation is to freeze the after-arm table
only if a freeze is wanted at all.** Both artifacts report
`"NOTHING HAS BEEN FROZEN"` and compare against the Phase 5 seed.

The two arms propose *different* tables (dense −77.74 ms before, +46.96 ms after),
and §2.1's rule is one frozen value **per corpus version** — but both arms share
`corpus_c5996129918d7e44`. That is not an ambiguity to resolve by picking the
average: the **after arm is the shipping corpus**, so if a table is frozen it is
the after-arm table, and the before-arm artifact is a study input rather than a
candidate.

Two caveats on that table:

* **It is thin.** 20 queries x 5 repetitions on the *legacy* Phase 2-6 dataset
  (`dense_eval_v1.jsonl`), not the frozen 107-query benchmark — that is
  `measure_strategy_cost.py`'s inherited default from Phase 7.0b, not a Phase 8
  choice, but it should be stated wherever the numbers appear.
* **Its error is currently inert.** The router divides each cost by the maximum, so
  the after table normalises to bm25 0.0006 / dense 0.141 / hybrid 0.139 /
  rerank 1.0. E5 shows all five `cost_weight` values produce identical recall@5
  (0.8830), because the sufficiency gate never opens (§4.3) — so a ±130 ms error in
  an API-bound entry cannot currently change any routing decision. It would start
  to matter the moment the gate is recalibrated per ADR-028 Finding 2, which is
  exactly when the table should be re-measured on the full benchmark.

## 7. Artifacts

| Artifact | Contents |
|---|---|
| `experiments/phase8/column_validation.json` | Step 1 gates: residual interleaving, spot check, regression, reading order, chunk delta |
| `experiments/phase8/gold_label_audit.json` | Step 4: 47/140 damaged section labels, per-label resolution before/after |
| `experiments/phase8/run_integrity.json` | §5 audit, contaminated runs and their supersessions |
| `experiments/phase8/arm_comparison_full5.json` | §2–3, the pre-registered family on all five arms |
| `experiments/phase8/combined/{p8b_e1,p8a_e1}/` | the five clean E1 arms per corpus, as compared |
| `experiments/phase8/analysis_{before,after}_*/` | E2–E5 per-arm analyses |
| `experiments/phase8/strategy_cost_ms_phase8_{before,after}.json` | §6, per-arm cost by stage; neither frozen |
| `experiments/phase8/oracle_ceiling.json` | §8, the routing ceiling, frontier, detectability and the pre-registered gate verdict, per corpus |
| `scripts/oracle_routing_ceiling.py` | §8 driver; offline, and reuses the §5 contamination guard |
| `experiments/phase8/gold_label_repair.json` | label repair grading and the offline nDCG@5 recomputation |
| `scripts/repair_phase8_gold_labels.py` | label repair driver; offline, rewrites nothing |
| `docs/strategy-cost-freeze-decision.md` | the `strategy_cost_ms` freeze recommendation |
| `scripts/compare_phase8_arms.py` | the family driver, with the contamination guard |
| `data/processed_phase8_{before,after}/` | the two corpora (618 / 613 chunks) |

## 8. Oracle routing ceiling — is there any headroom for routing at all?

Full artifact: `experiments/phase8/oracle_ceiling.json`.
Driver: `scripts/oracle_routing_ceiling.py`. Tests:
`tests/test_oracle_routing_ceiling.py`. Offline and deterministic — verified
byte-identical across two runs apart from the timestamp. No index, no API.

> **Correction (v2).** The selectable set in the artifact above included
> `adaptive`, which is not correct: `adaptive` is the router's **output**, not a
> strategy it may select (`docs/phase7-closure-evidence.md` §4). The corrected
> run `experiments/phase8/oracle_ceiling_v2.json` uses the four selectable arms
> only. **recall@5, MRR, and every gate verdict are unchanged** — `adaptive` was
> credited on 6 queries (`after`) / 1 (`before`) but only as a *tie* at the
> recall maximum, so it never raised the ceiling. Only the credited-arm latencies
> move, and three figures below reflect that: oracle mean ms 92.63 → **96.35**
> (same for the frontier rows), and frontier(0.01) saving −845.84 → **−842.11 ms**
> (`after`), −398.30 → **−398.21 ms** (`before`). Full correction, the added
> bootstrap CI, both tie-breaks and the rank-aware frontier: `docs/phase7-closure-evidence.md` §8.

### 8.1 Why this question comes first

Phase 7 concluded `adaptive ≡ hybrid`. That conclusion is ambiguous between two
very different worlds:

1. routing by query characteristic is worthless on this corpus; or
2. the router **never routes**, so the equivalence was forced before routing was
   ever tested.

World 2 is the live one, and all three of the plan's structural claims check out
in the code:

* the initial pick is `hybrid` on **104 of 107** queries, and the sufficiency
  gate never fires at 0.5 (Phase 7 E8: observed sufficiency never drops below
  0.6364);
* `next_strategy("hybrid")` returns `None` once the ladder terminates, so
  escalation *from* the default pick is a recorded no-op
  (`routing/escalation.py:71-76`);
* the ladder orders `bm25 < dense < hybrid < hybrid_rerank`
  (`schemas/config.py:380`), so escalation can only ever climb **away** from
  `dense` — the strategy with the best measured recall@5.

The literal ADR-028 re-run would therefore have measured the escalation
mechanism, not the routing idea. This section measures the ceiling instead: the
best any per-query selection over the five strategies could achieve, which is
fully offline because the five clean E1 arms already contain every strategy's
per-query outcome. The bar is the **best fixed** strategy — `dense` — not
`hybrid`.

### 8.2 The pre-registered gate, and the results

Declared as module constants and copied into the artifact before any number was
read:

* **quality headroom** — oracle recall@5 − best-fixed recall@5 **> +0.02**; or
* **latency headroom** — frontier(ε=0.01) mean latency − best-fixed mean
  latency **< −100 ms**, *and* the needed-vs-sufficient split is predictable
  from `category` or the recorded sufficiency signals;
* **exhausted** — oracle − best-fixed ≤ +0.01 **and** frontier(0.01) saves
  < 50 ms.

| policy | recall@5 | MRR | Hit@5 | mean ms | median ms |
|---|---|---|---|---|---|
| bm25 | 0.7866 | 0.7313 | 0.8505 | 1.98 | 1.85 |
| **dense** (best fixed) | 0.8988 | 0.8645 | 0.9346 | 938.46 | 632.08 |
| hybrid | 0.8583 | 0.8678 | 0.9252 | 663.33 | 577.12 |
| hybrid_rerank | 0.7056 | 0.6021 | 0.7664 | 3626.51 | 3595.22 |
| adaptive | 0.8583 | 0.8664 | 0.9252 | 530.91 | 479.61 |
| **ORACLE** (per-query) | **0.9206** | 0.8224 | **0.9626** | 96.35 | 2.03 |
| frontier ε=0.005 / 0.01 / 0.02 | 0.9206 | 0.8224 | 0.9626 | 96.35 | 2.03 |

`after` arm, n=107. The `before` arm replicates every ordering: oracle 0.9486,
best fixed `dense` 0.9143, delta **+0.0343**, frontier saving −398.21 ms.

The gate returns **`routing_has_headroom`** on both corpora — via the quality
branch, at +0.0218 and +0.0343 against a +0.02 threshold. **That verdict should
not be read at face value**, for the three reasons in §8.3–§8.5.

### 8.3 The ceiling is real but carried by four queries

`recall_at_5` on this benchmark takes only **four distinct values**
(`0, 1/3, 1/2, 1`) because 92 of the 107 queries carry exactly one relevant
document — recall can only be 0.0 or 1.0 for them. Five strategies therefore
**tie at the maximum on 98 of 107 queries**, and the latency tie-break sends
every one of those to bm25.

The consequence is that the oracle's mean advantage is not a broad improvement:

| | queries carrying gain | share | total gain mass | unique-max queries |
|---|---|---|---|---|
| `after` | **4** / 107 | 3.7% | 2.333 | 1 |
| `before` | 6 / 107 | 5.6% | 3.667 | 2 |

The after-arm delta clears the +0.02 threshold by **+0.0018** — roughly two
queries changing sides. The concentration block is in the artifact and printed by
the script precisely so this number cannot be quoted without it.

### 8.4 The latency prize is not detectable, so the latency branch fails

At ε = 0.01 the split is **8 queries `needed` / 99 `sufficient`**, and nothing
separates the two groups:

| test | after | before |
|---|---|---|
| `sufficiency_score` mean, needed vs sufficient | 0.8059 vs 0.8248, p = 0.4989 | 0.8053 vs 0.8220, p = 0.5426 |
| `coverage` mean | 0.8028 vs 0.8078, p = 0.9078 | 0.8005 vs 0.8047, p = 0.9193 |
| `category` association | p = 0.0567 | p = 0.3135 |
| sufficiency-signal association | no test reached α = 0.05 | no test reached α = 0.05 |

Max mean separation 0.019 against a pre-registered minimum effect of 0.05. Note
the **direction** as well: `sufficiency_score` is *higher* on the sufficient
group, so the signal runs backwards for this purpose, and mean reference latency
is *not* higher where `dense` is needed (736.65 ms needed vs 954.77 ms
sufficient) — the expensive strategy is not concentrated on the queries that
need it.

The gate's latency branch requires detectability, so it returns **false**
despite the −842.11 ms headline. The saving exists only under a signal that does
not exist.

### 8.5 The oracle costs MRR, and the gate does not price that

The oracle's **MRR is worse than the best fixed strategy's**: 0.8224 vs 0.8645
(`after`), 0.8338 vs 0.8714 (`before`). Its recall@5 and Hit@5 are higher only
because it maximises recall per query and ignores rank. The ε=0.01 frontier
picks bm25 on 91 of 107 queries, costing ranking quality the pre-registered gate
does not measure.

This is a limitation of the **gate**, not a reason to discard it: the gate was
registered on recall@5 and latency only, exactly as the plan specified. It is
stated here so `routing_has_headroom` is not read as "routing is free".

### 8.6 What this settles, and what it does not

* **Settled:** a per-query oracle *can* beat the best fixed strategy on recall@5
  here (+0.0218 / +0.0343). The ceiling is not zero, so "routing is worthless on
  this corpus" is **not** established.
* **Settled:** that ceiling is **not reachable by the existing sufficiency
  mechanism**. No category, signal, score, or coverage association separates
  needed from sufficient at α = 0.05, and the two scalars run the wrong way.
* **Not settled:** whether a different feature set or a learned router could
  find those 4–6 queries. That is the learned-router question Phase 7 deferred,
  and 4–6 positive examples is far too few to fit one without overfitting this
  benchmark (ADR-027's held-out-split problem).
* **Not settled:** any real latency saving. The −842.11 ms is an oracle figure
  gated on a detectability test that failed.

**Recommendation: do not run Stage 2's confirmatory re-run as a routing test.**
It is still worth running to characterise the escalation mechanism directly,
which is a different question from whether routing can pay. Under ADR-028's
rerank-rung removal, escalation from `hybrid` becomes a no-op by construction, so
Stage 2 would measure an unreachable path.

**Stage 2 prerequisite — answered.** `build_experiment_config` **can** express
the reduced strategy set: a `RoutingConfig(available_strategies=[bm25, dense,
hybrid], escalation_ladder=[bm25, dense, hybrid], sufficiency_threshold=0.7)`
constructs, validates, and is accepted by `build_experiment_config(routing=...)`
(verified). No code change is needed to *express* it. However
`scripts/run_phase7_suite.py` — the command the plan specifies — exposes **no**
routing flags, so the override cannot be passed through it; only
`scripts/run_experiment.py` has `--routing-strategies` / `--routing-ladder` /
`--sufficiency-threshold`. Running Stage 2 through the suite as written would
silently use the shipped 0.5 threshold and the four-rung ladder. This is flagged
rather than changed, per the plan.

### 8.7 Deviations from the plan, and one methodology note

* The plan pointed detectability at `routing.metadata.signals`; the traces store
  signals at **`routing.sufficiency.signals`**. The script reads the path the
  traces actually use.
* The frontier and detectability reference is the **best-fixed strategy**
  (`dense` here) rather than a hard-coded `dense`, so the reference cannot
  silently disagree with the measured winner.
* The gate's middle band — a delta of +0.015 with a 70 ms saving satisfies
  neither threshold — is reported as **`inconclusive`** rather than rounded to a
  verdict. Both corpora landed outside that band, so it did not trigger here.
* `nDCG@5` excluded throughout, as in §0.

**Methodology note.** The contingency test originally permuted labels *within
each row*, which is the natural-looking shortcut but has almost no power on a 2×2
table: a 2×2 table has only four row-internal rearrangements, two of which
reproduce the observed statistic, so p is floored near 0.5 however strong the
association. Pooling all observations and re-splitting at the original group
sizes gives the correct conditional null and a p-value that can approach zero.
The fixed version reports `category` p = 0.0567 where the broken test produced
1.0000 — the fix moved a result materially, which is why it is recorded. Both
variants are covered by tests.

## 9. Open items

1. **Cost table not frozen** (§6). Recommendation now made and **awaiting a user
   call**: do not freeze. The two arms' 25% disagreement on `dense` is provider
   variance on the query-embedding call, and it shrinks to 5.3% on the normalised
   ratio the router actually consumes — a 0.008 score-penalty shift against rule
   weights of 0.2–1.5. Evidence: `docs/strategy-cost-freeze-decision.md`.
   `strategy_cost_ms` remains untouched.
2. **`nDCG@5` and E8 remain not measured**, and the repair route is now
   **measured and found wanting** (`experiments/phase8/gold_label_repair.json`):
   of 47 corrupt labels only **22** are safely repairable, not the 37 the audit
   called high-confidence — 15 of those 37 have no clean reconstruction, and 3
   resolve only to paths that are themselves spliced. Separately, the **fixed
   corpus still carries 42 spliced section paths** (from 52), so repairing labels
   alone cannot make `nDCG@5` trustworthy. Unblocking it needs the corpus's
   section paths fixed first, then a human read of the remainder.
3. **`sufficiency_threshold=0.7` regression** (§4.3) needs a paired test before it
   is reported as a finding.
4. **ADR-028** revised: the recommendation stands, the corpus-defect rationale does
   not.
5. **Stage 2 cannot run through `run_phase7_suite.py` as specified** (§8.6). The
   reduced strategy set is expressible in `RoutingConfig`, but the suite runner
   exposes no routing flags, so it would silently use the shipped 0.5 threshold
   and the four-rung ladder. Flagged, not changed.
6. **Latency measurement hygiene.** Provider-latency variance is large enough
   (±130 ms on API-bound stages) to produce paired-test significance on a signal
   the corpus cannot influence. Any future latency comparison on this hardware
   needs a no-API control arm and stage-level breakdown, not just `total_latency_ms`.