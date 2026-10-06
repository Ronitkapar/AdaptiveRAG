# Phase 13 Results — Next Research Direction

Selection: **NEW RESEARCH TRACK** — cheap-first adaptive retrieval
(BM25 → Dense need prediction).
Plan of record: [`docs/phases/phase-13.md`](phases/phase-13.md).
Phases 8–12 unchanged. No retrieval, embeddings, API calls, router,
threshold, classifier, or code change; only read-only frozen-row counts.

## 1. Project-state model (from the repository)

**Implemented.** Four fixed strategies (`bm25`, `dense`, `hybrid` RRF,
`hybrid_rerank` ONNX cross-encoder) + adaptive orchestration layer
(rule-based router, `QueryFeatures`, sufficiency checker, escalation
ladder) over one canonical corpus; strategy-agnostic evaluators +
suite driver/registry/row export + statistics (Holm, bootstrap,
permutation) + dispersion/agreement/signals/escalation/mechanism modules;
107-query benchmark (`phase7_eval_v1`, 47 calib / 60 test, sha
`f0695189903c8ef8`) × dual corpus arms; offline fakes for all arms.

**Validated.** Fixed-strategy measurement (E1), cost table (medians;
`adaptive` excluded), gate harness, dual-arm replication throughout.

**Failed.** Shipped adaptive (gate never fires → collapses to hybrid +
59 ms); DDR escalation (AUC 0.55, dominated by dense-only and random);
Gate-3 observable-signal search (AUC 0.534, 0/24 Holm).

**Ruled out.** Rerank rung (ADR-028); DDR routing; retrieval-state-only
post-dense Dense→Hybrid routing on this setting (Position C).

**Plausible, untested.** Evidence-sufficiency judgment (the shipped
checker never fired — never evaluated); cheap-first BM25→Dense need
prediction; other rungs/regimes/benchmarks.

**Unexplored.** Generation-grounded answerability (judge exists, cached,
never used for routing); dense-weak regimes; headroom-designed
benchmarks; joint threshold×weight calibration (flagged, never run).

## 2. New feasibility evidence (read-only frozen-row counts)

| Property (recall@5, ε=0.01) | after | before |
| --- | --- | --- |
| Dense mean / BM25 mean / Hybrid mean | 0.8988 / 0.7866 / 0.8583 | 0.9143 / 0.7757 / 0.8489 |
| Dense < 1.0 (insufficient class) | 14 | 13 |
| Dense−BM25 ≥ ε (**cheap-first positives**) | **16** (calib 5 / test 11) | **21** (calib 10 / test 11) |
| BM25 ≥ Dense (cheap suffices or ties) | 91 (both-at-max: 77) | 86 (both-at-max: 76) |
| Dense−Hybrid ≥ ε (old-track positives, cf.) | 4 | 4 |

Three facts change the feasibility picture: (i) 4–5× more positives than
any old-track framing (4–8); (ii) the decision sits **before** the
embedding API call (~2 ms local BM25 vs ~450 ms provider call) — the
savings currency is API calls avoided, not 31 ms of local fusion;
(iii) the rung is EV-positive (dense beats BM25 +0.11–0.14 R@5, both
arms) — the first rung in the project with upside worth routing to.
A cheap-first policy study is fully offline-evaluable from frozen rows,
exactly as Phase 10 was.

## 3. Resolved sub-questions

**Sufficiency operationalization (§7).** O1 lexical coverage: tested,
runs the wrong way (AUC 0.346/0.357) — rejected. O2 score confidence:
tested, ranges overlap — rejected as standalone. O3 label-derived
recall-at-max: valid as an *oracle* (labels score the rule, never enter
it — Phases 10/11 discipline); on 92/107 single-doc queries sufficiency
is binary hit/miss, which is exactly the practical question (is the
answer document present?). O4 generation-grounded answerability:
decision-time-possible via a judge call but changes the cost model and is
unvalidated — deferred, not adopted. Verdict: sufficiency is
operationalizable as O3-oracle + decision-time query/BM25-state inputs,
without circularity — as the *framing* of the new track (§4), not a
separate track.

**Stopping vs escalation (§8).** Genuinely different as a *measurement*
problem (sufficiency needs no counterfactual rung outcome; oracle from
current-evidence labels alone) but converges at *action* time: CONTINUE
needs a useful rung, and none exists dense-side. So stopping is the right
formulation of the new question ("stop at BM25 vs pay for dense"), not a
rival track. The old failure is thereby sidestepped structurally rather
than relitigated: new decision point (T0/T1, pre-cost), new rung
direction, new currency.

**Interventions (§9).** `bm25` ~2 ms local, weaker (−0.11–0.14 R@5 vs
dense); `dense`/`hybrid` ~450–720 ms API-dominated, dense best fixed;
`hybrid_rerank` ~3.5 s, last on quality. Only BM25→Dense has meaningful
headroom. Project constraint documented: on this stack, adaptive
retrieval can only mean deciding when to pay for Dense — every other
rung is flat or negative.

**Benchmark (§10).** 107 queries; dense-insufficient 13–14; old-track
usable cases 4–6; cheap-first usable cases 16–21 with 86–91
cheap-suffices; diversity thin (8 Gate-3 positives over 6 categories; E7
hard cells small). Verdict: the current benchmark **supports** a
cheap-first study and **cannot** support post-dense routing — same
corpus, different question, no benchmark change needed now. Benchmark
expansion (dense-weak slices, synthetic challenges) is the named fallback
if the new track fails on calibration power (after-calib positives = 5),
not a prerequisite.

## 4. Candidates (dossiers condensed; full A–J scoring in §5)

**C1 — Cheap-first need prediction (BM25 → Dense).** RQ: after cheap BM25,
can decision-time info predict whether the embedding call for Dense is
worthwhile? Follows: restores the *original* project question ("when does
a query need expensive retrieval") — Phases 9–11 tested the inverted,
expensive-first variant. Support: §2 table (positives, rung upside,
pre-cost decision, offline-evaluable). Against: Gate-3 query-feature
failure (0/24, related label) and P9 query-only +0.194 — signal-side risk
is real; after-calib positives = 5 demands a low-dimensional
pre-registered rule + cross-arm replication; false stops cost quality
(BM25 0.78 vs 0.90). New info: BM25-state at decision time (2 ms, no API
call — explicitly admitted, not smuggled). Differs on all four closed
elements: point, rung, information, currency. Contribution even if
negative: first EV-positive-rung routing test + pre-cost decision
evidence. Complexity: low (frozen rows + existing harness).

**C2 — Sufficiency/stopping as standalone track.** Same oracle question
without the BM25→Dense action design. Rejected as standalone (§3:
measurement without an actionable rung repeats the Phase 9 pattern of
association without intervention); adopted as C1's framing.

**C3 — Answerability-aware (generation-grounded) routing.** RQ: can the
system tell whether current evidence supports a reliable answer before
retrieving more? Genuinely new information (generator/judge side).
Against: needs generation-bearing runs (cost, protocol, judge
calibration); same 13–14 insufficient cases bound power; highest
complexity; failure would be ambiguous (judge vs retrieval blamed).
Deferred behind C1.

**C4 — Benchmark/methods track.** Design a headroom-rich benchmark +
protocol. Only repair for power, but: large scope, new-data validation
burden, risks re-finding Position C without rung preconditions, and §2
shows the current benchmark suffices for C1. Fallback, not lead.

**C5 — Research stop.** Legitimate iff no candidate clears the bar. Not
taken: C1 clears it (§5), and stopping with an untested EV-positive rung
and 16–21 counted positives would be premature, not rigorous.

## 5. Decision matrix (qualitative; no false-precision scores)

| Candidate | Novelty | Feasibility | Falsifiability | Data adequacy | Scientific value | Repeats old failure? | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 cheap-first | High (new point/rung/currency) | High (frozen rows, existing harness) | High (TP≈0 / random-wins closes it, as P10) | Adequate (16/21; calib-5 is the watch item) | High either way | No — all four closed elements replaced | **ADOPT** |
| C2 stopping-alone | Medium (framing only) | Medium | Low (no action to fail) | Same as C1 | Medium | Partially (association w/o intervention) | Fold into C1 |
| C3 answerability | High | Low (new gen protocol + cost) | Medium (ambiguous attribution) | Weak (13–14 cases) | High but delayed | No | Defer |
| C4 new benchmark | Medium (methods, not finding) | Low (large scope) | Low (benchmarks don't fail cleanly) | N/A (builds its own) | Medium | Risk (re-find C elsewhere) | Fallback |
| C5 stop now | N/A | N/A | N/A | N/A | Loses C1's cheap test | N/A | Reject (C1 clears bar) |

## 6. Selection: NEW RESEARCH TRACK

**Research question.** After running cheap BM25 retrieval, can
decision-time-available information predict whether paying for Dense
retrieval is worthwhile — so the system stops at BM25 when its evidence
suffices and pays the embedding call only when needed?

**Hypothesis (falsifiable).** A pre-registered low-dimensional rule over
query + BM25-output state (computed before any embedding call) identifies
the BM25→Dense oracle positives out of sample, such that the adaptive
policy matches dense-only quality within margin at substantially fewer
embedding calls (or beats BM25-only quality at lower cost than
dense-only) — else the hypothesis fails.

**Decision point.** T0 (query string only) and T1 (post-BM25, ~2 ms
local) — both strictly before the T2 embedding call. Pre-routing in the
sense Phase 9 found missing: the expensive path is genuinely avoidable.

**Observable variables.** Phase 6 `QueryFeatures` + BM25-output state
(ranks/scores/IDs of the local top-k, single-arm quantities only —
DDR-style; cross-arm quantities excluded by construction, since dense has
not run). Labels score only.

**Required data.** Frozen `p8{a,b}_e1` bm25+dense rows (present);
ADR-027 splits (present); no new retrieval for oracle/policy evaluation.

**Intervention.** BM25 → Dense escalation (stop-vs-pay), rung EV-positive
(+0.11–0.14 R@5). Currency: embedding API calls avoided + local latency.

**Evaluation metric.** Recall@5 primary (oracle-compatible), MRR
secondary; cost as API-call rate + median incremental latency; random
escalation at equal rate as the Phase-10-style ablation.

**Success criteria.** On frozen test, one pass: adaptive quality within
pre-registered margin of dense-only at a substantially lower call rate,
or above BM25-only quality at lower cost than dense-only, with TP > 0 and
random ablation not winning — replicated in direction on the second arm.

**Failure criteria.** TP ≈ 0, CIs covering no-benefit, or random-at-rate
winning (the Phase 10 signature) → closes cheap-first routing on this
benchmark and triggers the C4 benchmark-expansion fallback. A negative
is publishable as the rung-positive complement to Phases 10–12.

**First experiment (Phase 14, step 0).** Fixed rung screening from frozen
rows (BM25-only / dense-only / always-escalate means + oracle-positive
counts — §2 already establishes the headroom) then oracle build
(escalate ⟺ dense−BM25 ≥ 0.01, ties → stop), calibration rule freeze on
`after`-calib with cross-arm consistency check, one-pass test evaluation
with random ablation. Design to be frozen in a Phase 14 preregistration
before any number is read.

## 7. Preserved narrative (§15)

P9: DDR measures dispersion. P10: DDR does not predict escalation value.
P11: disagreement ≠ insufficiency (helps: BM25 rescue; harms: State-3
fusion corruption). P12: retrieval-state-only routing not demonstrated;
STOP. P13: new track restores the original question at the pre-cost
decision point with an EV-positive rung — every lesson above is a design
constraint on it, not a bygone.
