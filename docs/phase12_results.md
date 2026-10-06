# Phase 12 Results — Research Boundary of Adaptive Retrieval

Recommendation: **STOP CURRENT ADAPTIVE-ROUTING TRACK**
(Positions: **C adopted**, B acknowledged as limitation, A rejected, D
rejected. Re-entry conditions in §9, not a continuation.)
Plan of record: [`docs/phases/phase-12.md`](phases/phase-12.md).
Phases 8–11 unchanged. Offline analysis only; one read-only artifact
verification was performed (Phase 11 tables/figures/provenance present;
oracle `phase10_escalation_v1` reused unchanged; dataset sha
`f0695189903c8ef8…`).

## 1. Evidence matrix (exact repository evidence)

| Question | Evidence | Result | Status |
| --- | --- | --- | --- |
| Does retrieval disagreement exist? | P9: T-disp3>0 on 21/107 after, 29/107 before; T-disp4 34/41 | Real, replicated | Established |
| Does DDR measure disagreement? | P9: DDR@dense ρ +0.343/+0.406 vs T-disp3, Holm p 0.0012; test ρ +0.410, no fitting | Replicated association, generalizes OOS | Established |
| Does DDR predict useful escalation? | P10 calib: ROC-AUC 0.55, Spearman −0.13 (wrong way), positives at DDR 0.2 and 0.8 | Chance-level | Falsified under setting |
| Does adaptive DDR escalation improve trade-off? | P10 test: Δ −0.050 [−0.117,0] / −0.033 [−0.083,0], TP 0/0, P(random≥adaptive) 1.000/0.954, +4–6 ms | Dominated by dense-only and by random | Falsified under setting |
| Why does Hybrid help? | P11: 5 unique helps; dense recall 0–0.33, BM25 holds/ranks missing evidence (`bm25_only` 4/5), RRF keeps it top-5 | BM25-side rescue of dense-missed evidence | Established (mechanism, n=5) |
| Why does Hybrid hurt? | P11: 9–12 harms/arm, ALL State 3; H1 intruder promotion + H2 pure re-rank with no new doc entering | Dense-sufficient queries corrupted by fusion | Established (mechanism) |
| Can retrieval-state signals distinguish the cases? | P11: top-1 0.49/0.48 vs 0.63/0.56 + jaccard 0.13/0.27 vs 0.41/0.40 directionally, ranges overlap, separation 1/6 + 0/6; P8G3: AUC 0.534, 0/24 Holm | Directional hints only, nothing establishable | Not established |
| Is another intervention currently justified? | P11 gate (≤3 helps/split/arm, no surviving candidate) + §8 gate below (criteria 3, 4 fail) | No | Decided: NO-GO |

Supporting constants: oracle helps 4/arm, harms 9/12, neutral ~92 (always-escalate net-negative by construction); E1 dense R@5 0.9283/MRR 0.8604 vs hybrid 0.8723/0.8576 vs bm25 0.7788/0.7264; ceiling +0.0218/+0.0343 (CIs exclude zero) on 4/107 + 6/107 carriers with 98 ties at max; Gate 3 fit-split AUC 1.000/0.962 vs OOS 0.534 (overfitting signature); DDR 5 values {0.2…1.0}; best query-only test |ρ| 0.194.

## 2. Classification

**Established.** Dispersion is real and DDR tracks it (P9, replicated both
arms + OOS split). DDR does not predict escalation gain (P10, both arms).
The frozen DDR policy is dominated (P10 test, both arms, incl. random
ablation). Helps = BM25 rescue of dense-missed evidence; harms = fusion
corrupting dense-sufficient results via H1/H2 (P11 per-query records).
All harms are State 3 (exact, both arms).

**Not established (inconclusive).** Whether weak-dense × disagreement
separates State 2 from State 3 (n=5, overlapping ranges, 1/6 + 0/6
separation). Whether DDR is a degraded disagreement proxy or a distinct
coverage quantity (P9 §9.4 left open; P10 resolves only its usefulness:
neither reading predicts gain). Any supervised routing claim on 4–8
positives (all framings: Gate 3's 8, P10's 4/arm, P11's 5 unique).

**Falsified under the tested setting.** "DDR identifies
useful-escalation queries." "Adaptive DDR escalation improves the
quality–latency trade-off." "Dispersion implies escalation value"
(P10 mechanism reading + P11 H2: dispersion here is substantially dense
succeeding where RRF fails). "A calibratable gate on current signals
opens usefully" (shipped 0.5 never fires; 0.7 fires 3/47 with MRR down).

**Still plausible but untested.** Query/task-aware need signals (D1);
retrieved-evidence sufficiency judgment — the shipped checker never fired,
so it was never evaluated (D2); answerability/coverage beyond lexical
`top1_coverage`, which runs the wrong way (D3, with that counter-evidence
noted); a second-stage rung with positive expected value (D4 — only
RRF-hybrid tested); any corpus/regime with dozens of positives and a
dense-weak margin (D5).

## 3. Boundary (derived, not assumed)

> Retrieval disagreement can be measured — including post-dense from a
> single arm — and it is real. Under the tested corpus/configuration
> (107-query academic-RAG benchmark, dense the strongest fixed arm,
> RRF-hybrid rung, post-dense retrieval-state-only decision), disagreement
> does not reliably indicate retrieval insufficiency, and no available
> post-dense retrieval-state information reliably predicted whether
> Hybrid escalation would help or harm. The tested escalation is
> net-negative (harms 2–3× helps), so there is no rung worth routing to.

## 4. Why the setting limits (A–G, judged)

* **A positive scarcity — SUPPORTED.** 4–8 positives in every framing;
  Gate 3 min-detectable AUC 0.825–0.850 at 80% power; P10's 2/arm makes
  even the `before` AUC 0.86 noise. No routing claim is fittable — measured.
* **B fixed ceiling / C corpus heterogeneity — MIXED, reported as measured
  properties not defects.** Dispersion exists (21–29/107) but *useful*
  disagreement is ~4%: 92/107 single-relevant-doc, 98 ties at max, gain on
  4–6 queries. The corpus offers disagreement with little escalation
  headroom — a property of the oracle counts, not an inadequacy verdict.
* **D query distribution — limitation, SUPPORTED as such.** 8 Gate-3
  positives span 6 categories; E7 hard cells (multi_document 0.4762) are
  pooled/small-cell descriptive only. Diversity unproven either way.
* **E dense already strong — SUPPORTED as observation.** Dense leads every
  E1 quality metric except deep recall; ceiling above it is +0.02–0.03.
  Small headroom + few positives jointly bind.
* **F intervention quality — SUPPORTED.** Hybrid trails dense on E1
  (−0.056 R@5, MRR −0.003…−0.13 with RRF flattening since Phase 4); H2
  shows harm needs no BM25 disagreement at all. A router cannot fix a rung
  with 2–3:1 harm:help asymmetry — the rung needs re-justification first.
* **G observable information — SUPPORTED, the central measured negative.**
  Three independent failures: Gate 3 AUC ≈ 0.53 (0/24 Holm), P10 AUC 0.55
  + wrong-way Spearman, P11 separation 1/6 + 0/6. The decision point
  exposes disagreement, not insufficiency — demonstrated per query (helps
  need label-visible missing evidence; harms need only confident dense +
  RRF).

## 5. Positions adjudicated

* **A (feasible, signal inadequate) — REJECTED.** Exhaustion is thorough:
  24 Gate-3 features, 12 P9 hypotheses, 6 P11 separation features, plus BM25
  at decision time considered — and positives number 4–8 in all framings.
  No "plausibly available signal" remains unnamed.
* **B (setting insufficient) — TRUE AS LIMITATION, insufficient as
  verdict.** A–G above show the setting binds; but B alone would license
  "same experiment, more data," which the rung finding (F) forecloses on
  this configuration. B motivates D5 as *new* research, not continuation.
* **C (retrieval-state-only routing not demonstrated under this setting) —
  ADOPTED.** Every available-evidence test fails consistently; scope is
  explicit; no impossibility claimed.
* **D (abandon the hypothesis) — REJECTED.** Helps exist, mechanism is
  understood, and whole families (D1–D5: sufficiency judgment, other rungs,
  other regimes) are untested. Evidence supports stopping *this track*,
  not closing the question.

## 6. Directions assessment (hypotheses, not proposals)

| Direction | Motivation + support | Counter-evidence / risk | Needs |
| --- | --- | --- | --- |
| 1 query/task-aware need | Only lexical query features tested (+0.194 best); information-requirement reasoning untested | Gate 3 0/24 included query features; risk: repeating P9.3A with new names | Task-grounded need labels, dozens of positives, OOS protocol |
| 2 evidence sufficiency | Closest to State-2/State-3 framing; shipped checker NEVER fired → never evaluated, nearest untested neighbor | `top1_coverage` runs wrong way (0.346/0.357); lexical coverage specifically counter-indicated | Sufficiency definition tied to task success, not term overlap |
| 3 answerability/coverage | Directly asks the State-2 question (is evidence enough?) | Same coverage counter-evidence as D2; needs generation-side ground truth | Answerability labels + generator in the loop |
| 4 different intervention | Only one rung tested; routing needs EV-positive rung first | Current rung net-negative 2–3:1; new rung must beat dense somewhere substantial before any router | Rung screening (fixed comparison) BEFORE routing work |
| 5 different corpus/regime | Only direction repairing power (A); dense-weak regimes untested | New-benchmark cost; risk of re-finding C under a new name without rung + power pre-conditions | Pre-registered positives floor, power analysis upfront, §5-style gate |

All five require what §8 demands; none is authorized now.

## 7. Unified story — arrow verification

P8 fixed strategies characterized (E1 five-arm + Gates 2/3 numbers ✓) →
P9 dispersion identified + DDR selected (5 survivors, DDR OOS +0.410 ✓) →
P10 escalation tested → FAILURE (frozen policy, one-pass test, random
ablation ✓) → P11 mechanism (5 helps BM25-rescue; harms State-3 H1/H2;
separation 1/6+0/6 ✓) → available signals cannot distinguish (three
independent failures agree ✓) → P12 boundary (§3) + STOP (gate §8 fails
on criteria 3 and 4: ≤3 positives/split/arm; rung net-negative). Every
arrow cites a frozen artifact above; no arrow crosses a split boundary
unlabelled.

## 8. Twelve framework answers

1. P9 proved: dispersion real; DDR tracks it incl. OOS; no pre-routing
   signal; usefulness untested (no oracle, no intervention).
2. P10 falsified: DDR predicts gain; DDR escalation improves trade-off;
   dispersion implies value — under this setting.
3. P11 explained: helps (rescue) vs harms (H1 intrusion / H2 re-rank),
   all harms State 3; hints exist but inseparable at n=5.
4. DDR: a retrieval-dispersion diagnostic, not a routing signal.
5. Disagreement: measurable, real, multidimensional (§9.4: not one latent
   quantity) — and ≠ insufficiency ≠ value.
6. Post-dense escalation (Dense→RRF-hybrid, retrieval-state-only): not
   demonstrated; tested policy dominated.
7. Failure cause: combination — uninformative-at-decision signals (G,
   measured) × net-negative rung (F, measured) × few positives + strong
   dense (A/E, measured); corpus-headroom (B/C) and query-mix (D) as
   compatible limitations.
8. Established vs speculative: §2 split; A/E/F/G established, B–D
   limitations, D1–D5 untested.
9. Another adaptive-routing experiment: NO — gate criteria 3 (positives)
   and 4 (rung upside) fail on this corpus/configuration.
10. What would change it (§9): dozens of positives + EV-positive rung +
    pre-registered rule + split discipline = new research, not Phase 13.
11. Final claim (§3 boundary + preserved negatives in §1 table).
12. Strongest contribution: a complete, replicated, mechanism-explained
    negative result with its boundary conditions — engineering (5
    strategies + adaptive layer + gate harness), empirical (headroom
    +0.02–0.03 concentrated; two chance-level AUCs; dominated policy),
    negative (DDR escalation fails; random beats it), mechanistic
    (State-2/State-3, H1/H2, disagreement≠insufficiency≠value),
    boundary-limited (107-query academic-RAG, dense-strong, RRF rung,
    retrieval-state-only), with D1–D5 as genuine remaining uncertainty.

## 9. Go/No-Go

**STOP CURRENT ADAPTIVE-ROUTING TRACK.** Preserved: all oracles, tables,
per-query mechanism records, figures, frozen policies, negative verdicts.
Re-entry (new question, not continuation): (i) benchmark with dozens of
oracle-positives + upfront power analysis; (ii) a rung beating dense
somewhere substantial (fixed comparison first); (iii) pre-registered
confidence×disagreement-family rule with BM25-at-decision admitted;
(iv) ADR-027 split discipline; (v) a §5-style gate that can say no again.
Absent all five, further routing experiments on this track are not
scientifically justified — a rigorous stopping point, preferred per
project rule over another weak experiment.
