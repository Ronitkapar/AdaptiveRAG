# AdaptiveRAG — Query-Aware Retrieval Optimization for Efficient RAG

An investigation into **query-adaptive retrieval**: can a system decide *before
paying for it* whether a query needs expensive retrieval, and save that cost
without losing too much quality?

**Research status: complete (Phases 1–15). No final optimal routing policy is
claimed.** The project's contribution is as much its carefully preserved
negative results as its one replicated positive finding.

---

## 1. The problem

Retrieval in RAG sits on a quality–cost frontier, and on this corpus the gap is
steep. Median latency and Recall@5, 107 paired queries (Phase 7 E1):

| Strategy | Recall@5 | Median latency | Cost per query |
| --- | --- | --- | --- |
| BM25 | 0.7788 | **1.88 ms** | free (no API call) |
| Dense | **0.9283** | 472.84 ms | one embedding API call (~374 ms of it) |
| Hybrid (RRF) | 0.8723 | 465.65 ms | as Dense |
| Hybrid + rerank | 0.7165 | 3491.69 ms | + local cross-encoder |

Dense is ~250× more expensive than BM25 and clearly better — but **every query
pays for it**, including the many whose answer BM25 already retrieves. Adaptive
retrieval asks: can the expensive call be skipped when it adds nothing?

## 2. Research question

> **When does a query actually need expensive retrieval, and can retrieval effort
> be adapted automatically without sacrificing too much retrieval quality?**

Pursued through two decision points: *after* Dense (should we escalate to
Hybrid?) in Phases 6–12, and *before* Dense (after cheap BM25, is Dense worth
buying?) in Phases 13–15.

## 3. Architecture

```text
Query
  ↓
Query / Retrieval Analysis      deterministic, model-free query features
  ↓
Routing / Decision Logic        rule-based router → structured decision
  ↓
Retriever                      ├── BM25            (local, ~2 ms)
  │                            ├── Dense          (embedding API + Qdrant)
  ├── adaptive ────────────────┤
  │                            ├── Hybrid         (RRF of dense + BM25)
  │                            └── Hybrid + Rerank (ONNX cross-encoder)
  ↓
Retrieved Evidence
  ↓
Evaluation / Trace             ExperimentTrace → strategy-agnostic evaluators
```

For `adaptive`, a label-free sufficiency check sits between the first retrieval
and any escalation, bounded by a total-order ladder. Router, retriever,
evaluation and experiment infrastructure are separate layers, enforced by
architecture guards in the test suite. Details:
[`docs/architecture/system_overview.md`](docs/architecture/system_overview.md).

## 4. Retrieval strategies

**BM25** (cheap lexical) · **Dense** (`text-embedding-3-large` + Qdrant) ·
**Hybrid** (RRF of the two) · **Hybrid + rerank** (MS-MARCO cross-encoder on
ONNX). All conform to one `Retriever` interface and are evaluated by the same
strategy-agnostic evaluators, so no comparison is confounded by the harness.

## 5. Main findings

**Detection works; decision-making is much harder.**

* **Retrieval disagreement is real, measurable and mechanistic.** Dense
  distinct-document ratio tracks per-query dispersion across strategies
  (ρ +0.343 / +0.406, generalizing out of sample at +0.410) — the
  evidence-diffuseness structure.
* **But disagreement does not predict escalation value.** Acting on it failed:
  chance-level AUC (0.550), zero true positives, and the adaptive policy was
  beaten by *random* escalation at the same spend (Phase 10, `FAILURE`).
* **The two outcomes have different mechanisms.** Escalation helps when BM25
  holds evidence Dense missed; it harms when fusion corrupts an already-correct
  Dense result. Not separable beforehand (1/6 and 0/6 on pre-declared
  features).
* **Cheap first-stage information does select Dense calls better than random.**
  On 220 fresh queries, a rule frozen in advance caught **18 of 25** oracle
  positives at ~30% Dense spend (avoiding ~70% of calls) and beat random
  spending decisively (**P = 0.001** combined).
* **It still was not good enough to ship.** Adaptive trailed full Dense by
  0.018–0.030 Recall@5, missing the pre-registered margin bar by 0.004 →
  `INSUFFICIENT EVIDENCE` (Phase 15).

## 6. Important negative result

> **Retrieval disagreement ≠ retrieval insufficiency ≠ escalation value.**

The most valuable single finding. Phase 9 found a reproducible correlation
between "strategies disagree" and "evidence is diffuse"; Phase 10 tested it as an
intervention and it failed, because on this benchmark dispersion is dominated by
*Dense succeeding where RRF fusion degrades the ranking*. The signal fires on
queries where escalating **hurts**. Correlation did not survive intervention,
and reporting that cleanly is the point.

## 7. Final conclusion

Adaptive retrieval is plausible, and cheap pre-cost information carries real,
replicated selection signal — but on this evidence the quality gap to full Dense
retrieval was too large to justify a replacement policy. The research track was
**stopped deliberately** after two consecutive inconclusive phases, rather than
continued through feature engineering that would have fitted noise.

## 8. Repository structure

```text
AdaptiveRAG/
├── README.md                    ← you are here
├── AGENTS.md                    agent conventions (research complete; no new phases)
├── src/adaptive_rag/            implementation (retrieval, routing, evaluation, experiments)
├── frontend/                    Streamlit demo (presentation layer; §9a)
├── scripts/                     corpus/index pipelines, experiment drivers, phase analyses
├── tests/                       1056 offline tests (2 integration deselected)
│   └── test_frontend/           24 offline presentation-layer tests
├── data/
│   ├── metadata/papers.json     canonical corpus manifest (14 papers, SHA-256)
│   ├── evaluation/              frozen benchmarks (107-query + 110-query) with ledgers
│   └── processed*/              derived corpora (reproducible; gitignored)
├── storage/                     indexes and the ONNX reranker (gitignored)
├── experiments/                 run artifacts, gitignored; superseded runs under archive/
└── docs/
    ├── ADAPTIVERAG_REPORT.md    ← central research report
    ├── research/                question · methodology · experiments · findings ·
    │                            negative_results · conclusions · limitations
    ├── architecture/            system overview (as implemented)
    ├── reproducibility/         commands and caveats
    ├── history/                 experiment timeline + archived planning notes
    ├── phases/                  per-phase plan-of-record (phases 1–15)
    ├── phase*_results.md        per-phase result records
    ├── decision.md              ADRs (locked architectural decisions)
    ├── architecture.md          detailed layer reference
    └── progress.md              full chronological project log
```

## 9. Reproducing the work

```bash
pip install -e ".[dev,plots]"     # or use the existing .venv
.venv/bin/pytest -q               # 1056 offline tests, no credentials needed

# Offline BM25 baseline end-to-end
.venv/bin/python scripts/build_bm25_index.py
.venv/bin/python scripts/run_experiment.py --retriever bm25 --no-judge --no-generation

# Phase 9–15 analyses are offline over frozen rows
.venv/bin/python scripts/gate9_mechanism_screening.py
.venv/bin/python scripts/phase10_evaluate_policy.py
```

Dense/hybrid arms need `AICREDITS_API_KEY` (copy `.env.example`). Full commands,
including how to reproduce each phase's analysis, are in
[`docs/reproducibility/reproduction.md`](docs/reproducibility/reproduction.md).

## 9a. Streamlit demo frontend

A recruiter-facing presentation layer over the frozen project — live query
playground, strategy comparison, adaptive-decision trace viewer, frozen
benchmark graphs, experiment explorer, research journey and findings. It
reimplements no retrieval: live queries go through the project's own
`instantiate_components` factory, and every benchmark number comes from a
recorded artifact.

```bash
pip install -e ".[frontend]"
streamlit run frontend/app.py
```

BM25 runs fully offline; dense/hybrid/rerank/adaptive light up when
`AICREDITS_API_KEY` is set. On a deployment without indexes or credentials
(e.g. a fresh clone, where `storage/` and `experiments/` are gitignored) the
demo degrades honestly: live retrieval reports itself unavailable and the
frozen benchmark results are shown instead, clearly labelled. No result is
ever simulated.

## 10. Research status

```text
Status:
Research investigation completed.
Adaptive-routing experimentation paused.
No final optimal routing policy claimed.
Project consolidated for documentation, presentation, and future research.
```

Phase verdicts: Phase 9 `PROMISING_SIGNAL` · Phase 10 `FAILURE` · Phase 11
outcome B · Phase 12 **STOP the post-dense track** · Phases 14 and 15
`INSUFFICIENT EVIDENCE`.

## 11. Limitations

Single 14-paper academic corpus (no second domain); binding constraint was
**positives, not queries** (4–25 per framing, where AUC claims need tens);
`nDCG@5` excluded entirely (corrupt section labels); latency figures are
provider- and hardware-specific; experiment artifacts are gitignored and
recoverable only from the working tree. Full list:
[`docs/research/limitations.md`](docs/research/limitations.md).

## 12. What this project does not claim

Not an optimal adaptive algorithm · not a production-ready routing policy · no
universal generalization · no causal claims from the observed correlations · no
claim that adaptive retrieval outperforms Dense · no claim the final policy can
replace full Dense retrieval. Stated explicitly in
[`docs/ADAPTIVERAG_REPORT.md`](docs/ADAPTIVERAG_REPORT.md) §9.

## 13. Future work

Possible directions, not completed work and not authorized by this evidence:
learned routing over the State-2/State-3 interaction; larger or multiple corpora
with upfront power analysis; cost-aware objectives that price the quality gap;
better-calibrated utility estimation; alternative retrieval architectures
(rungs other than the net-negative RRF-hybrid); answerability-grounded routing.

---

**Start here:** [`docs/ADAPTIVERAG_REPORT.md`](docs/ADAPTIVERAG_REPORT.md) ·
**Timeline:** [`docs/history/experiment_timeline.md`](docs/history/experiment_timeline.md) ·
**Findings:** [`docs/research/findings.md`](docs/research/findings.md) ·
**Corpus:** [`docs/corpus.md`](docs/corpus.md)