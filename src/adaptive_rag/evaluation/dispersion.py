"""
evaluation.dispersion
---------------------
Phase 9 per-query **outcome dispersion** across retrieval strategies: how much
the strategies disagree about how well they did on a given query.

This module exists because Phase 7's Gate 3 measured the wrong predicate, and
the reason belongs in the code rather than only in a report. Gate 3 labelled a
query `needed` when *no* alternative came within `EPSILON` of the reference
strategy. A query carries oracle gain only when some strategy *beats* the
reference. Those predicates are mutually exclusive, so every headroom-carrying
query landed in Gate 3's negative class and all 8 of its positives carried
`oracle_gain == 0.0`. Its AUC near 0.53 is a statement about that label, not
about strategy sensitivity.

Dispersion replaces the label with a continuous quantity that lacks the failure
mode: `max - min` over the strategies, zero when they tie and positive exactly
when they disagree about the outcome.

**Why the three-strategy default.** `SOUND_STRATEGIES` excludes `hybrid_rerank`
deliberately. On this benchmark 26 of 34 dispersed queries (`after`) and 30 of 41
(`before`) are dispersed *only* because the reranker is strictly worse than every
other strategy, and ADR-028 already adjudicated that rung as net-negative.
Including it would mostly re-measure a closed finding. `dispersion()` takes an
explicit strategy tuple and `reranker_decomposition` reports the split, so the
choice is auditable rather than implied.

**No metric is reimplemented.** `recall_at_5` and `mrr` are read off the row the
evaluator itself wrote, following the rule `evaluation.rows` sets: a dispersion
computed from re-derived metrics would silently disagree with every published
table the first time a metric definition changed.

This module builds **targets** from ground truth, which is its one privilege and
one responsibility -- see `docs/phases/phase-9.md` §2.9. Nothing here is
admissible as a routing feature.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

DISPERSION_VERSION = "phase9_dispersion_v1"

#: Same epsilon as Gate 2 and Gate 3, so `needed` means the same thing in all
#: three phases and a Phase 9 control is comparable to the number it reproduces.
EPSILON: float = 0.01

#: Strategies whose *disagreement* is the open question. `hybrid_rerank` is
#: excluded: its contribution is a known, already-adjudicated defect (ADR-028).
SOUND_STRATEGIES: tuple[str, ...] = ("bm25", "dense", "hybrid")

#: The frozen selectable set, for the decomposition only. `adaptive` is absent by
#: construction -- it is the router's own output, not a strategy it may select.
SELECTABLE_STRATEGIES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


class StrategyOutcome(BaseModel):
    """What one strategy scored on one query, plus what it retrieved.

    Only what dispersion and the agreement signals need. It deliberately does
    **not** carry `query_id`, `relevant_documents`, or any other ground truth: a
    target builder may hold labels, and a signal builder must not be able to reach
    them from here (docs/phases/phase-9.md §2.9). Omitting the id is the same
    discipline applied to identity -- the id belongs to the grouping, not to one
    strategy's slice of it.
    """

    model_config = ConfigDict(extra="forbid")

    system: str
    recall_at_5: float
    mrr: float
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    #: Rank-ordered retrieval scores, aligned with `retrieved_document_ids`.
    #: Retrieval-observable, so admissible as a signal input (docs/phases/
    #: phase-9.md §2.9) and read from `traces.jsonl` -- `rows.jsonl` does not
    #: carry them. Empty when only rows were available, which makes the score
    #: geometry signals undefined rather than silently zero.
    scores: list[float | None] = Field(default_factory=list)


class QueryDispersion(BaseModel):
    """One query's dispersion across a strategy set.

    `dispersion` is `max - min` of `recall_at_5` over `strategies`: zero when
    every strategy ties, positive exactly when they disagree on the outcome.
    `ties_at_max` records how many strategies reached the top value, because a
    dispersion of zero is only *benign* when several share the top -- a lone
    winner at zero dispersion is the near-binary-recall case that made Gate 2's
    ceiling fragile, and the two must not be conflated.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str
    strategies: tuple[str, ...]
    dispersion: float
    dispersion_mrr: float
    ties_at_max: int
    # Per-strategy scores, so any aggregate can be re-derived and checked.
    recall_by_strategy: dict[str, float] = Field(default_factory=dict)
    mrr_by_strategy: dict[str, float] = Field(default_factory=dict)


class RerankerDecomposition(BaseModel):
    """How much 4-arm dispersion the net-negative reranker accounts for.

    Recorded because excluding the reranker is a choice, and a choice whose
    justification is not measured cannot be audited.

    The three counts **overlap and do not partition** the dispersed queries, which
    is a property of the question rather than a defect in the counting. A query
    can be dispersed among the sound three *and* have the reranker strictly below
    all of them, so it is counted twice. Measured on this benchmark 26 and 21
    sum to more than the 34 dispersed, which is exactly this overlap.

    * `reranker_strictly_worst` -- the reranker is strictly below the best of the
      others. ADR-028 re-measured. Includes queries also counted below.
    * `reranker_strictly_best` -- the converse; expected near zero.
    * `dispersion_among_sound_only` -- dispersion surviving the reranker's
      removal, i.e. the queries this phase actually studies.
    """

    model_config = ConfigDict(extra="forbid")

    n_dispersed_four_arm: int
    reranker_strictly_worst: int
    reranker_strictly_best: int
    dispersion_among_sound_only: int
def dispersion(
    query_id: str,
    outcomes: Mapping[str, StrategyOutcome],
    strategies: Sequence[str] = SOUND_STRATEGIES,
) -> QueryDispersion:
    """`max - min` recall (and MRR) across `strategies` for one query.

    Deterministic: the same outcomes and strategies always give the same result.
    Raises `ValueError` if a requested strategy is absent rather than computing
    over whatever is present -- a missing arm would otherwise look like a tie and
    enter the analysis as a zero.
    """
    missing = [s for s in strategies if s not in outcomes]
    if missing:
        raise ValueError(f"missing outcomes for strategies: {missing}")

    recalls = {s: float(outcomes[s].recall_at_5) for s in strategies}
    mrrs = {s: float(outcomes[s].mrr) for s in strategies}
    best = max(recalls.values())
    return QueryDispersion(
        query_id=query_id,
        strategies=tuple(strategies),
        dispersion=round(best - min(recalls.values()), 6),
        dispersion_mrr=round(max(mrrs.values()) - min(mrrs.values()), 6),
        ties_at_max=sum(1 for v in recalls.values() if abs(v - best) <= EPSILON),
        recall_by_strategy=recalls,
        mrr_by_strategy=mrrs,
    )


def needed_split(
    outcomes: Mapping[str, StrategyOutcome],
    strategies: Sequence[str],
    reference: str,
    epsilon: float = EPSILON,
) -> str:
    """Gate 3's label, recomputed so Phase 9 can reproduce it as a control.

    `needed` means *nothing else* comes within `epsilon` of the reference. Kept
    faithful in behaviour to `scripts/oracle_routing_ceiling.py:needed_split`,
    including its known pathology: this predicate is mutually exclusive with
    "some strategy beats the reference", so it cannot label a headroom-carrying
    query `needed`. That is why it is a **control** here and not a target.
    """
    if reference not in outcomes:
        raise ValueError(f"missing reference outcome: {reference}")
    reference_recall = float(outcomes[reference].recall_at_5)
    if any(
        float(outcomes[s].recall_at_5) >= reference_recall - epsilon
        for s in strategies
        if s != reference
    ):
        return "sufficient"
    return "needed"


def is_tie_break_artefact(
    outcomes: Mapping[str, StrategyOutcome],
    strategies: Sequence[str] = SELECTABLE_STRATEGIES,
    epsilon: float = EPSILON,
) -> bool:
    """True when every strategy lands within `epsilon` of the best one.

    The H7 control. Dispersion built mostly from such queries would be an
    artefact of near-binary recall rather than a property of retrieval, and the
    phase would have to stop and enlarge the benchmark rather than keep searching
    on the same data.
    """
    recalls = [float(outcomes[s].recall_at_5) for s in strategies]
    return (max(recalls) - min(recalls)) <= epsilon


def reranker_decomposition(
    per_query: Mapping[str, Mapping[str, StrategyOutcome]],
) -> RerankerDecomposition:
    """Partition four-arm dispersion by whether the reranker explains it."""
    worst = best = sound_only = dispersed = 0
    for outcomes in per_query.values():
        if not all(s in outcomes for s in SELECTABLE_STRATEGIES):
            continue
        recalls = {s: float(outcomes[s].recall_at_5) for s in SELECTABLE_STRATEGIES}
        if max(recalls.values()) - min(recalls.values()) <= 0.0:
            continue
        dispersed += 1
        others = [recalls[s] for s in SOUND_STRATEGIES]
        rerank = recalls["hybrid_rerank"]
        if rerank < max(others):
            worst += 1
        elif rerank > max(others):
            best += 1
        if max(others) - min(others) > 0.0:
            sound_only += 1
    return RerankerDecomposition(
        n_dispersed_four_arm=dispersed,
        reranker_strictly_worst=worst,
        reranker_strictly_best=best,
        dispersion_among_sound_only=sound_only,
    )


def scores_from_trace(record: Mapping[str, Any]) -> list[float | None]:
    """Rank-ordered scores from one `traces.jsonl` record.

    `rows.jsonl` does not persist scores, so the score-geometry signals (H4) are
    unreadable without the trace. Returns an empty list for a failed or
    score-less trace, which leaves the signals `None` rather than zero.
    """
    retrieval = record.get("retrieval") or {}
    results = retrieval.get("results") or []
    return [r.get("score") for r in results]


def outcome_from_row(
    record: Mapping[str, Any], scores: Sequence[float | None] | None = None
) -> StrategyOutcome:
    """Build a `StrategyOutcome` from one persisted `rows.jsonl` record.

    Read field by field rather than through `EvaluationRow.model_validate`, so
    this module does not couple to the row schema and a row gaining a column
    cannot silently change what a target reads. Filtering on `status` is the
    caller's job. `scores` is supplied separately because it lives in the trace,
    not the row.
    """
    return StrategyOutcome(
        system=str(record["system"]),
        recall_at_5=float(record["recall_at_5"]),
        mrr=float(record["mrr"]),
        retrieved_document_ids=list(record.get("retrieved_document_ids") or []),
        retrieved_chunk_ids=list(record.get("retrieved_chunk_ids") or []),
        scores=[None if s is None else float(s) for s in (scores or [])],
    )


def per_query_outcomes(
    records: Iterable[Mapping[str, Any]],
    strategies: Sequence[str] = SOUND_STRATEGIES,
    scores_by_query: Mapping[str, Mapping[str, Sequence[float | None]]] | None = None,
) -> dict[str, dict[str, StrategyOutcome]]:
    """Group persisted rows by query, keeping only `ok` rows for `strategies`.

    Failed rows are dropped rather than scored as `0.0`. Phase 8 measured why
    that matters: 13 arms lost embedding calls, every suite still reported a full
    trace count, and a contaminated arm read 0.6384 where its true value was
    0.352 -- a sign-flipping error that `status` catches and an implicit zero
    would not.

    `scores_by_query` maps `{query_id: {system: scores}}` from the traces. It is
    optional so that a rows-only caller still works; the score-geometry signals
    are then `None`, never zero.
    """
    score_map = scores_by_query or {}
    grouped: dict[str, dict[str, StrategyOutcome]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        system = str(record.get("system", ""))
        if system not in strategies:
            continue
        query_id = str(record["query_id"])
        scores = (score_map.get(query_id) or {}).get(system)
        grouped.setdefault(query_id, {})[system] = outcome_from_row(record, scores)
    return grouped


def complete_queries(
    per_query: Mapping[str, Mapping[str, StrategyOutcome]],
    strategies: Sequence[str] = SOUND_STRATEGIES,
) -> list[str]:
    """Query IDs present for **every** requested strategy, sorted.

    Incomplete queries are excluded rather than computed over a subset:
    dispersion across three strategies with one missing is not comparable to
    dispersion across all three, and narrowing per query would move the
    denominator underneath the analysis.
    """
    return sorted(
        query_id
        for query_id, outcomes in per_query.items()
        if all(s in outcomes for s in strategies)
    )


def dispersion_table(
    per_query: Mapping[str, Mapping[str, StrategyOutcome]],
    strategies: Sequence[str] = SOUND_STRATEGIES,
) -> list[QueryDispersion]:
    """Dispersion for every query complete under `strategies`, in id order.

    The single entry point Gate 9.1 uses. Enforcing completeness here, once,
    means no downstream caller has to remember to -- a per-query check is the
    kind of omission that leaves a denominator nobody can reconstruct.
    """
    return [
        dispersion(query_id, per_query[query_id], strategies)
        for query_id in complete_queries(per_query, strategies)
    ]