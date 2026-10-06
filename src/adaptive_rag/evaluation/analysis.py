"""
evaluation.analysis
-------------------
The Phase 7 *analysis* layer: eight studies (E1..E8) over the flat row exports
that `evaluation.rows` produces.

This module is a consumer, not a second source of truth. It reads rows and
groups them; it never re-computes a metric, never re-derives a relevance label,
and never re-implements a percentile. A row's `recall_at_5` was computed by
`RetrievalEvaluator`'s own helper, and this file summarises it. p95 comes from
`evaluation.base.percentile`, which is nearest-rank and does not interpolate, so
the p95 column of a Phase 7 table is literally the 95th observed sample rather
than a number between two samples.

Three rules shape every analyzer here, and all three exist because the failure
they prevent is a published result that cannot be defended:

1. **No composite score, anywhere.** Quality and cost stay on separate axes and
   are compared separately. `analysis.stats` says the same thing about the tests
   it runs; a single "winner" number would re-appear the moment someone wanted
   one headline, and it would hide the only question Phase 7 exists to answer --
   *does adaptive routing earn its complexity*, which is a trade-off, not an
   ordering. `analyze_main_comparison` therefore reports every axis and refuses
   to collapse them, and the guard for that is a test, not a convention.

2. **A cell below the minimum is flagged, not quietly reported.** The dataset
   gate (`scripts/validate_phase7_dataset.py`, `--min-per-category`) already
   established that a per-category cell under five queries is noise. Every cell
   here carries `n` and an `insufficient_data` flag at that same threshold, so a
   three-query category cannot be read as a result even if the table is skimmed.

3. **Only what was observed is reported.** E8 emits a row per transition that
   actually occurred and never a zero row for a pair that did not happen, because
   a table of "0.0% of escalations went bm25 -> dense" reads identically to one
   of "no escalation from bm25 to dense ever happened" unless the reader is told
   which, and the second is a claim about the ladder while the first is a claim
   about the router's decisions. Absent evidence is `None` with a reason attached,
   never `0.0`.

The E7 study is the sharpest version of rule 3. The benchmark's `category`
column is the only ground-truth query-type label that exists: six hand-assigned
values, validated by the dataset gate. The router's runtime `QueryFeatures`
(`question_type`, `complexity_score`, `concept_count`, ...) are *derivation
inputs*, not labels -- they are computed from the query text by the same
heuristics the router then acts on, so a "routing wins on complex queries"
finding read off them is the router agreeing with itself. E7 reports the
dataset's categories and says explicitly which column is which.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median, stdev
from typing import Any, Iterable, Mapping, Sequence

from adaptive_rag.config.paths import CHUNKS_DIR
from adaptive_rag.evaluation.ablation import (
    cost_weight_sweep,
    escalation_variants,
    feature_group_variants,
    threshold_sweep,
)
from adaptive_rag.evaluation.base import percentile
from adaptive_rag.evaluation.dataset import VALID_CATEGORIES
from adaptive_rag.experiments.arms import ARM_NAMES

ANALYSIS_VERSION = "phase7_analysis_v1"

# The per-cell minimum for reporting a generalisation. Five is not a
# statistical constant: it is the same threshold the dataset gate enforces
# (`--min-per-category`, `TARGET_MIN_PER_CATEGORY`), chosen so a cell that the
# gate would refuse to validate is also a cell this layer refuses to generalise
# from. Below it the per-cell numbers are still *reported* -- the data exists and
# hiding it would be dishonest -- but every such cell is flagged.
MIN_CELL = 5

# Quality axes, in the order tables should print them. Every one of these is a
# value `RetrievalEvaluator` computed for the row; this list only says which of
# its columns are quality rather than cost or latency.
QUALITY_METRICS: tuple[str, ...] = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "mrr",
    "ndcg_at_5",
)

# The primary quality axis, used wherever a study needs one headline quality
# number. Chosen over recall@10 because the retrieval runs report at k=5.
PRIMARY_QUALITY_METRIC = "recall_at_5"

# Latency columns carried by a row, in decomposition order: the cheapest and most
# innermost first, so a reader can see what a millisecond was spent on.
LATENCY_COLUMNS: tuple[str, ...] = (
    "routing_latency_ms",
    "retrieval_latency_ms",
    "reranking_latency_ms",
    "generation_latency_ms",
    "query_embedding_latency_ms",
    "search_latency_ms",
    "candidate_generation_latency_ms",
    "total_latency_ms",
)

# Resource and money columns.
COST_COLUMNS: tuple[str, ...] = (
    "context_tokens",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost_usd",
)

# The five E1 systems, in the order `experiments.arms.ARM_NAMES` declares them.
# Read from the arm module rather than re-spelled so the comparison table cannot
# list a system no arm ever built, or omit one that did.
DEFAULT_SYSTEMS: tuple[str, ...] = tuple(ARM_NAMES)

# The E2/E3/E4/E5 variant grids are read from `evaluation.ablation` -- the single
# definition of what each study runs. E3's list is therefore guaranteed to carry
# all six real feature groups from `schemas.config.FEATURE_GROUPS` (including
# `entity`, which is the one most often forgotten when a sweep is written by
# hand) and E5's to carry the `cost_weight` values the ablation module actually
# implements rather than ones invented here.
DEFAULT_ESCALATION_VARIANTS: tuple[str, ...] = tuple(
    v.name for v in escalation_variants()
)
DEFAULT_FEATURE_VARIANTS: tuple[str, ...] = tuple(
    v.name for v in feature_group_variants()
)
DEFAULT_THRESHOLD_VARIANTS: tuple[str, ...] = tuple(
    v.name for v in threshold_sweep()
)
DEFAULT_COST_WEIGHT_VARIANTS: tuple[str, ...] = tuple(
    v.name for v in cost_weight_sweep()
)

# The study each analyzer answers for. Carried in every result so an artifact
# never has to be matched to a study by its filename.
E1 = "E1_baseline_comparison"
E2 = "E2_escalation_ablation"
E3 = "E3_feature_ablation"
E4 = "E4_threshold_calibration"
E5 = "E5_cost_weight"
E6 = "E6_routing_overhead"
E7 = "E7_query_type"
E8 = "E8_escalation_analysis"

# E4's grid is two sweeps, not one. `ablation.threshold_sweep` names the
# sufficiency-threshold arms and `ablation.escalation_step_sweep` the
# `max_escalation_steps` arms; they are the same study id and the same 47
# queries, so a `system` column cannot tell them apart. The step arms are
# detected by this prefix and reported separately rather than being folded into
# the threshold table, whose every other cell would be false for them.
ESCALATION_STEP_PREFIX = "max_escalation_steps="

# Why the step sweep is disclosed rather than tabulated. Wording is
# deliberately a statement about the *design*, not about a measured null: at the
# shipped sufficiency_threshold the gate returns "evidence was sufficient" at
# adaptive.py:196 before the budget is ever consulted at :206, so no arm
# escalates and max_escalation_steps is never read. And because
# `allows_escalation` is called once with a hardcoded `steps_taken=0` inside a
# `retrieve()` that has no loop, steps 1, 2 and 3 are the same configuration
# whatever the gate does. Only step 0 differs from the rest, and only by
# refusing an escalation that never happens.
STEP_SWEEP_NOT_ANALYSED = (
    "The max_escalation_steps arms are not analysed here and are not analysable "
    "as configured -- a design limitation to disclose, not a null result. All "
    "four ran at the shipped sufficiency_threshold=0.5, and that gate never fires "
    "on this dataset (every observed sufficiency score is above it), so no arm "
    "escalates and the bound is never consulted. Independently, the retriever "
    "calls allows_escalation exactly once, with a hardcoded steps_taken=0 and no "
    "loop, so steps 1, 2 and 3 are structurally identical to each other; only "
    "step 0 could differ, by declining an escalation that never happens. A flat "
    "row of equal numbers here would be an artefact of that structure and must "
    "not be read as evidence that the bound does not matter."
)

NO_COMPOSITE_POLICY = (
    "No composite quality/cost score is produced, and none may be derived from "
    "this result. Quality and cost are reported on separate axes because the "
    "Phase 7 question is a trade-off ('does adaptive routing earn its "
    "complexity'), not an ordering; a single number would hide the trade-off it "
    "collapses."
)

# Every `system` / `variant` string an analysis may key on. Sorting is by this
# index so a report lists arms in the order they were built rather than in
# alphabetical order, and an unknown string sorts last rather than first.
_SYSTEM_ORDER: dict[str, int] = {
    name: index for index, name in enumerate(DEFAULT_SYSTEMS)
}

RowLike = Any  # EvaluationRow | Mapping[str, Any]


# --------------------------------------------------------------------------
# row access
# --------------------------------------------------------------------------


def _get(row: RowLike, field: str) -> Any:
    """One field of a row, whether the row is a model or a loaded dict.

    `build_rows` returns `EvaluationRow` instances; `rows.read_rows_jsonl`
    returns plain dicts. Accepting both means the analysis layer can run over a
    freshly built row set in a test and over a run's persisted export in the
    CLI, with no conversion step that could quietly rename a column.
    """
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def row_field(row: RowLike, field: str) -> Any:
    """Public accessor for one field of a row.

    `evaluation.statistics` needs the same model-or-dict field access this module
    uses; making it public here keeps the column names it reads identical to the
    ones this module reads, instead of a second module spelling them out.
    """
    return _get(row, field)


def _values(rows: Sequence[RowLike], field: str) -> list[float]:
    """Every non-`None` numeric value of one column, in row order.

    `None` means "not recorded", which is the normal state of every routing
    column on a fixed-strategy arm and of every latency column on a run that
    failed before that clock started. Dropping those is correct for a latency
    distribution (a missing clock is not a zero-millisecond clock) and it makes
    the reported `n` differ per column, which is why `distribution` always
    carries its own `n` rather than inheriting the cell's.
    """
    out: list[float] = []
    for row in rows:
        value = _get(row, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out.append(float(value))
    return out


def distribution(values: Sequence[float]) -> dict[str, Any]:
    """Mean, median, p95, min, max and stdev of a numeric sample.

    p95 is `evaluation.base.percentile`, i.e. nearest-rank with no interpolation,
    matching the estimator Phase 7 froze the cost table with. Every other
    measure is `None` rather than zero for an empty sample, and stdev is `None`
    for a single sample because a sample standard deviation needs two.
    """
    data = [float(v) for v in values]
    if not data:
        return {"n": 0, "mean": None, "median": None, "p95": None,
                "min": None, "max": None, "stdev": None}
    return {
        "n": len(data),
        "mean": round(sum(data) / len(data), 6),
        "median": round(float(median(data)), 6),
        "p95": round(float(percentile(data, 95)), 6),
        "min": round(min(data), 6),
        "max": round(max(data), 6),
        "stdev": round(float(stdev(data)), 6) if len(data) > 1 else None,
    }


def _cell(n: int, *, min_cell: int = MIN_CELL) -> dict[str, Any]:
    """The `n` / `insufficient_data` pair every reported cell carries.

    `insufficient_data` is `True` whenever the cell holds fewer than `min_cell`
    observations, including when it holds none. A cell with no observations is
    the case that most needs the flag: an empty `{}` reads as "nothing to say",
    when in fact it means "the thing that was supposed to produce this number
    produced nothing", which is a finding.
    """
    return {"n": n, "insufficient_data": n < min_cell, "min_cell": min_cell}


def _sort_key(system: str) -> tuple[int, str]:
    return (_SYSTEM_ORDER.get(system, len(_SYSTEM_ORDER)), system)


def _group_by_system(
    rows: Sequence[RowLike], systems: Iterable[str] | None
) -> dict[str, list[RowLike]]:
    """Rows bucketed by `system`, restricted to the requested systems.

    The restriction is explicit rather than incidental: a caller that passes
    `systems=("bm25", "dense")` gets a two-row table, and a system in the row
    set that the caller did not ask about is not silently appended to it. A
    requested system with no rows is kept as an empty bucket, so a missing arm
    is reported as a missing arm.
    """
    wanted = None if systems is None else list(systems)
    grouped: dict[str, list[RowLike]] = {}
    for row in rows:
        system = _get(row, "system")
        if system is None:
            continue
        if wanted is not None and system not in wanted:
            continue
        grouped.setdefault(str(system), []).append(row)
    if wanted is not None:
        for system in wanted:
            grouped.setdefault(system, [])
    return {k: grouped[k] for k in sorted(grouped, key=_sort_key)}


def _counts(rows: Sequence[RowLike], field: str) -> dict[str, int]:
    """Frequency table of a categorical column, skipping absent values."""
    counts: dict[str, int] = {}
    for row in rows:
        value = _get(row, field)
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _strategy_distribution(rows: Sequence[RowLike]) -> dict[str, dict[str, Any]]:
    """Initial and final strategy selection, as counts and shares.

    Shares are taken over the rows where the column is present, not over the
    whole cell: a fixed-strategy arm has `initial_strategy is None` on every
    row, and dividing by the cell size would report 0.0% for a distribution that
    is simply undefined. The denominator is therefore `observed`.
    """
    out: dict[str, dict[str, Any]] = {}
    for field in ("initial_strategy", "final_strategy"):
        counts = _counts(rows, field)
        observed = sum(counts.values())
        out[field] = {
            "observed": observed,
            "counts": counts,
            "shares": {
                key: (round(value / observed, 6) if observed else None)
                for key, value in counts.items()
            },
        }
    return out


def _escalation_rate(rows: Sequence[RowLike]) -> dict[str, Any]:
    """Escalation rate over the rows that recorded a routing decision.

    `escalated` is read as the boolean the row carries, which `rows.py` took
    from the escalation *decision* rather than inferring it from
    `initial_strategy != final_strategy` -- a router can reach the same strategy
    twice, and a rate computed from the strings would disagree with the run's
    own `escalation_rate` metric.
    """
    decisions = [r for r in rows if _get(r, "escalated") is not None]
    escalated = sum(1 for r in decisions if _get(r, "escalated"))
    return {
        "n_decisions": len(decisions),
        "n_escalated": escalated,
        "escalation_rate": round(escalated / len(decisions), 6) if decisions else None,
    }


def _observed_transitions(rows: Sequence[RowLike]) -> list[dict[str, Any]]:
    """Escalation transitions that actually happened, with counts.

    Only observed pairs appear. The `to` side prefers `escalation_target` (the
    strategy the escalation decision named) and falls back to `final_strategy`
    for a row whose escalation record is absent but whose final strategy
    differs -- a shape the current router does not produce, accepted rather than
    dropped so a trace from a future version is still analysable.
    """
    observed: dict[tuple[str, str], int] = {}
    for row in rows:
        if not _get(row, "escalated"):
            continue
        source = _get(row, "initial_strategy")
        target = _get(row, "escalation_target") or _get(row, "final_strategy")
        if source is None or target is None or source == target:
            continue
        observed[(str(source), str(target))] = observed.get((str(source), str(target)), 0) + 1
    return [
        {"from": source, "to": target, "count": count}
        for (source, target), count in sorted(observed.items())
    ]


def _quality(
    rows: Sequence[RowLike], metrics: Sequence[str] = QUALITY_METRICS
) -> dict[str, Any]:
    """Mean and median of every quality axis in one pass over the rows."""
    return {
        metric: {
            "mean": (round(sum(values) / len(values), 6) if values else None),
            "median": round(float(median(values)), 6) if values else None,
        }
        for metric, values in ((m, _values(rows, m)) for m in metrics)
    }


def _cost(rows: Sequence[RowLike]) -> dict[str, Any]:
    """Resource and money columns, each as a distribution."""
    return {column: distribution(_values(rows, column)) for column in COST_COLUMNS}


def _routing(rows: Sequence[RowLike]) -> dict[str, Any]:
    """Router-side signals: confidence, sufficiency, stage count, per-stage times."""
    return {
        "routing_confidence": distribution(_values(rows, "routing_confidence")),
        "sufficiency_score": distribution(_values(rows, "sufficiency_score")),
        "stage_count": distribution(_values(rows, "stage_count")),
        "pooled_stage_latency_ms": distribution(
            [v for row in rows for v in (_get(row, "stage_latencies_ms") or [])]
        ),
    }


def _status(rows: Sequence[RowLike]) -> dict[str, int]:
    """Row-status breakdown, so a failure cannot be averaged away silently."""
    return _counts(rows, "status")


def _system_summary(rows: Sequence[RowLike], *, min_cell: int = MIN_CELL) -> dict[str, Any]:
    """Everything one system's rows say, with no axis collapsed into another."""
    escalation = _escalation_rate(rows)
    return {
        **_cell(len(rows), min_cell=min_cell),
        "n_successful": sum(1 for r in rows if str(_get(r, "status")) == "success"),
        "n_failed": sum(1 for r in rows if str(_get(r, "status")) != "success"),
        "status_counts": _status(rows),
        "quality": _quality(rows),
        "latency": {column: distribution(_values(rows, column)) for column in LATENCY_COLUMNS},
        "routing": _routing(rows),
        "strategy_distribution": _strategy_distribution(rows),
        "escalation": escalation,
        "escalation_transitions": _observed_transitions(rows),
        "cost": _cost(rows),
    }


# --------------------------------------------------------------------------
# E1 -- main comparison
# --------------------------------------------------------------------------


def analyze_main_comparison(
    rows: Sequence[RowLike],
    *,
    systems: Sequence[str] = DEFAULT_SYSTEMS,
    min_cell: int = MIN_CELL,
) -> dict[str, Any]:
    """E1: the five benchmark systems side by side, one axis at a time.

    Every system gets the same block: n (with the `insufficient_data` flag),
    all eleven quality metrics, the latency decomposition, the strategy
    distribution, the escalation rate and observed transitions, the mean
    retrieval-stage count, and the resource columns.

    What this deliberately does not return is a single number ranking the five
    systems. The mandated table keeps quality and cost in separate columns for
    the same reason, and `NO_COMPOSITE_POLICY` is carried in the result so the
    reason travels with the numbers instead of living only in this docstring --
    a reader holding the JSON has no way to see the prose otherwise.
    """
    grouped = _group_by_system(rows, systems)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "experiment_id": E1,
        "study": "baseline comparison of the five benchmark arms",
        "systems_requested": list(systems),
        "systems_present": [name for name in grouped if grouped[name]],
        "systems_absent": [name for name in grouped if not grouped[name]],
        "n_rows": len(rows),
        "composite_score_policy": NO_COMPOSITE_POLICY,
        "per_system": {
            name: _system_summary(grouped[name], min_cell=min_cell) for name in grouped
        },
    }


# --------------------------------------------------------------------------
# E2 / E3 / E4 / E5 -- the ablation grids
# --------------------------------------------------------------------------


def _variant_report(rows: Sequence[RowLike], *, min_cell: int) -> dict[str, Any]:
    """The shared body of E2..E5: one variant's full observation set."""
    escalation = _escalation_rate(rows)
    return {
        **_cell(len(rows), min_cell=min_cell),
        "n_failed": sum(1 for r in rows if str(_get(r, "status")) != "success"),
        "quality": _quality(rows),
        "latency": {column: distribution(_values(rows, column)) for column in LATENCY_COLUMNS},
        "routing": _routing(rows),
        "strategy_distribution": _strategy_distribution(rows),
        "escalation": escalation,
        "escalation_rate": escalation["escalation_rate"],
        "cost": _cost(rows),
    }


def _rows_outside_variants(
    rows: Sequence[RowLike], variants: Sequence[str]
) -> dict[str, int]:
    """Row counts for `system` values the caller did not ask to be described.

    The mirror image of `variants_absent`: that field names a requested arm with
    no rows, this one names rows of an arm the analysis never described. Both are
    needed, because a study's rows can hold an arm its variant list excludes --
    E4's grid is two sweeps sharing one study id -- and dropping those rows
    without saying so leaves `n_rows` larger than the table with nothing to
    explain the difference.
    """
    wanted = set(variants)
    return {
        name: count
        for name, count in _counts(rows, "system").items()
        if name not in wanted
    }


def _variant_study(
    rows: Sequence[RowLike],
    *,
    experiment_id: str,
    study: str,
    variants: Sequence[str],
    min_cell: int,
) -> dict[str, Any]:
    """Group rows by `system` and describe each named variant.

    E2..E5 are one shape: a grid of named variants, each a `system` value in the
    row export, each measured on the same queries. The `variants` list is
    explicit so a grid with a missing arm reports that arm as empty rather than
    silently shrinking the table to the arms that happen to have rows -- and so a
    grid with an *extra* arm reports those rows as undescribed rather than
    folding them into `n_rows` unexplained.
    """
    grouped = _group_by_system(rows, variants)
    unanalysed = _rows_outside_variants(rows, variants)
    result: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "experiment_id": experiment_id,
        "study": study,
        "variants_requested": list(variants),
        "variants_absent": [name for name in grouped if not grouped[name]],
        "variants_not_analysed": unanalysed,
        "n_rows": len(rows),
        "n_rows_analysed": sum(len(rows_for) for name, rows_for in grouped.items()
                               if name in set(variants)),
        "composite_score_policy": NO_COMPOSITE_POLICY,
        "per_variant": {
            name: _variant_report(grouped[name], min_cell=min_cell) for name in grouped
        },
    }
    if unanalysed:
        result["variants_not_analysed_note"] = (
            f"{len(unanalysed)} arm(s) contributed rows that no per_variant cell "
            "describes: "
            + ", ".join(f"{name} (n={count})" for name, count in sorted(unanalysed.items()))
            + ". They are counted in n_rows and in no table row, so "
            "n_rows_analysed plus these counts equals n_rows. They are reported "
            "here rather than dropped because a row that exists and is never "
            "looked at is indistinguishable from a row that was never run."
        )
    return result


def analyze_escalation_ablation(
    rows: Sequence[RowLike],
    *,
    variants: Sequence[str] = DEFAULT_ESCALATION_VARIANTS,
    min_cell: int = MIN_CELL,
) -> dict[str, Any]:
    """E2: route-and-stop vs measure-and-stop vs measure-and-recover.

    The three variants are read from `evaluation.ablation.escalation_variants`,
    which defines them as *configuration*: A turns off both the sufficiency check
    and escalation, B computes sufficiency and records its verdict without acting
    on it, C is the shipped configuration. The separation matters -- A vs B is
    the cost of measuring sufficiency, B vs C is the value of acting on it, and
    only reading the variants from one definition keeps those two deltas from
    quietly describing different systems.
    """
    return _variant_study(
        rows,
        experiment_id=E2,
        study="sufficiency check and bounded escalation, as three configurations",
        variants=variants,
        min_cell=min_cell,
    )


def analyze_feature_ablation(
    rows: Sequence[RowLike],
    *,
    variants: Sequence[str] = DEFAULT_FEATURE_VARIANTS,
    min_cell: int = MIN_CELL,
) -> dict[str, Any]:
    """E3: the full router against one arm per disabled feature group.

    The grid is `evaluation.ablation.feature_group_variants`, which iterates
    `schemas.config.FEATURE_GROUPS` -- all six of `lexical`, `semantic`,
    `entity`, `complexity`, `question_type`, `multi_concept`. `entity` carries
    the largest weight for BM25 in the shipped rule table (1.5, against 1.0 for
    `lexical`), so an ablation that skipped it would leave the most influential
    signal unmeasured; deriving the list from the config rather than writing it
    out is what prevents that. A group already disabled in the base config is
    skipped by `feature_group_variants` for the same reason: dropping it again
    would report a phantom effect.
    """
    return _variant_study(
        rows,
        experiment_id=E3,
        study="leave-one-out ablation over the router's six feature groups",
        variants=variants,
        min_cell=min_cell,
    )


def analyze_threshold_sweep(
    rows: Sequence[RowLike],
    *,
    variants: Sequence[str] = DEFAULT_THRESHOLD_VARIANTS,
    min_cell: int = MIN_CELL,
) -> dict[str, Any]:
    """E4: one sufficiency threshold at a time.

    The candidate values are `evaluation.ablation.threshold_sweep`'s, and the
    field is `sufficiency_threshold` -- the repository states the policy, so the
    sweep names settings worth measuring rather than inventing a scoring rule.
    Each candidate reports quality, latency, cost, escalation rate and the
    strategy distribution, which is what makes the trade-off legible: a lower
    threshold escalates more often, and the question the table answers is
    whether the extra escalations bought quality.

    E4's registered arms also include `ablation.escalation_step_sweep`'s four
    `max_escalation_steps` values, which this variant list does not describe.
    Those rows are named by `_variant_study` and, when they are present, this
    function adds why they are not tabulated: they are degenerate by
    construction, so a row of equal numbers for them would be an artefact of
    the retriever's structure rather than a measurement. Widening `variants`
    would produce exactly the table that misleads, so the disclosure is the
    output instead.
    """
    result = _variant_study(
        rows,
        experiment_id=E4,
        study="sufficiency-threshold sweep, one parameter at a time",
        variants=variants,
        min_cell=min_cell,
    )
    skipped = [
        name
        for name in result.get("variants_not_analysed", {})
        if name.startswith(ESCALATION_STEP_PREFIX)
    ]
    if skipped:
        result["escalation_step_sweep_not_analysed"] = {
            "variants": sorted(skipped),
            "rows": sum(result["variants_not_analysed"][name] for name in skipped),
            "reason": STEP_SWEEP_NOT_ANALYSED,
        }
    return result


def analyze_cost_weight(
    rows: Sequence[RowLike],
    *,
    variants: Sequence[str] = DEFAULT_COST_WEIGHT_VARIANTS,
    min_cell: int = MIN_CELL,
) -> dict[str, Any]:
    """E5: the quality-versus-cost knob.

    `variants` defaults to `evaluation.ablation.cost_weight_sweep`'s values,
    which are the ones the router actually implements: 0.0 is pure evidence
    (cost ignored entirely), 0.25 is the shipped default, 0.5 weights measured
    cost equally with a full-strength signal, 0.75 and 1.0 push further toward
    efficiency, and the router's own rule is that higher values subtract
    `cost_weight * cost(strategy) / max_cost` from each strategy's score. No
    value is invented here; the regimes are the ablation module's descriptions.
    """
    return _variant_study(
        rows,
        experiment_id=E5,
        study="cost_weight sweep over RoutingConfig.cost_weight",
        variants=variants,
        min_cell=min_cell,
    )


# --------------------------------------------------------------------------
# E6 -- routing overhead
# --------------------------------------------------------------------------


def analyze_routing_overhead(rows: Sequence[RowLike]) -> dict[str, Any]:
    """E6: what the decision layer itself costs, measured rather than asserted.

    Routing latency is a real clock: the feature extraction, the weighted
    evidence table, the cost normalisation and the sufficiency check all happen
    before retrieval and are inside `total_latency_ms`. The right question is
    not "is it negligible" -- that is a conclusion, and conclusions do not belong
    in a measurement function -- but "what fraction is it". So this reports the
    four clocks side by side, the per-query ratio, and the aggregate ratio, and
    says nothing beyond them.

    The ratio is computed two ways because they answer different questions. The
    per-query distribution (`routing_share_of_total`) is the honest one: it is
    the fraction each individual query paid, and it is invariant to a single slow
    retrieval diluting a shared total. The aggregate ratio (sum routing / sum
    total) is reported beside it because it is the number that would be billed,
    and the two can differ by a lot on a right-skewed workload.
    """
    routed = [r for r in rows if _get(r, "routing_latency_ms") is not None]
    latency = {column: distribution(_values(routed, column)) for column in LATENCY_COLUMNS}

    paired: list[tuple[float, float]] = []
    for row in routed:
        routing = _get(row, "routing_latency_ms")
        total = _get(row, "total_latency_ms")
        if isinstance(total, (int, float)) and float(total) > 0.0:
            paired.append((float(routing), float(total)))
    shares = [routing / total for routing, total in paired]
    total_routing = sum(routing for routing, _ in paired)
    total_all = sum(total for _, total in paired)

    return {
        "analysis_version": ANALYSIS_VERSION,
        "experiment_id": E6,
        "study": "routing decision overhead, measured against the arms that have one",
        "n_rows": len(rows),
        **_cell(len(routed), min_cell=1),
        "n_rows_with_total": len(paired),
        "n_rows_without_total": len(routed) - len(paired),
        "note": (
            "Only rows carrying a routing_latency_ms are measured; a fixed-strategy "
            "arm records None there because no decision was taken. The share is "
            "reported per query and in aggregate, and no threshold for 'acceptable' "
            "is applied: the measured ratio is the finding."
        ),
        "latency": latency,
        "routing_share_of_total": {
            "per_query": distribution(shares),
            "aggregate": (round(total_routing / total_all, 6) if total_all > 0.0 else None),
        },
        "mean_stage_latency_ms": distribution(
            [v for row in routed for v in (_get(row, "stage_latencies_ms") or [])]
        ),
    }


# --------------------------------------------------------------------------
# E7 -- per query category
# --------------------------------------------------------------------------


def _query_trace(row: RowLike) -> dict[str, Any]:
    """The per-query record E7 preserves for inspection.

    One row, flattened. E7's aggregate cells are means over these rows, and a
    mean with no rows behind it cannot be checked, so they travel with the
    result.
    """
    return {
        "query_id": _get(row, "query_id"),
        "system": _get(row, "system"),
        "category": _get(row, "category"),
        "split": _get(row, "split"),
        "status": _get(row, "status"),
        "initial_strategy": _get(row, "initial_strategy"),
        "final_strategy": _get(row, "final_strategy"),
        "escalated": _get(row, "escalated"),
        "escalation_target": _get(row, "escalation_target"),
        "routing_confidence": _get(row, "routing_confidence"),
        "sufficiency_score": _get(row, "sufficiency_score"),
        "stage_count": _get(row, "stage_count"),
        "recall_at_5": _get(row, "recall_at_5"),
        "recall_at_10": _get(row, "recall_at_10"),
        "mrr": _get(row, "mrr"),
        "ndcg_at_5": _get(row, "ndcg_at_5"),
        "retrieval_latency_ms": _get(row, "retrieval_latency_ms"),
        "routing_latency_ms": _get(row, "routing_latency_ms"),
        "total_latency_ms": _get(row, "total_latency_ms"),
        "context_tokens": _get(row, "context_tokens"),
        "estimated_cost_usd": _get(row, "estimated_cost_usd"),
    }


def _category_block(rows: Sequence[RowLike], *, min_cell: int) -> dict[str, Any]:
    escalation = _escalation_rate(rows)
    strategies = _strategy_distribution(rows)
    return {
        **_cell(len(rows), min_cell=min_cell),
        "strategy_selection": strategies,
        "escalation": escalation,
        "escalation_rate": escalation["escalation_rate"],
        "quality": _quality(rows),
        "latency": {column: distribution(_values(rows, column)) for column in LATENCY_COLUMNS},
        "cost": _cost(rows),
    }


def _category_order(category: str) -> int:
    """Report categories in `VALID_CATEGORIES` order, unknown ones last."""
    order = sorted(VALID_CATEGORIES)
    return order.index(category) if category in order else len(order)


def analyze_query_type(
    rows: Sequence[RowLike],
    *,
    min_cell: int = MIN_CELL,
    include_traces: bool = True,
) -> dict[str, Any]:
    """E7: the per-category breakdown, keyed on the dataset's own label.

    `category` is the only ground-truth query-type column that exists. It is one
    of six hand-assigned values (`factual`, `terminology`, `conceptual`,
    `comparative`, `multi_document`, `fine_grained`), validated by
    `validate_phase7_dataset.py` and therefore usable as a grouping key.

    The router's runtime `QueryFeatures` are a different thing and are absent
    from this result. `question_type`, `complexity_score`, `concept_count`,
    `entity_count` and friends are *derived from the query text by the router's
    own heuristics*, and the router then selects a strategy using them, so
    "queries the router calls complex get the complex strategy" is the router
    agreeing with itself and says nothing about whether the routing helped. They
    are also not columns on a row at all, which is the second reason: producing
    them here would mean re-running the analyzer over the query text, silently
    duplicating a phase-6 derivation in the reporting layer. `derived_breakdown`
    records that absence explicitly, with `ground_truth: false`, instead of
    leaving a reader to wonder whether a breakdown was run and found empty.

    Categories present in the rows are reported even when below `min_cell`; a
    category present in the dataset but absent from the rows is reported as an
    empty cell rather than omitted, so "the router never saw comparative
    queries" is visible.
    """
    by_category: dict[str, list[RowLike]] = {}
    for row in rows:
        by_category.setdefault(str(_get(row, "category")), []).append(row)

    ordered = sorted(by_category, key=lambda c: (_category_order(c), c))
    return {
        "analysis_version": ANALYSIS_VERSION,
        "experiment_id": E7,
        "study": "per-category breakdown on the dataset's ground-truth label",
        "n_rows": len(rows),
        "min_cell": min_cell,
        "categories_present": ordered,
        "categories_absent": sorted(VALID_CATEGORIES - set(by_category)),
        "category_source": (
            "EvaluationExample.category -- a hand-assigned dataset label, validated "
            "by scripts/validate_phase7_dataset.py"
        ),
        "derived_breakdown": {
            "ground_truth": False,
            "available": False,
            "reason": (
                "The router's QueryFeatures (question_type, complexity_score, "
                "concept_count, entity_count, ...) are runtime derivations from the "
                "query text, not dataset labels, and are not columns on an exported "
                "row. A breakdown over them would measure the router against its own "
                "inputs, so none is reported here. Any such table must be labelled "
                "'derived' wherever it appears."
            ),
            "fields_not_reported": [
                "question_type",
                "complexity_score",
                "concept_count",
                "entity_count",
                "is_multi_hop",
                "has_numerical",
            ],
        },
        "per_category": {
            category: _category_block(by_category[category], min_cell=min_cell)
            for category in ordered
        },
        "traces": (
            [
                _query_trace(row)
                for row in sorted(
                    rows,
                    key=lambda r: (str(_get(r, "query_id")), str(_get(r, "system"))),
                )
            ]
            if include_traces
            else []
        ),
    }


# --------------------------------------------------------------------------
# E8 -- escalation transitions
# --------------------------------------------------------------------------


def gold_document_map(examples: Iterable[Any]) -> dict[str, set[str]]:
    """`query_id -> relevant document ids`, read straight off the dataset.

    The document-level gold set is a label, so it needs no corpus: it is
    `EvaluationExample.relevant_documents` verbatim. This is the mapping E8 uses
    for its document-level quality-before when a caller has no chunk-level gold.
    """
    return {str(e.example_id): set(e.relevant_documents) for e in examples}


def gold_chunk_map(
    examples: Iterable[Any], *, chunks_dir: Path | str = CHUNKS_DIR
) -> dict[str, set[str]]:
    """`query_id -> relevant chunk ids`, using the dataset's own relevance rule.

    `evaluation.dataset.relevant_chunk_ids` defines relevance for a *trace* by
    intersecting the chunks that trace retrieved with the example's labels. E8
    needs the same set for chunks the row records as the *pre-escalation*
    retrieval, which no trace object is available for, so the rule is applied
    here over the whole corpus instead of over one retrieval: a chunk is
    relevant when its document is in `relevant_documents` and, where the example
    labels sections for that document, its `section_path` starts with one of the
    labelled prefixes. Identical rule, different enumeration -- and it never
    reads the retrieval output, so the labels still cannot depend on what a
    system returned.

    Returns an empty mapping when the corpus is absent rather than raising: the
    chunk files are gitignored and regenerable, and E8 without chunk-level gold
    still reports its transitions, document-level quality, latency and cost. The
    caller sees the missing gold through the result's `gold_chunk_ids_available`
    flag rather than through an exception.
    """
    materialised = list(examples)
    directory = Path(chunks_dir)
    if not directory.is_dir():
        return {}

    by_id: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.chunks.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_document = record["document_id"]
                section_path = list(record["metadata"]["section_path"])
                for example in materialised:
                    if chunk_document not in example.relevant_documents:
                        continue
                    prefixes = [
                        ref.section_path_prefix
                        for ref in example.relevant_sections
                        if ref.document_id == chunk_document
                    ]
                    if prefixes and not any(
                        section_path[: len(prefix)] == prefix for prefix in prefixes
                    ):
                        continue
                    by_id.setdefault(str(example.example_id), set()).add(record["chunk_id"])
    return by_id


def _pre_escalation_quality(
    row: RowLike,
    *,
    gold_chunk_ids: Mapping[str, set[str]] | None,
    top_k: int,
) -> tuple[float | None, str | None]:
    """Quality of the *pre-escalation* retrieval for one escalated row.

    Returns `(value, reason)`. Exactly one of the two is ever `None`, and the
    distinction is the whole point of this function:

    * `pre_escalation_chunk_ids is None` -- the field is absent, which
      `rows.py` documents as "the query never escalated, or nothing was
      discarded". Quality before is **unknown**. It is never reported as 0.0: a
      zero is a real observation, it is what a stage that retrieved nothing
      scores, and conflating the two would understate escalation by counting
      unmeasured queries as measured failures.
    * `pre_escalation_chunk_ids == []` -- a real observation that the first stage
      returned nothing. Quality before is 0.0, legitimately.
    * a non-empty list with gold available -- the real metric, computed the way
      `RetrievalEvaluator._recall_at_k` computes it: gold chunks found among the
      first `k` retrieved, divided by the number of gold chunks.
    * a non-empty list with no gold supplied -- unknown, with the reason.

    Without gold the function returns `(None, reason)` rather than 0.0 for the
    same reason it does for an absent field: the measurement was not made, and
    reporting it as a measurement is the one failure E8 cannot make.
    """
    identifiers = _get(row, "pre_escalation_chunk_ids")
    if identifiers is None:
        return None, (
            "pre_escalation_chunk_ids is absent (the row recorded no pre-escalation "
            "retrieval)"
        )
    if identifiers == []:
        return 0.0, "the pre-escalation stage returned no chunks"
    if not gold_chunk_ids:
        return None, "no chunk-level relevance labels were supplied"

    gold = gold_chunk_ids.get(str(_get(row, "query_id")))
    if not gold:
        return None, "no relevant chunks are labelled for this query"
    retrieved = list(identifiers)[:top_k]
    hits = sum(1 for identifier in retrieved if identifier in gold)
    return round(hits / len(gold), 6), None


def _pre_escalation_document_quality(
    row: RowLike,
    *,
    gold_document_ids: Mapping[str, set[str]] | None,
) -> tuple[float | None, str | None]:
    """Same, at document granularity, using `pre_escalation_document_ids`."""
    identifiers = _get(row, "pre_escalation_document_ids")
    if identifiers is None:
        return None, (
            "pre_escalation_document_ids is absent (the row recorded no "
            "pre-escalation retrieval)"
        )
    if identifiers == []:
        return 0.0, "the pre-escalation stage returned no documents"
    if not gold_document_ids:
        return None, "no document-level relevance labels were supplied"
    gold = gold_document_ids.get(str(_get(row, "query_id")))
    if not gold:
        return None, "no relevant documents are labelled for this query"
    hits = sum(1 for identifier in identifiers if identifier in gold)
    return round(hits / len(gold), 6), None


def _quality_change(
    before_values: Sequence[float], after_values: Sequence[float]
) -> dict[str, Any]:
    """Before/after/improvement over the escalated queries that have both.

    The improvement rate is the share of *paired* queries whose metric increased,
    computed only over queries where both numbers exist. Denominators are
    reported next to every rate, because a rate computed over a silently
    filtered subset is the most common way a transition table overstates itself.
    """
    pairs = list(zip(before_values, after_values))
    n = len(pairs)
    improved = sum(1 for before, after in pairs if after > before)
    worsened = sum(1 for before, after in pairs if after < before)
    return {
        "n_paired": n,
        "n_improved": improved,
        "n_worsened": worsened,
        "n_unchanged": n - improved - worsened,
        "improvement_rate": round(improved / n, 6) if n else None,
        "worsening_rate": round(worsened / n, 6) if n else None,
        "mean_before": round(sum(b for b, _ in pairs) / n, 6) if n else None,
        "mean_after": round(sum(a for _, a in pairs) / n, 6) if n else None,
        "median_delta": (
            round(float(median([a - b for b, a in pairs])), 6) if n else None
        ),
    }


def analyze_escalation_transitions(
    rows: Sequence[RowLike],
    *,
    gold_chunk_ids: Mapping[str, set[str]] | None = None,
    gold_document_ids: Mapping[str, set[str]] | None = None,
    metric: str = PRIMARY_QUALITY_METRIC,
    top_k: int = 5,
    min_cell: int = MIN_CELL,
) -> dict[str, Any]:
    """E8: what escalation actually bought, per observed transition.

    For each `(from -> to)` pair that occurred: occurrences, the share of
    escalated queries, quality before, quality after, the improvement rate, the
    added latency and the added cost. Unobserved pairs are absent from
    `transitions` entirely -- never a zero row -- so "no escalation from bm25 to
    dense ever happened" is stated by the pair's absence rather than by a zero
    that would be indistinguishable from a real 0.0%.

    Quality before is measured against `pre_escalation_chunk_ids` /
    `pre_escalation_document_ids`. Those columns are populated **only on
    escalated rows**, so a non-escalated row has `None` there; `None` is treated
    as "not recorded" and is never read as zero. See
    `_pre_escalation_quality` for the three-way distinction between absent
    (unknown), empty (a measured zero) and populated (a real metric).

    `metric` must be a quality column of the row export. A latency column would
    make "quality improvement" a statement about milliseconds.
    """
    if metric not in QUALITY_METRICS:
        raise ValueError(
            f"metric {metric!r} is not a quality column; E8 measures the effect of "
            f"escalation on quality, so expected one of {list(QUALITY_METRICS)}"
        )

    escalated = [r for r in rows if _get(r, "escalated")]
    groups: dict[tuple[str, str], list[RowLike]] = {}
    for row in escalated:
        source = _get(row, "initial_strategy")
        target = _get(row, "escalation_target") or _get(row, "final_strategy")
        if source is None or target is None or source == target:
            continue
        groups.setdefault((str(source), str(target)), []).append(row)

    before_missing = sum(
        1
        for row in escalated
        if _get(row, "pre_escalation_chunk_ids") is None
        and _get(row, "pre_escalation_document_ids") is None
    )

    transitions: list[dict[str, Any]] = []
    for (source, target), group in sorted(groups.items()):
        chunk_before: list[float] = []
        chunk_after: list[float] = []
        doc_before: list[float] = []
        doc_after: list[float] = []
        reasons: dict[str, int] = {}
        for row in group:
            after_value = _get(row, metric)
            chunk_value, chunk_reason = _pre_escalation_quality(
                row, gold_chunk_ids=gold_chunk_ids, top_k=top_k
            )
            if chunk_value is not None and isinstance(after_value, (int, float)):
                chunk_before.append(chunk_value)
                chunk_after.append(float(after_value))
            elif chunk_reason:
                reasons[chunk_reason] = reasons.get(chunk_reason, 0) + 1

            doc_value, doc_reason = _pre_escalation_document_quality(
                row, gold_document_ids=gold_document_ids
            )
            if doc_value is not None and isinstance(after_value, (int, float)):
                doc_before.append(doc_value)
                doc_after.append(float(after_value))
            elif doc_reason:
                reasons[doc_reason] = reasons.get(doc_reason, 0) + 1

        transitions.append(
            {
                "from": source,
                "to": target,
                **_cell(len(group), min_cell=min_cell),
                "occurrences": len(group),
                "share_of_escalated": (
                    round(len(group) / len(escalated), 6) if escalated else None
                ),
                "quality": {
                    "metric": metric,
                    "chunk_level": _quality_change(chunk_before, chunk_after),
                    "document_level": _quality_change(doc_before, doc_after),
                },
                "unpaired_before_reasons": dict(sorted(reasons.items())),
                "after_quality": distribution(_values(group, metric)),
                "added_latency": {
                    "total_latency_ms": distribution(_values(group, "total_latency_ms")),
                    "reranking_latency_ms": distribution(
                        _values(group, "reranking_latency_ms")
                    ),
                    "retrieval_latency_ms": distribution(
                        _values(group, "retrieval_latency_ms")
                    ),
                },
                "added_cost": {
                    "context_tokens": distribution(_values(group, "context_tokens")),
                    "estimated_cost_usd": distribution(_values(group, "estimated_cost_usd")),
                    "total_tokens": distribution(_values(group, "total_tokens")),
                },
            }
        )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "experiment_id": E8,
        "study": "escalation transitions, as observed",
        "metric": metric,
        "n_rows": len(rows),
        "n_escalated": len(escalated),
        "n_escalated_without_pre_escalation_evidence": before_missing,
        "pre_escalation_note": (
            "pre_escalation_* is populated only on escalated rows. A None there "
            "means 'not recorded' and is reported as unknown, never as 0.0; an "
            "empty list is a real observation of an empty first stage and does "
            "score 0.0."
        ),
        "gold_chunk_ids_available": bool(gold_chunk_ids),
        "gold_document_ids_available": bool(gold_document_ids),
        "n_transitions_observed": len(transitions),
        "transitions": transitions,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "COST_COLUMNS",
    "DEFAULT_COST_WEIGHT_VARIANTS",
    "DEFAULT_ESCALATION_VARIANTS",
    "DEFAULT_FEATURE_VARIANTS",
    "DEFAULT_SYSTEMS",
    "DEFAULT_THRESHOLD_VARIANTS",
    "ESCALATION_STEP_PREFIX",
    "LATENCY_COLUMNS",
    "MIN_CELL",
    "NO_COMPOSITE_POLICY",
    "PRIMARY_QUALITY_METRIC",
    "QUALITY_METRICS",
    "STEP_SWEEP_NOT_ANALYSED",
    "analyze_cost_weight",
    "analyze_escalation_ablation",
    "analyze_escalation_transitions",
    "analyze_feature_ablation",
    "analyze_main_comparison",
    "analyze_query_type",
    "analyze_routing_overhead",
    "analyze_threshold_sweep",
    "distribution",
    "gold_chunk_map",
    "gold_document_map",
    "row_field",
]
