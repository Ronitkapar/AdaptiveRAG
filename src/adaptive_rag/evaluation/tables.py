"""
evaluation.tables
-----------------
Deterministic table builders for the Phase 7 report.

Every table is built once, in memory, as a `Table`: a fixed column order, a
list of rows, and the notes that belong with it. `render_markdown`,
`render_csv` and `render_json` are three renderings of the *same* cells, so the
markdown in a write-up, the CSV someone opens in a spreadsheet and the JSON a
downstream script reads cannot disagree -- a difference between them is the
classic way a published number stops matching the file it came from.

Determinism is a property of the cells, not of the caller. Values are formatted
once, by explicit per-column format, and every `None` renders as an empty cell
rather than `0.0`, `None` or `nan`: a metric that was not measured must not be
printable as a number. Row order is the order `evaluation.analysis` produced
(arms in build order, variants in grid order, transitions sorted), so the same
inputs always give the same bytes.

The E1 table is the mandated one:

    System | Recall@5 | MRR | Median latency | P95 latency | Routing latency |
    Retrieval latency | Reranking latency | Total latency | Cost/resource

and it keeps quality and cost in separate columns on purpose. There is no
"overall" column, and `_assert_no_composite` refuses to build a table whose
header claims one -- the research design forbids collapsing quality and cost into
a single number, and a header is exactly where such a number would appear.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.evaluation.analysis import NO_COMPOSITE_POLICY

TABLES_VERSION = "phase7_tables_v1"

# Header fragments that would mean a table had collapsed two axes into one
# number. Matched case-insensitively against every column of every table built
# here, so the guard is not limited to the E1 table -- a future builder cannot
# reintroduce the composite through a different study.
FORBIDDEN_COMPOSITE_TOKENS: tuple[str, ...] = (
    "composite",
    "overall_score",
    "overall score",
    "winner",
    "best_system",
    "utility_score",
    "combined_score",
    "tradeoff_score",
)

# The mandated E1 columns, in the mandated order.
MAIN_COMPARISON_COLUMNS: tuple[str, ...] = (
    "System",
    "Recall@5",
    "MRR",
    "Median latency (ms)",
    "P95 latency (ms)",
    "Routing latency (ms)",
    "Retrieval latency (ms)",
    "Reranking latency (ms)",
    "Total latency (ms)",
    "Cost/resource",
    "n",
    "insufficient_data",
)

# Every table ends with these two, so a reader of any one of them can see the
# sample size and whether the cell is large enough to generalise from.
_TRAILER = ("n", "insufficient_data")


class Table(BaseModel):
    """One rendered table: fixed columns, string cells, and its notes."""

    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    experiment_id: str
    columns: tuple[str, ...]
    rows: list[dict[str, str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Set by a builder that produced no rows *and* knows why, e.g. E8 when no
    # query escalated. It is a statement about the study, not a cell, so it never
    # becomes a row: `rows` stays empty and every renderer says why, because a
    # header with nothing under it is otherwise indistinguishable from a study
    # that was run and produced no findings. Left `None` when rows exist, and when
    # a builder genuinely does not know why a table is empty -- silence is still
    # better than an invented reason.
    empty_reason: str | None = None

    def model_post_init(self, _context: Any) -> None:
        _assert_no_composite(self.columns)
        for row in self.rows:
            unknown = [key for key in row if key not in self.columns]
            if unknown:
                raise ValueError(
                    f"table {self.name!r} has cells for unknown columns {unknown}; "
                    f"columns are {list(self.columns)}"
                )
        if self.empty_reason is not None and self.rows:
            raise ValueError(
                f"table {self.name!r} declares an empty_reason but has rows; the "
                "reason would contradict them"
            )


# --------------------------------------------------------------------------
# cell formatting
# --------------------------------------------------------------------------


def _assert_no_composite(columns: Sequence[str]) -> None:
    """Refuse a header that claims a single combined quality/cost number."""
    lowered = [column.lower() for column in columns]
    hits = [
        token
        for token in FORBIDDEN_COMPOSITE_TOKENS
        if any(token in column for column in lowered)
    ]
    if hits:
        raise ValueError(
            f"table columns {list(columns)} name a composite score ({hits}). "
            f"{NO_COMPOSITE_POLICY}"
        )


def _num(value: Any, digits: int) -> str:
    """A fixed-decimal number, or an empty cell when the value is absent.

    The empty cell is the important half. Latency and cost columns are `None` on
    a fixed-strategy arm and on a run that failed before that clock started;
    printing those as `0.0` would make a system look infinitely fast.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    """A rate as a percentage, empty when absent."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def _int(value: Any) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return str(int(value))


def _flag(insufficient: Any) -> str:
    """`yes` / `no` for the sample-size flag, empty when the cell has no verdict."""
    if insufficient is None:
        return ""
    return "yes" if insufficient else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _md(value: str) -> str:
    """Escape a cell for a markdown table."""
    return value.replace("|", "\\|")


def _mean(summary: Mapping[str, Any] | None, *path: str) -> Any:
    """Walk a nested summary mapping, returning `None` if any step is missing."""
    node: Any = summary
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _latency(cell: Mapping[str, Any], column: str, statistic: str) -> str:
    return _num(_mean(cell.get("latency"), column, statistic), 2)


def _quality_mean(cell: Mapping[str, Any], metric: str) -> str:
    return _num(_mean(cell.get("quality"), metric, "mean"), 4)


def _quality_median(cell: Mapping[str, Any], metric: str) -> str:
    return _num(_mean(cell.get("quality"), metric, "median"), 4)


def _trailer(cell: Mapping[str, Any]) -> dict[str, str]:
    return {
        "n": _int(cell.get("n")),
        "insufficient_data": _flag(cell.get("insufficient_data")),
    }


def _top_strategy(distribution: Mapping[str, Any] | None, field: str) -> str:
    """The most-selected strategy in one distribution, with its share.

    Ties are broken by name, because the counts dict is already sorted by name
    and `max` on `(count, name)` is order-independent either way -- but making
    the rule explicit is cheaper than explaining a flaky table later.
    """
    if not distribution:
        return ""
    field_data = distribution.get(field) or {}
    counts = field_data.get("counts") or {}
    observed = field_data.get("observed") or 0
    if not counts or not observed:
        return ""
    name = max(sorted(counts), key=lambda key: (counts[key], key))
    return f"{name} ({_pct(counts[name] / observed, 0)})"


def _resource(cost: Mapping[str, Any] | None) -> str:
    """The `Cost/resource` cell: tokens and money, both or neither.

    One column rather than three because the mandated E1 header has one
    `Cost/resource` column, and splitting it would change the header. The
    format is `ctx=<n> total=<n> usd=<v>`; `usd` is omitted when no run
    recorded a cost, so an absent cost is not printed as `$0.0000`.
    """
    if not cost:
        return ""
    parts: list[str] = []
    context = cost.get("context_tokens") or {}
    total = cost.get("total_tokens") or {}
    money = cost.get("estimated_cost_usd") or {}
    if context.get("mean") is not None:
        parts.append(f"ctx={_num(context['mean'], 0)}")
    if total.get("mean") is not None:
        parts.append(f"total={_num(total['mean'], 0)}")
    if money.get("mean") is not None:
        parts.append(f"usd={_num(money['mean'], 6)}")
    return " ".join(parts)


# --------------------------------------------------------------------------
# table builders
# --------------------------------------------------------------------------


def main_comparison_table(analysis: Mapping[str, Any]) -> Table:
    """E1: the mandated main-comparison table.

    `Median latency` and `P95 latency` are the median and p95 of
    `total_latency_ms`; the three stage columns are the medians of
    `routing_latency_ms`, `retrieval_latency_ms` and `reranking_latency_ms`; and
    `Total latency` is the **mean** of `total_latency_ms`, printed beside the
    median and p95 precisely so the right tail's effect on it is visible. The
    `Cost/resource` cell carries context tokens, total tokens and the estimated
    dollar cost.
    """
    columns = MAIN_COMPARISON_COLUMNS
    rows: list[dict[str, str]] = []
    for system, cell in analysis.get("per_system", {}).items():
        rows.append(
            {
                "System": _text(system),
                "Recall@5": _quality_mean(cell, "recall_at_5"),
                "MRR": _quality_mean(cell, "mrr"),
                "Median latency (ms)": _latency(cell, "total_latency_ms", "median"),
                "P95 latency (ms)": _latency(cell, "total_latency_ms", "p95"),
                "Routing latency (ms)": _latency(cell, "routing_latency_ms", "median"),
                "Retrieval latency (ms)": _latency(cell, "retrieval_latency_ms", "median"),
                "Reranking latency (ms)": _latency(cell, "reranking_latency_ms", "median"),
                "Total latency (ms)": _latency(cell, "total_latency_ms", "mean"),
                "Cost/resource": _resource(cell.get("cost")),
                **_trailer(cell),
            }
        )
    return Table(
        name="e1_main_comparison",
        title="E1 -- main comparison of the five benchmark arms",
        experiment_id="E1_baseline_comparison",
        columns=columns,
        rows=rows,
        notes=[
            "Latency columns: 'Median latency' and 'P95 latency' are the median and "
            "nearest-rank p95 of total_latency_ms; the three stage columns are stage "
            "medians; 'Total latency' is the mean, shown so the right tail's effect "
            "is visible rather than hidden behind a median.",
            "p95 is nearest-rank with no interpolation, matching the estimator the "
            "Phase 7 cost table was frozen with.",
            NO_COMPOSITE_POLICY,
        ],
    )


def escalation_ablation_table(analysis: Mapping[str, Any]) -> Table:
    """E2: the A/B/C sufficiency-and-escalation ablation."""
    columns = (
        "Variant",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Median total (ms)",
        "P95 total (ms)",
        "Escalation rate",
        "Mean stage count",
        "Cost/resource",
        *_TRAILER,
    )
    rows: list[dict[str, str]] = []
    for variant, cell in analysis.get("per_variant", {}).items():
        rows.append(
            {
                "Variant": _text(variant),
                "Recall@5": _quality_mean(cell, "recall_at_5"),
                "MRR": _quality_mean(cell, "mrr"),
                "nDCG@5": _quality_mean(cell, "ndcg_at_5"),
                "Median total (ms)": _latency(cell, "total_latency_ms", "median"),
                "P95 total (ms)": _latency(cell, "total_latency_ms", "p95"),
                "Escalation rate": _pct(cell.get("escalation_rate")),
                "Mean stage count": _num(
                    _mean(cell.get("routing"), "stage_count", "mean"), 2
                ),
                "Cost/resource": _resource(cell.get("cost")),
                **_trailer(cell),
            }
        )
    return Table(
        name="e2_escalation_ablation",
        title="E2 -- sufficiency check and bounded escalation (A/B/C)",
        experiment_id="E2_escalation_ablation",
        columns=columns,
        rows=rows,
        notes=[
            "A routes once and stops; B computes sufficiency and records it without "
            "acting; C is the shipped configuration (measure and act, bounded by "
            "max_escalation_steps). A-vs-B is the cost of measuring sufficiency, "
            "B-vs-C is the value of acting on it.",
            "Escalation rate is over the rows that recorded a routing decision.",
        ],
    )


def feature_ablation_table(analysis: Mapping[str, Any]) -> Table:
    """E3: the full router against each leave-one-out feature group."""
    columns = (
        "Variant",
        "Recall@5",
        "MRR",
        "Median total (ms)",
        "P95 total (ms)",
        "Mean routing confidence",
        "Top initial strategy",
        "Top final strategy",
        "Escalation rate",
        "Mean stage count",
        "Cost/resource",
        *_TRAILER,
    )
    rows: list[dict[str, str]] = []
    for variant, cell in analysis.get("per_variant", {}).items():
        strategies = cell.get("strategy_distribution") or {}
        rows.append(
            {
                "Variant": _text(variant),
                "Recall@5": _quality_mean(cell, "recall_at_5"),
                "MRR": _quality_mean(cell, "mrr"),
                "Median total (ms)": _latency(cell, "total_latency_ms", "median"),
                "P95 total (ms)": _latency(cell, "total_latency_ms", "p95"),
                "Mean routing confidence": _num(
                    _mean(cell.get("routing"), "routing_confidence", "mean"), 4
                ),
                "Top initial strategy": _top_strategy(strategies, "initial_strategy"),
                "Top final strategy": _top_strategy(strategies, "final_strategy"),
                "Escalation rate": _pct(cell.get("escalation_rate")),
                "Mean stage count": _num(
                    _mean(cell.get("routing"), "stage_count", "mean"), 2
                ),
                "Cost/resource": _resource(cell.get("cost")),
                **_trailer(cell),
            }
        )
    return Table(
        name="e3_feature_ablation",
        title="E3 -- leave-one-out ablation over the router's six feature groups",
        experiment_id="E3_feature_ablation",
        columns=columns,
        rows=rows,
        notes=[
            "'full' is the shipped router; each 'without_*' variant disables exactly "
            "one group from schemas.config.FEATURE_GROUPS (lexical, semantic, entity, "
            "complexity, question_type, multi_concept).",
            "Mean routing confidence is over the rows that recorded a decision; a "
            "fixed-strategy row has none and is excluded rather than counted as zero.",
        ],
    )


def threshold_sweep_table(analysis: Mapping[str, Any]) -> Table:
    """E4: one sufficiency threshold at a time.

    E4 registered two sweeps under one study id, so the step-sweep arms may be
    present in the rows without appearing in this table. When they are, the note
    says so here rather than only in the JSON: the markdown is the artifact that
    gets pasted into a write-up, and a silently narrower table is the failure
    this module exists to prevent.
    """
    columns = (
        "Threshold",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Median total (ms)",
        "Escalation rate",
        "Mean stage count",
        "Top initial strategy",
        "Cost/resource",
        *_TRAILER,
    )
    rows: list[dict[str, str]] = []
    for variant, cell in analysis.get("per_variant", {}).items():
        rows.append(
            {
                "Threshold": _text(variant),
                "Recall@5": _quality_mean(cell, "recall_at_5"),
                "MRR": _quality_mean(cell, "mrr"),
                "nDCG@5": _quality_mean(cell, "ndcg_at_5"),
                "Median total (ms)": _latency(cell, "total_latency_ms", "median"),
                "Escalation rate": _pct(cell.get("escalation_rate")),
                "Mean stage count": _num(
                    _mean(cell.get("routing"), "stage_count", "mean"), 2
                ),
                "Top initial strategy": _top_strategy(
                    cell.get("strategy_distribution"), "initial_strategy"
                ),
                "Cost/resource": _resource(cell.get("cost")),
                **_trailer(cell),
            }
        )
    notes = [
        "The swept field is RoutingConfig.sufficiency_threshold; the candidate "
        "values are evaluation.ablation.threshold_sweep's, not chosen here.",
        "The escalation rate is the axis to read alongside quality: a lower "
        "threshold escalates more often, and this table is what says whether "
        "the extra escalations bought quality.",
    ]
    step_sweep = analysis.get("escalation_step_sweep_not_analysed")
    if step_sweep:
        notes.append(
            "NOT ANALYSED -- "
            + ", ".join(step_sweep.get("variants", []))
            + f" ({step_sweep.get('rows', 0)} rows): "
            + str(step_sweep.get("reason", ""))
        )
    return Table(
        name="e4_threshold_sweep",
        title="E4 -- sufficiency-threshold sweep",
        experiment_id="E4_threshold_calibration",
        columns=columns,
        rows=rows,
        notes=notes,
    )


def cost_weight_table(analysis: Mapping[str, Any]) -> Table:
    """E5: the quality-versus-cost knob, on the sweep's own values."""
    columns = (
        "cost_weight",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Median total (ms)",
        "P95 total (ms)",
        "Escalation rate",
        "Top initial strategy",
        "Cost/resource",
        *_TRAILER,
    )
    rows: list[dict[str, str]] = []
    for variant, cell in analysis.get("per_variant", {}).items():
        rows.append(
            {
                "cost_weight": _text(variant),
                "Recall@5": _quality_mean(cell, "recall_at_5"),
                "MRR": _quality_mean(cell, "mrr"),
                "nDCG@5": _quality_mean(cell, "ndcg_at_5"),
                "Median total (ms)": _latency(cell, "total_latency_ms", "median"),
                "P95 total (ms)": _latency(cell, "total_latency_ms", "p95"),
                "Escalation rate": _pct(cell.get("escalation_rate")),
                "Top initial strategy": _top_strategy(
                    cell.get("strategy_distribution"), "initial_strategy"
                ),
                "Cost/resource": _resource(cell.get("cost")),
                **_trailer(cell),
            }
        )
    return Table(
        name="e5_cost_weight",
        title="E5 -- cost_weight sweep over RoutingConfig.cost_weight",
        experiment_id="E5_cost_weight",
        columns=columns,
        rows=rows,
        notes=[
            "The swept values are evaluation.ablation.cost_weight_sweep's: 0.0 is "
            "pure evidence (cost ignored), 0.25 is the shipped default, 0.5 weights "
            "measured cost equally with a full-strength signal, 0.75 and 1.0 push "
            "further toward efficiency. The router subtracts "
            "cost_weight * cost(strategy) / max_cost from each strategy's score.",
        ],
    )


def routing_overhead_table(analysis: Mapping[str, Any]) -> Table:
    """E6: the decision layer's own cost, measured (supplementary to the mandated set)."""
    latency = analysis.get("latency") or {}
    share = analysis.get("routing_share_of_total") or {}
    per_query = share.get("per_query") or {}
    columns = (
        "Clock",
        "n",
        "mean (ms)",
        "median (ms)",
        "p95 (ms)",
        "min (ms)",
        "max (ms)",
    )
    rows: list[dict[str, str]] = []
    for column in (
        "routing_latency_ms",
        "retrieval_latency_ms",
        "reranking_latency_ms",
        "total_latency_ms",
    ):
        cell = latency.get(column) or {}
        rows.append(
            {
                "Clock": _text(column),
                "n": _int(cell.get("n")),
                "mean (ms)": _num(cell.get("mean"), 2),
                "median (ms)": _num(cell.get("median"), 2),
                "p95 (ms)": _num(cell.get("p95"), 2),
                "min (ms)": _num(cell.get("min"), 2),
                "max (ms)": _num(cell.get("max"), 2),
            }
        )
    return Table(
        name="e6_routing_overhead",
        title="E6 -- routing decision overhead",
        experiment_id="E6_routing_overhead",
        columns=columns,
        rows=rows,
        notes=[
            f"Routing share of total latency: per-query mean "
            f"{_pct(per_query.get('mean'), 2)}, median {_pct(per_query.get('median'), 2)}, "
            f"aggregate {_pct(share.get('aggregate'), 2)}.",
            "Only rows carrying a routing_latency_ms are measured; a fixed-strategy "
            "arm records None there because no decision was taken.",
            "No threshold for 'acceptable' is applied. The measured ratio is the "
            "finding, and a claim that routing overhead is negligible is a "
            "conclusion this table does not make on the measurement's behalf.",
        ],
    )


def query_type_table(analysis: Mapping[str, Any]) -> Table:
    """E7: the per-category breakdown, on the dataset's own label."""
    columns = (
        "Category",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Median total (ms)",
        "Escalation rate",
        "Top initial strategy",
        "Top final strategy",
        "Cost/resource",
        *_TRAILER,
    )
    rows: list[dict[str, str]] = []
    for category, cell in analysis.get("per_category", {}).items():
        strategies = cell.get("strategy_selection") or {}
        rows.append(
            {
                "Category": _text(category),
                "Recall@5": _quality_mean(cell, "recall_at_5"),
                "MRR": _quality_mean(cell, "mrr"),
                "nDCG@5": _quality_mean(cell, "ndcg_at_5"),
                "Median total (ms)": _latency(cell, "total_latency_ms", "median"),
                "Escalation rate": _pct(cell.get("escalation_rate")),
                "Top initial strategy": _top_strategy(strategies, "initial_strategy"),
                "Top final strategy": _top_strategy(strategies, "final_strategy"),
                "Cost/resource": _resource(cell.get("cost")),
                **_trailer(cell),
            }
        )
    derived = analysis.get("derived_breakdown") or {}
    notes = [
        "Category is EvaluationExample.category -- a hand-assigned dataset label "
        "validated by scripts/validate_phase7_dataset.py. It is the only "
        "ground-truth query-type column that exists.",
        "No breakdown over the router's runtime QueryFeatures (question_type, "
        "complexity_score, concept_count, ...) is reported. Those are router-internal "
        "derivations from the query text, not dataset labels, so a table over them "
        "would measure the router against its own inputs. Any such table must be "
        "labelled 'derived' wherever it appears.",
        f"insufficient_data=yes marks a cell below {analysis.get('min_cell', 5)} "
        "observations, the same threshold the dataset gate enforces.",
    ]
    if not derived.get("available", False):
        notes.append("derived_breakdown.ground_truth=false in the JSON artifact.")
    absent = analysis.get("categories_absent") or []
    if absent:
        notes.append(
            "Categories present in the dataset but absent from these rows: "
            + ", ".join(absent)
            + "."
        )
    return Table(
        name="e7_query_type",
        title="E7 -- per-category breakdown (dataset ground-truth label)",
        experiment_id="E7_query_type",
        columns=columns,
        rows=rows,
        notes=notes,
    )


def escalation_transition_table(analysis: Mapping[str, Any]) -> Table:
    """E8: what escalation bought, per observed transition only."""
    columns = (
        "Transition",
        "Occurrences",
        "Share of escalated",
        "Recall@5 before",
        "Recall@5 after",
        "Improvement rate",
        "Worsening rate",
        "Paired queries",
        "Median added total (ms)",
        "P95 added total (ms)",
        "Mean added cost (USD)",
        *_TRAILER,
    )
    rows: list[dict[str, str]] = []
    for transition in analysis.get("transitions", []):
        quality = transition.get("quality") or {}
        chunk_level = quality.get("chunk_level") or {}
        added_latency = transition.get("added_latency") or {}
        added_cost = transition.get("added_cost") or {}
        total = added_latency.get("total_latency_ms") or {}
        money = added_cost.get("estimated_cost_usd") or {}
        rows.append(
            {
                "Transition": f"{transition.get('from')} -> {transition.get('to')}",
                "Occurrences": _int(transition.get("occurrences")),
                "Share of escalated": _pct(transition.get("share_of_escalated")),
                "Recall@5 before": _num(chunk_level.get("mean_before"), 4),
                "Recall@5 after": _num(chunk_level.get("mean_after"), 4),
                "Improvement rate": _pct(chunk_level.get("improvement_rate")),
                "Worsening rate": _pct(chunk_level.get("worsening_rate")),
                "Paired queries": _int(chunk_level.get("n_paired")),
                "Median added total (ms)": _num(total.get("median"), 2),
                "P95 added total (ms)": _num(total.get("p95"), 2),
                "Mean added cost (USD)": _num(money.get("mean"), 6),
                **_trailer(transition),
            }
        )
    notes = [
        "Only transitions that actually occurred are listed. A pair that never "
        "fired is absent from this table rather than shown as 0.0%, because a zero "
        "row is indistinguishable from a real zero.",
        "Quality before is measured against pre_escalation_chunk_ids, which is "
        "populated only on escalated rows. A None there means 'not recorded' and is "
        "reported as unknown -- never as 0.0. An empty list is a real observation "
        "of an empty first stage and does score 0.0.",
        "'Paired queries' is the number of escalations where a quality-before was "
        "actually available; the before/after columns and the improvement rate are "
        "over exactly that many queries, never over all of them.",
        "'Median added total' and 'P95 added total' are the total latency of the "
        "escalated queries, i.e. the cost of the escalation path including its first "
        "stage, not an isolated delta between stages.",
    ]
    if not analysis.get("gold_chunk_ids_available"):
        notes.append(
            "No chunk-level relevance labels were supplied, so the chunk-level "
            "before/after cells are empty: the measurement was not made, which is "
            "not the same as it being zero."
        )
    # Zero escalations is an outcome, not a missing result, and a header with
    # nothing under it reads as either. The reason carries the two counts that
    # distinguish them, and is rendered by all three writers.
    empty_reason = None
    if not rows and not analysis.get("transitions"):
        empty_reason = (
            f"NO TRANSITIONS OBSERVED: {analysis.get('n_escalated', 0)} of "
            f"{analysis.get('n_rows', 0)} routed rows escalated "
            f"(n_transitions_observed={analysis.get('n_transitions_observed', 0)}). "
            "This table is empty because no query escalated -- not because the "
            "analysis is missing, and not because escalation was found to be "
            "harmless. Every question E8 asks (what escalation buys, per "
            "transition) is therefore unmeasured on this run."
        )
    return Table(
        name="e8_escalation_transitions",
        title="E8 -- escalation transitions, as observed",
        experiment_id="E8_escalation_analysis",
        columns=columns,
        rows=rows,
        notes=notes,
        empty_reason=empty_reason,
    )


#: Analysis key -> builder. The order is report order (E1..E8).
BUILDERS: tuple[tuple[str, Any], ...] = (
    ("E1", main_comparison_table),
    ("E2", escalation_ablation_table),
    ("E3", feature_ablation_table),
    ("E4", threshold_sweep_table),
    ("E5", cost_weight_table),
    ("E6", routing_overhead_table),
    ("E7", query_type_table),
    ("E8", escalation_transition_table),
)


def build_tables(analyses: Mapping[str, Mapping[str, Any]]) -> list[Table]:
    """Every table the available analyses support, in report order.

    An analysis that was not run contributes no table rather than an empty one;
    the report's index says which studies are present, so a missing table is
    visibly missing instead of looking like a study with no findings.
    """
    tables: list[Table] = []
    for key, builder in BUILDERS:
        payload = analyses.get(key)
        if payload is None:
            continue
        tables.append(builder(payload))
    return tables


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_markdown(table: Table) -> str:
    """A GitHub-flavoured markdown table with a title and its notes."""
    lines: list[str] = [f"## {table.title}", ""]
    if table.empty_reason:
        lines.extend([f"**{table.empty_reason}**", ""])
    lines.append("| " + " | ".join(_md(column) for column in table.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
    for row in table.rows:
        lines.append(
            "| " + " | ".join(_md(row.get(column, "")) for column in table.columns) + " |"
        )
    if table.notes:
        lines.extend(["", "Notes:", ""])
        lines.extend(f"- {note}" for note in table.notes)
    lines.append("")
    return "\n".join(lines)


def render_csv(table: Table) -> str:
    """RFC 4180 CSV with the same cells and column order as the markdown.

    An `empty_reason` becomes a single marker row carrying it in the first
    column, with every other cell blank. That is the only thing RFC 4180 allows
    -- there is no comment syntax -- and a header-only CSV is the exact artifact
    that reads as a result with no finding in it. The marker is not an
    observation: its `Occurrences`-style cells are blank rather than zero, and it
    says in the cell itself that it is a marker.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(table.columns),
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    body = table.rows
    if table.empty_reason:
        marker = {column: "" for column in table.columns}
        marker[table.columns[0]] = f"(no data rows) {table.empty_reason}"
        body = [marker, *table.rows]
    for row in body:
        writer.writerow({column: row.get(column, "") for column in table.columns})
    return buffer.getvalue()


def render_json(table: Table) -> str:
    """The table as JSON, column order preserved rather than alphabetised."""
    payload = {
        "tables_version": TABLES_VERSION,
        "name": table.name,
        "title": table.title,
        "experiment_id": table.experiment_id,
        "columns": list(table.columns),
        "notes": list(table.notes),
        "rows": [{column: row.get(column, "") for column in table.columns} for row in table.rows],
    }
    if table.empty_reason:
        payload["empty_reason"] = table.empty_reason
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def render_report(tables: Sequence[Table]) -> str:
    """One markdown document containing every table, in the order given."""
    parts: list[str] = [
        "# Phase 7 analysis",
        "",
        f"(_tables_version: {TABLES_VERSION})",
        "",
        "## Contents",
        "",
    ]
    parts.extend(f"- [{table.title}](#{_anchor(table.title)})" for table in tables)
    parts.append("")
    parts.extend(render_markdown(table) for table in tables)
    return "\n".join(parts)


def _anchor(title: str) -> str:
    return title.lower().replace(" ", "-").replace("|", "").replace(".", "")


def write_table(
    table: Table, directory: Path | str, *, stem: str | None = None
) -> dict[str, Path]:
    """Write one table as `.md`, `.csv` and `.json` with a shared stem."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    base = stem or table.name
    paths = {
        "md": target / f"{base}.md",
        "csv": target / f"{base}.csv",
        "json": target / f"{base}.json",
    }
    paths["md"].write_text(render_markdown(table), encoding="utf-8")
    paths["csv"].write_text(render_csv(table), encoding="utf-8")
    paths["json"].write_text(render_json(table), encoding="utf-8")
    return paths


def write_tables(
    tables: Iterable[Table], directory: Path | str
) -> dict[str, dict[str, str]]:
    """Write every table plus one combined markdown report.

    Returns `{table name: {"md": path, "csv": path, "json": path}}` with string
    paths, so the result can go straight into a JSON artifact.
    """
    materialised = list(tables)
    written = {
        table.name: {key: str(value) for key, value in write_table(table, directory).items()}
        for table in materialised
    }
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "analysis_report.md"
    report_path.write_text(render_report(materialised), encoding="utf-8")
    written["_report"] = {"md": str(report_path)}
    return written


__all__ = [
    "BUILDERS",
    "FORBIDDEN_COMPOSITE_TOKENS",
    "MAIN_COMPARISON_COLUMNS",
    "TABLES_VERSION",
    "Table",
    "build_tables",
    "cost_weight_table",
    "escalation_ablation_table",
    "escalation_transition_table",
    "feature_ablation_table",
    "main_comparison_table",
    "query_type_table",
    "render_csv",
    "render_json",
    "render_markdown",
    "render_report",
    "routing_overhead_table",
    "threshold_sweep_table",
    "write_table",
    "write_tables",
]
