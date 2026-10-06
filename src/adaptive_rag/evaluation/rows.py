"""
evaluation.rows
---------------
Query-level row export: one flat row per (query, system) with every metric a
Phase 7 analysis needs.

The evaluators report *aggregates* over a run, and `RetrievalEvaluator` alone
emits just `recall_at_5` and `mrr` per example. That is enough to publish a
table and not enough to analyse one: a paired significance test between two arms
needs the per-query numbers, a category breakdown needs the category, an
efficiency-versus-quality scatter needs the latency, and the escalation story
needs the routing decision. Recomputing any of that from `traces.jsonl` means
re-opening a private metric on a `RoutingTrace` in four different places, which
is how four slightly different recalls get published.

So this module is the single place where a trace becomes a row. Two rules:

* **No metric arithmetic is reimplemented.** Every recall, precision, hit, MRR
  and nDCG value comes from calling `RetrievalEvaluator`'s own helpers on the
  same `(trace, example)` pair the evaluator would have used. If the definition
  of recall changes, a row changes with it; if it is reimplemented here, a row
  silently keeps the old one.
* **Rows are a model, not a dict.** `EvaluationRow` forbids extra fields, so a
  field added here but not to the schema is a construction error, and its
  `COLUMNS` tuple is the single definition of the CSV and JSONL column order --
  which is what makes those exports comparable between runs.

Failed queries are rows too. A trace whose retrieval raised is exported with its
`status` and with the metric helpers' own `0.0` (they return zero for an absent
retrieval), rather than being dropped: silently dropping failures inflates every
average in a report, which is the error the per-trace status breakdown in the
manifest exists to make visible.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.errors import EvaluationError
from adaptive_rag.evaluation.retrieval import RetrievalEvaluator
from adaptive_rag.schemas import EvaluationExample, ExperimentTrace

ROWS_VERSION = "phase7_rows_v1"

# The k values a row carries. Fixed rather than taken from the run's k_grid so a
# row exported from a `k_grid=[1, 3]` run is still directly comparable with one
# exported from a `k_grid=[1, 5, 10]` run -- which is the whole point of a flat
# row, and the reason the suite cannot simply dump whatever the evaluator had.
ROW_K_VALUES: tuple[int, int, int] = (1, 5, 10)


class EvaluationRow(BaseModel):
    """One query, one system, one observation of everything measured about it."""

    model_config = ConfigDict(extra="forbid")

    # --- identity ---
    query_id: str
    # The arm or ablation variant that produced the row ("bm25", "adaptive",
    # "without_technical"). Named `system` because that is what a comparison
    # table calls it; `variant` is the registry's word for the same string.
    system: str
    experiment_id: str
    category: str
    split: str
    status: str

    # --- what was retrieved ---
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    # --- retrieval quality (computed by RetrievalEvaluator) ---
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    ndcg_at_5: float

    # --- latency, in milliseconds ---
    retrieval_latency_ms: float | None = None
    reranking_latency_ms: float | None = None
    routing_latency_ms: float | None = None
    total_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    # Decomposition of the retrieval clock. Present because a dense arm's
    # retrieval latency is mostly a live embedding call, and a reader deciding
    # what it paid for needs to see that without re-opening the trace.
    query_embedding_latency_ms: float | None = None
    search_latency_ms: float | None = None
    candidate_generation_latency_ms: float | None = None
    stage_latencies_ms: list[float] = Field(default_factory=list)

    # --- routing decision (absent on fixed-strategy runs) ---
    initial_strategy: str | None = None
    final_strategy: str | None = None
    escalated: bool | None = None
    escalation_target: str | None = None
    sufficiency_score: float | None = None
    sufficiency_threshold: float | None = None
    sufficient: bool | None = None
    routing_confidence: float | None = None
    stage_count: int | None = None
    # E8's pre-escalation evidence. `None` means "not recorded", which for this
    # field means the query never escalated -- `retrieval/adaptive.py` only
    # captures it when something was discarded. `[]` is a real observation: the
    # first stage returned nothing.
    pre_escalation_chunk_ids: list[str] | None = None
    pre_escalation_document_ids: list[str] | None = None

    # --- resources and cost ---
    context_tokens: int | None = None
    estimated_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    rerank_candidate_count: int | None = None
    rerank_result_count: int | None = None
    rerank_fallback: bool | None = None

    # --- failure detail ---
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None


# Frozen column order for the flat exports. `list()`-ing a pydantic model would
# give field-declaration order, which happens to be stable but is not stated
# anywhere a reader can check; declaring it makes the export's schema a contract
# rather than an accident of how the model was written.
COLUMNS: tuple[str, ...] = tuple(EvaluationRow.model_fields)

_LIST_COLUMNS: frozenset[str] = frozenset(
    {
        "retrieved_document_ids",
        "retrieved_chunk_ids",
        "stage_latencies_ms",
        "pre_escalation_chunk_ids",
        "pre_escalation_document_ids",
    }
)


def row_to_record(row: EvaluationRow) -> dict[str, Any]:
    """One row as a flat JSON-ready mapping.

    List-valued columns are joined with a newline rather than JSON-encoded: the
    identifier lists are short and rank-ordered, a newline keeps them readable in
    a CSV cell and greppable in a JSONL line, and a nested object per row would
    make the flat export stop being flat.

    Elements are stringified rather than joined raw, because not every list
    column holds identifiers: `stage_latencies_ms` holds measured floats.
    Joining those unconverted raises `TypeError: sequence item 0: expected str
    instance, float found` on the first routed trace, and it can only ever
    surface on a run that routed -- a fixed-strategy arm leaves the column empty,
    so the empty join succeeds and hides the defect until an ablation arm, the
    only kind that routes, reaches the export.
    """
    record: dict[str, Any] = {}
    for column in COLUMNS:
        value = getattr(row, column)
        record[column] = (
            "\n".join(str(item) for item in value)
            if column in _LIST_COLUMNS and value is not None
            else value
        )
    return record


def _routing_fields(trace: ExperimentTrace) -> dict[str, Any]:
    """Routing columns, straight off the trace's `RoutingTrace`.

    Every value is `None` on a fixed-strategy run. `escalated` is read the same
    way `evaluation.routing.RoutingEvaluator` reads it -- from the escalation
    decision, not inferred from `initial_strategy != final_strategy` -- so the
    column and the published `escalation_rate` cannot disagree.
    """
    routing = trace.routing
    if routing is None:
        return {}
    escalation = routing.escalation
    sufficiency = routing.sufficiency
    return {
        "initial_strategy": routing.initial_strategy,
        "final_strategy": routing.final_strategy,
        "escalated": bool(escalation.escalated) if escalation else None,
        "escalation_target": escalation.to_strategy if escalation else None,
        "sufficiency_score": sufficiency.score if sufficiency else None,
        "sufficiency_threshold": sufficiency.threshold if sufficiency else None,
        "sufficient": sufficiency.sufficient if sufficiency else None,
        "routing_confidence": routing.decision.confidence,
        "stage_count": routing.stage_count,
        "pre_escalation_chunk_ids": routing.initial_chunk_ids,
        "pre_escalation_document_ids": routing.initial_document_ids,
        "stage_latencies_ms": list(routing.stage_latencies_ms),
    }


def _row_from_trace(
    trace: ExperimentTrace,
    example: EvaluationExample,
    *,
    system: str,
    experiment_id: str,
    evaluator: RetrievalEvaluator,
    k_values: Sequence[int],
) -> EvaluationRow:
    """One (trace, example) pair rendered as a row, using the evaluator's math.

    The metric helpers are the *same* methods `RetrievalEvaluator.evaluate` calls
    on the same paired inputs, so a row's `recall_at_5` is the value that went
    into the run's `metrics.json` mean -- not a re-derivation of it.
    """
    if len(k_values) < 3:
        raise EvaluationError(
            f"a row needs three k values (1, 5, 10); got {list(k_values)}"
        )
    k1, k5, k10 = k_values[0], k_values[1], k_values[2]
    results = trace.retrieval.results if trace.retrieval is not None else []
    metadata = trace.retrieval.retrieval_metadata if trace.retrieval is not None else None

    # Computed once per k and reused by both the recall and the hit column: the
    # hit column is defined *as* the recall test, and calling the helper twice
    # would be two chances for the two columns to disagree for no reason.
    recall = {k: evaluator._recall_at_k(trace, example, k) for k in (k1, k5, k10)}

    return EvaluationRow(
        query_id=trace.example_id,
        system=system,
        experiment_id=experiment_id,
        category=example.category,
        split=example.split,
        status=trace.status,
        retrieved_document_ids=[r.metadata.document_id for r in results],
        retrieved_chunk_ids=[r.chunk_id for r in results],
        recall_at_1=recall[k1],
        recall_at_5=recall[k5],
        recall_at_10=recall[k10],
        precision_at_1=evaluator._precision_at_k(trace, example, k1),
        precision_at_5=evaluator._precision_at_k(trace, example, k5),
        precision_at_10=evaluator._precision_at_k(trace, example, k10),
        # Hit@k is the evaluator's own definition -- "any relevant document was
        # retrieved" -- expressed as a test on the recall value just computed,
        # so it cannot drift from `recall_at_k` in `metrics.json`.
        hit_at_1=1.0 if recall[k1] > 0 else 0.0,
        hit_at_5=1.0 if recall[k5] > 0 else 0.0,
        hit_at_10=1.0 if recall[k10] > 0 else 0.0,
        mrr=evaluator._reciprocal_rank(trace, example),
        ndcg_at_5=evaluator._ndcg_at_k(trace, example, k5),
        retrieval_latency_ms=trace.retrieval_latency_ms,
        reranking_latency_ms=trace.rerank_latency_ms,
        routing_latency_ms=trace.routing.routing_latency_ms if trace.routing else None,
        total_latency_ms=trace.total_latency_ms,
        generation_latency_ms=trace.generation_latency_ms,
        query_embedding_latency_ms=metadata.query_embedding_latency_ms if metadata else None,
        search_latency_ms=metadata.search_latency_ms if metadata else None,
        candidate_generation_latency_ms=trace.candidate_generation_latency_ms,
        context_tokens=trace.context_tokens,
        estimated_cost_usd=trace.estimated_cost_usd,
        input_tokens=trace.usage.input_tokens if trace.usage else None,
        output_tokens=trace.usage.output_tokens if trace.usage else None,
        total_tokens=trace.usage.total_tokens if trace.usage else None,
        rerank_candidate_count=trace.rerank_candidate_count,
        rerank_result_count=trace.rerank_result_count,
        rerank_fallback=trace.rerank_fallback,
        error_stage=trace.error.stage if trace.error else None,
        error_type=trace.error.error_type if trace.error else None,
        error_message=trace.error.message if trace.error else None,
        **_routing_fields(trace),
    )


def build_rows(
    traces: Iterable[ExperimentTrace],
    examples: Iterable[EvaluationExample],
    *,
    system: str,
    experiment_id: str,
    k_values: Sequence[int] = ROW_K_VALUES,
    evaluator: RetrievalEvaluator | None = None,
) -> list[EvaluationRow]:
    """One row per trace, paired with its labelled example.

    Traces without a matching example are skipped, for the same reason
    `RetrievalEvaluator` skips them: a row whose relevance labels are unknown
    would carry zeros that look like measured failure. Traces whose retrieval
    *failed* are kept, because a measured failure is a result.
    """
    metric_evaluator = evaluator or RetrievalEvaluator()
    example_by_id = {example.example_id: example for example in examples}
    rows: list[EvaluationRow] = []
    for trace in traces:
        example = example_by_id.get(trace.example_id)
        if example is None:
            continue
        rows.append(
            _row_from_trace(
                trace,
                example,
                system=system,
                experiment_id=experiment_id,
                evaluator=metric_evaluator,
                k_values=k_values,
            )
        )
    return rows


def rows_to_csv(rows: Sequence[EvaluationRow], path: Path | str) -> Path:
    """Write rows as CSV with the frozen column order and no index column."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_record(row))
    return destination


def rows_to_jsonl(rows: Sequence[EvaluationRow], path: Path | str) -> Path:
    """Write rows as JSON Lines, one object per query, keys in `COLUMNS` order."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for row in rows:
            record = {column: getattr(row, column) for column in COLUMNS}
            handle.write(
                json.dumps(record, sort_keys=False, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
    return destination


def write_rows(
    rows: Sequence[EvaluationRow],
    directory: Path | str,
    *,
    stem: str = "rows",
) -> dict[str, Path]:
    """Write both flat exports for one run and return their paths."""
    target = Path(directory)
    return {
        "csv": rows_to_csv(rows, target / f"{stem}.csv"),
        "jsonl": rows_to_jsonl(rows, target / f"{stem}.jsonl"),
    }


def read_rows_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read back a JSONL row export, validating nothing.

    A convenience for the analysis stage, which reads a previous run's rows
    rather than its traces. It returns plain dicts because the point is to load
    what was written, not to re-validate it.
    """
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records

