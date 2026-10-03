"""
tests.test_rows
---------------
Offline checks for the query-level row export.

The row export exists so Phase 7 analysis never re-implements a metric, and the
whole value of it rests on one property: a row's numbers must be *the same
numbers* `RetrievalEvaluator` published in `metrics.json`. If the two ever drift
by a definition, the report is describing a different system from the one that
was measured. So the tests here pin exact agreement, per example and in
aggregate, rather than checking only that the columns exist.
"""

import csv
import json
from pathlib import Path

import pytest

from adaptive_rag.evaluation.retrieval import RetrievalEvaluator
from adaptive_rag.evaluation.rows import (
    COLUMNS,
    ROW_K_VALUES,
    ROWS_VERSION,
    EvaluationRow,
    build_rows,
    read_rows_jsonl,
    row_to_record,
    rows_to_csv,
    rows_to_jsonl,
    write_rows,
)
from adaptive_rag.schemas import (
    ChunkMetadata,
    ChunkProvenance,
    EvaluationExample,
    ExperimentTrace,
    ReferenceInfo,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
)

CORPUS_VERSION = "corpus_test_fixture_v1"


def make_example(
    example_id: str = "ex_001",
    *,
    documents: list[str] | None = None,
    category: str = "factual",
    split: str = "calibration",
) -> EvaluationExample:
    return EvaluationExample(
        example_id=example_id,
        query=f"what is question {example_id}",
        reference_answer="a reference answer",
        relevant_documents=documents if documents is not None else ["doc_a", "doc_b"],
        category=category,
        split=split,
        dataset_version="dense_eval_v1",
    )


def make_result(chunk_id: str, doc_id: str, rank: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"passage {chunk_id}",
        score=score,
        rank=rank,
        metadata=ChunkMetadata(
            document_id=doc_id,
            doc_title=doc_id,
            section_path=["1. Test"],
            headings=["Test"],
            element_ids=["e1"],
            element_types=["paragraph"],
            token_count=10,
            char_count=20,
        ),
        provenance=ChunkProvenance(document_id=doc_id, pages=[1], source_sha256="s"),
    )


def make_response(results: list[RetrievalResult], *, method: str = "dense") -> RetrievalResponse:
    return RetrievalResponse(
        query="q",
        results=results,
        retrieval_method=method,
        status="ok" if results else "no_results",
        retrieval_metadata=RetrievalMetadata(
            top_k=len(results),
            retriever_version=f"{method}_v1",
            index_id="idx",
            corpus_version=CORPUS_VERSION,
            latency_ms=1.25,
            search_latency_ms=0.5,
            query_embedding_latency_ms=0.75,
        ),
    )


def make_trace(
    example: EvaluationExample,
    results: list[RetrievalResult],
    *,
    status: str = "ok",
    response: RetrievalResponse | None = None,
    routing=None,
) -> ExperimentTrace:
    return ExperimentTrace(
        trace_id=f"run:{example.example_id}",
        experiment_id="run",
        example_id=example.example_id,
        query=example.query,
        category=example.category,
        retrieval=response if response is not None else make_response(results),
        reference=ReferenceInfo(
            reference_answer=example.reference_answer,
            relevant_documents=example.relevant_documents,
        ),
        status=status,
        config_hash="cfg",
        corpus_version=CORPUS_VERSION,
        routing=routing,
    )


def mixed_traces() -> tuple[list[ExperimentTrace], list[EvaluationExample]]:
    """Two examples: one where the relevant document is at rank 2, one with no hit."""
    first = make_example("ex_001", documents=["doc_a", "doc_b"])
    second = make_example("ex_002", documents=["doc_x"])

    traces = [
        make_trace(
            first,
            [
                make_result("c_noise", "doc_noise", rank=1, score=0.9),
                make_result("c_hit", "doc_a", rank=2, score=0.8),
            ],
        ),
        make_trace(
            second,
            [
                make_result("c_miss", "doc_y", rank=1, score=0.7),
                make_result("c_miss2", "doc_z", rank=2, score=0.6),
            ],
        ),
    ]
    return traces, [first, second]


# --- exact agreement with the evaluator ----------------------------------------


def test_a_rows_recall_and_mrr_are_the_evaluators_own_numbers():
    """The load-bearing property: one definition of recall, not two."""
    traces, examples = mixed_traces()
    report = RetrievalEvaluator().evaluate(traces, examples)
    per_example = report.aggregates["per_example"]

    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    for row in rows:
        expected = per_example[row.query_id]
        assert row.recall_at_5 == pytest.approx(expected["recall_at_5"])
        assert row.mrr == pytest.approx(expected["mrr"])


def test_row_means_equal_the_metrics_json_aggregates():
    """The aggregate a report publishes must be the mean of the rows it ships.

    Compared at 4 decimal places because `evaluation.base.metric` rounds every
    published metric to 4 dp for a byte-stable artifact; the per-example
    agreement test above is the exact one.
    """
    traces, examples = mixed_traces()
    report = RetrievalEvaluator().evaluate(traces, examples)
    by_key = {(m.name, m.k): m.value for m in report.metrics}

    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")
    for k in ROW_K_VALUES:
        for name, field in (
            ("recall_at_k", "recall_at_{k}"),
            ("precision_at_k", "precision_at_{k}"),
            ("hit_at_k", "hit_at_{k}"),
        ):
            if (name, k) not in by_key:
                continue
            mean = sum(getattr(row, field.format(k=k)) for row in rows) / len(rows)
            assert mean == pytest.approx(by_key[(name, k)], abs=1e-4), f"{name}@{k} disagrees"
    # nDCG is carried at k=5 only, which is the depth the report quotes.
    if ("ndcg_at_k", 5) in by_key:
        ndcg_mean = sum(row.ndcg_at_5 for row in rows) / len(rows)
        assert ndcg_mean == pytest.approx(by_key[("ndcg_at_k", 5)], abs=1e-4)
    mrr_mean = sum(row.mrr for row in rows) / len(rows)
    assert mrr_mean == pytest.approx(by_key[("mrr", None)], abs=1e-4)


def test_hit_is_defined_as_the_recall_test_not_a_separate_path():
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    for row in rows:
        assert row.hit_at_5 == (1.0 if row.recall_at_5 > 0 else 0.0)
        assert row.hit_at_1 == (1.0 if row.recall_at_1 > 0 else 0.0)
        assert row.hit_at_10 == (1.0 if row.recall_at_10 > 0 else 0.0)


def test_partial_recall_is_reported_as_a_fraction_not_a_hit():
    """One of two relevant documents found: recall 0.5, hit 1.0, MRR 0.5."""
    example = make_example("ex_001", documents=["doc_a", "doc_b"])
    trace = make_trace(
        example, [make_result("c_hit", "doc_a", rank=1, score=0.9)]
    )

    row = build_rows([trace], [example], system="dense", experiment_id="E1_baseline_comparison")[0]

    assert row.recall_at_5 == pytest.approx(0.5)
    assert row.hit_at_5 == 1.0
    assert row.mrr == pytest.approx(1.0)


# --- completeness --------------------------------------------------------------


def test_every_mandated_column_exists_and_is_exported():
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    record = row_to_record(rows[0])
    for column in COLUMNS:
        assert column in record, f"missing column {column}"


def test_the_column_set_covers_the_task_fields():
    required = {
        "query_id", "system", "experiment_id", "category", "split",
        "retrieved_document_ids", "retrieved_chunk_ids",
        "recall_at_1", "recall_at_5", "recall_at_10",
        "precision_at_1", "precision_at_5", "precision_at_10",
        "hit_at_1", "hit_at_5", "hit_at_10",
        "mrr", "ndcg_at_5",
        "retrieval_latency_ms", "reranking_latency_ms", "routing_latency_ms",
        "total_latency_ms", "status",
        "initial_strategy", "final_strategy", "escalated", "escalation_target",
        "sufficiency_score", "routing_confidence", "stage_count",
        "estimated_cost_usd",
    }
    assert required <= set(COLUMNS)


def test_the_columns_are_declared_in_a_frozen_order():
    """A row export whose column order depends on model field order is not a contract."""
    assert COLUMNS == tuple(EvaluationRow.model_fields)
    assert COLUMNS[0] == "query_id"


def test_the_module_declares_a_version():
    assert ROWS_VERSION == "phase7_rows_v1"


def test_extra_fields_cannot_be_added_to_a_row():
    with pytest.raises(Exception):
        EvaluationRow(query_id="q", system="s", unexpected=1)


def test_rows_carry_the_identity_and_the_retrieved_evidence():
    example = make_example("ex_001", category="comparative", split="test")
    trace = make_trace(
        example, [make_result("c_hit", "doc_a", rank=1, score=0.9)]
    )

    row = build_rows(
        [trace], [example], system="adaptive", experiment_id="E2_escalation_ablation"
    )[0]

    assert row.query_id == "ex_001"
    assert row.system == "adaptive"
    assert row.experiment_id == "E2_escalation_ablation"
    assert row.category == "comparative"
    assert row.split == "test"
    assert row.status == "ok"
    assert row.retrieved_document_ids == ["doc_a"]
    assert row.retrieved_chunk_ids == ["c_hit"]


def test_the_retrieval_clock_decomposition_is_carried():
    """A dense arm's latency is mostly a live embedding call; the row must show it."""
    example = make_example()
    trace = make_trace(example, [make_result("c", "doc_a", 1, 0.5)])
    # The runner copies the response clock onto the trace; the row reads it there.
    trace = trace.model_copy(update={"retrieval_latency_ms": 1.25})

    row = build_rows([trace], [example], system="dense", experiment_id="E1_baseline_comparison")[0]

    assert row.retrieval_latency_ms == 1.25
    assert row.search_latency_ms == 0.5
    assert row.query_embedding_latency_ms == 0.75


# --- failures are rows too ------------------------------------------------------


def test_a_failed_retrieval_is_a_row_rather_than_a_dropped_query():
    """Dropping failures would inflate every average a report publishes."""
    example = make_example("ex_001", documents=["doc_a"])
    trace = ExperimentTrace(
        trace_id="run:ex_001",
        experiment_id="run",
        example_id="ex_001",
        query=example.query,
        category=example.category,
        reference=ReferenceInfo(
            reference_answer="a", relevant_documents=["doc_a"]
        ),
        status="retrieval_failed",
        error={"stage": "retrieval", "error_type": "RetrievalError", "message": "boom"},
        config_hash="cfg",
        corpus_version=CORPUS_VERSION,
    )

    rows = build_rows([trace], [example], system="dense", experiment_id="E1_baseline_comparison")

    assert len(rows) == 1
    row = rows[0]
    assert row.status == "retrieval_failed"
    assert row.recall_at_5 == 0.0
    assert row.mrr == 0.0
    assert row.retrieved_chunk_ids == []
    assert row.retrieval_latency_ms is None
    assert row.error_type == "RetrievalError"
    assert row.error_stage == "retrieval"


def test_a_trace_without_a_labelled_example_is_skipped():
    """An unpaired row's zeros would look like a measured failure."""
    labelled = make_example("ex_001")
    unlabelled = make_trace(make_example("ex_999"), [])

    rows = build_rows(
        [unlabelled], [labelled], system="dense", experiment_id="E1_baseline_comparison"
    )

    assert rows == []


# --- routing fields -------------------------------------------------------------


def routing_trace(**overrides):
    """A minimal `RoutingTrace`, defaulting to a non-escalating bm25 decision."""
    from adaptive_rag.schemas.routing import (
        EscalationDecision,
        QueryFeatures,
        RoutingDecision,
        RoutingTrace,
        SufficiencyDecision,
    )

    fields = {
        "query_length_words": 6,
        "query_length_chars": 32,
        "content_terms": ["cost"],
        "content_term_count": 1,
        "lexical_density": 0.17,
        "entity_indicator_count": 0,
        "entity_ratio": 0.0,
        "technical_term_count": 0,
        "technical_ratio": 0.0,
        "semantic_indicator_count": 0,
        "semantic_ratio": 0.0,
        "question_type": "what",
        "concept_count": 1,
        "multi_concept": False,
        "comparison_indicator_count": 0,
        "complexity_score": 0.2,
        "analyzer_version": "analyzer_v1",
    }
    values = {
        "features": QueryFeatures(**fields),
        "decision": RoutingDecision(
            strategy="bm25",
            confidence=0.42,
            candidate_k=20,
            final_top_k=5,
            reranking_required=False,
            router_version="rule_based_router_v1",
            analyzer_version="analyzer_v1",
            cost_weight=0.25,
        ),
        "initial_strategy": "bm25",
        "initial_latency_ms": 2.0,
        "initial_result_count": 5,
        "sufficiency": SufficiencyDecision(
            sufficient=True,
            score=0.7,
            threshold=0.5,
            reason="coverage ok",
            checker_version="sufficiency_v1",
            result_count=5,
        ),
        "escalation": EscalationDecision(
            escalated=False,
            from_strategy="bm25",
            reason="sufficient",
            step_index=0,
            max_steps=2,
            policy_version="escalation_v1",
        ),
        "final_strategy": "bm25",
        "stage_count": 1,
        "stage_latencies_ms": [2.0],
        "routing_latency_ms": 0.3,
    }
    values.update(overrides)
    return RoutingTrace(**values)  # type: ignore[arg-type]


def test_a_routed_trace_exposes_the_decision_fields():
    example = make_example()
    trace = make_trace(example, [make_result("c", "doc_a", 1, 0.5)], routing=routing_trace())

    row = build_rows([trace], [example], system="adaptive", experiment_id="E1_baseline_comparison")[0]

    assert row.initial_strategy == "bm25"
    assert row.final_strategy == "bm25"
    assert row.escalated is False
    assert row.escalation_target is None
    assert row.sufficiency_score == 0.7
    assert row.sufficiency_threshold == 0.5
    assert row.routing_confidence == 0.42
    assert row.stage_count == 1
    assert row.routing_latency_ms == 0.3


def test_a_fixed_strategy_trace_leaves_the_routing_columns_null():
    example = make_example()
    trace = make_trace(example, [make_result("c", "doc_a", 1, 0.5)])

    row = build_rows([trace], [example], system="bm25", experiment_id="E1_baseline_comparison")[0]

    assert row.initial_strategy is None
    assert row.escalated is None
    assert row.sufficiency_score is None
    assert row.routing_latency_ms is None


def test_an_escalating_trace_reports_the_target_and_the_discarded_evidence():
    """E8's hook: `initial_chunk_ids is None` means nothing was discarded."""
    from adaptive_rag.schemas.routing import EscalationDecision

    escalated = routing_trace(
        initial_chunk_ids=["c_before"],
        initial_document_ids=["doc_before"],
        initial_latency_ms=2.0,
        stage_count=2,
        stage_latencies_ms=[2.0, 3.0],
        final_strategy="dense",
        escalation=EscalationDecision(
            escalated=True,
            from_strategy="bm25",
            to_strategy="dense",
            reason="insufficient",
            step_index=1,
            max_steps=2,
            policy_version="escalation_v1",
        ),
    )
    example = make_example()
    trace = make_trace(
        example, [make_result("c_after", "doc_a", 1, 0.5)], routing=escalated
    )

    row = build_rows([trace], [example], system="adaptive", experiment_id="E8_escalation_analysis")[0]

    assert row.escalated is True
    assert row.escalation_target == "dense"
    assert row.pre_escalation_chunk_ids == ["c_before"]
    assert row.pre_escalation_document_ids == ["doc_before"]
    assert row.stage_latencies_ms == [2.0, 3.0]


def test_a_query_that_never_escalated_records_no_pre_escalation_evidence():
    """`None` and `[]` are different observations and must stay distinguishable."""
    example = make_example()
    trace = make_trace(example, [make_result("c", "doc_a", 1, 0.5)], routing=routing_trace())

    row = build_rows([trace], [example], system="adaptive", experiment_id="E8_escalation_analysis")[0]

    assert row.pre_escalation_chunk_ids is None
    assert row.pre_escalation_document_ids is None


# --- flat exports ---------------------------------------------------------------


def test_rows_to_csv_uses_the_frozen_column_order(tmp_path: Path):
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    path = rows_to_csv(rows, tmp_path / "rows.csv")
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COLUMNS)
        written = list(reader)

    assert len(written) == 2
    assert written[0]["query_id"] == "ex_001"
    assert written[0]["retrieved_chunk_ids"] == "c_noise\nc_hit"


def test_rows_to_jsonl_is_one_object_per_query(tmp_path: Path):
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    path = rows_to_jsonl(rows, tmp_path / "rows.jsonl")
    records = read_rows_jsonl(path)

    assert len(records) == 2
    assert list(records[0]) == list(COLUMNS)
    assert records[0]["retrieved_chunk_ids"] == ["c_noise", "c_hit"]


def test_both_exports_are_byte_stable_across_writes(tmp_path: Path):
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    first = rows_to_csv(rows, tmp_path / "a.csv")
    second = rows_to_csv(rows, tmp_path / "b.csv")
    assert first.read_bytes() == second.read_bytes()

    first_jsonl = rows_to_jsonl(rows, tmp_path / "a.jsonl")
    second_jsonl = rows_to_jsonl(rows, tmp_path / "b.jsonl")
    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()


def test_write_rows_produces_both_exports(tmp_path: Path):
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    written = write_rows(rows, tmp_path)

    assert set(written) == {"csv", "jsonl"}
    assert written["csv"].is_file()
    assert written["jsonl"].is_file()


def test_the_jsonl_export_carries_the_list_columns_as_lists(tmp_path: Path):
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="dense", experiment_id="E1_baseline_comparison")

    path = rows_to_jsonl(rows, tmp_path / "rows.jsonl")
    first_line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert isinstance(first_line["retrieved_document_ids"], list)
    assert isinstance(first_line["stage_latencies_ms"], list)



# --- routed rows reach the flat export (regression) ----------------------------
#
# `row_to_record` newline-joins every column in `_LIST_COLUMNS`, and one of those
# columns -- `stage_latencies_ms` -- holds measured floats rather than
# identifiers. Joining those unconverted raised `TypeError: sequence item 0:
# expected str instance, float found` inside `suite.run_suite`'s row export, on
# the *first* row of the *first* arm. Every ablation arm (E2/E3/E4/E5) routes, so
# every one of them failed at exactly that line, and E2/E3/E4/E5 have never once
# completed. Fixed-strategy E1 arms were unaffected only because a trace without a
# `RoutingTrace` leaves the column empty and an empty join succeeds -- which is why
# the defect hid behind a clean E1. These tests keep the routed path covered.


def routed_traces() -> tuple[list[ExperimentTrace], list[EvaluationExample]]:
    """Two routed traces with non-empty float stage latencies."""
    first = make_example("ex_001", documents=["doc_a"])
    second = make_example("ex_002", documents=["doc_b"])
    traces = [
        make_trace(
            first,
            [make_result("c_a", "doc_a", rank=1, score=0.9)],
            routing=routing_trace(stage_latencies_ms=[2.0]),
        ),
        make_trace(
            second,
            [make_result("c_b", "doc_b", rank=1, score=0.7)],
            routing=routing_trace(stage_latencies_ms=[3.5, 4.25]),
        ),
    ]
    return traces, [first, second]


def test_a_routed_row_exports_through_both_flat_writers(tmp_path: Path):
    """The regression: a routed row must survive `write_rows`, which every ablation arm calls."""
    traces, examples = routed_traces()
    rows = build_rows(traces, examples, system="adaptive", experiment_id="E2_escalation_ablation")

    written = write_rows(rows, tmp_path)

    # A stage-latency cell is itself newline-joined, so the physical line count is
    # not the record count; both are asserted because the bug's fingerprint was a
    # header-only file.
    text = written["csv"].read_text(encoding="utf-8")
    assert len(list(csv.DictReader(text.splitlines()))) == 2
    assert len(text.splitlines()) > 3
    assert len(written["jsonl"].read_text(encoding="utf-8").splitlines()) == 2


def test_the_csv_renders_float_stage_latencies_while_the_jsonl_keeps_them_numeric(tmp_path: Path):
    """Newline-joined for the CSV cell, still a real list in the JSONL line."""
    traces, examples = routed_traces()
    rows = build_rows(traces, examples, system="adaptive", experiment_id="E2_escalation_ablation")

    written = write_rows(rows, tmp_path)

    cell = next(
        record["stage_latencies_ms"]
        for record in csv.DictReader(written["csv"].read_text(encoding="utf-8").splitlines())
    )
    assert cell == "2.0"
    first_line = json.loads(written["jsonl"].read_text(encoding="utf-8").splitlines()[0])
    assert first_line["stage_latencies_ms"] == [2.0]


def test_the_identifier_list_columns_are_unchanged_by_the_stringification(tmp_path: Path):
    """Joining `str(item)` must be a no-op for the columns that were already strings."""
    traces, examples = routed_traces()
    rows = build_rows(traces, examples, system="adaptive", experiment_id="E2_escalation_ablation")

    record = row_to_record(rows[0])

    assert record["retrieved_document_ids"] == "doc_a"
    assert record["retrieved_chunk_ids"] == "c_a"
    assert record["stage_latencies_ms"] == "2.0"


def test_a_fixed_strategy_row_still_leaves_the_stage_latencies_cell_empty():
    """The `None`/empty guard must survive: an empty list is still an empty cell."""
    traces, examples = mixed_traces()
    rows = build_rows(traces, examples, system="bm25", experiment_id="E1_baseline_comparison")

    assert row_to_record(rows[0])["stage_latencies_ms"] == ""


def _every_ablation_variant_name() -> list[tuple[str, str]]:
    """Every variant `ablation.py` can mint, as (experiment_id, name) pairs.

    Taken from the real builders rather than hand-listed, so a future parameterised
    variant is covered by construction instead of by remembering to edit a list.
    """
    from adaptive_rag.evaluation.ablation import (
        cost_weight_sweep,
        escalation_step_sweep,
        escalation_variants,
        feature_group_variants,
        threshold_sweep,
    )

    variants = [
        *escalation_variants(),
        *feature_group_variants(),
        *threshold_sweep(),
        *cost_weight_sweep(),
        *escalation_step_sweep(),
    ]
    return [(variant.experiment_id, variant.name) for variant in variants]


ABLATION_VARIANTS = _every_ablation_variant_name()


def test_the_arms_built_by_ablation_cover_a_float_an_int_and_a_float_again():
    """Guard the guard: a float threshold, an int step count and a float cost weight."""
    names = [name for _, name in ABLATION_VARIANTS]

    assert any(name == "sufficiency_threshold=0.3" for name in names)
    assert any(name == "cost_weight=0.25" for name in names)
    assert any(name == "max_escalation_steps=2" for name in names)


@pytest.mark.parametrize(
    ("experiment_id", "variant_name"),
    ABLATION_VARIANTS,
    ids=[f"{eid}/{name}" for eid, name in ABLATION_VARIANTS],
)
def test_every_ablation_arm_exports_its_rows(
    tmp_path: Path, experiment_id: str, variant_name: str
):
    """Each parameterised arm's row export completes and lands both files.

    The variant name is the run directory's own label and the row's `system`
    column, so it is also what the float parameters have to render as.
    """
    traces, examples = routed_traces()
    rows = build_rows(traces, examples, system=variant_name, experiment_id=experiment_id)

    target = tmp_path / f"{experiment_id}__{variant_name}"
    written = write_rows(rows, target)

    assert written["csv"].is_file() and written["jsonl"].is_file()
    records = list(csv.DictReader(written["csv"].read_text(encoding="utf-8").splitlines()))
    assert len(records) == 2
    assert {record["system"] for record in records} == {variant_name}


@pytest.mark.parametrize(
    ("experiment_id", "variant_name"),
    ABLATION_VARIANTS,
    ids=[f"{eid}/{name}" for eid, name in ABLATION_VARIANTS],
)
def test_every_ablation_arms_variant_label_renders_verbatim(
    tmp_path: Path, experiment_id: str, variant_name: str
):
    """A swept float must print as the label spells it: `0.3`, not `0.30000000000000004`.

    These labels are what `registry.json` keys on and what the run directory is
    named, so the export must reproduce the spelling byte for byte and must not
    normalise or reformat it.
    """
    traces, examples = routed_traces()
    rows = build_rows(traces, examples, system=variant_name, experiment_id=experiment_id)

    record = row_to_record(rows[0])

    assert record["system"] == variant_name
    assert "0.30000000000000004" not in record["system"]
