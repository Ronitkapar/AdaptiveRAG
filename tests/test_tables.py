"""
tests.test_tables
-----------------
Offline, deterministic checks for the Phase 7 table builders.

The property under test is agreement: the markdown, the CSV and the JSON are
three renderings of one set of cells, so they cannot drift. On top of that, two
presentation rules are pinned, because both are how a wrong number becomes
invisible rather than absent:

* an unmeasured value renders as an **empty cell**, never as `0.0` -- printing
  `0.0` for a latency clock that never started makes a system look free; and
* no table may carry a column that claims a combined quality/cost score, so the
  E1 header keeps quality and cost apart.

Determinism is checked by rendering the same table twice and comparing bytes, and
by re-running a whole analysis over shuffled input rows and checking the bytes
do not move.
"""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

import pytest

from adaptive_rag.evaluation.analysis import (
    analyze_cost_weight,
    analyze_escalation_ablation,
    analyze_escalation_transitions,
    analyze_feature_ablation,
    analyze_main_comparison,
    analyze_query_type,
    analyze_routing_overhead,
    analyze_threshold_sweep,
)
from adaptive_rag.evaluation.tables import (
    MAIN_COMPARISON_COLUMNS,
    TABLES_VERSION,
    Table,
    _top_strategy,
    build_tables,
    cost_weight_table,
    escalation_ablation_table,
    escalation_transition_table,
    feature_ablation_table,
    main_comparison_table,
    query_type_table,
    render_csv,
    render_json,
    render_markdown,
    render_report,
    routing_overhead_table,
    threshold_sweep_table,
    write_table,
    write_tables,
)


def make_row(**overrides):
    row = {
        "query_id": "q0",
        "system": "adaptive",
        "category": "factual",
        "split": "calibration",
        "status": "success",
        "recall_at_5": 0.5,
        "mrr": 0.4,
        "ndcg_at_5": 0.45,
        "total_latency_ms": 100.0,
        "retrieval_latency_ms": 80.0,
        "routing_latency_ms": 2.0,
        "reranking_latency_ms": 10.0,
        "initial_strategy": "bm25",
        "final_strategy": "bm25",
        "escalated": False,
        "escalation_target": None,
        "routing_confidence": 0.6,
        "stage_count": 1,
        "context_tokens": 1000,
        "total_tokens": 1000,
        "estimated_cost_usd": 0.001,
        "pre_escalation_chunk_ids": None,
        "pre_escalation_document_ids": None,
    }
    row.update(overrides)
    return row


def sample_rows(n=6):
    """Six BM25 rows (no routing clock) and six adaptive rows (one 12 s sample)."""
    rows = []
    for i in range(n):
        rows.append(
            make_row(
                query_id=f"q{i}",
                system="bm25",
                category="factual" if i < 4 else "comparative",
                recall_at_5=0.4,
                total_latency_ms=50.0,
                retrieval_latency_ms=50.0,
                routing_latency_ms=None,
                reranking_latency_ms=None,
                initial_strategy=None,
                final_strategy=None,
                escalated=None,
                routing_confidence=None,
                stage_count=None,
            )
        )
        rows.append(
            make_row(
                query_id=f"q{i}",
                system="adaptive",
                category="factual" if i < 4 else "comparative",
                recall_at_5=0.8,
                total_latency_ms=12000.0 if i == 5 else 100.0,
                retrieval_latency_ms=90.0,
                reranking_latency_ms=8.0,
                escalated=i < 2,
                final_strategy="hybrid" if i < 2 else "bm25",
                escalation_target="hybrid" if i < 2 else None,
                pre_escalation_chunk_ids=["c1"] if i < 2 else None,
                pre_escalation_document_ids=None,
                stage_count=2 if i < 2 else 1,
            )
        )
    return rows


ANALYSES = {
    "E1": lambda rows: analyze_main_comparison(rows),
    "E2": lambda rows: analyze_escalation_ablation(rows),
    "E3": lambda rows: analyze_feature_ablation(rows),
    "E4": lambda rows: analyze_threshold_sweep(rows),
    "E5": lambda rows: analyze_cost_weight(rows),
    "E6": lambda rows: analyze_routing_overhead(rows),
    "E7": lambda rows: analyze_query_type(rows),
    "E8": lambda rows: analyze_escalation_transitions(rows),
}


# --------------------------------------------------------------------------
# the mandated E1 header
# --------------------------------------------------------------------------


def test_e1_table_has_the_mandated_columns_in_the_mandated_order():
    table = main_comparison_table(ANALYSES["E1"](sample_rows()))

    mandated = (
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
    )
    assert table.columns[: len(mandated)] == mandated
    assert list(table.columns) == list(MAIN_COMPARISON_COLUMNS)
    # Every table carries its sample size and the small-cell flag.
    assert table.columns[-2:] == ("n", "insufficient_data")


def test_e1_table_keeps_quality_and_cost_in_separate_columns():
    table = main_comparison_table(ANALYSES["E1"](sample_rows()))

    assert "Cost/resource" in table.columns
    # Nothing in the header collapses the two, and the reason is on the table.
    assert not any("composite" in c.lower() or "overall" in c.lower() for c in table.columns)
    assert any("No composite" in note for note in table.notes)


def test_a_table_declaring_a_composite_column_is_refused():
    """The guard is in the model, so no builder -- present or future -- can add one."""
    with pytest.raises(ValueError, match="composite"):
        Table(
            name="bad",
            title="bad",
            experiment_id="E1_baseline_comparison",
            columns=("System", "Composite score"),
            rows=[{"System": "a", "Composite score": "0.9"}],
        )


def test_a_table_with_cells_for_an_unknown_column_is_refused():
    with pytest.raises(ValueError, match="unknown columns"):
        Table(
            name="bad",
            title="bad",
            experiment_id="E1_baseline_comparison",
            columns=("System",),
            rows=[{"System": "a", "Recall@5": "0.5"}],
        )


def test_e1_table_renders_the_known_values():
    """BM25 median 50 ms with no routing clock; adaptive p95 12000 ms."""
    table = main_comparison_table(ANALYSES["E1"](sample_rows()))
    rows = {row["System"]: row for row in table.rows}

    assert rows["bm25"]["Recall@5"] == "0.4000"
    assert rows["bm25"]["Median latency (ms)"] == "50.00"
    assert rows["bm25"]["P95 latency (ms)"] == "50.00"
    assert rows["bm25"]["n"] == "6"
    assert rows["bm25"]["insufficient_data"] == "no"
    assert rows["adaptive"]["Recall@5"] == "0.8000"
    assert rows["adaptive"]["P95 latency (ms)"] == "12000.00"
    assert rows["adaptive"]["Routing latency (ms)"] == "2.00"
    assert "ctx=1000" in rows["adaptive"]["Cost/resource"]
    assert "usd=0.001000" in rows["adaptive"]["Cost/resource"]


def test_e1_table_leaves_an_unrecorded_clock_empty_rather_than_zero():
    """BM25 took no routing decision. Printing 0.00 would make it look free, which
    is the same class of error as the missing-clock check in the 7.0 gate."""
    table = main_comparison_table(ANALYSES["E1"](sample_rows()))
    rows = {row["System"]: row for row in table.rows}

    assert rows["bm25"]["Routing latency (ms)"] == ""
    assert rows["bm25"]["Reranking latency (ms)"] == ""


def test_e1_table_marks_an_arm_with_no_rows():
    table = main_comparison_table(ANALYSES["E1"](sample_rows()))
    rows = {row["System"]: row for row in table.rows}

    assert rows["dense"]["n"] == "0"
    assert rows["dense"]["insufficient_data"] == "yes"
    assert rows["dense"]["Recall@5"] == ""


# --------------------------------------------------------------------------
# the other seven builders
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,builder,expected_name",
    [
        ("E2", escalation_ablation_table, "e2_escalation_ablation"),
        ("E3", feature_ablation_table, "e3_feature_ablation"),
        ("E4", threshold_sweep_table, "e4_threshold_sweep"),
        ("E5", cost_weight_table, "e5_cost_weight"),
        ("E6", routing_overhead_table, "e6_routing_overhead"),
        ("E7", query_type_table, "e7_query_type"),
        ("E8", escalation_transition_table, "e8_escalation_transitions"),
    ],
)
def test_every_builder_produces_a_usable_table(key, builder, expected_name):
    rows = sample_rows()
    table = builder(ANALYSES[key](rows))

    assert table.name == expected_name
    assert table.columns
    assert table.experiment_id.startswith(key)
    assert render_markdown(table).startswith("## ")
    assert render_csv(table).splitlines()[0] == ",".join(table.columns)
    assert json.loads(render_json(table))["columns"] == list(table.columns)


def test_e7_table_labels_router_internal_features_as_not_ground_truth():
    """The note has to be on the table, not only in the analysis JSON: the markdown
    is what gets pasted into a write-up."""
    table = query_type_table(ANALYSES["E7"](sample_rows()))

    notes = " ".join(table.notes)
    assert "QueryFeatures" in notes
    assert "derived" in notes.lower()
    assert "ground-truth" in notes or "ground truth" in notes
    # The breakdown itself is keyed on the dataset's own label only.
    assert {row["Category"] for row in table.rows} <= {
        "factual",
        "terminology",
        "conceptual",
        "comparative",
        "multi_document",
        "fine_grained",
    }
    assert "insufficient_data=yes marks a cell below 5" in notes


def test_e7_table_flags_a_thin_category():
    table = query_type_table(ANALYSES["E7"](sample_rows()))
    rows = {row["Category"]: row for row in table.rows}

    assert rows["factual"]["n"] == "8"
    assert rows["factual"]["insufficient_data"] == "no"
    assert rows["comparative"]["n"] == "4"
    assert rows["comparative"]["insufficient_data"] == "yes"


def test_e8_table_lists_only_observed_transitions_and_says_why():
    table = escalation_transition_table(ANALYSES["E8"](sample_rows()))

    assert [row["Transition"] for row in table.rows] == ["bm25 -> hybrid"]
    assert table.rows[0]["Occurrences"] == "2"
    assert table.rows[0]["Share of escalated"] == "100.0%"
    # No chunk-level gold was supplied, so the before/after cells are empty and
    # the note says the measurement was not made.
    assert table.rows[0]["Recall@5 before"] == ""
    assert table.rows[0]["Paired queries"] == "0"
    notes = " ".join(table.notes)
    assert "Only transitions that actually occurred" in notes
    assert "never as 0.0" in notes
    assert "not the same as it being zero" in notes


def test_e8_table_marks_the_zero_transition_case_instead_of_writing_a_bare_header():
    """On the shipped configuration the router escalates on nothing, so `transitions`
    is empty. A header with no rows under it is indistinguishable from a study that
    ran and found nothing -- the two opposite claims."""
    analysis = ANALYSES["E8"]([row for row in sample_rows() if row["system"] != "adaptive"])

    table = escalation_transition_table(analysis)

    assert analysis["n_escalated"] == 0
    assert analysis["n_transitions_observed"] == 0
    # Still zero data rows: the marker is a statement, not an invented transition.
    assert table.rows == []
    assert table.empty_reason is not None
    assert "NO TRANSITIONS OBSERVED" in table.empty_reason
    assert f"{analysis['n_escalated']} of {analysis['n_rows']}" in table.empty_reason
    assert "not because the analysis is missing" in table.empty_reason
    # All three renderings carry it, so none of the files is silently empty.
    assert "NO TRANSITIONS OBSERVED" in render_markdown(table)
    assert "NO TRANSITIONS OBSERVED" in render_csv(table)
    assert json.loads(render_json(table))["empty_reason"] == table.empty_reason
    # The CSV's marker row is labelled and leaves every other cell blank, so it
    # cannot be read as a transition with zero occurrences.
    marker = list(csv.DictReader(io.StringIO(render_csv(table))))[0]
    assert marker["Transition"].startswith("(no data rows)")
    assert marker["Occurrences"] == ""


def test_e8_table_with_no_escalations_writes_three_files_not_two(tmp_path):
    written = write_table(escalation_transition_table(
        ANALYSES["E8"]([row for row in sample_rows() if row["system"] != "adaptive"])
    ), tmp_path)

    assert set(written) == {"md", "csv", "json"}
    for path in written.values():
        assert Path(path).stat().st_size > 0


def test_a_table_with_rows_may_not_declare_an_empty_reason():
    with pytest.raises(ValueError, match="empty_reason"):
        Table(
            name="t",
            title="t",
            experiment_id="E1_baseline_comparison",
            columns=("A",),
            rows=[{"A": "1"}],
            empty_reason="nothing to see",
        )


def test_e4_table_carries_the_step_sweep_disclosure():
    """The markdown is the artifact pasted into a write-up, so the disclosure the
    JSON carries has to be on the table too."""
    analysis = ANALYSES["E4"](sample_rows())
    analysis["escalation_step_sweep_not_analysed"] = {
        "variants": ["max_escalation_steps=0", "max_escalation_steps=1"],
        "rows": 12,
        "reason": "not analysable as configured -- a design limitation, not a null result",
    }

    table = threshold_sweep_table(analysis)

    assert "NOT ANALYSED" in table.notes[-1]
    assert "max_escalation_steps=0, max_escalation_steps=1" in table.notes[-1]
    assert "design limitation" in table.notes[-1]
    # And no such note when there is nothing to disclose.
    assert not any("NOT ANALYSED" in note for note in threshold_sweep_table(
        ANALYSES["E4"](sample_rows())
    ).notes)


def test_e6_table_reports_the_ratio_without_ruling_on_it():
    table = routing_overhead_table(ANALYSES["E6"](sample_rows()))

    assert [row["Clock"] for row in table.rows] == [
        "routing_latency_ms",
        "retrieval_latency_ms",
        "reranking_latency_ms",
        "total_latency_ms",
    ]
    notes = " ".join(table.notes)
    assert "per-query mean 1.67%" in notes  # five rows at 2%, one at 2/12000
    assert "median 2.00%" in notes
    assert "aggregate 0.10%" in notes  # the 12 s tail dominates the sum ratio
    assert "negligible" in notes  # quoted in order to be disclaimed, not asserted
    assert "No threshold for 'acceptable' is applied" in notes


def test_e3_table_names_every_group_the_router_actually_scores():
    table = feature_ablation_table(ANALYSES["E3"](sample_rows()))

    assert table.rows[0]["Variant"] == "full"
    assert "without_entity" in table.notes[0] or "entity" in " ".join(table.notes)
    assert "Mean routing confidence" in table.columns
    assert "Top initial strategy" in table.columns


def test_e5_table_uses_the_sweeps_own_values():
    table = cost_weight_table(ANALYSES["E5"](sample_rows()))

    assert table.columns[0] == "cost_weight"
    assert "0.0 is pure evidence" in " ".join(table.notes)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_the_three_renderings_agree_cell_for_cell():
    table = main_comparison_table(ANALYSES["E1"](sample_rows()))

    parsed = json.loads(render_json(table))
    csv_rows = list(csv.DictReader(io.StringIO(render_csv(table))))
    markdown_body = [
        line for line in render_markdown(table).splitlines() if line.startswith("| ")
    ][2:]

    assert len(parsed["rows"]) == len(csv_rows) == len(markdown_body)
    for index, expected in enumerate(parsed["rows"]):
        assert csv_rows[index] == expected
        cells = [
            cell.strip().replace("\\|", "|")
            for cell in markdown_body[index].strip("| ").split(" | ")
        ]
        assert cells == [expected[column] for column in table.columns]


def test_rendering_is_deterministic_across_repeated_calls():
    table = main_comparison_table(ANALYSES["E1"](sample_rows()))

    assert render_markdown(table) == render_markdown(table)
    assert render_csv(table) == render_csv(table)
    assert render_json(table) == render_json(table)


def test_rendering_does_not_depend_on_input_row_order():
    """Rows arrive from a run's export; the table's order comes from the analysis,
    so shuffling the input cannot move a single cell."""
    rows = sample_rows()
    straight = main_comparison_table(analyze_main_comparison(rows))
    shuffled = list(rows)
    random.Random(4).shuffle(shuffled)
    reordered = main_comparison_table(analyze_main_comparison(shuffled))

    assert render_csv(straight) == render_csv(reordered)
    assert render_json(straight) == render_json(reordered)


def test_markdown_escapes_a_pipe_in_a_cell():
    table = Table(
        name="t",
        title="t",
        experiment_id="E1_baseline_comparison",
        columns=("A", "B"),
        rows=[{"A": "x|y", "B": "1"}],
    )

    body = render_markdown(table).splitlines()[4]

    assert body == r"| x\|y | 1 |"


def test_csv_uses_unix_line_endings_and_a_header_only_when_empty():
    table = Table(
        name="t", title="t", experiment_id="E1_baseline_comparison",
        columns=("A", "B"), rows=[],
    )

    rendered = render_csv(table)

    assert rendered == "A,B\n"
    assert "\r" not in rendered


def test_build_tables_covers_every_available_analysis_in_report_order():
    rows = sample_rows()
    analyses = {key: builder(rows) for key, builder in ANALYSES.items()}

    tables = build_tables(analyses)

    assert [t.name for t in tables] == [
        "e1_main_comparison",
        "e2_escalation_ablation",
        "e3_feature_ablation",
        "e4_threshold_sweep",
        "e5_cost_weight",
        "e6_routing_overhead",
        "e7_query_type",
        "e8_escalation_transitions",
    ]


def test_build_tables_skips_a_study_that_was_not_run():
    """A missing study is visibly missing rather than looking like a study with no
    findings."""
    tables = build_tables({"E1": ANALYSES["E1"](sample_rows())})

    assert [t.name for t in tables] == ["e1_main_comparison"]


def test_render_report_contains_every_table_and_a_contents_list():
    rows = sample_rows()
    tables = build_tables({key: builder(rows) for key, builder in ANALYSES.items()})

    report = render_report(tables)

    assert report.startswith("# Phase 7 analysis")
    assert TABLES_VERSION in report
    for table in tables:
        assert table.title in report
    assert report.count("| System |") == 1


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def test_write_tables_emits_three_formats_per_table_plus_one_report(tmp_path):
    rows = sample_rows()
    tables = build_tables({key: builder(rows) for key, builder in ANALYSES.items()})

    written = write_tables(tables, tmp_path)

    assert set(written) == {table.name for table in tables} | {"_report"}
    for name, paths in written.items():
        for kind, path in paths.items():
            assert Path(path).is_file()
            if kind != "md":
                assert path.endswith(f".{kind}")
    report = (tmp_path / "analysis_report.md").read_text(encoding="utf-8")
    assert report.count("## E") == 8


def test_written_files_are_byte_identical_across_two_writes(tmp_path):
    rows = sample_rows()
    tables = build_tables({"E1": ANALYSES["E1"](rows)})

    write_tables(tables, tmp_path / "first")
    write_tables(tables, tmp_path / "second")

    for name in ("e1_main_comparison.md", "e1_main_comparison.csv", "e1_main_comparison.json"):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


# --------------------------------------------------------------------------
# regression: _top_strategy reads nested counts/observed
# --------------------------------------------------------------------------


def test_top_strategy_reads_nested_structure():
    """_top_strategy must read counts/observed from the field's sub-dict, not
    the outer mapping. The structure from _strategy_distribution is:
        {"initial_strategy": {"observed": 47, "counts": {"hybrid": 47}, ...}}
    """
    distribution = {
        "initial_strategy": {
            "observed": 47,
            "counts": {"hybrid": 47, "bm25": 0},
            "shares": {"hybrid": 1.0, "bm25": 0.0},
        },
        "final_strategy": {
            "observed": 47,
            "counts": {"hybrid": 47},
            "shares": {"hybrid": 1.0},
        },
    }

    # initial_strategy has hybrid at 47/47 = 100%
    assert _top_strategy(distribution, "initial_strategy") == "hybrid (100%)"

    # final_strategy same
    assert _top_strategy(distribution, "final_strategy") == "hybrid (100%)"

    # Empty distribution
    assert _top_strategy({}, "initial_strategy") == ""
    assert _top_strategy(None, "initial_strategy") == ""

    # Field missing
    assert _top_strategy({"other": {"observed": 1, "counts": {"x": 1}}}, "initial_strategy") == ""

    # Zero observed
    assert _top_strategy({"initial_strategy": {"observed": 0, "counts": {}}}, "initial_strategy") == ""

    # Tie broken by name (max on (count, name) picks lexicographically last on tie)
    tie_dist = {
        "initial_strategy": {
            "observed": 10,
            "counts": {"bm25": 5, "dense": 5},
            "shares": {"bm25": 0.5, "dense": 0.5},
        }
    }
    assert _top_strategy(tie_dist, "initial_strategy") == "dense (50%)"
