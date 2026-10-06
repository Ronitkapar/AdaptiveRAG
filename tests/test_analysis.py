"""
tests.test_analysis
-------------------
Offline, deterministic checks for the Phase 7 analysis layer.

Every expected value here is hand-computed from the synthetic row set rather than
copied out of the module, so a change in the analysis arithmetic fails a test
instead of quietly changing a published number. The row set is tiny and its
values are chosen so the right answer is obvious: six BM25 latencies of
10..60 ms have a median of 35 and a nearest-rank p95 of 60, not an interpolated
57.5.
"""

from __future__ import annotations

import json

import pytest

from adaptive_rag.evaluation.analysis import (
    DEFAULT_COST_WEIGHT_VARIANTS,
    DEFAULT_ESCALATION_VARIANTS,
    DEFAULT_FEATURE_VARIANTS,
    DEFAULT_SYSTEMS,
    MIN_CELL,
    analyze_cost_weight,
    analyze_escalation_ablation,
    analyze_escalation_transitions,
    analyze_feature_ablation,
    analyze_main_comparison,
    analyze_query_type,
    analyze_routing_overhead,
    analyze_threshold_sweep,
    distribution,
    gold_chunk_map,
    gold_document_map,
)
from adaptive_rag.evaluation.dataset import VALID_CATEGORIES
from adaptive_rag.evaluation.rows import COLUMNS, EvaluationRow
from adaptive_rag.schemas.config import FEATURE_GROUPS
from adaptive_rag.schemas.experiment import EvaluationExample, SectionRef

#: Keys that would mean quality and cost had been collapsed into one number.
#: Checked recursively against every analyzer result, because a composite would
#: be as easy to add to a nested block as to the top level.
COMPOSITE_TOKENS = (
    "composite",
    "overall_score",
    "overall score",
    "winner",
    "best_system",
    "utility_score",
    "combined",
    "tradeoff_score",
)


def make_row(**overrides):
    """One flat row with every column present, so overrides are the only variable."""
    row = {name: None for name in COLUMNS}
    row.update(
        query_id="q0",
        system="adaptive",
        experiment_id="E1_baseline_comparison",
        category="factual",
        split="calibration",
        status="success",
        retrieved_document_ids=[],
        retrieved_chunk_ids=[],
        recall_at_1=0.0,
        recall_at_5=0.0,
        recall_at_10=0.0,
        precision_at_1=0.0,
        precision_at_5=0.0,
        precision_at_10=0.0,
        hit_at_1=0.0,
        hit_at_5=0.0,
        hit_at_10=0.0,
        mrr=0.0,
        ndcg_at_5=0.0,
        stage_latencies_ms=[],
        context_tokens=1000,
        estimated_cost_usd=0.001,
        input_tokens=900,
        output_tokens=100,
        total_tokens=1000,
    )
    row.update(overrides)
    return row


def keys_of(payload):
    """Every mapping key in a nested JSON-ready structure."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from keys_of(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from keys_of(item)


def bm25_rows(n=6):
    """BM25 over queries q0..q{n-1}: recall 1,1,1,0,0,0 and 10..60 ms."""
    recalls = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    return [
        make_row(
            query_id=f"q{i}",
            system="bm25",
            recall_at_5=recalls[i],
            mrr=recalls[i],
            ndcg_at_5=recalls[i],
            total_latency_ms=latencies[i],
            retrieval_latency_ms=latencies[i],
            routing_latency_ms=None,
            reranking_latency_ms=None,
        )
        for i in range(n)
    ]


def adaptive_rows(n=6):
    """Adaptive over the same queries, with a 12 s tail and two escalations."""
    recalls = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # The 12 000 ms sample stands in for the documented 12.0 s rate-limited
    # embedding call that the frozen cost table records against `hybrid`.
    latencies = [100.0, 110.0, 120.0, 130.0, 140.0, 12000.0]
    rows = []
    for i in range(n):
        escalated = i < 2
        rows.append(
            make_row(
                query_id=f"q{i}",
                system="adaptive",
                recall_at_5=recalls[i],
                mrr=0.5,
                ndcg_at_5=0.5,
                total_latency_ms=latencies[i],
                retrieval_latency_ms=latencies[i],
                reranking_latency_ms=10.0,
                routing_latency_ms=2.0,
                initial_strategy="bm25",
                final_strategy="hybrid" if escalated else "bm25",
                escalated=escalated,
                escalation_target="hybrid" if escalated else None,
                sufficiency_score=0.4,
                routing_confidence=0.6,
                stage_count=2 if escalated else 1,
                stage_latencies_ms=[latencies[i]],
                pre_escalation_chunk_ids=["c1", "c2"] if escalated else None,
                pre_escalation_document_ids=["d1"] if escalated else None,
            )
        )
    return rows


# --------------------------------------------------------------------------
# distribution
# --------------------------------------------------------------------------


def test_distribution_p95_is_nearest_rank_not_interpolated():
    """Six samples 10..60: median 35, p95 60 -- not the interpolated 57.5."""
    result = distribution([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    assert result["n"] == 6
    assert result["median"] == 35.0
    assert result["p95"] == 60.0
    assert result["mean"] == 35.0
    assert result["min"] == 10.0
    assert result["max"] == 60.0


def test_distribution_of_one_sample_has_no_stdev_and_of_none_is_empty():
    """A sample standard deviation needs two observations, and an empty sample
    reports `None` for every measure rather than a fabricated zero."""
    single = distribution([42.0])
    assert single["n"] == 1
    assert single["mean"] == 42.0
    assert single["stdev"] is None

    empty = distribution([])
    assert empty["n"] == 0
    assert all(empty[key] is None for key in ("mean", "median", "p95", "min", "max"))


# --------------------------------------------------------------------------
# E1
# --------------------------------------------------------------------------


def test_main_comparison_reports_every_axis_with_known_values():
    """E1 over the six-query set: BM25 median 35 ms / p95 60 ms, adaptive 125 / 12000."""
    result = analyze_main_comparison(bm25_rows() + adaptive_rows())

    assert result["experiment_id"] == "E1_baseline_comparison"
    assert result["systems_present"] == ["bm25", "adaptive"]
    assert result["systems_absent"] == ["dense", "hybrid", "hybrid_rerank"]

    bm25 = result["per_system"]["bm25"]
    assert bm25["n"] == 6
    assert bm25["insufficient_data"] is False
    assert bm25["quality"]["recall_at_5"]["mean"] == 0.5
    assert bm25["quality"]["recall_at_5"]["median"] == 0.5
    assert bm25["latency"]["total_latency_ms"]["median"] == 35.0
    assert bm25["latency"]["total_latency_ms"]["p95"] == 60.0
    assert bm25["latency"]["total_latency_ms"]["mean"] == 35.0
    # A fixed-strategy arm took no routing decision, so the routing clock is
    # undefined there -- `None`, not 0.0 ms.
    assert bm25["latency"]["routing_latency_ms"]["n"] == 0
    assert bm25["latency"]["routing_latency_ms"]["mean"] is None

    adaptive = result["per_system"]["adaptive"]
    assert adaptive["quality"]["recall_at_5"]["mean"] == 1.0
    assert adaptive["latency"]["total_latency_ms"]["median"] == 125.0
    assert adaptive["latency"]["total_latency_ms"]["p95"] == 12000.0
    assert adaptive["latency"]["total_latency_ms"]["mean"] == 2100.0
    assert adaptive["latency"]["routing_latency_ms"]["mean"] == 2.0
    assert adaptive["latency"]["retrieval_latency_ms"]["mean"] == 2100.0
    assert adaptive["latency"]["reranking_latency_ms"]["mean"] == 10.0
    assert adaptive["strategy_distribution"]["initial_strategy"]["counts"] == {"bm25": 6}
    assert adaptive["strategy_distribution"]["initial_strategy"]["shares"]["bm25"] == 1.0
    assert adaptive["escalation"]["escalation_rate"] == pytest.approx(2 / 6, abs=1e-6)
    assert adaptive["escalation_transitions"] == [
        {"from": "bm25", "to": "hybrid", "count": 2}
    ]
    assert adaptive["routing"]["stage_count"]["mean"] == pytest.approx(8 / 6, abs=1e-6)


def test_main_comparison_emits_no_composite_score_anywhere():
    """The research design forbids collapsing quality and cost into one number,
    so no key anywhere in the E1 result may name one."""
    result = analyze_main_comparison(bm25_rows() + adaptive_rows())

    offenders = [
        key
        for key in keys_of(result)
        if key != "composite_score_policy"
        and any(token in key.lower() for token in COMPOSITE_TOKENS)
    ]
    assert offenders == []
    # And the reason travels with the numbers, not only in the docstring.
    assert "composite" in result["composite_score_policy"].lower()


def test_main_comparison_flags_a_cell_below_the_minimum():
    """Three observations is below the documented threshold of five."""
    result = analyze_main_comparison(bm25_rows(n=3))

    cell = result["per_system"]["bm25"]
    assert cell["n"] == 3
    assert cell["insufficient_data"] is True
    assert cell["min_cell"] == MIN_CELL == 5


def test_main_comparison_reports_an_absent_system_as_an_empty_cell():
    """A requested arm with no rows is an empty cell, not a missing key -- and an
    empty cell is flagged, because 'the study produced nothing' is a finding."""
    result = analyze_main_comparison(bm25_rows())

    assert result["systems_absent"] == ["dense", "hybrid", "hybrid_rerank", "adaptive"]
    absent = result["per_system"]["adaptive"]
    assert absent["n"] == 0
    assert absent["insufficient_data"] is True
    assert absent["quality"]["recall_at_5"]["mean"] is None


def test_main_comparison_accepts_evaluation_row_objects_as_well_as_dicts():
    """`build_rows` returns models and `read_rows_jsonl` returns dicts; both work."""
    models = [EvaluationRow.model_validate(row) for row in bm25_rows()]

    from_models = analyze_main_comparison(models)
    from_dicts = analyze_main_comparison(bm25_rows())

    assert from_models == from_dicts


def test_main_comparison_defaults_to_the_five_declared_arms():
    assert DEFAULT_SYSTEMS == ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")


# --------------------------------------------------------------------------
# E2 / E3 / E4 / E5
# --------------------------------------------------------------------------


def test_escalation_ablation_uses_the_three_declared_variants():
    """A/B/C come from `evaluation.ablation.escalation_variants`, not from here."""
    assert DEFAULT_ESCALATION_VARIANTS == (
        "A_no_sufficiency_no_escalation",
        "B_sufficiency_no_escalation",
        "C_sufficiency_bounded_escalation",
    )

    rows = [
        make_row(query_id=f"q{i}", system="A_no_sufficiency_no_escalation",
                 recall_at_5=0.2, total_latency_ms=80.0, escalated=False, stage_count=1)
        for i in range(6)
    ] + [
        make_row(query_id=f"q{i}", system="B_sufficiency_no_escalation",
                 recall_at_5=0.2, total_latency_ms=85.0, escalated=False, stage_count=1,
                 sufficiency_score=0.4)
        for i in range(6)
    ] + [
        make_row(query_id=f"q{i}", system="C_sufficiency_bounded_escalation",
                 recall_at_5=0.6, total_latency_ms=140.0,
                 escalated=i < 3, stage_count=2 if i < 3 else 1)
        for i in range(6)
    ]

    result = analyze_escalation_ablation(rows)
    assert result["experiment_id"] == "E2_escalation_ablation"
    assert list(result["per_variant"]) == list(DEFAULT_ESCALATION_VARIANTS)
    assert result["variants_absent"] == []

    a = result["per_variant"]["A_no_sufficiency_no_escalation"]
    assert a["quality"]["recall_at_5"]["mean"] == 0.2
    assert a["escalation_rate"] == 0.0
    assert a["routing"]["stage_count"]["mean"] == 1.0

    c = result["per_variant"]["C_sufficiency_bounded_escalation"]
    assert c["quality"]["recall_at_5"]["mean"] == 0.6
    assert c["escalation_rate"] == 0.5
    assert c["routing"]["stage_count"]["mean"] == 1.5
    assert c["latency"]["total_latency_ms"]["mean"] == 140.0
    assert c["cost"]["estimated_cost_usd"]["mean"] == 0.001


def test_feature_ablation_covers_all_six_groups_including_entity():
    """`entity` carries the largest BM25 weight in the shipped rule table (1.5
    against lexical's 1.0), so a sweep that skipped it would leave the most
    influential signal unmeasured."""
    assert FEATURE_GROUPS == (
        "lexical",
        "semantic",
        "entity",
        "complexity",
        "question_type",
        "multi_concept",
    )
    assert DEFAULT_FEATURE_VARIANTS == ("full", *(f"without_{g}" for g in FEATURE_GROUPS))
    assert "without_entity" in DEFAULT_FEATURE_VARIANTS

    rows = [
        make_row(query_id=f"q{i}", system="full", initial_strategy="bm25",
                 final_strategy="bm25", routing_confidence=0.7, escalated=False,
                 recall_at_5=0.5, total_latency_ms=100.0)
        for i in range(6)
    ] + [
        make_row(query_id=f"q{i}", system="without_entity", initial_strategy="dense",
                 final_strategy="dense", routing_confidence=0.4, escalated=True,
                 escalation_target="hybrid", recall_at_5=0.3, total_latency_ms=400.0)
        for i in range(6)
    ]

    result = analyze_feature_ablation(rows)
    assert result["experiment_id"] == "E3_feature_ablation"
    full = result["per_variant"]["full"]
    assert full["routing"]["routing_confidence"]["mean"] == 0.7
    assert full["strategy_distribution"]["initial_strategy"]["counts"] == {"bm25": 6}
    assert full["escalation_rate"] == 0.0

    ablated = result["per_variant"]["without_entity"]
    assert ablated["routing"]["routing_confidence"]["mean"] == 0.4
    assert ablated["strategy_distribution"]["initial_strategy"]["counts"] == {"dense": 6}
    assert ablated["escalation_rate"] == 1.0
    assert ablated["quality"]["recall_at_5"]["mean"] == 0.3
    assert ablated["latency"]["total_latency_ms"]["mean"] == 400.0
    # The four arms that were not run are reported absent, not omitted.
    assert set(result["variants_absent"]) == {
        "without_lexical",
        "without_semantic",
        "without_complexity",
        "without_question_type",
        "without_multi_concept",
    }


def test_threshold_sweep_reports_each_candidate_with_its_own_escalation_rate():
    """A lower threshold escalates more often; the table has to show both sides."""
    result = analyze_threshold_sweep(
        [
            make_row(query_id=f"q{i}", system="sufficiency_threshold=0.3",
                     recall_at_5=0.4, total_latency_ms=200.0, escalated=i < 4,
                     escalation_target="hybrid" if i < 4 else None,
                     initial_strategy="bm25",
                     final_strategy="hybrid" if i < 4 else "bm25")
            for i in range(6)
        ]
        + [
            make_row(query_id=f"q{i}", system="sufficiency_threshold=0.7",
                     recall_at_5=0.8, total_latency_ms=110.0, escalated=False,
                     initial_strategy="bm25", final_strategy="bm25")
            for i in range(6)
        ]
    )

    assert result["experiment_id"] == "E4_threshold_calibration"
    low = result["per_variant"]["sufficiency_threshold=0.3"]
    high = result["per_variant"]["sufficiency_threshold=0.7"]
    assert low["escalation_rate"] == pytest.approx(4 / 6, abs=1e-6)
    assert high["escalation_rate"] == 0.0
    assert low["latency"]["total_latency_ms"]["mean"] == 200.0
    assert high["latency"]["total_latency_ms"]["mean"] == 110.0
    assert low["strategy_distribution"]["initial_strategy"]["counts"] == {"bm25": 6}
    assert low["cost"]["estimated_cost_usd"]["mean"] == 0.001


def _step_sweep_rows():
    """Rows for E4's second sweep, which shares the study id with the first."""
    return [
        make_row(query_id=f"q{i}", system=f"max_escalation_steps={steps}",
                 recall_at_5=0.5, total_latency_ms=300.0, escalated=False,
                 initial_strategy="bm25", final_strategy="bm25")
        for steps in range(4)
        for i in range(6)
    ]


def test_threshold_sweep_names_the_step_arms_it_did_not_analyse():
    """E4 registered nine arms and the threshold variant list covers five. The
    other four are degenerate by construction, so they must be *named* -- a
    `variants_absent` of [] plus an n_rows nobody can reconcile reads exactly
    like a complete table."""
    threshold_rows = [
        make_row(query_id=f"q{i}", system=f"sufficiency_threshold={value}",
                 recall_at_5=0.5, total_latency_ms=300.0, escalated=False)
        for value in ("0.3", "0.4", "0.5", "0.6", "0.7")
        for i in range(6)
    ]
    step_rows = _step_sweep_rows()

    result = analyze_threshold_sweep(threshold_rows + step_rows)

    assert result["variants_absent"] == []  # every *requested* arm has rows...
    assert result["variants_not_analysed"] == {
        "max_escalation_steps=0": 6,
        "max_escalation_steps=1": 6,
        "max_escalation_steps=2": 6,
        "max_escalation_steps=3": 6,
    }
    assert "max_escalation_steps=0" not in result["per_variant"]
    # The arithmetic a reader needs in order to notice the drop is checkable.
    assert result["n_rows"] == 54  # 5 threshold arms x 6 + 4 step arms x 6
    assert result["n_rows_analysed"] == 30
    assert result["n_rows_analysed"] + sum(result["variants_not_analysed"].values()) == (
        result["n_rows"]
    )
    assert "n_rows_analysed" in result["variants_not_analysed_note"]


def test_threshold_sweep_discloses_why_the_step_sweep_is_not_tabulated():
    result = analyze_threshold_sweep(_step_sweep_rows())

    disclosure = result["escalation_step_sweep_not_analysed"]
    assert disclosure["variants"] == [f"max_escalation_steps={s}" for s in range(4)]
    assert disclosure["rows"] == 24
    reason = disclosure["reason"]
    # A design limitation, stated as one, and not dressed up as a measurement.
    assert "design limitation" in reason
    assert "not a null result" in reason
    assert "sufficiency_threshold=0.5" in reason
    assert "steps_taken=0" in reason


def test_threshold_sweep_makes_no_disclosure_when_those_arms_never_ran():
    """The disclosure describes real rows. With none, claiming it would be the
    same error in the opposite direction."""
    result = analyze_threshold_sweep(
        [
            make_row(query_id=f"q{i}", system="sufficiency_threshold=0.5",
                     recall_at_5=0.5, total_latency_ms=300.0)
            for i in range(6)
        ]
    )

    assert result["variants_not_analysed"] == {}
    assert "variants_not_analysed_note" not in result
    assert "escalation_step_sweep_not_analysed" not in result


def test_every_variant_study_accounts_for_all_of_its_rows():
    """Generic guard: no study's rows may vanish between n_rows and the table."""
    rows = _step_sweep_rows() + [
        make_row(query_id=f"q{i}", system=name, recall_at_5=0.5,
                 total_latency_ms=300.0)
        for name in DEFAULT_ESCALATION_VARIANTS
        for i in range(6)
    ]

    for analyzer in (analyze_escalation_ablation, analyze_threshold_sweep):
        result = analyzer(rows)
        described = sum(cell["n"] for cell in result["per_variant"].values())
        assert described + sum(result["variants_not_analysed"].values()) == result["n_rows"]


def test_cost_weight_uses_the_values_the_ablation_module_implements():
    """No invented parameters: the grid is `cost_weight_sweep`'s, 0.0 through 1.0."""
    assert DEFAULT_COST_WEIGHT_VARIANTS == (
        "cost_weight=0.0",
        "cost_weight=0.25",
        "cost_weight=0.5",
        "cost_weight=0.75",
        "cost_weight=1.0",
    )

    result = analyze_cost_weight(
        [
            make_row(query_id=f"q{i}", system="cost_weight=0.0", recall_at_5=0.9,
                     total_latency_ms=300.0, initial_strategy="hybrid_rerank",
                     final_strategy="hybrid_rerank", escalated=False)
            for i in range(6)
        ]
        + [
            make_row(query_id=f"q{i}", system="cost_weight=1.0", recall_at_5=0.5,
                     total_latency_ms=20.0, initial_strategy="bm25",
                     final_strategy="bm25", escalated=False)
            for i in range(6)
        ]
    )

    assert result["experiment_id"] == "E5_cost_weight"
    quality_first = result["per_variant"]["cost_weight=0.0"]
    efficiency_first = result["per_variant"]["cost_weight=1.0"]
    assert quality_first["quality"]["recall_at_5"]["mean"] == 0.9
    assert quality_first["latency"]["total_latency_ms"]["mean"] == 300.0
    assert quality_first["strategy_distribution"]["initial_strategy"]["counts"] == {
        "hybrid_rerank": 6
    }
    assert efficiency_first["quality"]["recall_at_5"]["mean"] == 0.5
    assert efficiency_first["latency"]["total_latency_ms"]["mean"] == 20.0


# --------------------------------------------------------------------------
# E6
# --------------------------------------------------------------------------


def test_routing_overhead_reports_the_measured_ratio_and_no_verdict():
    """2 ms of routing inside 100 ms of total is 2%. The function reports that
    number; it does not conclude anything about whether it is acceptable."""
    result = analyze_routing_overhead(
        [make_row(query_id=f"q{i}", routing_latency_ms=2.0, total_latency_ms=100.0)
         for i in range(6)]
    )

    assert result["experiment_id"] == "E6_routing_overhead"
    assert result["n"] == 6
    assert result["latency"]["routing_latency_ms"]["mean"] == 2.0
    assert result["latency"]["total_latency_ms"]["mean"] == 100.0
    assert result["routing_share_of_total"]["per_query"]["mean"] == 0.02
    assert result["routing_share_of_total"]["aggregate"] == 0.02
    assert result["routing_share_of_total"]["per_query"]["min"] == 0.02
    assert result["routing_share_of_total"]["per_query"]["max"] == 0.02
    assert "negligible" not in json.dumps(result).lower()


def test_routing_overhead_ignores_arms_that_never_routed():
    """A fixed-strategy arm has no routing clock, so it contributes no ratio --
    and inventing a zero-millisecond decision would understate the share."""
    rows = adaptive_rows() + bm25_rows()

    result = analyze_routing_overhead(rows)

    assert result["n"] == 6
    assert result["n_rows"] == 12
    assert result["routing_share_of_total"]["aggregate"] is not None


def test_routing_overhead_keeps_the_tail_visible():
    """The mean and the p95 of the same 12 s-bearing sample, both reported."""
    result = analyze_routing_overhead(adaptive_rows())

    total = result["latency"]["total_latency_ms"]
    assert total["mean"] == 2100.0
    assert total["median"] == 125.0
    assert total["p95"] == 12000.0
    assert total["max"] == 12000.0


# --------------------------------------------------------------------------
# E7
# --------------------------------------------------------------------------


def test_query_type_breaks_down_on_the_dataset_label_and_nothing_else():
    """`category` is the only ground-truth type column; the router's own
    QueryFeatures are reported as absent and explicitly not ground truth."""
    rows = [
        make_row(query_id=f"q{i}", category="factual", recall_at_5=0.5,
                 total_latency_ms=100.0, initial_strategy="bm25", final_strategy="bm25",
                 escalated=False)
        for i in range(5)
    ] + [
        make_row(query_id=f"q5", category="factual", recall_at_5=0.9,
                 initial_strategy="bm25", final_strategy="bm25", escalated=False),
        make_row(query_id="r0", category="comparative", recall_at_5=0.2,
                 escalated=True, escalation_target="hybrid", initial_strategy="bm25",
                 final_strategy="hybrid"),
        make_row(query_id="r1", category="comparative", recall_at_5=0.4,
                 escalated=False, initial_strategy="bm25", final_strategy="bm25"),
    ]

    result = analyze_query_type(rows)

    assert result["experiment_id"] == "E7_query_type"
    assert set(result["per_category"]) <= VALID_CATEGORIES
    assert set(result["per_category"]) == {"factual", "comparative"}
    assert result["categories_absent"] == sorted(
        VALID_CATEGORIES - {"factual", "comparative"}
    )

    derived = result["derived_breakdown"]
    assert derived["ground_truth"] is False
    assert derived["available"] is False
    assert "QueryFeatures" in derived["reason"]
    for field in ("question_type", "complexity_score", "concept_count"):
        assert field in derived["fields_not_reported"]
        assert field not in result["per_category"]["factual"]["quality"]

    factual = result["per_category"]["factual"]
    assert factual["n"] == 6
    assert factual["insufficient_data"] is False
    assert factual["quality"]["recall_at_5"]["mean"] == pytest.approx(3.4 / 6, abs=1e-6)
    initial = factual["strategy_selection"]["initial_strategy"]
    assert initial["observed"] == 6
    assert initial["counts"] == {"bm25": 6}
    assert initial["shares"]["bm25"] == 1.0

    comparative = result["per_category"]["comparative"]
    assert comparative["n"] == 2
    assert comparative["insufficient_data"] is True
    # The rate is over the rows that recorded a decision (both of them here),
    # not over the cell size, so it is one of two rather than one of three.
    assert comparative["escalation_rate"] == 0.5
    assert comparative["escalation"]["n_decisions"] == 2


def test_query_type_preserves_one_trace_per_query():
    """The per-query rows travel with the aggregate, sorted and complete."""
    rows = adaptive_rows()
    result = analyze_query_type(rows, include_traces=True)

    assert len(result["traces"]) == 6
    assert [trace["query_id"] for trace in result["traces"]] == [f"q{i}" for i in range(6)]
    assert result["traces"][0]["initial_strategy"] == "bm25"
    assert result["traces"][0]["escalated"] is True
    assert result["traces"][5]["escalated"] is False
    # A non-escalated row's pre-escalation evidence is None, and the trace keeps
    # that distinction rather than collapsing it to an empty list.
    assert result["traces"][5]["escalation_target"] is None

    assert analyze_query_type(rows, include_traces=False)["traces"] == []


# --------------------------------------------------------------------------
# E8
# --------------------------------------------------------------------------


def _escalated_row(query_id, source, target, *, before_chunks, after, **extra):
    return make_row(
        query_id=query_id,
        escalated=True,
        initial_strategy=source,
        final_strategy=target,
        escalation_target=target,
        pre_escalation_chunk_ids=before_chunks,
        pre_escalation_document_ids=None,
        recall_at_5=after,
        **extra,
    )


def test_escalation_transitions_reports_only_observed_pairs():
    """Two pairs fired; the ladder's other rungs must not appear as 0.0% rows."""
    rows = [
        _escalated_row(f"h{i}", "bm25", "hybrid", before_chunks=["c1", "c2"], after=0.5)
        for i in range(3)
    ] + [
        _escalated_row(f"d{i}", "bm25", "dense", before_chunks=["c1"], after=0.8)
        for i in range(2)
    ] + [
        # Not escalated at all: its pre-escalation fields are None and it
        # contributes no transition.
        make_row(query_id="q9", escalated=False, initial_strategy="bm25",
                 final_strategy="bm25", escalation_target=None,
                 pre_escalation_chunk_ids=None, recall_at_5=1.0),
    ]

    result = analyze_escalation_transitions(
        rows, gold_chunk_ids={**{f"h{i}": {"c1", "c2"} for i in range(3)},
                              **{f"d{i}": {"c2"} for i in range(2)}}
    )

    assert result["experiment_id"] == "E8_escalation_analysis"
    assert result["n_escalated"] == 5
    assert result["n_transitions_observed"] == 2
    observed = {(t["from"], t["to"]): t for t in result["transitions"]}
    assert set(observed) == {("bm25", "hybrid"), ("bm25", "dense")}
    assert ("hybrid", "hybrid_rerank") not in observed
    assert ("dense", "hybrid_rerank") not in observed

    hybrid = observed[("bm25", "hybrid")]
    assert hybrid["occurrences"] == 3
    assert hybrid["share_of_escalated"] == 0.6
    # Three occurrences is below the threshold of five, so the cell is flagged
    # even though its numbers are exact.
    assert hybrid["insufficient_data"] is True
    assert hybrid["min_cell"] == 5
    # 2 of 2 gold chunks in the first five slots is a perfect pre-escalation
    # recall, and the post-escalation mean is 0.5, so every query worsened.
    chunk = hybrid["quality"]["chunk_level"]
    assert chunk["n_paired"] == 3
    assert chunk["mean_before"] == 1.0
    assert chunk["mean_after"] == 0.5
    assert chunk["improvement_rate"] == 0.0
    assert chunk["worsening_rate"] == 1.0
    assert chunk["median_delta"] == -0.5
    assert hybrid["added_latency"]["total_latency_ms"]["mean"] is None

    dense = observed[("bm25", "dense")]
    chunk = dense["quality"]["chunk_level"]
    assert chunk["improvement_rate"] == 1.0
    assert chunk["worsening_rate"] == 0.0
    assert chunk["n_unchanged"] == 0


def test_escalation_transitions_treats_absent_pre_escalation_as_unknown():
    """`None` means 'not recorded'. It must not be read as a measured 0.0, and it
    must not be counted in the improvement rate's denominator."""
    rows = [
        _escalated_row("e0", "bm25", "hybrid", before_chunks=["c1"], after=0.9),
        _escalated_row("e1", "bm25", "hybrid", before_chunks=None, after=0.9),
        _escalated_row("e2", "bm25", "hybrid", before_chunks=["c9"], after=0.9),
    ]

    result = analyze_escalation_transitions(rows, gold_chunk_ids={"e0": {"c1"}})

    transition = result["transitions"][0]
    assert transition["occurrences"] == 3
    assert result["n_escalated_without_pre_escalation_evidence"] == 1
    # Only e0 has a measurable 'before'; e1 and e2 are excluded from the pairing
    # rather than counted as zeros.
    assert transition["quality"]["chunk_level"]["n_paired"] == 1
    assert transition["quality"]["chunk_level"]["improvement_rate"] == 0.0
    reasons = " ".join(transition["unpaired_before_reasons"])
    assert "absent" in reasons
    assert "no relevant chunks are labelled" in reasons
    assert "pre_escalation_document_ids is absent" in reasons
    assert result["gold_document_ids_available"] is False
    assert result["gold_chunk_ids_available"] is True


def test_escalation_transitions_treats_an_empty_first_stage_as_a_measured_zero():
    """`[]` is a real observation -- the first stage returned nothing -- and it
    does score 0.0. That is the difference the two cases exist to keep."""
    rows = [
        _escalated_row("e0", "bm25", "hybrid", before_chunks=[], after=1.0),
        _escalated_row("e1", "bm25", "hybrid", before_chunks=[], after=0.5),
    ]

    result = analyze_escalation_transitions(rows, gold_chunk_ids={"e0": {"c1"}})

    chunk = result["transitions"][0]["quality"]["chunk_level"]
    assert chunk["n_paired"] == 2
    assert chunk["mean_before"] == 0.0
    assert chunk["improvement_rate"] == 1.0
    assert result["transitions"][0]["occurrences"] == 2


def test_escalation_transitions_reports_document_level_quality_independently():
    """Document-level quality-before needs only the dataset's labels, so it is
    available even when no chunk-level gold was supplied."""
    rows = [
        make_row(
            query_id="e0",
            escalated=True,
            initial_strategy="bm25",
            final_strategy="hybrid",
            escalation_target="hybrid",
            pre_escalation_chunk_ids=None,
            pre_escalation_document_ids=["d1"],
            recall_at_5=1.0,
        )
    ]

    result = analyze_escalation_transitions(
        rows, gold_document_ids={"e0": {"d1", "d2"}}
    )

    document_level = result["transitions"][0]["quality"]["document_level"]
    assert document_level["n_paired"] == 1
    assert document_level["mean_before"] == 0.5
    assert document_level["improvement_rate"] == 1.0
    assert result["gold_document_ids_available"] is True
    assert result["gold_chunk_ids_available"] is False


def test_escalation_transitions_rejects_a_non_quality_metric():
    """'Quality improvement' measured in milliseconds would be nonsense."""
    with pytest.raises(ValueError, match="not a quality column"):
        analyze_escalation_transitions(adaptive_rows(), metric="total_latency_ms")


def test_escalation_transitions_with_no_escalations_reports_none():
    result = analyze_escalation_transitions(adaptive_rows(n=6)[:0] or bm25_rows())

    assert result["n_escalated"] == 0
    assert result["n_transitions_observed"] == 0
    assert result["transitions"] == []


# --------------------------------------------------------------------------
# gold relevance maps
# --------------------------------------------------------------------------


def _example(example_id, documents, sections=()):
    return EvaluationExample(
        example_id=example_id,
        query="q",
        reference_answer="a",
        relevant_documents=list(documents),
        relevant_sections=[
            SectionRef(document_id=document, section_path_prefix=list(prefix))
            for document, prefix in sections
        ],
        category="factual",
    )


def test_gold_document_map_reads_the_dataset_labels(tmp_path):
    examples = [_example("a", ["doc1"]), _example("b", ["doc2", "doc3"])]

    gold = gold_document_map(examples)

    assert gold == {"a": {"doc1"}, "b": {"doc2", "doc3"}}


def test_gold_chunk_map_applies_the_dataset_relevance_rule(tmp_path):
    """Same rule as `evaluation.dataset.relevant_chunk_ids`: a chunk counts when
    its document is relevant and, where sections are labelled, its path starts
    with a labelled prefix."""
    chunks = tmp_path / "corpus"
    chunks.mkdir()
    (chunks / "doc.chunks.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "chunk_id": "c1",
                    "document_id": "doc1",
                    "text": "x",
                    "metadata": {"section_path": ["2 Methods", "2.1 Models"]},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "doc1",
                    "text": "x",
                    "metadata": {"section_path": ["3 Results"]},
                },
                {
                    "chunk_id": "c3",
                    "document_id": "other",
                    "text": "x",
                    "metadata": {"section_path": ["2 Methods", "2.1 Models"]},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = [_example("a", ["doc1"], sections=[("doc1", ["2 Methods", "2.1 Models"])])]
    gold = gold_chunk_map(examples, chunks_dir=chunks)

    # c1 is inside the labelled prefix; c2 is not; c3 is in an irrelevant document.
    assert gold == {"a": {"c1"}}


def test_gold_chunk_map_without_a_prefix_matches_every_chunk_of_the_document(tmp_path):
    chunks = tmp_path / "corpus"
    chunks.mkdir()
    (chunks / "doc.chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "text": "x",
                "metadata": {"section_path": ["3 Results"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    gold = gold_chunk_map([_example("a", ["doc1"])], chunks_dir=chunks)

    assert gold == {"a": {"c1"}}


def test_gold_chunk_map_returns_empty_when_the_corpus_is_absent(tmp_path):
    """The chunk files are gitignored and regenerable, so an absent corpus is a
    reported absence (via `gold_chunk_ids_available`), not a crash."""
    assert gold_chunk_map([_example("a", ["doc1"])], chunks_dir=tmp_path / "nope") == {}


# --------------------------------------------------------------------------
# serialisability
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        lambda rows: analyze_main_comparison(rows),
        lambda rows: analyze_escalation_ablation(rows),
        lambda rows: analyze_feature_ablation(rows),
        lambda rows: analyze_threshold_sweep(rows),
        lambda rows: analyze_cost_weight(rows),
        lambda rows: analyze_routing_overhead(rows),
        lambda rows: analyze_query_type(rows),
        lambda rows: analyze_escalation_transitions(rows),
    ],
)
def test_every_analyzer_returns_json_serialisable_output(builder):
    rows = adaptive_rows() + bm25_rows()

    payload = builder(rows)

    assert json.loads(json.dumps(payload)) == payload
    assert payload["analysis_version"] == "phase7_analysis_v1"
