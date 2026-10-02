"""
Tests for the Phase 7.0b measurement protocol (`evaluation.measurement`).

Offline and deterministic, per AGENTS.md: no index, no API keys, no timing.
What is tested is the protocol's logic -- schedule construction, warm-up
exclusion, estimator choice, and the guards that stop a broken sweep from
becoming a frozen cost table.
"""

from __future__ import annotations

import statistics

import pytest

from adaptive_rag.errors import ConfigurationError
from adaptive_rag.evaluation.base import percentile
from adaptive_rag.evaluation.measurement import (
    COST_STRATEGIES,
    CostSample,
    aggregate_samples,
    arm_schedule,
    build_artifact,
    compare_cost_tables,
    proposed_cost_table,
    rotated,
    sample_order,
    summarize_all,
    warmup_queries,
)


class _Example:
    """Minimal stand-in for an EvaluationExample; the protocol only needs an id."""

    def __init__(self, example_id: str) -> None:
        self.example_id = example_id

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Example({self.example_id!r})"


def _samples(strategy: str, values: list[float]) -> list[CostSample]:
    return [
        CostSample(
            strategy=strategy,
            example_id=f"q{i}",
            repetition=i // 2,
            latency_ms=value,
        )
        for i, value in enumerate(values)
    ]


def _summary(strategy: str, values: list[float]):
    return aggregate_samples(_samples(strategy, values), strategy=strategy)


# --- schedule construction ----------------------------------------------------


def test_rotation_moves_every_item_and_preserves_membership():
    items = list("abcde")
    for offset in range(len(items)):
        rotated_items = rotated(items, offset)
        assert sorted(rotated_items) == sorted(items), "rotation must not lose items"
        assert len(rotated_items) == len(items), "rotation must not duplicate items"


def test_rotation_wraps_around_when_offset_exceeds_length():
    items = ["a", "b", "c"]
    assert rotated(items, 4) == ["b", "c", "a"]
    assert rotated(items, 0) == items
    assert rotated([], 3) == []


def test_arm_schedule_gives_every_arm_every_position():
    """The property the schedule exists for: no arm holds a fixed position.

    With at least as many repetitions as arms, each arm appears in each
    position exactly once, so an arm cannot systematically absorb the
    machine-warmup cost of running first, or whatever else the machine was
    doing when it ran last.
    """
    arms = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    schedule = arm_schedule(arms, len(arms))

    assert len(schedule) == len(arms)
    for position in range(len(arms)):
        at_position = [order[position] for order in schedule]
        assert sorted(at_position) == sorted(arms), (
            f"position {position} did not see every arm exactly once"
        )

    for order in schedule:
        assert sorted(order) == sorted(arms), "each repetition runs every arm once"


def test_arm_schedule_rejects_non_positive_repetitions():
    with pytest.raises(ConfigurationError):
        arm_schedule(["bm25"], 0)


def test_query_order_rotates_between_repetitions():
    queries = [_Example(f"q{i}") for i in range(4)]
    first = [ex.example_id for ex in sample_order(queries, 0)]
    second = [ex.example_id for ex in sample_order(queries, 1)]

    assert first == ["q0", "q1", "q2", "q3"]
    assert second == ["q1", "q2", "q3", "q0"]
    assert first != second, "a static query order would load position effects onto one arm"
    assert sorted(first) == sorted(second)


def test_warmup_queries_are_deterministic_and_excluded_from_the_query_set():
    examples = [_Example(f"q{i}") for i in range(10)]
    warm = warmup_queries(examples, 5)

    assert [ex.example_id for ex in warm] == ["q0", "q1", "q2", "q3", "q4"]
    # Deterministic across calls: two runs must warm the same queries.
    assert [ex.example_id for ex in warmup_queries(examples, 5)] == [
        ex.example_id for ex in warm
    ]


def test_warmup_rejects_more_queries_than_the_dataset_holds():
    with pytest.raises(ConfigurationError):
        warmup_queries([_Example("q0")], 5)


def test_warmup_rejects_a_negative_count():
    with pytest.raises(ConfigurationError):
        warmup_queries([_Example("q0")], -1)


# --- aggregation --------------------------------------------------------------


def test_median_is_the_midpoint_not_the_nearest_rank():
    """`statistics.median` and `base.percentile(_, 50)` disagree at even n.

    At n=4 the median is the midpoint of the two middle values; nearest-rank
    returns the lower one. The module docstring documents this, and the test
    pins it -- otherwise a well-meaning refactor to reuse `percentile` would
    silently shift every proposed value.
    """
    values = [10.0, 20.0, 30.0, 40.0]
    summary = _summary("dense", values)

    assert summary.median_ms == 25.0
    assert percentile(values, 50) == 20.0
    assert summary.median_ms != percentile(values, 50)


def test_p95_is_nearest_rank_over_observed_samples():
    values = [float(i) for i in range(1, 101)]  # 1..100
    summary = _summary("dense", values)

    # Nearest-rank: ceil(0.95 * 100) = 95th smallest, which is the value 95.
    assert summary.p95_ms == 95.0
    assert summary.p95_ms in values, "p95 must be an observed sample, not interpolated"
    assert summary.p95_ms != 95.5, "linear interpolation would give 95.5"


def test_aggregate_reports_full_dispersion_and_keeps_raw_samples():
    values = [10.0, 12.0, 14.0, 16.0]
    summary = _summary("hybrid", values)

    assert summary.n == 4
    assert summary.median_ms == statistics.median(values)
    assert summary.mean_ms == sum(values) / len(values)
    assert summary.min_ms == 10.0
    assert summary.max_ms == 16.0
    assert summary.stdev_ms == pytest.approx(statistics.stdev(values))
    assert sorted(summary.raw_ms) == sorted(values), "raw samples must travel with the summary"


def test_all_equal_samples_give_zero_dispersion_not_a_crash():
    summary = _summary("bm25", [2.5] * 8)

    assert summary.stdev_ms == 0.0
    assert summary.median_ms == 2.5
    assert summary.p95_ms == 2.5


def test_a_single_sample_reports_zero_stdev():
    summary = _summary("bm25", [3.0])

    assert summary.n == 1
    assert summary.stdev_ms == 0.0
    assert summary.median_ms == 3.0


def test_empty_samples_give_an_empty_summary_rather_than_a_crash():
    summary = aggregate_samples([], strategy="dense")

    assert summary.n == 0
    assert summary.median_ms is None
    assert summary.raw_ms == []


def test_aggregate_refuses_samples_from_another_strategy():
    with pytest.raises(ConfigurationError):
        aggregate_samples(_samples("dense", [1.0]), strategy="bm25")


def test_fallback_samples_are_counted():
    """A fallback reports a near-zero rerank latency that is not a real rerank."""
    samples = _samples("hybrid_rerank", [4000.0, 5.0])
    samples[1].rerank_fallback = True

    summary = aggregate_samples(samples, strategy="hybrid_rerank")
    assert summary.n_fallback == 1


# --- proposed table -----------------------------------------------------------


def _all_strategies() -> list:
    return [
        _summary("bm25", [1.7, 2.0, 1.5]),
        _summary("dense", [800.0, 900.0, 700.0]),
        _summary("hybrid", [850.0, 900.0, 800.0]),
        _summary("hybrid_rerank", [4500.0, 5200.0, 4800.0]),
    ]


def test_proposed_table_is_the_rounded_median_of_each_strategy():
    table = proposed_cost_table(_all_strategies())

    assert set(table) == set(COST_STRATEGIES)
    assert table["bm25"] == 1.7
    assert table["dense"] == 800.0
    assert table["hybrid"] == 850.0
    assert table["hybrid_rerank"] == 4800.0


def test_proposed_table_excludes_the_adaptive_arm():
    """The adaptive arm has no single per-strategy cost -- it is a mixture.

    Folding it in would be the single easiest way to make this table wrong,
    so it is rejected outright rather than warned about.
    """
    summaries = _all_strategies()
    summaries.append(_summary("adaptive", [100.0, 200.0, 150.0]))

    with pytest.raises(ConfigurationError, match="adaptive"):
        proposed_cost_table(summaries)


def test_proposed_table_rejects_an_incomplete_sweep():
    summaries = [s for s in _all_strategies() if s.strategy != "dense"]

    with pytest.raises(ConfigurationError, match="dense"):
        proposed_cost_table(summaries)


def test_proposed_table_refuses_fallback_samples():
    """A fallback's median does not describe the cross-encoder's real cost."""
    samples = _samples("hybrid_rerank", [4000.0, 5.0])
    samples[1].rerank_fallback = True
    summaries = [
        _summary("bm25", [1.7]),
        _summary("dense", [800.0]),
        _summary("hybrid", [850.0]),
        aggregate_samples(samples, strategy="hybrid_rerank"),
    ]

    with pytest.raises(ConfigurationError, match="fallback"):
        proposed_cost_table(summaries)


def test_comparison_reports_both_tables_side_by_side():
    current = {"bm25": 1.71, "dense": 621.35, "hybrid": 721.0, "hybrid_rerank": 4447.68}
    proposed = proposed_cost_table(_all_strategies())
    rows = compare_cost_tables(current, proposed)

    assert [row["strategy"] for row in rows] == list(COST_STRATEGIES)
    by_name = {row["strategy"]: row for row in rows}

    assert by_name["dense"]["current_ms"] == 621.35
    assert by_name["dense"]["proposed_ms"] == 800.0
    assert by_name["dense"]["changed"] is True
    # Normalised ratios are what the router consumes, so they are reported too.
    assert by_name["hybrid_rerank"]["ratio_proposed"] == 1.0


def test_comparison_flags_an_unchanged_value():
    current = {"bm25": 1.7, "dense": 800.0, "hybrid": 850.0, "hybrid_rerank": 4800.0}
    rows = compare_cost_tables(current, proposed_cost_table(_all_strategies()))
    by_name = {row["strategy"]: row for row in rows}

    assert all(row["changed"] is False for row in rows)


def test_comparison_tolerates_only_sub_hundredth_difference():
    """A 0.01 ms gap is a real difference, not noise to round away.

    BM25's latencies are single-digit milliseconds, so a tolerance loose enough
    to hide sub-millisecond drift would erase the entire signal for the
    cheapest arm.
    """
    proposed = proposed_cost_table(_all_strategies())
    # Spread order matters: `{**proposed, "bm25": ...}` overrides the key, while
    # `{"bm25": ..., **proposed}` would let proposed overwrite it back.
    rows = compare_cost_tables({**proposed, "bm25": 1.71}, proposed)
    by_name = {row["strategy"]: row for row in rows}

    assert by_name["bm25"]["changed"] is True
    assert by_name["dense"]["changed"] is False


# --- artifact -----------------------------------------------------------------


def test_stage_medians_separate_the_embedding_api_call_from_retrieval():
    """The end-to-end clock hides how much of it is a live API round-trip.

    `AICreditsEmbeddingModel` has no query cache, so for dense and hybrid the
    query embedding is a network call inside `latency_ms`. Reporting only the
    end-to-end number would present gateway latency as a property of the
    retrieval strategy.
    """
    samples = _samples("dense", [1200.0, 1100.0, 1300.0])
    for i, sample in enumerate(samples):
        sample.stage_latency_ms = {
            "query_embedding_latency_ms": 900.0 + i,
            "search_latency_ms": 200.0,
        }

    summary = aggregate_samples(samples, strategy="dense")

    assert summary.median_ms == 1200.0
    assert summary.stage_median_ms["query_embedding_latency_ms"] == 901.0
    assert summary.stage_median_ms["search_latency_ms"] == 200.0


def test_stage_medians_skip_absent_clocks_rather_than_padding_with_zero():
    """A clock missing from some samples must not be diluted by fake zeroes."""
    samples = _samples("bm25", [2.0, 3.0])
    samples[0].stage_latency_ms = {"search_latency_ms": 1.5}
    samples[1].stage_latency_ms = {"search_latency_ms": 2.5}

    summary = aggregate_samples(samples, strategy="bm25")

    # Only the samples that reported the clock are summarised.
    assert summary.stage_median_ms == {"search_latency_ms": 2.0}


def test_artifact_marks_a_partial_sweep_incomplete_and_omits_the_table():
    """A partial sweep must not look freeze-ready.

    With `dense` missing, a table built from the arms that did run would
    normalise against the wrong maximum and produce cost ratios the router
    would act on. The table is omitted rather than filled in.
    """
    summaries = [s for s in _all_strategies() if s.strategy != "dense"]

    artifact = build_artifact(
        summaries,
        proposed=None,
        current={"bm25": 1.71, "dense": 621.35, "hybrid": 721.0, "hybrid_rerank": 4447.68},
        environment={},
        provenance={},
        incomplete_reason="no measurements for ['dense']",
    )

    assert artifact["complete"] is False
    assert artifact["incomplete_reason"] == "no measurements for ['dense']"
    assert artifact["proposed_strategy_cost_ms"] is None
    assert artifact["comparison"] is None
    # The arms that did measure are still reported.
    assert [row["strategy"] for row in artifact["strategies"]] == [
        "bm25", "hybrid", "hybrid_rerank"
    ]


def test_artifact_round_trips_as_json():
    import json

    summaries = _all_strategies()
    artifact = build_artifact(
        summaries,
        proposed=proposed_cost_table(summaries),
        current={"bm25": 1.71, "dense": 621.35, "hybrid": 721.0, "hybrid_rerank": 4447.68},
        environment={},
        provenance={},
    )

    assert json.loads(json.dumps(artifact))["complete"] is True


def test_artifact_records_the_protocol_and_never_freezes_anything():
    summaries = _all_strategies()
    artifact = build_artifact(
        summaries,
        proposed=proposed_cost_table(summaries),
        current={"bm25": 1.71, "dense": 621.35, "hybrid": 721.0, "hybrid_rerank": 4447.68},
        environment={"cpu_count": 12},
        provenance={"git_commit": "deadbeef"},
        warmup=5,
        repetitions=5,
    )

    assert artifact["protocol"]["warmup_queries"] == 5
    assert artifact["protocol"]["repetitions"] == 5
    assert artifact["protocol"]["excluded_from_table"] == ["adaptive"]
    assert "nearest-rank" in artifact["protocol"]["p95_definition"]
    assert artifact["provenance"]["git_commit"] == "deadbeef"
    assert artifact["environment"]["cpu_count"] == 12

    # The proposed table is reported, never written back over the current one.
    assert artifact["proposed_strategy_cost_ms"] != artifact["current_strategy_cost_ms"]
    assert artifact["current_strategy_cost_ms"]["dense"] == 621.35

    # Raw samples travel with the artifact so aggregates can be recomputed.
    assert any(row["raw_ms"] for row in artifact["strategies"])


def test_summarize_all_returns_strategies_in_cost_order():
    summaries = summarize_all(
        [s for name in ("dense", "bm25", "hybrid", "hybrid_rerank")
         for s in _samples(name, [1.0, 2.0])]
    )
    assert [s.strategy for s in summaries] == list(COST_STRATEGIES)