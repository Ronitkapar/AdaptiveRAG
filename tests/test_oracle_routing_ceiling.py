"""
tests.test_oracle_routing_ceiling
----------------------------------
The offline routing-ceiling driver (`scripts/oracle_routing_ceiling.py`).

Three things are pinned here, each for a reason.

*The query-set assertion.* The Phase 8 contamination incident was a query-set
mismatch: an arm that silently lost embedding calls reported `n=59` metrics as
though they were `n=107`, and the resulting number inverted the reranker's
headline result. An oracle joins per query, so a ragged join would compare a
different question set per strategy and no aggregate of it would look wrong.

*The oracle-dominates assertion.* A per-query argmax cannot underperform the
best constant choice. That is a checkable invariant, and it is the cheapest
possible detector of a broken join, so it is asserted rather than assumed.

*The ceiling's fragility.* `recall_at_5` is near-binary on this benchmark, so
the oracle's mean advantage is carried by a handful of queries and clears or
misses a pre-registered threshold depending on those queries alone. The
concentration reporting is what stops that number being read at face value.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oracle_routing_ceiling", REPO_ROOT / "scripts" / "oracle_routing_ceiling.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle_routing_ceiling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle_routing_ceiling)

build_panel = oracle_routing_ceiling.build_panel
panel_latencies = oracle_routing_ceiling.panel_latencies
attach_latency = oracle_routing_ceiling.attach_latency
oracle_pick = oracle_routing_ceiling.oracle_pick
frontier_pick = oracle_routing_ceiling.frontier_pick
summarise = oracle_routing_ceiling.summarise
fixed_summaries = oracle_routing_ceiling.fixed_summaries
best_fixed = oracle_routing_ceiling.best_fixed
evaluate_gate = oracle_routing_ceiling.evaluate_gate
needed_split = oracle_routing_ceiling.needed_split
permutation_p_value = oracle_routing_ceiling.permutation_p_value
contingency_permutation_p = oracle_routing_ceiling.contingency_permutation_p
oracle_gain_concentration = oracle_routing_ceiling.oracle_gain_concentration
read_adaptive_signals = oracle_routing_ceiling.read_adaptive_signals
cross_tabs = oracle_routing_ceiling.cross_tabs
QuerySetMismatch = oracle_routing_ceiling.QuerySetMismatch

SYSTEMS = ("bm25", "dense", "hybrid")


def _row(query_id: str, system: str, recall: float, latency: float) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "system": system,
        "recall_at_5": recall,
        "mrr": recall,
        "hit_at_5": recall,
        "total_latency_ms": latency,
    }


def _arm(n: int = 3, **by_system: dict[str, tuple[float, float]]) -> dict[str, list[dict[str, Any]]]:
    """Build `n` queries per system; `by_system` maps system -> (recall, latency)."""
    return {
        system: [_row(f"q{i:02d}", system, *values) for i in range(n)]
        for system, values in by_system.items()
    }


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


def test_panel_joins_arms_on_query_id():
    rows = _arm(
        bm25=(0.5, 2.0),
        dense=(1.0, 100.0),
        hybrid=(0.5, 120.0),
    )
    panel = build_panel(rows, list(SYSTEMS))
    assert sorted(panel) == ["q00", "q01", "q02"]
    assert panel["q00"]["dense"]["recall_at_5"] == 1.0


def test_panel_rejects_a_short_arm_rather_than_joining_it():
    """The contamination failure mode: one arm short, everything else fine."""
    rows = _arm(bm25=(0.5, 2.0), dense=(1.0, 100.0), hybrid=(0.5, 120.0))
    rows["dense"] = rows["dense"][:1]
    with pytest.raises(QuerySetMismatch) as excinfo:
        build_panel(rows, list(SYSTEMS))
    # The message must name the offending arm, or a reader cannot act on it.
    assert "dense" in str(excinfo.value)


def test_panel_rejects_duplicate_query_ids():
    rows = _arm(bm25=(0.5, 2.0), dense=(1.0, 100.0))
    rows["bm25"].append(_row("q00", "bm25", 0.9, 2.0))
    with pytest.raises(QuerySetMismatch):
        build_panel(rows, ["bm25", "dense"])


def test_panel_latencies_rejects_a_null_rather_than_skipping_the_query():
    rows = _arm(bm25=(0.5, 2.0), dense=(1.0, 100.0))
    rows["dense"][0]["total_latency_ms"] = None
    with pytest.raises(QuerySetMismatch) as excinfo:
        panel_latencies(rows, ["bm25", "dense"])
    assert "total_latency_ms" in str(excinfo.value)


def test_attach_latency_folds_latency_into_every_system():
    rows = _arm(bm25=(0.5, 2.0), dense=(1.0, 100.0))
    panel = build_panel(rows, ["bm25", "dense"])
    latencies = panel_latencies(rows, ["bm25", "dense"])
    merged = attach_latency(panel, latencies)
    assert merged["q00"]["dense"]["total_latency_ms"] == 100.0
    # The quality panel itself stays latency-free.
    assert "total_latency_ms" not in panel["q00"]["dense"]


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------


def test_oracle_picks_the_best_recall_per_query():
    rows = _arm(
        bm25=(0.0, 2.0),
        dense=(1.0, 100.0),
        hybrid=(0.5, 120.0),
    )
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    assert set(oracle_pick(per_query, list(SYSTEMS)).values()) == {"dense"}


def test_oracle_breaks_recall_ties_on_latency_not_on_name():
    """A tie must resolve to the cheap arm, or the oracle inherits a bias."""
    rows = _arm(bm25=(1.0, 2.0), dense=(1.0, 100.0), hybrid=(1.0, 120.0))
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    assert set(oracle_pick(per_query, list(SYSTEMS)).values()) == {"bm25"}


def test_frontier_buys_the_cheapest_sufficient_strategy():
    rows = _arm(bm25=(1.0, 2.0), dense=(1.0, 100.0), hybrid=(1.0, 120.0))
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    # bm25 is within epsilon of the best, so the frontier should not pay for dense.
    assert set(frontier_pick(per_query, list(SYSTEMS), 0.01).values()) == {"bm25"}


def test_frontier_pays_up_when_the_cheap_arm_is_outside_epsilon():
    rows = _arm(bm25=(0.0, 2.0), dense=(1.0, 100.0), hybrid=(0.5, 120.0))
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    assert set(frontier_pick(per_query, list(SYSTEMS), 0.01).values()) == {"dense"}


def test_frontier_never_beats_the_oracle_on_recall():
    """A frontier that raised recall above the oracle would mean a broken join."""
    rows = _arm(
        bm25=(0.0, 2.0),
        dense=(1.0, 100.0),
        hybrid=(0.5, 120.0),
    )
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    latencies = panel_latencies(rows, list(SYSTEMS))
    oracle = summarise(oracle_pick(per_query, list(SYSTEMS)), per_query, latencies)
    for epsilon in (0.0, 0.005, 0.01, 0.5):
        frontier = summarise(
            frontier_pick(per_query, list(SYSTEMS), epsilon), per_query, latencies
        )
        assert frontier["mean_recall_at_5"] <= oracle["mean_recall_at_5"] + 1e-9


def test_frontier_at_zero_epsilon_equals_the_oracle():
    rows = _arm(bm25=(0.0, 2.0), dense=(1.0, 100.0), hybrid=(0.5, 120.0))
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    assert frontier_pick(per_query, list(SYSTEMS), 0.0) == oracle_pick(
        per_query, list(SYSTEMS)
    )


def test_frontier_rejects_a_negative_epsilon():
    with pytest.raises(ValueError):
        frontier_pick({}, list(SYSTEMS), -0.01)


# ---------------------------------------------------------------------------
# Aggregation, and the oracle-dominates invariant
# ---------------------------------------------------------------------------


def test_oracle_dominates_the_best_fixed_strategy_on_recall():
    """The invariant that makes a broken join detectable rather than plausible."""
    rows = _arm(
        bm25=(0.0, 2.0),
        dense=(1.0, 100.0),
        hybrid=(0.5, 120.0),
    )
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    latencies = panel_latencies(rows, list(SYSTEMS))
    fixed = fixed_summaries(per_query, latencies, list(SYSTEMS))
    oracle = summarise(oracle_pick(per_query, list(SYSTEMS)), per_query, latencies)
    assert oracle["mean_recall_at_5"] >= max(
        block["mean_recall_at_5"] for block in fixed.values()
    ) - 1e-9


def test_best_fixed_selects_the_highest_mean_recall():
    rows = _arm(
        bm25=(0.0, 2.0),
        dense=(1.0, 100.0),
        hybrid=(0.5, 120.0),
    )
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    fixed = fixed_summaries(per_query, latencies=panel_latencies(rows, list(SYSTEMS)), systems=list(SYSTEMS))
    assert best_fixed(fixed) == "dense"


def test_summarise_counts_the_strategies_it_actually_chose():
    rows = _arm(bm25=(1.0, 2.0), dense=(1.0, 100.0), hybrid=(1.0, 120.0))
    per_query = attach_latency(
        build_panel(rows, list(SYSTEMS)), panel_latencies(rows, list(SYSTEMS))
    )
    summary = summarise(
        oracle_pick(per_query, list(SYSTEMS)),
        per_query,
        panel_latencies(rows, list(SYSTEMS)),
    )
    assert summary["strategy_counts"] == {"bm25": 3}
    assert summary["mean_total_latency_ms"] == 2.0


# ---------------------------------------------------------------------------
# Detectability
# ---------------------------------------------------------------------------


def test_needed_split_marks_a_query_needed_only_when_nothing_else_is_close():
    per_query = {
        "q_needs": {s: {"recall_at_5": v} for s, v in
                    zip(SYSTEMS, (0.0, 1.0, 0.5))},
        "q_suffices": {s: {"recall_at_5": v} for s, v in
                       zip(SYSTEMS, (1.0, 1.0, 0.5))},
    }
    labels = needed_split(per_query, list(SYSTEMS), "dense", 0.01)
    assert labels["q_needs"] == "needed"
    assert labels["q_suffices"] == "sufficient"


def test_needed_split_treats_a_tie_as_sufficient():
    """A tie means there is a choice to be made, and paying for dense is waste."""
    per_query = {
        "q": {s: {"recall_at_5": 1.0} for s in SYSTEMS},
    }
    assert needed_split(per_query, list(SYSTEMS), "dense", 0.01)["q"] == "sufficient"


def test_permutation_test_finds_a_real_separation():
    low = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    high = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]
    test = permutation_p_value(low, high, n_resamples=2000)
    assert test["significant"]
    assert test["observed_difference"] > 0.6


def test_permutation_test_finds_nothing_between_identical_groups():
    values = [0.5, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50]
    test = permutation_p_value(values, list(values), n_resamples=2000)
    assert not test["significant"]


def test_permutation_test_is_deterministic_under_a_fixed_seed():
    low = [0.1, 0.2, 0.15, 0.18]
    high = [0.7, 0.8, 0.75, 0.72]
    first = permutation_p_value(low, high, n_resamples=500, seed=7)
    second = permutation_p_value(low, high, n_resamples=500, seed=7)
    assert first == second


def test_permutation_test_reports_an_empty_group_rather_than_a_fake_p():
    test = permutation_p_value([], [0.5, 0.6], n_resamples=100)
    assert test["p_value"] is None
    assert not test["significant"]


def test_contingency_test_detects_a_planted_association():
    table = {"needed": {"passed": 20, "failed": 0}, "sufficient": {"passed": 5, "failed": 75}}
    test = contingency_permutation_p(table, n_resamples=2000)
    assert test["significant"]


def test_contingency_test_finds_nothing_in_a_flat_table():
    table = {"needed": {"passed": 10, "failed": 10}, "sufficient": {"passed": 10, "failed": 10}}
    test = contingency_permutation_p(table, n_resamples=2000)
    assert not test["significant"]


def test_contingency_test_rejects_a_single_group_table():
    test = contingency_permutation_p({"only": {"a": 3}}, n_resamples=100)
    assert test["p_value"] is None
    assert not test["significant"]


# ---------------------------------------------------------------------------
# Gain concentration -- the ceiling's fragility, made explicit
# ---------------------------------------------------------------------------


def test_gain_concentration_counts_the_queries_carrying_the_oracle():
    """Contributors are the queries the *fixed* strategy loses, not the ones it wins."""
    per_query = {
        # bm25 beats dense outright: dense loses 1.0 here.
        "q_bm25_wins": {s: {"recall_at_5": v} for s, v in zip(SYSTEMS, (1.0, 0.0, 0.0))},
        # bm25 ties dense: dense loses nothing measurable, so no gain is booked.
        "q_tie": {s: {"recall_at_5": v} for s, v in zip(SYSTEMS, (1.0, 1.0, 0.0))},
        # hybrid beats dense by 0.5.
        "q_hybrid_wins": {s: {"recall_at_5": v} for s, v in zip(SYSTEMS, (0.0, 0.5, 1.0))},
        # dense wins: nothing to book.
        "q_dense_wins": {s: {"recall_at_5": v} for s, v in zip(SYSTEMS, (0.0, 1.0, 0.0))},
    }
    report = oracle_gain_concentration(per_query, list(SYSTEMS), "dense", 0.5)
    assert report["n_queries_contributing_gain"] == 2
    assert report["total_gain_mass"] == pytest.approx(1.5)
    assert report["top_contributors"][0]["query_id"] == "q_bm25_wins"
    # q_tie is not a contributor at all: the oracle gains nothing by switching.
    assert "q_tie" not in [c["query_id"] for c in report["top_contributors"]]


def test_gain_concentration_reports_zero_when_the_fixed_strategy_is_the_oracle():
    per_query = {
        f"q{i}": {s: {"recall_at_5": 1.0} for s in SYSTEMS} for i in range(4)
    }
    report = oracle_gain_concentration(per_query, list(SYSTEMS), "dense", 1.0)
    assert report["n_queries_contributing_gain"] == 0
    assert report["total_gain_mass"] == 0.0
    assert report["mean_gain_per_contributing_query"] == 0.0


# ---------------------------------------------------------------------------
# The pre-registered gate
# ---------------------------------------------------------------------------


def test_gate_reports_headroom_on_a_large_quality_delta():
    gate = evaluate_gate(oracle_delta=0.05, frontier_latency_delta_ms=-10.0, detectable=True)
    assert gate["verdict"] == "routing_has_headroom"
    assert gate["quality_headroom"]


def test_gate_reports_exhaustion_when_neither_ceiling_is_cleared():
    gate = evaluate_gate(oracle_delta=0.005, frontier_latency_delta_ms=-20.0, detectable=False)
    assert gate["verdict"] == "routing_exhausted_on_this_benchmark"
    assert gate["exhausted"]


def test_gate_refuses_latency_headroom_without_detectability():
    """A large saving no query-time signal can see is not a bankable prize."""
    gate = evaluate_gate(oracle_delta=0.005, frontier_latency_delta_ms=-500.0, detectable=False)
    assert not gate["latency_headroom"]
    assert gate["verdict"] == "inconclusive"


def test_gate_reports_the_middle_band_as_inconclusive():
    """The gap between the two thresholds must not be rounded to a verdict."""
    gate = evaluate_gate(oracle_delta=0.015, frontier_latency_delta_ms=-70.0, detectable=True)
    assert gate["verdict"] == "inconclusive"
    assert not gate["quality_headroom"]
    assert not gate["exhausted"]


def test_gate_is_strict_at_the_quality_threshold():
    """Exactly 0.02 does not clear a '> 0.02' gate."""
    gate = evaluate_gate(oracle_delta=0.02, frontier_latency_delta_ms=-10.0, detectable=True)
    assert not gate["quality_headroom"]


# ---------------------------------------------------------------------------
# Reading the router's own signals
# ---------------------------------------------------------------------------


def test_read_adaptive_signals_uses_routing_sufficiency_signals(tmp_path):
    """The plan's `routing.metadata.signals` path does not exist in the traces."""
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "example_id": "q01",
                "category": "factual",
                "routing": {
                    "initial_strategy": "hybrid",
                    "features": {"question_type": "what"},
                    "sufficiency": {
                        "score": 0.83,
                        "coverage": 0.86,
                        "signals": [
                            {"name": "result_count", "passed": True},
                            {"name": "top1_coverage", "passed": False},
                        ],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    signals = read_adaptive_signals(traces)
    assert signals["q01"]["sufficiency_score"] == 0.83
    assert signals["q01"]["passed_signals"] == ["result_count"]
    assert signals["q01"]["question_type"] == "what"


def test_read_adaptive_signals_errors_on_a_missing_trace_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_adaptive_signals(tmp_path / "absent.jsonl")


def test_cross_tabs_count_strategies_by_category():
    picks = {"q1": "bm25", "q2": "dense"}
    signals = {"q1": {"category": "factual"}, "q2": {"category": "factual"}}
    assert cross_tabs(picks, signals, "category") == {
        "factual": {"bm25": 1, "dense": 1}
    }


# ---------------------------------------------------------------------------
# End to end over a synthetic suite
# ---------------------------------------------------------------------------


def _write_suite(root: Path, *, per_query: dict[str, dict[str, float]]) -> Path:
    """Materialise a two-arm suite whose per-query recalls are `per_query`.

    The number of queries is the length of `per_query`. One query gives bm25 a
    hard miss and dense a hit, so best-fixed is dense and the oracle has
    something to win; the rest tie, so the frontier should route them to bm25.
    """
    # `adaptive` is given the hybrid-like recall (0.5) so it cannot tie `dense`
    # for best-fixed; the fixture is about the ceiling, not about reproducing
    # Phase 7's adaptive-equals-hybrid result.
    for system, recall_key in (("bm25", "bm25"), ("dense", "dense"), ("adaptive", "hybrid")):
        run_dir = root / f"p8__E1_baseline_comparison__{system}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rows, traces = [], []
        for query_id in sorted(per_query):
            recall = per_query[query_id].get(recall_key, 0.5)
            latency = 2.0 if system == "bm25" else 100.0
            rows.append(
                {
                    "query_id": query_id,
                    "system": system,
                    "recall_at_5": recall,
                    "mrr": recall,
                    "hit_at_5": recall,
                    "total_latency_ms": latency,
                }
            )
            traces.append(
                {
                    "example_id": query_id,
                    "category": "factual",
                    "status": "ok",
                    "routing": {
                        "initial_strategy": "hybrid",
                        "features": {"question_type": "what"},
                        "sufficiency": {
                            "score": 0.8,
                            "coverage": 0.8,
                            "signals": [{"name": "result_count", "passed": True}],
                        },
                    },
                }
            )
        (run_dir / "rows.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        (run_dir / "traces.jsonl").write_text(
            "\n".join(json.dumps(t) for t in traces) + "\n", encoding="utf-8"
        )
    return root


def test_run_arm_end_to_end(tmp_path):
    # q00: only dense hits -> the oracle books a gain over dense here.
    # q01: only bm25 hits -> the oracle books a second gain.
    # q02-q04: all tie -> the frontier routes them to cheap bm25.
    # dense wins 4/5, so dense is best-fixed and the oracle's advantage is
    # exactly the one query dense alone gets right.
    per_query = {
        "q00": {"bm25": 0.0, "dense": 1.0},
        "q01": {"bm25": 1.0, "dense": 0.0},
        "q02": {"bm25": 1.0, "dense": 1.0},
        "q03": {"bm25": 1.0, "dense": 1.0},
        "q04": {"bm25": 1.0, "dense": 1.0},
    }
    # dense 4/5 = 0.8, bm25 4/5 = 0.8 -> tie. Break it toward dense by having
    # dense also win q04 outright, so dense is unambiguously best-fixed.
    per_query["q04"] = {"bm25": 0.0, "dense": 1.0}
    root = _write_suite(tmp_path / "suite", per_query=per_query)
    arm = oracle_routing_ceiling.run_arm(root, ["bm25", "dense", "adaptive"])

    assert arm["n_queries"] == 5
    assert arm["best_fixed_strategy"] == "dense"
    # The oracle wins recall@5 by one query's worth and buys latency everywhere else.
    assert arm["oracle"]["mean_recall_at_5"] > arm["fixed"]["dense"]["mean_recall_at_5"]
    assert arm["oracle"]["mean_total_latency_ms"] < arm["fixed"]["dense"]["mean_total_latency_ms"]
    assert arm["gate"]["verdict"] in {
        "routing_has_headroom",
        "routing_exhausted_on_this_benchmark",
        "inconclusive",
    }
    # The whole gain rides on one query, and the artifact must say so.
    assert arm["oracle_gain_concentration"]["n_queries_contributing_gain"] == 1
    assert arm["oracle_gain_concentration"]["share_of_queries_contributing"] == 0.2


def test_run_arm_rejects_a_ragged_suite(tmp_path):
    """One short arm must abort the whole run, not shrink the panel silently."""
    root = _write_suite(
        tmp_path / "suite",
        per_query={f"q{i:02d}": {"bm25": 1.0, "dense": 1.0} for i in range(3)},
    )
    dense_rows = root / "p8__E1_baseline_comparison__dense" / "rows.jsonl"
    dense_rows.write_text(dense_rows.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(QuerySetMismatch):
        oracle_routing_ceiling.run_arm(root, ["bm25", "dense", "adaptive"])
