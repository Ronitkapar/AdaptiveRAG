"""Contract tests for Phase 14 cheap-first need prediction.

Covers the frozen oracle rule, the pre-routing signal allowlist (the
structural guarantee that no dense/hybrid/reranker-derived quantity can
enter a rule), threshold grids and rule application, the 1-SE selection
rule, and the deterministic random ablation. No index, provider, or
network; no frozen artifacts are read.
"""

from __future__ import annotations

import pytest

from adaptive_rag.evaluation.cheapfirst import (
    CANDIDATE_SIGNALS,
    PRE_DENSE_SIGNALS,
    apply_rule,
    classify_decision,
    confusion_counts,
    decile_thresholds,
    oracle_label,
    random_escalation,
    roc_auc,
    select_rule,
    signal_value,
    sweep_rule,
)


def _record(signal_val, *, label=True, bm25_q=0.5, dense_q=0.8,
            bm25_lat=2.0, dense_lat=450.0):
    return {
        "signals": {"ddr_bm25": signal_val},
        "oracle_escalate": label,
        "bm25_recall_at_5": bm25_q,
        "dense_recall_at_5": dense_q,
        "bm25_latency_ms": bm25_lat,
        "dense_latency_ms": dense_lat,
    }


# --- frozen oracle ---------------------------------------------------------

@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.5, True), (0.01, True), (0.009, False), (0.0, False), (-0.3, False)],
)
def test_oracle_label_boundaries(delta, expected):
    assert oracle_label(delta) is expected


# --- pre-routing allowlist ---------------------------------------------------

ALLOWLIST_SOURCES = {"signal_table", "bm25_trace", "adaptive_trace"}


def test_candidate_family_is_bounded_and_sourced_pre_dense():
    assert len(CANDIDATE_SIGNALS) == 6
    assert {s["name"] for s in CANDIDATE_SIGNALS} == set(PRE_DENSE_SIGNALS)
    assert {s["direction"] for s in CANDIDATE_SIGNALS} <= {"high", "low"}
    for spec in CANDIDATE_SIGNALS:
        assert spec["source"] in ALLOWLIST_SOURCES


def test_signal_value_serves_only_allowlisted_names():
    signals = {"ddr_bm25": 0.6}
    assert signal_value(signals, "ddr_bm25") == 0.6
    for forbidden in (
        "distinct_doc_ratio.dense", "jaccard@bm25|dense", "dense_top1",
        "hybrid_mrr", "rerank_score", "oracle_escalate", "recall_at_5",
    ):
        with pytest.raises(ValueError):
            signal_value({forbidden: 1.0}, forbidden)


def test_signal_value_missing_and_nonfinite_read_as_missing():
    assert signal_value({}, "bm25_gap") is None
    assert signal_value({"bm25_gap": None}, "bm25_gap") is None
    assert signal_value({"bm25_gap": float("nan")}, "bm25_gap") is None
    assert signal_value({"bm25_gap": float("inf")}, "bm25_gap") is None
    assert signal_value({"bm25_gap": True}, "bm25_gap") is None


def test_missing_signal_maps_to_stop_never_escalate():
    assert apply_rule(None, 0.5, "high") is False
    assert apply_rule(None, 0.5, "low") is False


def test_apply_rule_directions_and_rejects_unknown():
    assert apply_rule(0.8, 0.5, "high") is True
    assert apply_rule(0.2, 0.5, "high") is False
    assert apply_rule(0.2, 0.5, "low") is True
    assert apply_rule(0.8, 0.5, "low") is False
    with pytest.raises(ValueError):
        apply_rule(0.5, 0.5, "sideways")


def test_sweep_rule_rejects_non_pre_dense_signal():
    records = [_record(0.5)]
    with pytest.raises(ValueError):
        sweep_rule(records, signal="dense_top1", thresholds=[0.5],
                   direction="high")


# --- threshold grids ---------------------------------------------------------

def test_decile_thresholds_are_label_free_and_deterministic():
    values = [0.1 * i for i in range(1, 21)]
    first = decile_thresholds(values)
    assert first == decile_thresholds(list(reversed(values)))
    assert len(first) == 9
    assert all(a < b for a, b in zip(first, first[1:]))
    assert first[0] > min(values) and first[-1] < max(values)


def test_decile_thresholds_collapse_ties_and_empty():
    assert decile_thresholds([0.5] * 10) == [0.5]
    assert decile_thresholds([]) == []


# --- decision taxonomy ---------------------------------------------------------

def test_classify_decision_four_way():
    assert classify_decision(True, True) == "true_escalation"
    assert classify_decision(False, True) == "false_escalation"
    assert classify_decision(True, False) == "missed_escalation"
    assert classify_decision(False, False) == "correct_stop"


def test_confusion_counts_and_length_guard():
    decisions = [True, True, False, False]
    labels = [True, False, True, False]
    assert confusion_counts(decisions, labels) == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 1,
    }
    with pytest.raises(ValueError):
        confusion_counts([True], [True, False])


# --- latency model -----------------------------------------------------------

def test_sweep_latency_charges_dense_only_when_escalated():
    records = [_record(0.9, bm25_lat=2.0, dense_lat=450.0),
               _record(0.1, bm25_lat=3.0, dense_lat=500.0)]
    (row,) = sweep_rule(records, signal="ddr_bm25", thresholds=[0.5],
                        direction="high")
    assert row["n_escalated"] == 1
    assert row["escalation_rate"] == pytest.approx(0.5)
    # First query escalated (2+450), second stopped at BM25 (3.0).
    assert row["mean_latency_ms"] == pytest.approx((452.0 + 3.0) / 2.0)
    assert row["mean_quality"] == pytest.approx((0.8 + 0.5) / 2.0)


# --- selection ---------------------------------------------------------------

def _sweep_row(threshold, rate, mean, se=0.01):
    return {"signal": "ddr_bm25", "threshold": threshold, "direction": "high",
            "escalation_rate": rate, "mean_quality": mean,
            "quality_stderror": se}


def test_select_rule_cheapest_within_one_se():
    rows = [_sweep_row(0.2, 0.80, 0.90), _sweep_row(0.6, 0.30, 0.895),
            _sweep_row(0.8, 0.10, 0.85)]
    chosen = select_rule(rows)
    # 0.895 is within 1 SE (0.01) of the 0.90 max and escalates least there.
    assert chosen["threshold"] == 0.6
    assert chosen["degenerate"] is False


def test_select_rule_degenerate_when_nothing_eligible():
    rows = [_sweep_row(0.2, 1.0, 0.90), _sweep_row(1.0, 0.0, 0.80)]
    chosen = select_rule(rows)
    assert chosen["degenerate"] is True
    assert chosen["threshold"] == 0.2


# --- ranking metric (descriptive-only helper) -----------------------------------

def test_roc_auc_perfect_inverse_and_single_class():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert roc_auc([0.5, 0.6], [True, True]) is None
    assert roc_auc([0.5, 0.6], [False, False]) is None


# --- random ablation -----------------------------------------------------------

def test_random_escalation_fixed_counts_and_deterministic():
    records = [_record(0.5, bm25_q=0.2, dense_q=0.9) for _ in range(20)]
    first = random_escalation(records, rate=0.25, n_draws=50, seed=20250101)
    second = random_escalation(records, rate=0.25, n_draws=50, seed=20250101)
    assert first["k_escalated_per_draw"] == 5
    assert first["mean_quality_values"] == second["mean_quality_values"]
    # 5 of 20 escalated per draw: mean = (5*0.9 + 15*0.2)/20 every draw.
    assert first["mean_quality"] == pytest.approx(0.375)
    # Seed sensitivity needs heterogeneous records: identical rows make
    # every subset equal, so vary the dense outcomes here.
    varied = [_record(0.5, bm25_q=0.2, dense_q=0.05 * i) for i in range(20)]
    seeded_a = random_escalation(varied, rate=0.25, n_draws=50, seed=20250101)
    seeded_b = random_escalation(varied, rate=0.25, n_draws=50, seed=7)
    assert seeded_a["mean_quality_values"] != seeded_b["mean_quality_values"]
