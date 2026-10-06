"""Phase 10 escalation oracle -- unit tests over synthetic records.

Every function under test is pure: no index, no provider, no network, no
artifact on disk. The tests pin the frozen research contract from
`docs/phases/phase-10.md` §4/§8 (oracle rule, threshold families, selection
rule, random-ablation protocol) so a later edit cannot silently redefine what
"worth escalating" means.
"""

from __future__ import annotations

import pytest

from adaptive_rag.evaluation.escalation import (
    DIRECTIONS,
    EPSILON,
    RANDOM_ABLATION_DRAWS,
    RANDOM_ABLATION_SEED,
    THRESHOLDS,
    apply_threshold,
    average_precision,
    classify_decision,
    confusion_counts,
    incremental_latency_ms,
    oracle_label,
    random_escalation,
    recompute_ddr,
    roc_auc,
    select_threshold,
    sweep_thresholds,
)


def _record(
    query_id="q",
    ddr=0.6,
    oracle=True,
    dense_r5=0.0,
    hybrid_r5=1.0,
    dense_mrr=0.0,
    hybrid_mrr=1.0,
    dense_lat=600.0,
    incr_lat=30.0,
):
    return {
        "query_id": query_id,
        "ddr_dense": ddr,
        "oracle_escalate": oracle,
        "dense_recall_at_5": dense_r5,
        "hybrid_recall_at_5": hybrid_r5,
        "dense_mrr": dense_mrr,
        "hybrid_mrr": hybrid_mrr,
        "dense_latency_ms": dense_lat,
        "incremental_latency_ms": incr_lat,
    }


# --- frozen contract surface ---------------------------------------------


def test_threshold_value_set_is_the_full_signal_range():
    assert THRESHOLDS == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert DIRECTIONS == ("high", "low")
    assert EPSILON == 0.01


# --- signal recomputation -------------------------------------------------


def test_recompute_ddr_dedupes_before_dividing():
    ids = ["d1", "d1", "d1", "d1", "d1", "d2"]
    assert recompute_ddr(ids) == pytest.approx(0.2)


def test_recompute_ddr_all_distinct():
    assert recompute_ddr(["a", "b", "c", "d", "e"]) == pytest.approx(1.0)


def test_recompute_ddr_empty_is_one():
    assert recompute_ddr([]) == pytest.approx(1.0)


# --- oracle rule ----------------------------------------------------------


def test_oracle_label_strict_improvement():
    assert oracle_label(0.5) is True
    assert oracle_label(EPSILON) is True


def test_oracle_label_tie_and_harm_are_no():
    assert oracle_label(0.0) is False
    assert oracle_label(0.009) is False
    assert oracle_label(-0.5) is False


# --- latency model --------------------------------------------------------


def test_incremental_latency_sums_stages():
    value, method = incremental_latency_ms(
        bm25_latency_ms=3.0,
        fusion_latency_ms=0.5,
        search_latency_ms=25.0,
        fallback_latency_ms=600.0,
    )
    assert value == pytest.approx(28.5)
    assert method == "stages"


def test_incremental_latency_falls_back_when_a_stage_is_missing():
    value, method = incremental_latency_ms(
        bm25_latency_ms=3.0,
        fusion_latency_ms=None,
        search_latency_ms=25.0,
        fallback_latency_ms=600.0,
    )
    assert value == pytest.approx(600.0)
    assert method == "full_fallback"


def test_incremental_latency_raises_when_nothing_is_usable():
    with pytest.raises(ValueError):
        incremental_latency_ms(
            bm25_latency_ms=None,
            fusion_latency_ms=None,
            search_latency_ms=None,
            fallback_latency_ms=None,
        )


# --- threshold policy -----------------------------------------------------


def test_apply_threshold_both_directions_and_boundaries():
    assert apply_threshold(0.6, 0.6, "high") is True
    assert apply_threshold(0.4, 0.6, "high") is False
    assert apply_threshold(0.6, 0.6, "low") is True
    assert apply_threshold(0.8, 0.6, "low") is False
    with pytest.raises(ValueError):
        apply_threshold(0.6, 0.6, "sideways")


def test_confusion_counts_taxonomy():
    counts = confusion_counts(
        [True, True, False, False], [True, False, True, False]
    )
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_confusion_counts_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        confusion_counts([True], [True, False])


def test_classify_decision_covers_all_four_cells():
    assert classify_decision(True, True) == "true_escalation"
    assert classify_decision(False, True) == "false_escalation"
    assert classify_decision(True, False) == "missed_escalation"
    assert classify_decision(False, False) == "correct_stop"


# --- sweep ----------------------------------------------------------------


def _sweep_records():
    # ddr:      0.2    0.4    0.8    1.0
    # oracle:   NO     YES    YES    NO
    # At t=0.4 (high): escalate 0.4,0.8,1.0 -> tp=2, fp=1, fn=0, tn=1.
    return [
        _record("a", ddr=0.2, oracle=False, dense_r5=1.0, hybrid_r5=0.0),
        _record("b", ddr=0.4, oracle=True, dense_r5=0.0, hybrid_r5=1.0),
        _record("c", ddr=0.8, oracle=True, dense_r5=0.0, hybrid_r5=1.0),
        _record("d", ddr=1.0, oracle=False, dense_r5=1.0, hybrid_r5=0.5),
    ]


def test_sweep_threshold_reports_rate_quality_and_confusion():
    (row,) = [
        r
        for r in sweep_thresholds(_sweep_records(), thresholds=[0.4])
        if r["threshold"] == 0.4
    ]
    assert row["n"] == 4
    assert row["n_escalated"] == 3
    assert row["escalation_rate"] == pytest.approx(0.75)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 1, 0, 1)
    assert row["oracle_agreement"] == pytest.approx(0.75)
    # adaptive quality: escalated take hybrid (1,1,0.5), stopped takes dense (1)
    assert row["mean_quality"] == pytest.approx((1.0 + 1.0 + 0.5 + 1.0) / 4)
    # adaptive latency: 600 + 30 on each escalated query
    assert row["mean_latency_ms"] == pytest.approx((630 * 3 + 600) / 4)


def test_sweep_always_escalate_at_bottom_threshold():
    (row,) = sweep_thresholds(_sweep_records(), thresholds=[0.2])
    assert row["escalation_rate"] == pytest.approx(1.0)


def test_sweep_supports_the_secondary_metric():
    (row,) = sweep_thresholds(
        _sweep_records(), thresholds=[0.4], quality="mrr"
    )
    assert row["quality_metric"] == "mrr"
    # MRR defaults are dense 0.0 / hybrid 1.0, so escalating b, c, d gives 0.75.
    assert row["mean_quality"] == pytest.approx(0.75)


# --- selection ------------------------------------------------------------


def test_select_threshold_prefers_cheapest_within_one_se():
    sweep = [
        {"threshold": 0.4, "direction": "high", "mean_quality": 0.80,
         "quality_stderror": 0.05, "escalation_rate": 0.60},
        {"threshold": 0.6, "direction": "high", "mean_quality": 0.78,
         "quality_stderror": 0.05, "escalation_rate": 0.30},
        {"threshold": 0.8, "direction": "high", "mean_quality": 0.60,
         "quality_stderror": 0.05, "escalation_rate": 0.10},
    ]
    pick = select_threshold(sweep)
    assert pick["degenerate"] is False
    assert pick["threshold"] == 0.6  # within 1 SE of 0.80, cheaper than 0.4


def test_select_threshold_excludes_degenerate_rates():
    sweep = [
        {"threshold": 0.2, "direction": "high", "mean_quality": 0.99,
         "quality_stderror": 0.01, "escalation_rate": 1.0},
        {"threshold": 0.6, "direction": "high", "mean_quality": 0.70,
         "quality_stderror": 0.05, "escalation_rate": 0.30},
    ]
    pick = select_threshold(sweep)
    assert pick["threshold"] == 0.6


def test_select_threshold_reports_degeneracy_instead_of_resweeping():
    sweep = [
        {"threshold": 0.2, "direction": "high", "mean_quality": 0.99,
         "quality_stderror": 0.01, "escalation_rate": 1.0},
        {"threshold": 1.0, "direction": "high", "mean_quality": 0.50,
         "quality_stderror": 0.05, "escalation_rate": 0.02},
    ]
    pick = select_threshold(sweep)
    assert pick["degenerate"] is True
    assert pick["threshold"] == 0.2


# --- ranking metrics ------------------------------------------------------


def test_roc_auc_perfect_worst_and_chance():
    assert roc_auc([0.1, 0.9], [False, True]) == pytest.approx(1.0)
    assert roc_auc([0.9, 0.1], [False, True]) == pytest.approx(0.0)
    assert roc_auc([0.5, 0.5], [False, True]) == pytest.approx(0.5)


def test_roc_auc_none_without_both_classes():
    assert roc_auc([0.1, 0.9], [True, True]) is None
    assert roc_auc([0.1, 0.9], [False, False]) is None


def test_average_precision_perfect_and_none():
    assert average_precision([0.1, 0.9], [False, True]) == pytest.approx(1.0)
    assert average_precision([0.1], [False]) is None


# --- random ablation ------------------------------------------------------


def test_random_escalation_is_deterministic_and_sized():
    records = _sweep_records()
    first = random_escalation(records, rate=0.5, n_draws=50, seed=7)
    second = random_escalation(records, rate=0.5, n_draws=50, seed=7)
    assert first["mean_quality_values"] == second["mean_quality_values"]
    assert first["k_escalated_per_draw"] == 2
    assert first["n_draws"] == 50


def test_random_escalation_rate_extremes_recover_fixed_arms():
    records = _sweep_records()
    never = random_escalation(records, rate=0.0, n_draws=5, seed=1)
    always = random_escalation(records, rate=1.0, n_draws=5, seed=1)
    dense_mean = sum(r["dense_recall_at_5"] for r in records) / len(records)
    hybrid_mean = sum(r["hybrid_recall_at_5"] for r in records) / len(records)
    assert never["mean_quality"] == pytest.approx(dense_mean)
    assert always["mean_quality"] == pytest.approx(hybrid_mean)


def test_random_ablation_protocol_constants():
    assert RANDOM_ABLATION_DRAWS == 1000
    assert RANDOM_ABLATION_SEED == 20250101
