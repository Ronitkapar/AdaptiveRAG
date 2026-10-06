"""Chart-spec builder tests (pure data, no Streamlit runtime)."""

from frontend.ui import benchmark, charts


def test_quality_bars_cover_five_systems():
    snapshot = benchmark.load_headline_snapshot()
    specs = charts.e1_quality_bars(snapshot["phase7_e1"])
    assert len(specs) == 5
    for spec in specs:
        assert spec["Recall@5"] > 0
        assert spec["MRR"] > 0


def test_frontier_has_latency_spread():
    snapshot = benchmark.load_headline_snapshot()
    specs = charts.quality_vs_latency(snapshot["phase7_e1"])
    latencies = [spec["median_latency_ms"] for spec in specs]
    # BM25 (~2 ms) to the reranked arm (~3.5 s): three orders of magnitude.
    assert max(latencies) / min(latencies) > 100


def test_chart_rows_accept_legacy_strategy_name():
    rows = [
        {
            "strategy": "bm25",
            "recall_at_5": 0.7,
            "mrr": 0.6,
            "hit_at_5": 0.8,
            "median_latency_ms": 2,
        },
        {
            "system": "dense",
            "recall_at_5": 0.9,
            "mrr": 0.8,
            "hit_at_5": 0.95,
            "median_latency_ms": 500,
        },
    ]
    specs = charts.e1_quality_bars({"rows": rows})
    assert [spec["system"] for spec in specs] == ["bm25", "dense"]
    assert specs[0]["Recall@5"] == 0.7


def test_quality_bars_accept_normalized_working_tree_rows():
    table, _source = benchmark.load_e1_table()
    assert table is not None
    # This is the exact call chain that raised KeyError: 'recall_at_5'.
    specs = charts.e1_quality_bars({"rows": table["rows"]})
    assert len(specs) == 5
    frontier = charts.quality_vs_latency({"rows": table["rows"]})
    assert len(frontier) == 5


def test_before_after_deltas_match_snapshot():
    snapshot = benchmark.load_headline_snapshot()
    specs = charts.phase8_before_after(snapshot["phase8_e1"])
    rerank = next(spec for spec in specs if spec["arm"] == "hybrid_rerank")
    # The corpus fix did not rescue the reranker (-0.0016 recall@5).
    assert rerank["delta"] == round(0.7056 - 0.7072, 4)


def test_oracle_frame_contains_oracle_and_dense():
    snapshot = benchmark.load_headline_snapshot()
    frame = charts.oracle_comparison(snapshot["oracle_ceiling"])
    policies = {row["policy"] for row in frame}
    assert "ORACLE (per-query)" in policies
    assert "dense (best fixed)" in policies
