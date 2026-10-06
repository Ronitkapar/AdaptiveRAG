"""Benchmark snapshot and artifact loader tests."""

from frontend.ui import benchmark


def test_headline_snapshot_has_frozen_tables():
    snapshot = benchmark.load_headline_snapshot()
    e1 = snapshot["phase7_e1"]
    assert len(e1["rows"]) == 5
    systems = {row["system"] for row in e1["rows"]}
    assert systems == {"bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"}
    dense = next(row for row in e1["rows"] if row["system"] == "dense")
    assert dense["recall_at_5"] == 0.9283
    assert dense["median_latency_ms"] == 472.84
    # Every frozen table names its source artifact.
    assert snapshot["phase7_e1"].get("source")
    assert snapshot["phase8_e1"].get("source")
    assert snapshot["oracle_ceiling"].get("source")


def test_phase_timeline_has_fifteen_phases():
    phases = benchmark.load_phase_timeline()
    assert len(phases) == 15
    assert [phase["n"] for phase in phases] == list(range(1, 16))
    verdicts = {phase["n"]: phase["verdict"] for phase in phases}
    assert verdicts[10] == "FAILURE"
    assert verdicts[12] == "STOP"
    assert verdicts[15] == "INSUFFICIENT_EVIDENCE"


def test_to_float_parses_display_strings():
    assert benchmark._to_float("472.84") == 472.84
    assert benchmark._to_float(0.5) == 0.5
    assert benchmark._to_float("") is None
    assert benchmark._to_float(None) is None
    assert benchmark._to_float("ctx=2798") is None


def test_normalize_e1_table_accepts_working_tree_artifact_shape():
    raw = {
        "title": "E1 -- main comparison",
        "rows": [
            {
                "System": "bm25",
                "Recall@5": "0.7788",
                "MRR": "0.7264",
                "Median latency (ms)": "1.88",
                "P95 latency (ms)": "3.45",
            },
            {
                "System": "dense",
                "Recall@5": "0.9283",
                "MRR": "0.8604",
                "Median latency (ms)": "472.84",
                "P95 latency (ms)": "691.20",
            },
        ],
    }
    table = benchmark.normalize_e1_table(raw)
    assert table is not None
    assert len(table["rows"]) == 2
    bm25 = table["rows"][0]
    assert bm25["system"] == "bm25"
    assert bm25["recall_at_5"] == 0.7788
    assert bm25["mrr"] == 0.7264
    assert bm25["median_latency_ms"] == 1.88
    # The working-tree artifact carries no Hit column.
    assert bm25["hit_at_5"] is None


def test_normalize_e1_table_is_idempotent_on_snapshot_shape():
    snapshot = benchmark.load_headline_snapshot()
    table = benchmark.normalize_e1_table(snapshot["phase7_e1"])
    assert table is not None
    dense = next(row for row in table["rows"] if row["system"] == "dense")
    assert dense["recall_at_5"] == 0.9283
    assert dense["hit_at_5"] == 0.9720


def test_e1_table_prefers_working_tree_but_snapshot_fallback_exists():
    table, source = benchmark.load_e1_table()
    assert table is not None
    assert source in (benchmark.WORKING_TREE, benchmark.SNAPSHOT)
    assert len(table["rows"]) == 5


def test_arm_rows_load_or_report_absence():
    rows, source = benchmark.load_arm_rows("bm25")
    if source == benchmark.WORKING_TREE:
        assert len(rows) == 107
        assert rows[0]["query_id"] == "p7_001"
        assert "recall_at_5" in rows[0]
    else:
        assert rows == []
