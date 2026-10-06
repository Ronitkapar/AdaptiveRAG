"""Availability probe tests (offline, filesystem + settings only)."""

from frontend.ui import availability


def test_probe_covers_all_strategies():
    probe = availability.probe()
    assert set(probe) == {"bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"}
    for name, entry in probe.items():
        assert entry.strategy == name
        assert entry.status in (
            availability.LIVE,
            availability.NEEDS_CREDENTIALS,
            availability.UNAVAILABLE,
        )
        assert entry.reason, f"{name} must explain its status"


def test_bm25_is_live_offline_on_this_tree():
    # The BM25 index is present in storage/ on the research machine, so
    # the cheapest arm must be runnable with no credentials involved.
    probe = availability.probe()
    assert probe["bm25"].status == availability.LIVE
    assert probe["bm25"].offline is True


def test_missing_credentials_reported_not_hidden(monkeypatch):
    from adaptive_rag.config.settings import settings as live_settings

    monkeypatch.setattr(live_settings, "AICREDITS_API_KEY", None)
    probe = availability.probe()
    assert probe["dense"].status == availability.NEEDS_CREDENTIALS
    assert "AICREDITS_API_KEY" in probe["dense"].reason


def test_summarize_is_json_serialisable():
    import json

    summary = availability.summarize()
    json.dumps(summary)
    assert set(summary["strategies"]) == set(availability.STRATEGIES)
