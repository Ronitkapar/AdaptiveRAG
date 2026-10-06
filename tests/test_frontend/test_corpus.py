"""Corpus and benchmark-query loader tests (committed data only)."""

from frontend.ui import corpus


def test_manifest_has_fourteen_papers():
    papers = corpus.load_papers()
    assert len(papers) == 14
    by_id = corpus.papers_by_id()
    assert "rag_lewis_2020" in by_id
    assert "adaptive_rag_jeong_2024" in by_id
    for paper in papers:
        for field in ("document_id", "title", "year", "category", "concepts"):
            assert paper[field], f"{paper.get('document_id')} missing {field}"


def test_category_counts_cover_all_papers():
    counts = corpus.category_counts()
    assert sum(counts.values()) == 14
    assert counts["adaptive_retrieval"] == 1


def test_phase7_benchmark_has_107_queries():
    rows = corpus.load_phase7_queries()
    assert len(rows) == 107
    assert rows[0]["example_id"] == "p7_001"
    for row in rows:
        assert row["query"].strip()
        assert row["relevant_documents"]


def test_suggested_questions_are_real_benchmark_queries():
    suggestions = corpus.suggested_questions(limit=8)
    assert len(suggestions) == 8
    known_ids = {row["example_id"] for row in corpus.load_phase7_queries()}
    for suggestion in suggestions:
        assert suggestion["example_id"] in known_ids
        assert suggestion["query"].strip()
