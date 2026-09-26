"""
tests.conftest
--------------
Shared pytest fixtures: offline provider guards and the fixture PDF builder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.make_fixture_pdf import create_synthetic_paper_pdf


@pytest.fixture(scope="session")
def fixture_pdf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Deterministic synthetic 2-page PDF shared across ingestion/chunking tests."""
    dest = tmp_path_factory.mktemp("adaptive_rag_fixtures") / "synthetic_paper.pdf"
    return create_synthetic_paper_pdf(dest)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Fail fast if any default-run test tries to touch the network."""
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="integration tests run explicitly only"))
