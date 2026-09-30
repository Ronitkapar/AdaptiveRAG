"""
tests.conftest
--------------
Shared pytest fixtures: offline provider guards and the fixture PDF builder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.fixtures.make_fixture_pdf import create_synthetic_paper_pdf


@pytest.fixture(scope="session")
def fixture_pdf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Deterministic synthetic 2-page PDF shared across ingestion/chunking tests."""
    dest = tmp_path_factory.mktemp("adaptive_rag_fixtures") / "synthetic_paper.pdf"
    return create_synthetic_paper_pdf(dest)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail fast if any default-run test tries to touch the network.

    `addopts` already deselects `-m 'not integration'` by default, so this is a
    second, explicit guard. It must yield when the caller opts in with
    `-m integration`: a command-line `-m` overrides `addopts`, so those items are
    collected and intended to run. Skipping them unconditionally would make the
    documented `pytest -m integration tests/test_reranking.py` a no-op.
    """
    if _integration_explicitly_requested(config):
        return
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="integration tests run explicitly only"))


def _integration_explicitly_requested(config: pytest.Config) -> bool:
    """True when the caller passed a `-m` expression that selects integration tests.

    Parsed rather than substring-matched: `-m "not integration"` must not count as
    opting in, and a compound expression is only honoured when it mentions the
    marker at all.
    """
    markexpr = config.getoption("markexpr", default=None)
    if not markexpr:
        return False
    return bool(re.search(r"(?<![\w.])integration(?![\w.])", markexpr))
