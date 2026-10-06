"""
test_config.py
--------------
Unit tests for configuration, paths, hashing, and pricing.
"""

from pathlib import Path
import pytest

from adaptive_rag.config.hashing import canonical_json, compute_config_hash, compute_sha256
from adaptive_rag.config.paths import PAPERS_MANIFEST_PATH, REPO_ROOT
from adaptive_rag.config.pricing import calculate_cost_usd
from adaptive_rag.config.settings import Settings
from adaptive_rag.errors import MissingCredentialError


def test_repo_root_contains_expected_files():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert PAPERS_MANIFEST_PATH.is_file()


def test_canonical_json_sorting_and_compactness():
    d1 = {"b": 2, "a": 1, "c": [3, 4]}
    d2 = {"c": [3, 4], "a": 1, "b": 2}
    s1 = canonical_json(d1)
    s2 = canonical_json(d2)
    assert s1 == s2
    assert s1 == '{"a":1,"b":2,"c":[3,4]}'


def test_compute_config_hash_stability():
    c1 = {"model": "text-embedding-3-large", "dim": 3072}
    c2 = {"dim": 3072, "model": "text-embedding-3-large"}
    assert compute_config_hash(c1) == compute_config_hash(c2)
    assert compute_config_hash(c1) != compute_config_hash({"model": "other"})


def test_missing_credentials_raise_typed_error():
    empty_settings = Settings(AICREDITS_API_KEY=None, GROQ_API_KEY=None)
    with pytest.raises(MissingCredentialError):
        empty_settings.require_aicredits_key()

    with pytest.raises(MissingCredentialError):
        empty_settings.require_groq_key()


def test_cost_calculation():
    # gpt-oss-120b: $0.15 input / $0.60 output per 1M
    cost = calculate_cost_usd("openai/gpt-oss-120b", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 0.75) < 1e-6

    # embedding: $0.13 per 1M input
    embed_cost = calculate_cost_usd("text-embedding-3-large", input_tokens=100_000)
    assert abs(embed_cost - 0.013) < 1e-6
