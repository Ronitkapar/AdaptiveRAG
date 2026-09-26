"""
tests.test_architecture_guards
------------------------------
Guards that keep the Phase 2 dense baseline honest: no premature retrieval
strategies, no LangChain/LlamaIndex core dependencies, and no secrets in repo.
"""

from pathlib import Path
import re
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "adaptive_rag"

FORBIDDEN_STRATEGY_TOKENS = (
    "bm25",
    "hybrid",
    "rerank",
    "adaptive_rout",
    "query_classif",
    "strategy_select",
)
FORBIDDEN_HEAVY_DEPS = ("langchain", "llama-index", "llama_index", "sentence-transformers")


def _iter_source_files():
    return sorted(SRC_ROOT.rglob("*.py"))


def test_no_future_strategy_symbols_in_source():
    """BM25 / hybrid / rerank / adaptive symbols must not exist anywhere in src."""
    violations: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_STRATEGY_TOKENS:
            if token in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not violations, f"future-strategy leakage: {violations}"


def test_no_heavy_framework_dependencies():
    """LangChain / LlamaIndex / sentence-transformers are banned as core deps."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)["project"]
    declared = " ".join(project.get("dependencies", [])).lower()
    declared += " " + " ".join(project.get("optional-dependencies", {}).get("dev", [])).lower()
    for dep in FORBIDDEN_HEAVY_DEPS:
        assert dep not in declared, f"forbidden dependency declared: {dep}"

    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        for token in ("from langchain", "import langchain", "from llama_index", "import llama_index"):
            assert token not in text, f"{path} imports a banned framework"


def test_provider_isolation_between_generation_and_embeddings():
    """Generation modules must not depend on embedding providers and vice versa."""
    generation_files = list((SRC_ROOT / "generation").glob("*.py"))
    for path in generation_files:
        text = path.read_text(encoding="utf-8")
        assert "aicredits" not in text.lower(), f"{path.name} couples generation to embeddings"
        assert "embed_chunks" not in text, f"{path.name} couples generation to embeddings"

    embedding_files = list((SRC_ROOT / "embeddings").glob("*.py"))
    for path in embedding_files:
        text = path.read_text(encoding="utf-8").lower()
        assert "groq" not in text, f"{path.name} couples embeddings to generation"


def test_no_secrets_committed_in_repo():
    """No API keys or .env files may be committed to the repository."""
    assert not (REPO_ROOT / ".env").exists(), ".env must never exist in the working tree"
    key_pattern = re.compile(r"(sk-|gsk_)[A-Za-z0-9]{10,}")
    for path in list(_iter_source_files()) + [REPO_ROOT / ".env.example"]:
        text = path.read_text(encoding="utf-8")
        assert not key_pattern.search(text), f"possible secret in {path}"
