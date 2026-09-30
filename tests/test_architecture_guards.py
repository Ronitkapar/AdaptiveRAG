"""
tests.test_architecture_guards
------------------------------
Guards that keep the implemented phases honest: no premature retrieval
strategies, no LangChain/LlamaIndex core dependencies, and no secrets in repo.
"""

from pathlib import Path
import ast
import re
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "adaptive_rag"

# Phase 5 (second-stage reranking) is implemented; Phase 6+ tokens remain banned.
FORBIDDEN_STRATEGY_TOKENS = (
    "adaptive_rout",
    "query_classif",
    "strategy_select",
)
FORBIDDEN_HEAVY_DEPS = ("langchain", "llama-index", "llama_index", "sentence-transformers")

# Phase 5 runtime dependencies: ONNX inference only, no torch / transformers.
REQUIRED_RERANK_DEPS = ("onnxruntime", "tokenizers")

# Tokens that would mean hybrid.py reimplements a strategy it must only compose.
FORBIDDEN_HYBRID_TOKENS = (
    "rerank",
    "router",
    "route",
    "adaptive",
    "comb_sum",
    "comb_mnz",
    "weighted",
    "qdrant",
    "tokenize",
    "idf",
)

# The second-stage scorer holds no index, no retriever, and no corpus access.
FORBIDDEN_RERANKER_TOKENS = (
    "qdrant",
    "bm25",
    "idf",
    "collection",
)
FORBIDDEN_RERANKER_IMPORTS = (
    ".retrieval",
    ".indexing",
    ".ingestion",
    ".chunking",
)

# The wrapper composes a base retriever and a scorer; it owns neither an index
# nor a text pipeline. Base-strategy words are permitted only as the method-name
# literals ("<base>_rerank"), so those are not substring-forbidden here.
FORBIDDEN_RERANKED_TOKENS = (
    "qdrant",
    "idf",
    "tokenize",
    "embedding_model",
    "vector_store",
    "collection",
)
FORBIDDEN_RERANKED_IMPORTS = (
    "adaptive_rag.retrieval.dense",
    "adaptive_rag.retrieval.bm25",
    "adaptive_rag.retrieval.hybrid",
    "adaptive_rag.retrieval.fusion",
    "adaptive_rag.indexing",
    "adaptive_rag.embeddings",
)


def _iter_source_files():
    return sorted(SRC_ROOT.rglob("*.py"))


def test_no_future_strategy_symbols_in_source():
    """Adaptive-routing symbols must not exist anywhere in src."""
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

    # The second-stage scorer must run on ONNX Runtime, not on torch.
    for dep in REQUIRED_RERANK_DEPS:
        assert dep in declared, f"required reranking dependency not declared: {dep}"
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8").lower()
        for token in ("import torch", "from torch", "transformers"):
            assert token not in text, f"{path} pulls in a banned heavy runtime"


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


def test_retriever_isolation_between_dense_and_bm25():
    """Dense retriever must not import or depend on BM25 and vice versa."""
    dense_path = SRC_ROOT / "retrieval" / "dense.py"
    if dense_path.is_file():
        text = dense_path.read_text(encoding="utf-8").lower()
        assert "bm25" not in text, "dense retriever couples to bm25"

    bm25_retrieval_path = SRC_ROOT / "retrieval" / "bm25.py"
    if bm25_retrieval_path.is_file():
        text = bm25_retrieval_path.read_text(encoding="utf-8").lower()
        assert "dense" not in text, "bm25 retriever couples to dense"
        assert "vector_store" not in text, "bm25 retriever couples to vector_store"
        assert "embedding" not in text, "bm25 retriever couples to embeddings"

    bm25_indexing_path = SRC_ROOT / "indexing" / "bm25.py"
    if bm25_indexing_path.is_file():
        text = bm25_indexing_path.read_text(encoding="utf-8").lower()
        assert "qdrant" not in text, "bm25 indexing couples to qdrant"
        assert "embedding" not in text, "bm25 indexing couples to embeddings"


def _strip_package_prefix(text: str) -> str:
    """Drop `adaptive_rag` import paths so package naming never trips a token guard."""
    return text.replace("adaptive_rag", "")


def test_hybrid_composes_existing_retrievers():
    """Hybrid must only orchestrate dense + BM25; it must not reimplement anything."""
    hybrid_path = SRC_ROOT / "retrieval" / "hybrid.py"
    if not hybrid_path.is_file():
        return
    text = _strip_package_prefix(hybrid_path.read_text(encoding="utf-8").lower())
    for token in FORBIDDEN_HYBRID_TOKENS:
        assert token not in text, f"hybrid retriever leaks strategy logic: {token}"

    # Constituent failures must propagate untouched: no silent dense/BM25 fallback.
    tree = ast.parse(hybrid_path.read_text(encoding="utf-8"))
    try_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert not try_nodes, "hybrid retriever swallows constituent failures"

    fusion_path = SRC_ROOT / "retrieval" / "fusion.py"
    if fusion_path.is_file():
        fusion_text = _strip_package_prefix(fusion_path.read_text(encoding="utf-8").lower())
        for token in ("dense", "bm25", "qdrant", "tokenize"):
            assert token not in fusion_text, f"RRF fusion couples to {token}"


def test_no_secrets_committed_in_repo():
    """No API keys may be committed/tracked, and live .env must stay gitignored."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", ".env"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()
    assert tracked == "", ".env must never be tracked by git"
    key_pattern = re.compile(r"(sk-|gsk_)[A-Za-z0-9]{10,}")
    for path in list(_iter_source_files()) + [REPO_ROOT / ".env.example"]:
        text = path.read_text(encoding="utf-8")
        assert not key_pattern.search(text), f"possible secret in {path}"


# --- Phase 5 guards --------------------------------------------------------


def test_reranker_has_no_index_access():
    """The second-stage scorer must know nothing about indexes or the corpus."""
    reranking_dir = SRC_ROOT / "reranking"
    for path in sorted(reranking_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_RERANKER_TOKENS:
            assert token not in text, f"{path.name} reaches into an index: {token}"
        for module in FORBIDDEN_RERANKER_IMPORTS:
            assert module not in text, f"{path.name} imports {module}"


def test_reranked_retriever_only_composes():
    """The wrapper must compose base + scorer, never reach into a concrete strategy."""
    path = SRC_ROOT / "retrieval" / "reranked.py"
    if not path.is_file():
        return
    raw = path.read_text(encoding="utf-8")
    text = raw.lower()
    for token in FORBIDDEN_RERANKED_TOKENS:
        assert token not in text, f"reranked retriever leaks strategy logic: {token}"
    for module in FORBIDDEN_RERANKED_IMPORTS:
        assert module not in text, f"reranked retriever imports {module}"

    # Exactly one fallback handler is permitted, and it may only swallow the
    # failure when the flag is set: the handler must re-raise otherwise.
    tree = ast.parse(raw)
    try_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert len(try_nodes) == 1, "reranked retriever must not add unguarded failure handling"
    handler = try_nodes[0].handlers[0]
    handler_nodes = list(ast.walk(handler))
    assert any(isinstance(n, ast.Raise) for n in handler_nodes), (
        "the reranking fallback must re-raise when config.rerank_fallback is unset"
    )
    assert any(
        isinstance(n, ast.Attribute) and n.attr == "rerank_fallback" for n in handler_nodes
    ), "the reranking fallback must be gated on config.rerank_fallback"


def test_hybrid_is_independent_of_reranking():
    """Hybrid must remain unaware that a second stage exists."""
    path = SRC_ROOT / "retrieval" / "hybrid.py"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8").lower()
    for token in ("rerank", "route", "router"):
        assert token not in text, f"hybrid retriever couples to reranking: {token}"
    assert "reranking" not in text, "hybrid retriever imports the reranking package"


def test_reranking_does_not_leak_into_retrievers():
    """No single-strategy or fusion module may mention second-stage scoring."""
    for name in ("dense.py", "bm25.py", "hybrid.py", "fusion.py"):
        path = SRC_ROOT / "retrieval" / name
        if not path.is_file():
            continue
        assert "rerank" not in path.read_text(encoding="utf-8").lower(), (
            f"{name} couples to reranking"
        )
