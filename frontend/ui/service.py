"""Thin service layer: builds retrievers and runs live retrieval.

This module is the *only* place the presentation layer touches the
retrieval system, and it does so exclusively through the project's
own factories -- ``build_experiment_config``, ``build_arm`` and
``instantiate_components``. It contains no retrieval logic of its
own: no scoring, no fusion, no ranking.

Two constraints from the retrieval layer shape everything here:

1. **One embedded Qdrant client per process.** In local mode
   (``QDRANT_MODE=local``) Qdrant takes an exclusive lock on
   ``storage/qdrant``, so a second embedded client in the same
   process fails. Every Qdrant-backed strategy therefore shares the
   single client opened here and injects it via
   ``instantiate_components(config, client=...)``, exactly as the
   Phase 7 suite does.

2. **Arms run sequentially.** Because they share one client and the
   provider arms embed queries remotely, comparison mode runs one
   strategy at a time and paces provider-calling arms.

Retrievers are cached by configuration hash so repeated queries in a
Streamlit rerun do not rebuild indexes or reopen connections.
"""

from __future__ import annotations

import time
from typing import Any

from qdrant_client import QdrantClient

from adaptive_rag.config.paths import QDRANT_DIR
from adaptive_rag.config.settings import settings
from adaptive_rag.errors import (
    AdaptiveRAGError,
    IndexUnavailableError,
    MissingCredentialError,
)
from adaptive_rag.experiments.arms import build_arm
from adaptive_rag.experiments.config import build_experiment_config
from adaptive_rag.schemas.config import RoutingConfig
from adaptive_rag.schemas.experiment import ExperimentConfig

# The shipping corpus arm (Phase 8 after). Pointing at the wrong arm
# is caught by the corpus-version guard at index load, so it is named
# here once rather than scattered through the UI.
DEFAULT_CORPUS_ARM = "phase8_after"

# Pace between provider-calling retrievals in comparison mode, to
# avoid bursting the embedding API (the same fix Phase 8 applied to
# its contaminated sweep).
PROVIDER_PACE_SECONDS = 1.5

# Module-level caches. Streamlit reruns the script on every
# interaction, but imported modules persist in sys.modules, so this
# state survives across reruns within one server process.
_shared_client: QdrantClient | None = None
_common_config_cache: ExperimentConfig | None = None
_retriever_cache: dict[str, Any] = {}


def get_shared_qdrant_client() -> QdrantClient | None:
    """Open (once) the process-wide embedded Qdrant client.

    Returns ``None`` in server mode, where each store gets its own
    stateless handle. In local mode a single embedded client is
    shared by every Qdrant-backed strategy.
    """
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    if settings.QDRANT_MODE == "server" and settings.QDRANT_URL:
        _shared_client = QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )
        return _shared_client
    try:
        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        _shared_client = QdrantClient(path=str(QDRANT_DIR))
    except Exception as exc:  # noqa: BLE001 -- normalized to a typed error
        raise IndexUnavailableError(
            "Failed to open the shared local Qdrant client at "
            f"{settings.QDRANT_PATH}. The commonest cause is another "
            "process (a previous run, a REPL, or a stray script) still "
            "holding the lock on that folder -- stop it and retry."
        ) from exc
    return _shared_client


def _common_config() -> ExperimentConfig:
    """Build (once) the frozen shared configuration all arms derive from."""
    global _common_config_cache
    if _common_config_cache is None:
        _common_config_cache = build_experiment_config(name="frontend_common")
    return _common_config_cache


def _arm_config(strategy: str, corpus_arm: str, routing: RoutingConfig | None = None) -> ExperimentConfig:
    """Materialise one strategy as a full, hash-stamped ExperimentConfig.

    The config's ``index.corpus_arm`` is set to the requested arm so
    the arm reads the right collection and lexical index; the
    ``IndexConfig`` validator derives the exact namespace from the arm.
    """
    common = _common_config()
    # Re-point the shared index config at the requested corpus arm.
    # build_experiment_config stamps the hash, so the arm config is
    # rebuilt rather than mutated.
    common = common.model_copy(
        update={"index": common.index.model_copy(update={"corpus_arm": corpus_arm})}
    )
    arm = build_arm(common, strategy, routing=routing)
    return arm.build_config(common)


def build_retriever(
    strategy: str,
    *,
    corpus_arm: str = DEFAULT_CORPUS_ARM,
    routing: RoutingConfig | None = None,
) -> Any:
    """Build (and cache) a live retriever for one strategy.

    ``strategy`` is one of ``bm25``, ``dense``, ``hybrid``,
    ``hybrid_rerank`` or ``adaptive``. The retriever is built through
    ``instantiate_components`` and cached by configuration hash.
    """
    config = _arm_config(strategy, corpus_arm, routing=routing)
    cache_key = config.config_hash
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    client = get_shared_qdrant_client()
    components = instantiate(config, client)
    retriever = components[2]  # slot 2 is the retriever
    _retriever_cache[cache_key] = retriever
    return retriever


def instantiate(config: ExperimentConfig, client: QdrantClient | None):
    """Call the project's own component factory.

    Imported lazily so importing this module never opens an index.
    """
    from adaptive_rag.experiments.config import instantiate_components

    return instantiate_components(config, client=client)


def adaptive_routing(available_strategies: list[str]) -> RoutingConfig:
    """A routing config restricted to the strategies this deployment can run.

    BM25-only (the offline default) needs a single-rung ladder and
    ``max_escalation_steps=0``, because a rung that cannot run must
    never appear in the ladder.
    """
    ladder = list(available_strategies)
    return RoutingConfig(
        available_strategies=ladder,
        escalation_ladder=ladder,
        max_escalation_steps=max(0, len(ladder) - 1),
    )


def available_adaptive_strategies() -> list[str]:
    """The strategies the adaptive router may use in this deployment."""
    from frontend.ui import availability

    probe = availability.probe()
    live = [s for s in ("bm25", "dense", "hybrid", "hybrid_rerank") if probe[s].is_live]
    # BM25 is always the floor; keep at least it so the router has a
    # valid offline arm even when no credentials are set.
    return live or ["bm25"]


def retrieve(
    strategy: str,
    query: str,
    *,
    top_k: int = 10,
    corpus_arm: str = DEFAULT_CORPUS_ARM,
    routing: RoutingConfig | None = None,
) -> Any:
    """Run one live retrieval and return the ``RetrievalResponse``."""
    retriever = build_retriever(strategy, corpus_arm=corpus_arm, routing=routing)
    return retriever.retrieve(query, top_k=top_k)


def retrieve_sequential(
    strategies: list[str],
    query: str,
    *,
    top_k: int = 10,
    corpus_arm: str = DEFAULT_CORPUS_ARM,
) -> list[dict[str, Any]]:
    """Run several strategies over one query, sequentially.

    Returns a list of ``{"strategy", "response", "error"}`` records in
    the order requested. Provider-calling arms are paced so the
    embedding API is not burst. An arm that fails is reported as an
    error record rather than aborting the comparison.
    """
    records: list[dict[str, Any]] = []
    for index, strategy in enumerate(strategies):
        try:
            response = retrieve(strategy, query, top_k=top_k, corpus_arm=corpus_arm)
            records.append({"strategy": strategy, "response": response, "error": None})
        except AdaptiveRAGError as exc:
            records.append(
                {
                    "strategy": strategy,
                    "response": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
        # Pace between provider-calling arms (not after the last one).
        if index < len(strategies) - 1 and strategy != "bm25":
            time.sleep(PROVIDER_PACE_SECONDS)
    return records


def clear_cache() -> None:
    """Drop cached retrievers and the shared client (used by tests)."""
    global _shared_client, _common_config_cache
    _retriever_cache.clear()
    _shared_client = None
    _common_config_cache = None


def credential_hint(exc: AdaptiveRAGError) -> str:
    """A friendly, actionable message for a retrieval failure."""
    if isinstance(exc, MissingCredentialError):
        return (
            "This strategy needs an API key that is not set. Add "
            "AICREDITS_API_KEY to .env (copy .env.example) and restart "
            "the app. BM25 works offline without any key."
        )
    if isinstance(exc, IndexUnavailableError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"
