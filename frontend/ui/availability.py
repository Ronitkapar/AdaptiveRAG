"""Strategy / credential / artifact availability probe.

The demo must never present a fallback as a real result, so the
first job of every page is to know what this deployment can
actually run. This module answers that question cheaply and
without opening any index or provider connection: it inspects
settings and the filesystem only. Whether a Qdrant collection is
reachable is deferred to :mod:`frontend.ui.service`, which opens
the shared client exactly once and reports any failure as a typed
error rather than a silent empty result.

Availability is reported per strategy with one of three states:

* ``live``            -- runnable right now in this deployment.
* ``needs_credentials`` -- the code path works but a required API
  key is not set (``AICREDITS_API_KEY``).
* ``unavailable``     -- a required on-disk artifact (BM25 index or
  reranker model) is missing, so the strategy cannot run even
  with credentials.

A strategy is only ``live`` when everything it needs is present;
BM25 is the one strategy that is fully offline and therefore the
only one that is ``live`` on a fresh clone that has rebuilt its
index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_rag.config.paths import BM25_INDEX_DIR, RERANKER_DIR
from adaptive_rag.config.settings import settings

# The four selectable strategies plus the orchestration layer.
STRATEGIES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")

# Human-readable one-liners, kept beside the probe so the UI and the
# probe cannot drift apart.
STRATEGY_LABELS: dict[str, str] = {
    "bm25": "BM25 (lexical)",
    "dense": "Dense (semantic)",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + Rerank",
    "adaptive": "Adaptive (router)",
}

LIVE = "live"
NEEDS_CREDENTIALS = "needs_credentials"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Availability:
    """The availability of one strategy in this deployment."""

    strategy: str
    status: str
    reason: str
    offline: bool

    @property
    def is_live(self) -> bool:
        return self.status == LIVE


def _aicredits_present() -> bool:
    return bool(settings.AICREDITS_API_KEY and settings.AICREDITS_API_KEY.strip())


def _groq_present() -> bool:
    return bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())


def _bm25_index_present() -> bool:
    # Any arm's index counts; the service layer resolves the exact
    # file from IndexConfig and the corpus-version guard catches a
    # mismatch at load time.
    return BM25_INDEX_DIR.is_dir() and any(BM25_INDEX_DIR.glob("bm25_index*.json"))


def _reranker_model_present() -> bool:
    # The ONNX cross-encoder is downloaded into RERANKER_DIR on first
    # use; a populated directory means the artifact is cached and no
    # network download is required.
    if not RERANKER_DIR.is_dir():
        return False
    return any(RERANKER_DIR.rglob("*"))


def probe() -> dict[str, Availability]:
    """Return the availability of every strategy in this deployment.

    Cheap and side-effect free: reads settings and the filesystem,
    opens no index and makes no network call.
    """
    aicredits = _aicredits_present()
    bm25 = _bm25_index_present()
    reranker = _reranker_model_present()

    out: dict[str, Availability] = {}

    # BM25 is fully offline: it needs only the lexical index.
    out["bm25"] = Availability(
        "bm25",
        LIVE if bm25 else UNAVAILABLE,
        "offline; needs the BM25 index" if bm25 else "BM25 index not found in storage/",
        offline=True,
    )

    # Dense, hybrid and hybrid_rerank all embed the query remotely,
    # so they need AICREDITS_API_KEY. They also need the Qdrant
    # collection (dense/hybrid) and, for the reranked arm, the ONNX
    # model. Collection reachability is checked at runtime by the
    # service layer, not here.
    provider_reason = (
        "needs AICREDITS_API_KEY (one embedding call per query)"
        if not aicredits
        else "live; one embedding API call per query"
    )
    out["dense"] = Availability(
        "dense",
        LIVE if aicredits else NEEDS_CREDENTIALS,
        provider_reason,
        offline=False,
    )

    hybrid_reason = provider_reason
    if not bm25:
        hybrid_reason = "BM25 index not found in storage/"
        out["hybrid"] = Availability("hybrid", UNAVAILABLE, hybrid_reason, offline=False)
    else:
        out["hybrid"] = Availability(
            "hybrid",
            LIVE if aicredits else NEEDS_CREDENTIALS,
            hybrid_reason,
            offline=False,
        )

    if not bm25:
        out["hybrid_rerank"] = Availability(
            "hybrid_rerank", UNAVAILABLE, "BM25 index not found in storage/", offline=False
        )
    elif not reranker:
        out["hybrid_rerank"] = Availability(
            "hybrid_rerank",
            UNAVAILABLE,
            "ONNX reranker model not cached in storage/reranker/",
            offline=False,
        )
    else:
        out["hybrid_rerank"] = Availability(
            "hybrid_rerank",
            LIVE if aicredits else NEEDS_CREDENTIALS,
            "live; embedding call + local ONNX cross-encoder (~3 s CPU)",
            offline=False,
        )

    # Adaptive orchestrates whichever strategies are available. A
    # BM25-only adaptive run is fully offline, so it is live as long
    # as the BM25 index is; with credentials it can use all four.
    if bm25:
        out["adaptive"] = Availability(
            "adaptive",
            LIVE,
            "live; routes over the available strategies (BM25-only offline)"
            + ("" if aicredits else "; dense/hybrid/rerank rungs disabled without AICREDITS_API_KEY"),
            offline=not aicredits,
        )
    else:
        out["adaptive"] = Availability(
            "adaptive",
            NEEDS_CREDENTIALS if aicredits else UNAVAILABLE,
            "needs the BM25 index" if not bm25 else "needs AICREDITS_API_KEY",
            offline=False,
        )

    return out


def generation_available() -> tuple[bool, str]:
    """Whether Groq answer generation can run, and why not if not."""
    if _groq_present():
        return True, "live; Groq answer generation available"
    return False, "needs GROQ_API_KEY; retrieval-only results are shown without it"


def summarize(probe_result: dict[str, Availability] | None = None) -> dict[str, Any]:
    """A JSON-serialisable summary for badges and the deployment banner."""
    result = probe_result or probe()
    return {
        "strategies": {
            name: {
                "label": STRATEGY_LABELS[name],
                "status": av.status,
                "reason": av.reason,
                "offline": av.offline,
            }
            for name, av in result.items()
        },
        "generation": {"available": _groq_present(), "reason": generation_available()[1]},
    }
