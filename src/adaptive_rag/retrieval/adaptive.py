"""
retrieval.adaptive
------------------
Phase 6 adaptive retriever: the orchestrator for query-aware strategy selection
with bounded escalation.

This module *composes* and *decides*. It owns no index, no embedding model, no
tokenizer, and no ranking logic: candidates and scores always come from the
Phase 2-5 retrievers it is handed, and the decision of *which* to run comes from
the injected router. That separation is why Phases 2-5 stay unmodified and why a
learned router can replace the rule-based one without touching retrieval.

Flow (one bounded escalation maximum):

```text
query → analyzer → router → initial retrieval → sufficiency check
          ├─ sufficient                      → final results
          └─ insufficient → escalation policy → stronger retrieval → final results
```

Contract notes:

* Satisfies the existing `Retriever` protocol, so `ExperimentRunner` and both
  existing evaluators consume it unchanged.
* `score_threshold` is deliberately **not** forwarded to the stages, mirroring the
  Phase 4 and Phase 5 decisions: a threshold would silently shrink the candidate
  pool before the sufficiency check could judge the evidence. It is echoed into
  metadata for traceability only.
* An escalated query returns the stronger stage's results, not a merge of both.
  Merging two rankings is a new fusion algorithm and belongs to Phase 4's
  concerns, not this one.
* No `try`/`except` anywhere: a failing strategy propagates unchanged and surfaces
  as `status="retrieval_failed"`. Escalation is a routing decision, never an
  error-swallowing retry (enforced by an AST guard).
"""

import time
from typing import Any, Mapping

from adaptive_rag.errors import InvalidQueryError
from adaptive_rag.routing.base import QueryAnalyzer, Router
from adaptive_rag.routing.escalation import EscalationPolicy
from adaptive_rag.routing.sufficiency import SufficiencyChecker
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.schemas import (
    EscalationDecision,
    RetrievalConfig,
    RetrievalResponse,
    RoutingConfig,
    RoutingTrace,
)

ADAPTIVE_VERSION = "adaptive_v1"


class AdaptiveRetriever:
    """Routes each query to a retrieval strategy, escalating once if warranted."""

    method: str = "adaptive"

    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        router: Router,
        analyzer: QueryAnalyzer,
        sufficiency_checker: SufficiencyChecker,
        escalation_policy: EscalationPolicy,
        retrieval_config: RetrievalConfig | None = None,
        routing_config: RoutingConfig | None = None,
        corpus_version: str = "corpus_v1",
        index_id: str | None = None,
    ):
        self.retrievers = dict(retrievers)
        self.router = router
        self.analyzer = analyzer
        self.sufficiency_checker = sufficiency_checker
        self.escalation_policy = escalation_policy
        self.routing_config = routing_config or RoutingConfig()
        self.config = retrieval_config or RetrievalConfig(
            retrieval_method="adaptive", retriever_version=ADAPTIVE_VERSION
        )
        self.corpus_version = corpus_version
        self.index_id = index_id or "adaptive_multi_strategy_v1"

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Route, retrieve, judge sufficiency, and escalate at most once.

        `score_threshold` is echoed into metadata but not forwarded, and `filters`
        are forwarded to whichever stage(s) run.
        """
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")

        k = top_k if top_k is not None else self.config.top_k
        filt = filters if filters is not None else self.config.filters

        t0 = time.perf_counter()
        features = self.analyzer.analyze(query)
        decision = self.router.route(features, top_k=k)
        routing_latency = (time.perf_counter() - t0) * 1000.0

        initial_strategy = decision.strategy
        initial = self.retrievers[initial_strategy].retrieve(query, top_k=k, filters=filt)
        stage_latencies = [initial.retrieval_metadata.latency_ms]
        stage_count = 1

        sufficiency = None
        if self.routing_config.sufficiency_enabled:
            sufficiency = self.sufficiency_checker.check(features, initial, initial_strategy)

        final = initial
        final_strategy = initial_strategy
        escalation = self._decide_escalation(sufficiency, initial_strategy)

        # Pre-escalation evidence for Phase 7 E8, which must score quality
        # *before* and *after* each escalation transition. Left None unless the
        # query actually escalated: a query that settled kept `initial` as its
        # final results, so the discarded-before list would just restate what the
        # trace already carries. Recorded here, where the first stage's response
        # is still in hand and is about to be superseded. Rank-ordered ids are all
        # E8 needs; scores are omitted because BM25, cosine, and cross-encoder
        # scores are not comparable across stages.
        initial_chunk_ids: list[str] | None = None
        initial_document_ids: list[str] | None = None

        if escalation.escalated and escalation.to_strategy is not None:
            initial_chunk_ids = [r.chunk_id for r in initial.results]
            initial_document_ids = [r.metadata.document_id for r in initial.results]
            final = self.retrievers[escalation.to_strategy].retrieve(
                query, top_k=k, filters=filt
            )
            final_strategy = escalation.to_strategy
            stage_latencies.append(final.retrieval_metadata.latency_ms)
            stage_count += 1

        routing_trace = RoutingTrace(
            features=features,
            decision=decision,
            initial_strategy=initial_strategy,
            initial_latency_ms=stage_latencies[0],
            initial_result_count=len(initial.results),
            initial_chunk_ids=initial_chunk_ids,
            initial_document_ids=initial_document_ids,
            sufficiency=sufficiency,
            escalation=escalation,
            final_strategy=final_strategy,
            stage_count=stage_count,
            stage_latencies_ms=stage_latencies,
            routing_latency_ms=routing_latency,
        )

        return self._build_response(
            query=query,
            final=final,
            routing_trace=routing_trace,
            escalation=escalation,
            sufficiency=sufficiency,
            stage_latencies=stage_latencies,
            routing_latency=routing_latency,
            top_k=k,
            filters=filt,
        )

    # --- internals ---------------------------------------------------------

    def _decide_escalation(self, sufficiency: Any, current: str) -> EscalationDecision:
        """Apply the escalation policy; returns a recorded decision either way.

        Every branch is recorded, including "no escalation", so a trace can always
        answer why the pipeline stopped where it did. Ordering matters: the
        configuration checks come before the ladder lookup so a disabled policy is
        reported as disabled rather than as "no stronger strategy".
        """
        max_steps = self.routing_config.max_escalation_steps
        version = self.escalation_policy.version

        def record(escalated: bool, reason: str, to: str | None = None) -> EscalationDecision:
            return EscalationDecision(
                escalated=escalated,
                from_strategy=current,
                to_strategy=to,
                reason=reason,
                step_index=1 if escalated else 0,
                max_steps=max_steps,
                policy_version=version,
            )

        if sufficiency is None:
            return record(False, "sufficiency check disabled by configuration")
        if sufficiency.sufficient:
            return record(
                False,
                f"retrieved evidence was sufficient (score={sufficiency.score:.3f})",
            )
        if not self.routing_config.escalation_enabled:
            return record(
                False,
                "evidence insufficient but escalation disabled by configuration",
            )
        if not self.escalation_policy.allows_escalation(steps_taken=0):
            return record(False, "escalation budget exhausted")

        next_strategy = self.escalation_policy.next_strategy(current)
        if next_strategy is None:
            return record(
                False,
                f"evidence insufficient but '{current}' is the strongest available "
                f"strategy; no further escalation is permitted",
            )

        return record(
            True,
            f"evidence insufficient (score={sufficiency.score:.3f} < "
            f"{sufficiency.threshold:.3f}); escalating to a stronger strategy",
            to=next_strategy,
        )

    def _build_response(
        self,
        *,
        query: str,
        final: RetrievalResponse,
        routing_trace: RoutingTrace,
        escalation: EscalationDecision,
        sufficiency: Any,
        stage_latencies: list[float],
        routing_latency: float,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> RetrievalResponse:
        """Assemble the response, carrying the winning stage's metadata forward.

        The winning stage's provenance, scores, and rerank diagnostics survive via
        `model_copy`; the adaptive fields are layered on top. Results and ranks are
        passed through untouched, because the winning stage already numbered them.
        """
        base_meta = final.retrieval_metadata
        metadata = base_meta.model_copy(
            update={
                "top_k": top_k,
                "score_threshold": self.config.score_threshold,
                "filters": filters,
                "retriever_version": ADAPTIVE_VERSION,
                "index_id": self.index_id,
                "corpus_version": self.corpus_version,
                # The honest end-to-end cost: analysis + routing + every stage that
                # actually ran. Efficiency metrics read exactly this field, so an
                # adaptive run is priced truthfully with no downstream special case.
                "latency_ms": routing_latency + sum(stage_latencies),
                "routing": routing_trace,
                "adaptive_router_version": routing_trace.decision.router_version,
                "adaptive_initial_strategy": routing_trace.initial_strategy,
                "adaptive_final_strategy": routing_trace.final_strategy,
                "adaptive_stage_count": routing_trace.stage_count,
                "adaptive_routing_latency_ms": routing_latency,
                "adaptive_initial_latency_ms": routing_trace.initial_latency_ms,
                "adaptive_escalated": escalation.escalated,
                "adaptive_sufficient": (
                    sufficiency.sufficient if sufficiency is not None else None
                ),
            }
        )
        return RetrievalResponse(
            query=query,
            results=list(final.results),
            retrieval_method="adaptive",
            status=final.status,
            retrieval_metadata=metadata,
        )