"""
routing.base
------------
Routing abstractions: the query analyzer and the router.

This module holds protocols only. The separation is deliberate and load-bearing:

* `QueryAnalyzer` turns a raw query into explicit `QueryFeatures`.
* `Router` turns those features into a `RoutingDecision` -- it decides *what*
  retrieval strategy should run, never *how* retrieval works.

Because `Router.route` consumes `QueryFeatures` rather than the raw string, a
learned router can be substituted later without changing the retrieval
architecture, the sufficiency check, or the escalation policy.

Imports are limited to `schemas` and `errors`: this package holds no index, no
embedding model, and no retriever (architecture-guard enforced).
"""

from typing import Protocol, runtime_checkable

from adaptive_rag.schemas import QueryFeatures, RoutingDecision


@runtime_checkable
class QueryAnalyzer(Protocol):
    """Protocol for deterministic query feature extraction."""

    version: str

    def analyze(self, query: str) -> QueryFeatures:
        """Extract routing features from a raw query.

        Must be a pure, cheap function of the query string: no model, no index,
        no corpus, no network.
        """
        ...


@runtime_checkable
class Router(Protocol):
    """Protocol for strategy selection.

    A router returns a structured `RoutingDecision` rather than a bare strategy
    name, so the decision is reproducible, loggable, and comparable across
    implementations (rule-based, learned, or anything else).
    """

    version: str

    def route(self, features: QueryFeatures, *, top_k: int) -> RoutingDecision:
        """Select a retrieval strategy from analyzed query features."""
        ...