"""
routing.sufficiency
-------------------
Phase 6 retrieval-sufficiency check: "did the first retrieval produce enough
evidence to stop, or is escalation justified?"

This runs at **query time** and reads only the query and the retrieved text. It
has no access to benchmark labels, held-out judgements, or evaluation metrics,
which would all be unavailable outside an experiment. An architecture guard
enforces that boundary, because a sufficiency check that peeked at labels would
be meaningless in production while looking excellent offline.

Signals, all label free:

* `result_count` -- did the stage return anything at all?
* `lexical_coverage` -- what fraction of the query's content terms actually occur
  in the retrieved text? A query whose terms are absent from everything returned
  is a clear sign the first strategy missed.
* `top1_coverage` -- the same measure against the top result only, catching the
  case where the term appears somewhere in the tail but not in anything salient.

**Why there is no score floor by default.** BM25 magnitudes, cosine similarity,
RRF scores, and cross-encoder logits live on incomparable scales, so a single
threshold across strategies would be an invented number rather than a measured
one -- and Phase 5 already recorded a documented cost estimate that turned out to
be wrong by two orders of magnitude. `RoutingConfig.score_floor` stays available
for a per-strategy threshold that has actually been measured.
"""

from adaptive_rag.schemas import (
    QueryFeatures,
    RetrievalResponse,
    RoutingConfig,
    SufficiencyDecision,
    SufficiencySignal,
)

CHECKER_VERSION = "sufficiency_v1"


class SufficiencyChecker:
    """Decides whether retrieved evidence justifies stopping at the first stage."""

    version = CHECKER_VERSION

    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or RoutingConfig()

    def check(
        self,
        features: QueryFeatures,
        response: RetrievalResponse,
        strategy: str,
    ) -> SufficiencyDecision:
        """Judge sufficiency from retrieved evidence alone.

        Deterministic: the same features, response, and strategy always yield the
        same decision.
        """
        config = self.config
        result_count = len(response.results)

        terms = list(features.content_terms)
        coverage = _term_coverage(terms, [r.text for r in response.results])
        top1_coverage = (
            _term_coverage(terms, [response.results[0].text]) if response.results else 0.0
        )

        signals: list[SufficiencySignal] = [
            SufficiencySignal(
                name="result_count",
                value=_cap(result_count / config.min_results),
                passed=result_count >= config.min_results,
                weight=1.0,
            ),
            SufficiencySignal(
                name="lexical_coverage",
                value=coverage,
                passed=coverage >= config.coverage_threshold,
                weight=1.0,
            ),
            SufficiencySignal(
                name="top1_coverage",
                value=top1_coverage,
                passed=top1_coverage >= config.top1_coverage_threshold,
                # Advisory: it corroborates `lexical_coverage` without dominating
                # it, so a term present only in the tail does not force escalation.
                weight=0.5,
            ),
        ]

        if config.score_floor and strategy in config.score_floor:
            floor = float(config.score_floor[strategy])
            top_score = max((r.score for r in response.results), default=0.0)
            signals.append(
                SufficiencySignal(
                    name="score_floor",
                    value=top_score,
                    passed=top_score >= floor,
                    weight=1.0,
                )
            )

        total_weight = sum(s.weight for s in signals)
        score = (
            sum(s.value * s.weight for s in signals) / total_weight
            if total_weight > 0
            else 0.0
        )
        sufficient = score >= config.sufficiency_threshold

        return SufficiencyDecision(
            sufficient=sufficient,
            score=round(score, 6),
            threshold=config.sufficiency_threshold,
            signals=signals,
            reason=_explain(signals, score, sufficient),
            checker_version=self.version,
            result_count=result_count,
            coverage=round(coverage, 6),
            top1_coverage=round(top1_coverage, 6),
        )


def _term_coverage(terms: list[str], texts: list[str]) -> float:
    """Fraction of query content terms present in any of the given texts.

    Matching is whole-term and case-insensitive. An empty term list scores 1.0:
    a query with no content terms cannot be shown to be ungrounded.
    """
    if not terms:
        return 1.0
    haystack = " ".join(texts).lower()
    present = sum(1 for term in terms if term.lower() in haystack)
    return present / len(terms)


def _explain(
    signals: list[SufficiencySignal], score: float, sufficient: bool
) -> str:
    """Human-readable reason, naming the signals that drove the decision."""
    failed = [s.name for s in signals if not s.passed]
    verdict = "sufficient" if sufficient else "insufficient"
    if failed:
        return f"{verdict} (score={score:.3f}); failed signals: {', '.join(failed)}"
    return f"{verdict} (score={score:.3f}); all signals passed"


def _cap(value: float) -> float:
    """Clamp to [0, 1]."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)