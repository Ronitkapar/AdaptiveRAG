"""
routing.escalation
------------------
Phase 6 escalation policy: which stronger strategy, if any, may follow the one
that just ran.

The ladder is a **total order** drawn from configuration, not a graph of
per-strategy transitions. That choice is what makes the forbidden

```text
BM25 → Dense → Hybrid → BM25 → ...
```

impossible by construction: a strategy has exactly one successor (or none, at the
top of the ladder), so no policy bug, configuration mistake, or repeated failure
can produce a cycle. Determinism follows from the same property -- the next
strategy is a pure function of the current one.

`max_steps` bounds how far a single query may travel. Phase 6 defaults to 1: one
escalation, then stop. Keeping the bound explicit means adaptive behaviour stays
measurable -- an unbounded pipeline would have no well-defined per-query cost.
"""

from adaptive_rag.errors import ConfigurationError
from adaptive_rag.schemas import RoutingConfig, StrategyName

POLICY_VERSION = "escalation_v1"


class EscalationPolicy:
    """Resolves the allowed stronger successor for the current strategy."""

    version = POLICY_VERSION

    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or RoutingConfig()
        self.ladder: list[str] = list(self.config.escalation_ladder)
        self.max_steps: int = self.config.max_escalation_steps
        self._validate()

    def _validate(self) -> None:
        """Reject a ladder that cannot describe a bounded, runnable ascent."""
        available = set(self.config.available_strategies)
        unknown = [s for s in self.ladder if s not in available]
        if unknown:
            raise ConfigurationError(
                f"Escalation ladder references strategies that are not available "
                f"for routing: {unknown}. A rung that cannot run must never appear "
                f"in the ladder."
            )
        if not self.ladder:
            raise ConfigurationError("Escalation ladder must not be empty")
        if len(set(self.ladder)) != len(self.ladder):
            raise ConfigurationError(
                "Escalation ladder must not contain duplicate strategies; a cycle "
                "in the ladder would make escalation unbounded."
            )
        if not 0 <= self.max_steps <= len(self.ladder) - 1:
            raise ConfigurationError(
                f"max_escalation_steps={self.max_steps} is outside the ladder's "
                f"reachable range [0, {len(self.ladder) - 1}]"
            )

    def next_strategy(self, strategy: StrategyName) -> StrategyName | None:
        """Return the next stronger strategy, or `None` at the top of the ladder.

        `None` is the honest answer at the top rung: the system has no stronger
        strategy available, so continuing would mean re-running the same pipeline
        or inventing a new one. Both are out of Phase 6's scope.
        """
        if strategy not in self.ladder:
            return None
        index = self.ladder.index(strategy)
        if index + 1 >= len(self.ladder):
            return None
        return self.ladder[index + 1]  # type: ignore[return-value]

    def allows_escalation(self, steps_taken: int) -> bool:
        """Whether one more escalation is still within the configured bound."""
        return steps_taken < self.max_steps

    def is_terminal(self, strategy: StrategyName) -> bool:
        """True when `strategy` has no stronger successor on the ladder."""
        return self.next_strategy(strategy) is None