"""
evaluation.measurement
----------------------
The measurement protocol behind Phase 7 gate 7.0b.

`RoutingConfig.strategy_cost_ms` is the router's price list: `_normalised_costs`
in `routing/rule_based.py` divides each strategy's cost by the maximum across
available strategies, so what the router consumes is the *ratio* between
strategies, not the absolute milliseconds. That makes the table unusually
sensitive to measurement error -- an arm whose mean is unstable contributes an
unstable ratio, and the cost term then shifts with machine noise.

The values currently in `RoutingConfig` were seeded from a Phase 5 run that
recorded per-query means over 20 examples in a single pass, with no warm-up and
no repetitions. The 7.0 environment gate subsequently observed dense retrieval
between roughly 0.85 s and 4 s on this hardware -- a spread wider than the entire
dense/hybrid gap those means encode. This module defines how to re-measure the
table honestly.

The protocol:

1. Build every component outside timing.
2. Warm up each arm and **discard** those samples. This pays the ONNX
   session's first-call cost and establishes the embedding provider's HTTPS
   connection; it cannot remove per-query network latency, because query
   embedding is a live API call with no local cache (see `warmup_queries`).
3. Run R repetitions of the full query set per arm, with the arm order rotated
   between repetitions so machine drift over a long sweep cannot favour
   whichever arm ran first, and the query order rotated so that position effects
   do not load onto a single arm.
4. Aggregate per strategy with a median estimator, and record p95, mean,
   dispersion and the raw samples alongside.

Two deliberate choices, both recorded in the artifact rather than left implicit:

* **The median is `statistics.median`, not `base.percentile(values, 50)`.**
  `base.percentile` is nearest-rank: at n=100 it returns the 50th of 100 ordered
  samples, which is the lower of the two middle values, not their midpoint. That
  is the correct and reproducible definition for p95 -- see below -- but calling
  it a median would misname it.
* **p95 uses `base.percentile`, which is nearest-rank and does not interpolate.**
  At n=100 it is literally the 95th of 100 observed samples. The artifact says so,
  because "p95" otherwise reads as a smoothed curve value that this number is not.

This module measures and reports. It does not write to `RoutingConfig`; freezing
a proposed table into the default is a separate, reviewed change.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.errors import ConfigurationError
from adaptive_rag.evaluation.base import mean, percentile

MEASUREMENT_VERSION = "phase7_measurement_v1"

# The strategies `strategy_cost_ms` prices. This is deliberately not
# `experiments.arms.ARM_NAMES`: the adaptive arm is missing on purpose, because
# its cost is a per-query mixture of the others decided at runtime, and the
# router chooses between *strategies*, not between the adaptive system.
COST_STRATEGIES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")

# Repetitions and warm-up are parameters of the protocol rather than constants of
# the estimator, so a reviewer can see them in the artifact next to the numbers.
DEFAULT_WARMUP_QUERIES = 5
DEFAULT_REPETITIONS = 5


class CostSample(BaseModel):
    """One timed retrieval: a strategy, a query, and a repetition."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    example_id: str
    repetition: int
    latency_ms: float
    # Per-stage clocks, all optional in `RetrievalMetadata` too. Kept raw so the
    # aggregate can be re-derived, and so a suspiciously fast `hybrid_rerank`
    # sample can be traced to a stage that did not actually run.
    stage_latency_ms: dict[str, float | None] = Field(default_factory=dict)
    rerank_fallback: bool | None = None


class StrategyCostSummary(BaseModel):
    """Aggregated latency distribution for one strategy."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    n: int
    median_ms: float | None
    p95_ms: float | None
    mean_ms: float | None
    # Sample standard deviation (ddof=1), reported with the raw samples so the
    # reader can check it. `evaluation.stats.assess_symmetry` uses a population
    # std for shape descriptors; this is dispersion of a sample, so ddof=1.
    stdev_ms: float | None
    min_ms: float | None
    max_ms: float | None
    n_fallback: int = 0
    # Median of each per-stage clock across the same samples. Reported because a
    # single clock can be dominated by something that is not the strategy: the
    # query embedding is a live `text-embedding-3-large` API call with no local
    # cache, so for dense and hybrid it can account for most of the end-to-end
    # figure. A reader deciding what to freeze needs to see that split.
    stage_median_ms: dict[str, float | None] = Field(default_factory=dict)
    raw_ms: list[float] = Field(default_factory=list)


# --- schedule construction ----------------------------------------------------


def warmup_queries(examples: Sequence[Any], count: int = DEFAULT_WARMUP_QUERIES) -> list[Any]:
    """The queries used to warm each arm, in dataset order.

    Deterministic rather than sampled, so two runs of the protocol warm exactly
    the same queries. Warm-up pays the ONNX session's first-call cost and
    establishes the HTTPS connection the embedding provider reuses.

    It does **not** remove network latency from the dense and hybrid arms.
    `AICreditsEmbeddingModel` has no query cache -- `EmbeddingCache` belongs to
    the offline chunk-embedding build pipeline and is never consulted during
    retrieval -- so every query embedding is a live `text-embedding-3-large` call.
    That call is inside `latency_ms` for those arms, which is why the per-stage
    breakdown in `StrategyCostSummary` reports `query_embedding_latency_ms`:
    the end-to-end number and the retrieval compute are different quantities,
    and only a reader can decide which belongs in a frozen cost table.
    """
    if count < 0:
        raise ConfigurationError(f"warm-up count must be >= 0, got {count}")
    if count > len(examples):
        raise ConfigurationError(
            f"cannot warm up with {count} queries from a dataset of {len(examples)}"
        )
    return list(examples[:count])


def rotated(items: Sequence[Any], offset: int) -> list[Any]:
    """`items` rotated left by `offset`, wrapping around.

    Offset is taken modulo the length, so a caller rotating by the repetition
    index gets a valid order for any number of repetitions, including more
    repetitions than items -- the order then repeats, which is the correct
    behaviour rather than an error.
    """
    if not items:
        return []
    shift = offset % len(items)
    return list(items[shift:]) + list(items[:shift])


def arm_schedule(arms: Sequence[str], repetitions: int) -> list[list[str]]:
    """Per-repetition arm order, rotated so no arm holds a fixed position.

    With `len(arms)` arms and at least `len(arms)` repetitions, every arm
    occupies every position exactly once. That is the property the schedule
    exists to provide: an arm that always runs first pays for the machine
    warming up, and one that always runs last pays for whatever else the machine
    was doing. Both are systematic biases, and both would land in the cost table
    as real differences between strategies.
    """
    if repetitions <= 0:
        raise ConfigurationError(f"repetitions must be > 0, got {repetitions}")
    return [rotated(arms, rep) for rep in range(repetitions)]


def sample_order(examples: Sequence[Any], repetition: int) -> list[Any]:
    """Query order for one repetition, rotated by the repetition index.

    Complements `arm_schedule`: it prevents first-position effects within an
    arm's pass from loading onto the same queries every time.
    """
    return rotated(examples, repetition)


# --- aggregation --------------------------------------------------------------


def _stage_medians(samples: Sequence[CostSample]) -> dict[str, float | None]:
    """Median of each stage clock over the samples that reported it.

    A clock absent from some samples is summarised over the samples that have
    it, and the count is left implicit in the raw data rather than padded with
    zeros -- a zero would pull the median down and invent compute that never
    happened.
    """
    keys: list[str] = []
    for sample in samples:
        for key in sample.stage_latency_ms:
            if key not in keys:
                keys.append(key)
    medians: dict[str, float | None] = {}
    for key in keys:
        values = [
            sample.stage_latency_ms[key]
            for sample in samples
            if sample.stage_latency_ms.get(key) is not None
        ]
        medians[key] = statistics.median(values) if values else None
    return medians


def aggregate_samples(
    samples: Iterable[CostSample], *, strategy: str
) -> StrategyCostSummary:
    """Aggregate every sample for one strategy into a distribution summary.

    `latency_ms` is the arm's end-to-end retrieval latency -- the same clock the
    Phase 5 seed table recorded, and the one the router's cost term is meant to
    represent. Per-stage clocks stay on the raw samples and are summarised
    separately so the end-to-end figure can be decomposed.
    """
    owned: list[CostSample] = []
    values: list[float] = []
    fallback = 0
    for sample in samples:
        if sample.strategy != strategy:
            raise ConfigurationError(
                f"sample for {sample.strategy!r} passed to aggregate_samples("
                f"{strategy!r})"
            )
        owned.append(sample)
        values.append(sample.latency_ms)
        if sample.rerank_fallback:
            fallback += 1

    if not values:
        return StrategyCostSummary(
            strategy=strategy,
            n=0,
            median_ms=None,
            p95_ms=None,
            mean_ms=None,
            stdev_ms=None,
            min_ms=None,
            max_ms=None,
            n_fallback=fallback,
            raw_ms=[],
        )

    return StrategyCostSummary(
        strategy=strategy,
        n=len(values),
        median_ms=statistics.median(values),
        p95_ms=percentile(values, 95),
        mean_ms=mean(values),
        stdev_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
        min_ms=min(values),
        max_ms=max(values),
        n_fallback=fallback,
        stage_median_ms=_stage_medians(owned),
        raw_ms=list(values),
    )


def summarize_all(samples: Sequence[CostSample]) -> list[StrategyCostSummary]:
    """Aggregate every strategy present in `samples`, in `COST_STRATEGIES` order.

    Filters per strategy rather than passing the whole collection down:
    `aggregate_samples` rejects foreign samples precisely so a mixed collection
    cannot be mis-reported as one strategy's distribution, so this is the one
    place that has to do the splitting.
    """
    return [
        aggregate_samples(
            [sample for sample in samples if sample.strategy == strategy],
            strategy=strategy,
        )
        for strategy in COST_STRATEGIES
    ]


# --- proposed table -----------------------------------------------------------


def proposed_cost_table(
    summaries: Sequence[StrategyCostSummary],
) -> dict[str, float]:
    """The table proposed for freezing: each strategy's median, rounded to 2 dp.

    Raises rather than guessing when the sweep did not produce a usable table:
    a strategy with no samples has no median, and a strategy with fallback
    samples has a median that does not describe a real rerank. Either would
    silently change the router's cost term, so both are hard failures the
    reviewer has to resolve.
    """
    by_name = {summary.strategy: summary for summary in summaries}
    unexpected = sorted(set(by_name) - set(COST_STRATEGIES))
    if unexpected:
        raise ConfigurationError(
            "strategy_cost_ms covers "
            f"{list(COST_STRATEGIES)}, not {unexpected}. The adaptive arm is "
            "excluded because its cost is a per-query mixture of these "
            "strategies, not a per-strategy cost."
        )

    missing = [name for name in COST_STRATEGIES if name not in by_name]
    if missing:
        raise ConfigurationError(
            f"no measurements for {missing}; the cost table would be incomplete"
        )
    empty = [name for name in COST_STRATEGIES if by_name[name].n == 0]
    if empty:
        raise ConfigurationError(
            f"no measurements for {empty}; the cost table would be incomplete"
        )

    degraded = [s.strategy for s in summaries if s.n_fallback]
    if degraded:
        raise ConfigurationError(
            f"{degraded} recorded rerank fallback samples. A fallback reports a "
            "near-zero rerank latency that does not describe the cross-encoder, "
            "so its median would understate the cost. Resolve before freezing."
        )

    return {
        name: round(float(by_name[name].median_ms), 2)  # type: ignore[arg-type]
        for name in COST_STRATEGIES
    }


def compare_cost_tables(
    current: Mapping[str, float], proposed: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Row-per-strategy comparison of the seed table against the proposed one.

    Reported side by side and never written back over the current values. The
    comparison is the point: a table that changed a lot is a finding about the
    sweep, not a diff to be smoothed over.
    """
    rows: list[dict[str, Any]] = []
    for name in COST_STRATEGIES:
        old = float(current.get(name, 0.0))
        new = float(proposed[name])
        rows.append(
            {
                "strategy": name,
                "current_ms": old,
                "proposed_ms": new,
                "ratio_current": round(old / max(proposed.values()), 4) if max(proposed.values()) else None,
                "ratio_proposed": round(new / max(proposed.values()), 4) if max(proposed.values()) else None,
                "delta_ms": round(new - old, 2),
                "changed": abs(new - old) > 0.005,
            }
        )
    return rows


def build_artifact(
    summaries: Sequence[StrategyCostSummary],
    *,
    proposed: Mapping[str, float] | None,
    current: Mapping[str, float],
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    warmup: int = DEFAULT_WARMUP_QUERIES,
    repetitions: int = DEFAULT_REPETITIONS,
    adaptive: Mapping[str, Any] | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    """Assemble the `strategy_cost_ms.json` artifact.

    `proposed` is `None` when the sweep did not cover every costed strategy, and
    `incomplete_reason` then says why. The comparison and the proposed table are
    omitted in that case rather than filled in from whichever arms did run: a
    partial table would normalise against the wrong maximum and quietly produce
    cost ratios the router would act on.

    Raw samples travel with the artifact so every aggregate here can be
    recomputed, or recomputed differently, without re-running a ~40 minute
    sweep. `adaptive` is reported as its own observation and never folded into
    `proposed`.
    """
    complete = proposed is not None
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "complete": complete,
        "incomplete_reason": incomplete_reason,
        "protocol": {
            "warmup_queries": warmup,
            "repetitions": repetitions,
            "arm_order": "rotated per repetition",
            "query_order": "rotated per repetition",
            "estimator": "statistics.median over per-query retrieval latency_ms",
            "p95_definition": (
                "nearest-rank (evaluation.base.percentile, no interpolation): at "
                "n samples the p95 is the ceil(0.95*n)-th smallest observed value"
            ),
            "stdev_definition": "sample standard deviation, ddof=1",
            "excluded_from_table": ["adaptive"],
        },
        "provenance": dict(provenance),
        "environment": dict(environment),
        "strategies": [summary.model_dump(mode="json") for summary in summaries],
        "proposed_strategy_cost_ms": dict(proposed) if complete else None,
        "current_strategy_cost_ms": dict(current),
        "comparison": compare_cost_tables(current, proposed) if complete else None,
        "adaptive_observed": dict(adaptive) if adaptive else None,
        "notes": [
            "Median understates worst-case cost; compare against p95_ms when "
            "reasoning about the expensive arm the router may avoid.",
            "Latency is hardware- and load-dependent. This table describes this "
            "machine under this protocol and is not portable.",
        ],
    }