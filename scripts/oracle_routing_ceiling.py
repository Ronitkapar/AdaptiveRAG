#!/usr/bin/env python3
"""
oracle_routing_ceiling
----------------------
The ceiling for query-characteristic routing, measured offline from arms that
already exist.

Phase 7 concluded `adaptive == hybrid` and left that result ambiguous between
two very different explanations:

1. routing by query characteristic is worthless on this corpus; or
2. the router *never routes*, so the equivalence was forced before routing was
   ever tested.

Explanation 2 is the live one: the initial pick is `hybrid` on 104 of 107
queries, the sufficiency gate never fires at threshold 0.5, and the escalation
ladder terminates on `hybrid_rerank` -- so escalating *from* hybrid is a
recorded no-op (`adaptive.py`: `next_strategy("hybrid")` is `None`). The ladder
also orders `bm25 < dense < hybrid`, so escalation can never climb *toward*
`dense`, which is the strategy with the best measured recall@5. A near-null
re-run would therefore measure the mechanism, not the idea.

This script answers the prior question instead: **what is the best any
query-characteristic router built from these five strategies could do?** That is
an offline join over the five clean E1 arms, which already contain every
strategy's per-query outcome. No index, no network, no credentials.

Four things are computed per corpus arm:

* **Oracle quality ceiling** -- per-query `argmax_s recall(q, s)`, ties broken
  by lowest latency. The upper bound on what *any* per-query selection over
  these strategies can reach, using a signal that does not exist.
* **Best fixed** -- the single strategy with the highest mean recall@5. This,
  not `hybrid`, is the bar routing must clear. Phase 7 framed its comparison
  against `hybrid` because `hybrid` is what shipped; the ceiling asks whether
  routing beats the best *available* strategy.
* **Quality/latency frontier** -- for each epsilon, pick the cheapest strategy
  within epsilon recall@5 of that query's own best. This is the realistic
  routing prize: something cheap when it suffices, the expensive strategy only
  where it does not.
* **Detectability** -- split queries by whether the reference strategy is
  *strictly needed* (nothing else within epsilon) or merely *good enough*, and
  test whether the shipped `sufficiency_score`, `coverage`, `category`, or the
  recorded sufficiency signals separate the two. Without this the oracle number
  is unbankable: a ceiling no query-time signal can see is not headroom.

Two honesty constraints are load-bearing and are stated in the artifact too:

* The **decision gate is pre-registered** as module constants and copied into
  the artifact before any number is read. The oracle and the frontier are
  post-hoc by construction -- they mean something only against thresholds fixed
  in advance.
* The oracle **cannot** underperform best-fixed on recall@5. That is asserted,
  not assumed: if it fails the join is wrong and the run aborts.

`ndcg_at_5` is excluded throughout. The Step 4 gold-label audit found 47 of 140
section labels spliced and 40 unresolvable on the fixed corpus, so an nDCG
number here would compare two corrupt label sets (`docs/phase-8-results.md` §0).
Every metric used keys on `relevant_documents` only.

The `before` corpus arm is a replication of the `after` arm, which is the
shipping corpus. A conclusion holding on only one corpus is reported as such
rather than pooled away.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
import compare_phase8_arms  # noqa: E402
from compare_phase8_arms import ContaminatedArm, load_arm  # noqa: E402

RESULT_VERSION = "oracle_routing_ceiling_v1"

# ---------------------------------------------------------------------------
# Pre-registered decision gate.
#
# Fixed here, in code, before any number is computed, and copied verbatim into
# the artifact. The oracle and the frontier are post-hoc by construction; only
# these thresholds may not move after seeing a result.
# ---------------------------------------------------------------------------

#: Recall@5 the oracle must beat best-fixed by before routing counts as having
#: *quality* headroom.
QUALITY_HEADROOM_DELTA = 0.02

#: Mean-latency saving (ms, negative) the frontier at GATE_EPSILON must produce
#: before routing counts as having *latency* headroom.
LATENCY_HEADROOM_SAVE_MS = -100.0

#: A needed-vs-sufficient split this predictable means the shipped signal could
#: plausibly act on it. Below this separation the ceiling is unreachable in
#: practice however large it is.
DETECTABILITY_ALPHA = 0.05
DETECTABILITY_MIN_EFFECT = 0.05

#: "Routing is exhausted" requires *both* of these -- a small quality delta AND
#: a small latency saving.
EXHAUSTED_QUALITY_DELTA = 0.01
EXHAUSTED_LATENCY_SAVE_MS = -50.0

#: Frontier tolerances. 0.01 is the gate epsilon; the others bound how sharp the
#: prize is.
FRONTIER_EPSILONS: tuple[float, ...] = (0.005, 0.01, 0.02)
GATE_EPSILON = 0.01

#: Metrics the oracle aggregates. ndcg_at_5 is deliberately absent.
ORACLE_METRICS: tuple[str, ...] = ("recall_at_5", "mrr", "hit_at_5")
LATENCY_METRIC = "total_latency_ms"

#: Permutation defaults. Deterministic and seeded; scipy is deliberately not a
#: project dependency (`tests/test_stats.py` records that).
PERMUTATION_RESAMPLES = 10_000
PERMUTATION_SEED = 20250101

DEFAULT_SUITES: dict[str, Path] = {
    "phase8_after": REPO_ROOT / "experiments" / "phase8" / "combined" / "p8a_e1",
    "phase8_before": REPO_ROOT / "experiments" / "phase8" / "combined" / "p8b_e1",
}

#: The arm whose traces supply the query-time signals for detectability.
ADAPTIVE_SYSTEM = "adaptive"


class QuerySetMismatch(RuntimeError):
    """The arms do not cover an identical set of `query_id`s."""


class CeilingViolation(RuntimeError):
    """The oracle did not dominate best-fixed, so the join or selection is wrong."""


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Index one arm's rows by `query_id`, rejecting duplicates."""
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id"))
        if query_id in indexed:
            raise QuerySetMismatch(
                f"query_id {query_id!r} appears twice in one arm; the row export "
                "is not one row per query, and every per-query join below is invalid"
            )
        indexed[query_id] = row
    return indexed


def assert_same_queries(
    indexed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    systems: Sequence[str],
) -> None:
    """Abort unless every arm covers exactly the same `query_id` set.

    Not a formality. The Phase 8 contamination incident was precisely a
    query-set mismatch: an arm that silently lost embedding calls reported
    `n=59` metrics as though they were `n=107`, and the resulting number inverted
    the reranker's headline result. A per-query oracle over a ragged join would
    compare a different question set per strategy, and no aggregate of it would
    look wrong.
    """
    reference = systems[0]
    reference_ids = set(indexed[reference])
    if not reference_ids:
        raise QuerySetMismatch(f"arm {reference!r} has no rows")
    for system in systems[1:]:
        ids = set(indexed[system])
        if ids == reference_ids:
            continue
        raise QuerySetMismatch(
            f"{system!r} covers {len(ids)} queries and {reference!r} covers "
            f"{len(reference_ids)}; only in {reference!r}: "
            f"{sorted(reference_ids - ids)[:5]}; only in {system!r}: "
            f"{sorted(ids - reference_ids)[:5]}. A per-query oracle needs every "
            "arm on the same queries; re-run the short arm."
        )


def build_panel(
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    systems: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Join the arms into `{query_id: {system: {metric: value}}}`.

    Every arm is indexed and the query sets asserted equal *before* the join is
    built, so a mismatch cannot produce a partial panel that reads as complete.
    """
    indexed = {system: index_rows(rows) for system, rows in arm_rows.items()}
    missing = [system for system in systems if system not in indexed]
    if missing:
        raise QuerySetMismatch(f"no rows loaded for {missing}")
    assert_same_queries(indexed, systems)

    panel: dict[str, dict[str, dict[str, Any]]] = {}
    for query_id in sorted(indexed[systems[0]]):
        panel[query_id] = {
            system: {
                metric: indexed[system][query_id].get(metric)
                for metric in ORACLE_METRICS
            }
            for system in systems
        }
    return panel


def panel_latencies(
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    systems: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Latency per `{query_id: {system: ms}}`, indexed separately from quality.

    Latency lives in its own index so the quality panel stays a pure quality
    join: a null latency must not be able to silently drop a query from the
    ceiling, which is the more important of the two numbers.
    """
    indexed = {system: index_rows(rows) for system, rows in arm_rows.items()}
    assert_same_queries(indexed, systems)
    out: dict[str, dict[str, float]] = {}
    for query_id in sorted(indexed[systems[0]]):
        out[query_id] = {}
        for system in systems:
            value = indexed[system][query_id].get(LATENCY_METRIC)
            if value is None:
                raise QuerySetMismatch(
                    f"{system!r} has no {LATENCY_METRIC} for {query_id!r}; latency "
                    "is a first-class output here, so a null cannot be skipped "
                    "without changing what the frontier means"
                )
            out[query_id][system] = float(value)
    return out


def attach_latency(
    panel: Mapping[str, Mapping[str, Mapping[str, Any]]],
    latencies: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fold the latency index into the quality panel for the selection rules.

    `oracle_pick` and `frontier_pick` need quality and latency in one lookup
    because both compare across systems within a single query. The separation
    stays useful upstream: the panel is built and validated without latency, so a
    null latency aborts before any selection is attempted.
    """
    return {
        query_id: {
            system: {**systems_map[system], LATENCY_METRIC: latencies[query_id][system]}
            for system in systems_map
        }
        for query_id, systems_map in panel.items()
    }


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------


def oracle_pick(
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    systems: Sequence[str],
) -> dict[str, str]:
    """Per-query `argmax` recall@5, ties broken by lowest latency.

    The tie-break is latency rather than a fixed name so the oracle is
    deterministic without privileging any arm, and so a query where every
    strategy ties is answered the way a latency-aware router would answer it.
    """
    picks: dict[str, str] = {}
    for query_id, systems_map in per_query.items():
        best: tuple[float, float, str] | None = None
        for system in systems:
            candidate = (
                -float(systems_map[system]["recall_at_5"]),
                float(systems_map[system][LATENCY_METRIC]),
                system,
            )
            if best is None or candidate < best:
                best = candidate
        assert best is not None, "at least one system is required"
        picks[query_id] = best[2]
    return picks


def frontier_pick(
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    systems: Sequence[str],
    epsilon: float,
) -> dict[str, str]:
    """Cheapest strategy within `epsilon` recall@5 of that query's own best.

    This is the realistic routing prize: something cheap when it is good enough,
    the expensive strategy only where it is not. The candidate set is always
    non-empty, because the per-query argmax is trivially within epsilon of
    itself -- so the frontier can never raise recall@5 above the oracle's.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    picks: dict[str, str] = {}
    for query_id, systems_map in per_query.items():
        best_recall = max(float(systems_map[s]["recall_at_5"]) for s in systems)
        threshold = best_recall - epsilon
        candidates = [
            s for s in systems if float(systems_map[s]["recall_at_5"]) >= threshold
        ]
        picks[query_id] = min(
            candidates, key=lambda s: (float(systems_map[s][LATENCY_METRIC]), s)
        )
    return picks


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(
    picks: Mapping[str, str],
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    latencies: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Mean quality and latency of one per-query selection."""
    summary: dict[str, Any] = {"n_queries": len(picks)}
    for metric in ORACLE_METRICS:
        values = [float(per_query[q][picks[q]][metric]) for q in sorted(picks)]
        summary[f"mean_{metric}"] = round(float(np.mean(values)), 6)
    lat_values = [float(latencies[q][picks[q]]) for q in sorted(picks)]
    summary["mean_total_latency_ms"] = round(float(np.mean(lat_values)), 4)
    summary["median_total_latency_ms"] = round(float(np.median(lat_values)), 4)
    summary["strategy_counts"] = dict(sorted(Counter(picks.values()).items()))
    return summary


def fixed_summaries(
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    latencies: Mapping[str, Mapping[str, float]],
    systems: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Per-strategy always-that-strategy aggregates: the fixed-policy baselines."""
    query_ids = sorted(per_query)
    return {
        system: summarise({q: system for q in query_ids}, per_query, latencies)
        for system in systems
    }


def best_fixed(fixed: Mapping[str, Mapping[str, Any]]) -> str:
    """The fixed strategy with the highest mean recall@5.

    Ties resolve on mean MRR, then on the strategy name, so the choice is
    deterministic and does not depend on dict ordering.
    """
    return min(
        fixed,
        key=lambda s: (
            -float(fixed[s]["mean_recall_at_5"]),
            -float(fixed[s]["mean_mrr"]),
            s,
        ),
    )


def oracle_gain_concentration(
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    systems: Sequence[str],
    reference: str,
    fixed_recall: float,
) -> dict[str, Any]:
    """How many queries actually carry the oracle's gain over the best fixed?

    This exists because the ceiling number is fragile in a way its own precision
    hides. On this benchmark `recall_at_5` is near-binary -- 92 of the 107
    queries carry exactly one relevant document, so recall can only be 0.0 or
    1.0 for them. Five strategies therefore *tie* at the top on 98 of 107
    queries, and a latency tie-break sends all of them to bm25. The oracle's
    mean recall@5 advantage is then the average of a handful of full-1.0 wins,
    not a broad per-query improvement, and a threshold comparison like
    `oracle - best_fixed > 0.02` can be cleared or missed by two or three
    queries changing sides.

    So the gain is reported three ways: how many queries contribute any of it,
    how concentrated the total mass is, and how many queries uniquely attain
    the maximum at all. A ceiling carried by <5% of the benchmark is an upper
    bound worth reading as "a handful of queries", not as a +0.02 effect.
    """
    contributors: list[dict[str, Any]] = []
    total_mass = 0.0
    for query_id, systems_map in per_query.items():
        best_recall = max(float(systems_map[s]["recall_at_5"]) for s in systems)
        winners = [s for s in systems if float(systems_map[s]["recall_at_5"]) == best_recall]
        gain = best_recall - float(systems_map[reference]["recall_at_5"])
        if gain > 0:
            total_mass += gain
            contributors.append(
                {
                    "query_id": query_id,
                    "gain": round(gain, 6),
                    "unique_winner": winners[0] if len(winners) == 1 else None,
                    "n_tied_at_max": len(winners),
                }
            )
    contributors.sort(key=lambda c: (-c["gain"], c["query_id"]))

    distinct = sorted(
        {
            round(float(per_query[q][reference]["recall_at_5"]), 6)
            for q in per_query
        }
    )
    n_queries = len(per_query)
    unique_max = sum(1 for c in contributors if c["unique_winner"])
    return {
        "n_queries": n_queries,
        "n_queries_contributing_gain": len(contributors),
        "share_of_queries_contributing": round(len(contributors) / n_queries, 4),
        "total_gain_mass": round(total_mass, 6),
        "top_contributors": contributors[:10],
        "n_queries_with_a_unique_max": unique_max,
        "distinct_recall_at_5_values": distinct,
        "n_distinct_recall_values": len(distinct),
        "mean_gain_per_contributing_query": (
            round(total_mass / len(contributors), 6) if contributors else 0.0
        ),
        "caveat": (
            "recall_at_5 is near-binary on this benchmark (most queries carry a "
            "single relevant document), so strategies tie at the maximum on most "
            "queries and the oracle's mean advantage is the average of a few "
            "full-1.0 wins rather than a broad improvement. Read the delta "
            "against n_queries_contributing_gain, not on its own."
        ),
    }


# ---------------------------------------------------------------------------
# Detectability
# ---------------------------------------------------------------------------


def needed_split(
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    systems: Sequence[str],
    reference: str,
    epsilon: float,
) -> dict[str, str]:
    """Label each query ``needed`` or ``sufficient`` for the reference strategy.

    ``needed`` means *nothing else* comes within `epsilon` recall@5 of the
    reference -- the reference is the only acceptable answer. ``sufficient``
    means a cheaper alternative is already good enough, which is exactly the
    state in which a router could save latency. A query where several strategies
    tie for best is ``sufficient``: there is a choice to be made at all, and
    paying for the reference on it is the waste this study measures.
    """
    labels: dict[str, str] = {}
    for query_id, systems_map in per_query.items():
        reference_recall = float(systems_map[reference]["recall_at_5"])
        alternatives = [s for s in systems if s != reference]
        if any(
            float(systems_map[s]["recall_at_5"]) >= reference_recall - epsilon
            for s in alternatives
        ):
            labels[query_id] = "sufficient"
        else:
            labels[query_id] = "needed"
    return labels


def permutation_p_value(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    n_resamples: int = PERMUTATION_RESAMPLES,
    seed: int = PERMUTATION_SEED,
    alpha: float = DETECTABILITY_ALPHA,
) -> dict[str, Any]:
    """Two-sided permutation test on the difference between two group means.

    The null is label exchangeability, which is the right null here: the
    needed/sufficient split is *defined* by quality, and the question is whether
    the sufficiency signal carries information about it beyond that definition.

    The observed difference counts toward the reference distribution, the
    conservative choice -- it keeps small-n p-values from being under-reported.
    """
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    if a.size == 0 or b.size == 0:
        return {
            "n_a": int(a.size),
            "n_b": int(b.size),
            "observed_difference": None,
            "p_value": None,
            "significant": False,
            "note": "one group is empty; no separation can be estimated",
        }

    observed = float(b.mean() - a.mean())
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        rng.shuffle(pooled)
        replicate = float(pooled[n_a:].mean() - pooled[:n_a].mean())
        if abs(replicate) >= abs(observed) - 1e-12:
            count += 1
    p_value = (count + 1) / (n_resamples + 1)
    return {
        "n_a": int(n_a),
        "n_b": int(b.size),
        "observed_difference": round(observed, 6),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def _chi2(observed: np.ndarray) -> float:
    """Pearson chi-square of `observed` against its independence expectation."""
    grand = observed.sum()
    if grand == 0:
        return 0.0
    row_totals = observed.sum(axis=1, keepdims=True)
    expected = row_totals * observed.sum(axis=0, keepdims=True) / grand
    denominator = np.where(expected == 0, 1.0, expected)
    return float(np.sum((observed - expected) ** 2 / denominator))


def contingency_permutation_p(
    table: Mapping[str, Mapping[str, int]],
    *,
    n_resamples: int = PERMUTATION_RESAMPLES,
    seed: int = PERMUTATION_SEED,
    alpha: float = DETECTABILITY_ALPHA,
) -> dict[str, Any]:
    """Permutation test of association in a `{group: {label: count}}` table.

    The null is "labels are exchangeable across groups, given the group sizes",
    implemented by pooling every observation, shuffling the label vector, and
    splitting it back at the original group sizes. That is the exact conditional
    null for association and needs no chi-square approximation at these counts.

    Permuting *within each row* instead -- the tempting shortcut -- is not the
    same test and has almost no power on a 2x2 table: a 2x2 table has only four
    row-internal rearrangements, two of which reproduce the observed statistic,
    so the attainable p-value is floored near 0.5 no matter how strong the
    association. Pooling and re-splitting gives the full permutation set and a
    p-value that can actually approach zero.
    """
    groups = sorted(table)
    if len(groups) < 2:
        return {
            "p_value": None,
            "significant": False,
            "note": "fewer than two groups; association is undefined",
        }
    labels = sorted({label for group in groups for label in table[group]})
    observed = np.array(
        [[table[g].get(label, 0) for label in labels] for g in groups], dtype=float
    )
    group_sizes = [int(total) for total in observed.sum(axis=1)]
    if min(group_sizes) == 0:
        return {
            "groups": groups,
            "labels": labels,
            "table": {g: dict(sorted(table[g].items())) for g in groups},
            "p_value": None,
            "significant": False,
            "note": "a group has no observations; association is undefined",
        }

    # One entry per observation, carrying its label.
    pooled = np.concatenate(
        [np.repeat(label_index, int(observed[g, label_index]))
         for g in range(len(groups))
         for label_index in range(len(labels))]
    )
    observed_stat = _chi2(observed)

    rng = np.random.default_rng(seed)
    count = 0
    edges = np.cumsum(group_sizes)[:-1]
    for _ in range(n_resamples):
        rng.shuffle(pooled)
        shuffled = np.array(
            [
                np.bincount(chunk, minlength=len(labels)).astype(float)
                for chunk in np.split(pooled, edges)
            ],
            dtype=float,
        )
        if _chi2(shuffled) >= observed_stat - 1e-12:
            count += 1
    p_value = (count + 1) / (n_resamples + 1)
    return {
        "groups": groups,
        "labels": labels,
        "table": {g: dict(sorted(table[g].items())) for g in groups},
        "chi2": round(observed_stat, 6),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def detectability(
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    latencies: Mapping[str, Mapping[str, float]],
    signals: Mapping[str, Mapping[str, Any]],
    systems: Sequence[str],
    reference: str,
    epsilon: float,
) -> dict[str, Any]:
    """Can the shipped query-time signals tell needed from sufficient?

    Three layers, each assuming less than the last:

    1. **Signal separation** -- do the recorded `sufficiency_score` and
       `coverage` differ between the two groups? Permutation test on the means.
    2. **Category association** -- is the split predictable from `category`?
       Contingency permutation test.
    3. **Recorded-signal association** -- the same, per sufficiency signal's
       pass/fail flag, read from `routing.sufficiency.signals`.

    A large oracle delta with no layer reaching `DETECTABILITY_ALPHA` is a
    ceiling no router on these signals can bank, and the gate treats it as such.
    Strategy counts are reported either way: which strategies a latency-cheap
    oracle would pick is the concrete prize, independent of whether any test
    reached significance.
    """
    labels = needed_split(per_query, systems, reference, epsilon)
    needed = [q for q in sorted(labels) if labels[q] == "needed"]
    sufficient = [q for q in sorted(labels) if labels[q] == "sufficient"]

    def _values(query_ids: Sequence[str], field: str) -> list[float]:
        return [
            float(signals[q][field])
            for q in query_ids
            if q in signals and signals[q].get(field) is not None
        ]

    def _mean(values: Sequence[float]) -> float | None:
        return round(float(np.mean(values)), 6) if values else None

    score_test = permutation_p_value(
        _values(needed, "sufficiency_score"), _values(sufficient, "sufficiency_score")
    )
    coverage_test = permutation_p_value(
        _values(needed, "coverage"), _values(sufficient, "coverage")
    )

    category_table: dict[str, dict[str, int]] = defaultdict(
        lambda: {"needed": 0, "sufficient": 0}
    )
    for query_id in sorted(labels):
        category = str(signals.get(query_id, {}).get("category", "unknown"))
        category_table[category][labels[query_id]] += 1
    category_test = contingency_permutation_p(dict(category_table))

    signal_tests: dict[str, Any] = {}
    all_names = sorted(
        {name for q in signals for name in signals[q].get("passed_signals", [])}
    )
    for name in all_names:
        table: dict[str, dict[str, int]] = defaultdict(
            lambda: {"needed": 0, "sufficient": 0}
        )
        for query_id in sorted(labels):
            passed = name in set(signals.get(query_id, {}).get("passed_signals", []))
            table["passed" if passed else "failed"][labels[query_id]] += 1
        test = contingency_permutation_p(dict(table))
        test["signal"] = name
        signal_tests[name] = test

    tests = [score_test, coverage_test, category_test, *signal_tests.values()]
    significant = [t for t in tests if t.get("significant")]
    separations = [
        abs(float(t["observed_difference"]))
        for t in (score_test, coverage_test)
        if t.get("observed_difference") is not None
    ]
    max_effect = max(separations, default=0.0)

    def _reference_latency(query_ids: Sequence[str]) -> float | None:
        """Mean latency of the reference strategy in one group.

        Direction matters: if the reference is not markedly more expensive
        where it is needed, even a perfect oracle buys little latency.
        """
        values = [float(latencies[q][reference]) for q in query_ids]
        return round(float(np.mean(values)), 4) if values else None

    return {
        "reference_strategy": reference,
        "epsilon": epsilon,
        "n_needed": len(needed),
        "n_sufficient": len(sufficient),
        "sufficiency_score": {
            "mean_needed": _mean(_values(needed, "sufficiency_score")),
            "mean_sufficient": _mean(_values(sufficient, "sufficiency_score")),
            "test": score_test,
        },
        "coverage": {
            "mean_needed": _mean(_values(needed, "coverage")),
            "mean_sufficient": _mean(_values(sufficient, "coverage")),
            "test": coverage_test,
        },
        "category_association": category_test,
        "sufficiency_signal_association": signal_tests,
        "detectable": bool(significant),
        "significant_tests": [
            str(t.get("signal", "score_coverage_or_category")) for t in significant
        ],
        "max_mean_separation": round(max_effect, 6),
        "meets_min_effect": bool(max_effect >= DETECTABILITY_MIN_EFFECT),
        "reference_mean_latency_ms": {
            "needed": _reference_latency(needed),
            "sufficient": _reference_latency(sufficient),
        },
        "oracle_strategy_counts": dict(
            sorted(Counter(oracle_pick(per_query, systems).values()).items())
        ),
        "frontier_strategy_counts": dict(
            sorted(Counter(frontier_pick(per_query, systems, epsilon).values()).items())
        ),
    }


def read_adaptive_signals(traces_path: Path) -> dict[str, dict[str, Any]]:
    """Read the query-time signals the router actually recorded, by `query_id`.

    The plan pointed at `routing.metadata.signals`; the traces store them at
    `routing.sufficiency.signals` as a list of `{name, passed, value, weight}`.
    Both the score/coverage scalars and the per-signal pass/fail flags are read,
    so detectability is tested against the shipped mechanism rather than a
    reimplementation of it.
    """
    if not traces_path.is_file():
        raise FileNotFoundError(f"no traces at {traces_path}")

    signals: dict[str, dict[str, Any]] = {}
    with open(traces_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            routing = trace.get("routing") or {}
            sufficiency = routing.get("sufficiency") or {}
            query_id = str(trace.get("example_id") or trace.get("trace_id"))
            signals[query_id] = {
                "sufficiency_score": sufficiency.get("score"),
                "coverage": sufficiency.get("coverage"),
                "top1_coverage": sufficiency.get("top1_coverage"),
                "sufficient": sufficiency.get("sufficient"),
                "passed_signals": [
                    str(s.get("name"))
                    for s in sufficiency.get("signals", [])
                    if s.get("passed")
                ],
                "initial_strategy": routing.get("initial_strategy"),
                "final_strategy": routing.get("final_strategy"),
                "question_type": (routing.get("features") or {}).get("question_type"),
                "category": trace.get("category"),
            }
    return signals


def cross_tabs(
    picks: Mapping[str, str],
    signals: Mapping[str, Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, int]]:
    """Counts of chosen strategy by a query attribute (`category`, ...)."""
    table: dict[str, dict[str, int]] = defaultdict(dict)
    for query_id in sorted(picks):
        attribute = str(signals.get(query_id, {}).get(key, "unknown"))
        strategy = picks[query_id]
        table[attribute][strategy] = table[attribute].get(strategy, 0) + 1
    return {name: dict(sorted(v.items())) for name, v in sorted(table.items())}


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------


def evaluate_gate(
    oracle_delta: float,
    frontier_latency_delta_ms: float,
    detectable: bool,
) -> dict[str, Any]:
    """Apply the pre-registered gate. Registered before any number was read.

    The two headroom conditions and the two exhaustion conditions leave a gap by
    construction: a delta of +0.015 recall@5 with a 70 ms saving satisfies
    neither. That band is reported as ``inconclusive`` rather than resolved
    toward whichever verdict is more convenient, because a gate whose middle
    band is quietly rounded to a verdict is not pre-registration.
    """
    quality_headroom = oracle_delta > QUALITY_HEADROOM_DELTA
    latency_headroom = (
        frontier_latency_delta_ms < LATENCY_HEADROOM_SAVE_MS and detectable
    )
    exhausted = (
        oracle_delta <= EXHAUSTED_QUALITY_DELTA
        and frontier_latency_delta_ms > EXHAUSTED_LATENCY_SAVE_MS
    )

    if quality_headroom or latency_headroom:
        verdict = "routing_has_headroom"
    elif exhausted:
        verdict = "routing_exhausted_on_this_benchmark"
    else:
        verdict = "inconclusive"

    return {
        "verdict": verdict,
        "quality_headroom": bool(quality_headroom),
        "latency_headroom": bool(latency_headroom),
        "exhausted": bool(exhausted),
        "observed": {
            "oracle_minus_best_fixed_recall_at_5": round(float(oracle_delta), 6),
            "frontier_epsilon_0_01_minus_best_fixed_latency_ms": round(
                float(frontier_latency_delta_ms), 4
            ),
            "needed_vs_sufficient_detectable": bool(detectable),
        },
    }


def pre_registered_gate() -> dict[str, Any]:
    """The gate as declared, copied into the artifact beside the results."""
    return {
        "registered_before_measurement": True,
        "quality_headroom": {
            "condition": (
                f"oracle recall@5 - best-fixed recall@5 > {QUALITY_HEADROOM_DELTA}"
            ),
            "threshold": QUALITY_HEADROOM_DELTA,
        },
        "latency_headroom": {
            "condition": (
                f"frontier(epsilon={GATE_EPSILON}) mean latency - best-fixed mean "
                f"latency < {LATENCY_HEADROOM_SAVE_MS} ms AND the "
                "needed-vs-sufficient split is predictable from category or the "
                "recorded sufficiency signals"
            ),
            "threshold_ms": LATENCY_HEADROOM_SAVE_MS,
            "requires_detectability": True,
        },
        "exhausted": {
            "condition": (
                f"oracle - best-fixed <= {EXHAUSTED_QUALITY_DELTA} AND "
                f"frontier(epsilon={GATE_EPSILON}) saves < "
                f"{-EXHAUSTED_LATENCY_SAVE_MS} ms"
            ),
            "threshold_recall_at_5": EXHAUSTED_QUALITY_DELTA,
            "threshold_ms": EXHAUSTED_LATENCY_SAVE_MS,
        },
        "detectability": {
            "alpha": DETECTABILITY_ALPHA,
            "min_mean_separation": DETECTABILITY_MIN_EFFECT,
        },
        "frontier_epsilons": list(FRONTIER_EPSILONS),
        "excluded_metrics": {
            "ndcg_at_5": (
                "gold section labels are spliced (47 of 140) and 40 stop "
                "resolving on the fixed corpus; see docs/phase-8-results.md §0"
            )
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _traces_path(suite_root: Path, system: str) -> Path:
    matches = sorted(
        suite_root.glob(f"*__E1_baseline_comparison__{system}/traces.jsonl")
    )
    if not matches:
        raise FileNotFoundError(f"no traces for {system!r} under {suite_root}")
    return matches[0]


def run_arm(
    suite_root: Path,
    systems: Sequence[str],
    *,
    allow_contaminated: bool = False,
) -> dict[str, Any]:
    """Compute every table for one corpus arm."""
    # Reuses the Phase 8 arm loader, so the contamination guard that caught the
    # 13 invalid arms applies here without being restated. A partial arm would
    # make every per-query join below meaningless.
    arm_rows = load_arm(suite_root, systems, allow_contaminated=allow_contaminated)

    panel = build_panel(arm_rows, systems)
    latencies = panel_latencies(arm_rows, systems)
    per_query = attach_latency(panel, latencies)

    fixed = fixed_summaries(per_query, latencies, systems)
    reference = best_fixed(fixed)

    signals = read_adaptive_signals(_traces_path(suite_root, ADAPTIVE_SYSTEM))
    missing = sorted(set(per_query) - set(signals))
    if missing:
        raise QuerySetMismatch(
            f"the {ADAPTIVE_SYSTEM} arm's traces carry no routing signals for "
            f"{len(missing)} of {len(per_query)} queries (e.g. {missing[:5]}); "
            "detectability would silently be measured on a subset"
        )

    oracle_picks = oracle_pick(per_query, systems)
    oracle_summary = summarise(oracle_picks, per_query, latencies)
    best = fixed[reference]
    oracle_delta = float(oracle_summary["mean_recall_at_5"]) - float(
        best["mean_recall_at_5"]
    )

    if oracle_delta < -1e-9:
        raise CeilingViolation(
            f"oracle recall@5 {oracle_summary['mean_recall_at_5']} is below "
            f"best-fixed {best['mean_recall_at_5']} for {reference!r}; a per-query "
            "argmax cannot underperform the best constant choice, so the join or "
            "the selection is wrong and no other number in this run can be trusted"
        )

    frontier: dict[str, Any] = {}
    for epsilon in FRONTIER_EPSILONS:
        picks = frontier_pick(per_query, systems, epsilon)
        summary = summarise(picks, per_query, latencies)
        summary["mean_latency_delta_vs_best_fixed_ms"] = round(
            float(summary["mean_total_latency_ms"])
            - float(best["mean_total_latency_ms"]),
            4,
        )
        summary["mean_recall_delta_vs_best_fixed"] = round(
            float(summary["mean_recall_at_5"]) - float(best["mean_recall_at_5"]), 6
        )
        frontier[f"epsilon_{epsilon:g}"] = summary

    gate_summary = frontier[f"epsilon_{GATE_EPSILON:g}"]
    detection = detectability(
        per_query, latencies, signals, systems, reference, GATE_EPSILON
    )
    concentration = oracle_gain_concentration(
        per_query, systems, reference, float(best["mean_recall_at_5"])
    )

    return {
        "suite_root": str(suite_root),
        "systems": list(systems),
        "n_queries": len(per_query),
        "best_fixed_strategy": reference,
        "oracle": oracle_summary,
        "oracle_gain_concentration": concentration,
        "fixed": fixed,
        "frontier": frontier,
        "detectability": detection,
        "cross_tabs": {
            "oracle_by_category": cross_tabs(oracle_picks, signals, "category"),
            "frontier_by_category": cross_tabs(
                frontier_pick(per_query, systems, GATE_EPSILON), signals, "category"
            ),
            "oracle_by_question_type": cross_tabs(oracle_picks, signals, "question_type"),
            "oracle_by_initial_strategy": cross_tabs(
                oracle_picks, signals, "initial_strategy"
            ),
        },
        "gate": evaluate_gate(
            oracle_delta,
            float(gate_summary["mean_latency_delta_vs_best_fixed_ms"]),
            bool(detection["detectable"]),
        ),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render(result: Mapping[str, Any]) -> str:
    """Human-readable report. Every number here also lands in the JSON artifact."""
    lines: list[str] = ["Oracle routing ceiling", "=" * 60, ""]
    gate = result["pre_registered_gate"]
    lines.append("Pre-registered gate (registered before measurement)")
    lines.append(f"  quality headroom: {gate['quality_headroom']['condition']}")
    lines.append(f"  latency headroom: {gate['latency_headroom']['condition']}")
    lines.append(f"  exhausted:        {gate['exhausted']['condition']}")
    lines.append("")

    for arm_name, arm in result["corpus_arms"].items():
        lines.append(f"[{arm_name}] ({arm['role']}) - {arm['n_queries']} queries")
        lines.append(f"  best fixed strategy: {arm['best_fixed_strategy']}")
        lines.append("")
        header = (
            f"  {'policy':<26}{'recall@5':>10}{'MRR':>10}{'Hit@5':>9}"
            f"{'mean ms':>11}{'median ms':>11}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for system, block in arm["fixed"].items():
            marker = " *" if system == arm["best_fixed_strategy"] else ""
            lines.append(
                f"  {system + marker:<26}{_fmt(block['mean_recall_at_5']):>10}"
                f"{_fmt(block['mean_mrr']):>10}{_fmt(block['mean_hit_at_5']):>9}"
                f"{_fmt(block['mean_total_latency_ms'], 2):>11}"
                f"{_fmt(block['median_total_latency_ms'], 2):>11}"
            )
        policies = [("ORACLE (per-query)", arm["oracle"])] + [
            (key, arm["frontier"][key]) for key in arm["frontier"]
        ]
        for label, block in policies:
            lines.append(
                f"  {label:<26}{_fmt(block['mean_recall_at_5']):>10}"
                f"{_fmt(block['mean_mrr']):>10}{_fmt(block['mean_hit_at_5']):>9}"
                f"{_fmt(block['mean_total_latency_ms'], 2):>11}"
                f"{_fmt(block['median_total_latency_ms'], 2):>11}"
            )
        lines.append("")

        detection = arm["detectability"]
        lines.append(
            f"  detectability (needed vs sufficient, eps={detection['epsilon']}): "
            f"{detection['n_needed']} needed / {detection['n_sufficient']} sufficient, "
            f"detectable={_fmt(detection['detectable'])}"
        )
        for field in ("sufficiency_score", "coverage"):
            block = detection[field]
            test = block["test"] or {}
            lines.append(
                f"    {field:<20} mean {_fmt(block['mean_needed'])} vs "
                f"{_fmt(block['mean_sufficient'])}  p={_fmt(test.get('p_value'))}"
            )
        lines.append(
            "    category association p="
            f"{_fmt((detection['category_association'] or {}).get('p_value'))}"
        )
        for name, test in detection["sufficiency_signal_association"].items():
            lines.append(f"    signal {name:<20} p={_fmt((test or {}).get('p_value'))}")
        lines.append("")

        observed = arm["gate"]["observed"]
        lines.append(f"  VERDICT [{arm_name}]: {arm['gate']['verdict']}")
        lines.append(
            "    oracle - best fixed recall@5 = "
            f"{_fmt(observed['oracle_minus_best_fixed_recall_at_5'])}"
        )
        lines.append(
            "    frontier(0.01) - best fixed latency = "
            f"{_fmt(observed['frontier_epsilon_0_01_minus_best_fixed_latency_ms'], 2)} ms"
        )
        concentration = arm["oracle_gain_concentration"]
        lines.append(
            f"    oracle gain carried by {concentration['n_queries_contributing_gain']}"
            f"/{concentration['n_queries']} queries "
            f"({_fmt(concentration['share_of_queries_contributing'] * 100, 1)}%), "
            f"total mass {_fmt(concentration['total_gain_mass'], 3)}, "
            f"{concentration['n_distinct_recall_values']} distinct recall@5 values"
        )
        if arm["gate"]["verdict"] != "inconclusive":
            lines.append(
                "    NOTE: the delta above clears a threshold that this few "
                "queries can move either way. Read it with the concentration "
                "line and the detectability block, not alone."
            )
        lines.append("")

    lines.append("An oracle is an upper bound computed with a signal that does not")
    lines.append("exist. It bounds routing; it does not evidence it. The verdict")
    lines.append("above means nothing without the detectability block above it.")
    lines.append("")
    lines.append("ndcg_at_5 is excluded: the gold section labels are corrupt")
    lines.append("(docs/phase-8-results.md section 0).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the quality and latency ceiling for query-characteristic "
            "routing, offline, from the clean E1 arms of each corpus arm."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--after-suite",
        type=Path,
        default=DEFAULT_SUITES["phase8_after"],
        help="Suite directory for the phase8_after (shipping corpus) E1 arms.",
    )
    parser.add_argument(
        "--before-suite",
        type=Path,
        default=DEFAULT_SUITES["phase8_before"],
        help="Suite directory for the phase8_before E1 arms (replication).",
    )
    parser.add_argument(
        "--systems",
        default="bm25,dense,hybrid,hybrid_rerank,adaptive",
        help="Comma-separated E1 arms to join, in any order.",
    )
    parser.add_argument(
        "--allow-contaminated",
        action="store_true",
        help=(
            "Proceed even if an arm carries failed traces. A partial arm makes "
            "every per-query join below meaningless; the shortfall is recorded "
            "in the artifact and printed above the results."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "phase8" / "oracle_ceiling.json",
    )
    args = parser.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if len(systems) < 2:
        parser.error("need at least two arms to compare a ceiling against")

    suites = {
        "phase8_after": (args.after_suite, "primary - the shipping corpus"),
        "phase8_before": (args.before_suite, "replication - the pre-fix corpus"),
    }

    corpus_arms: dict[str, Any] = {}
    try:
        for name, (root, role) in suites.items():
            print(f"computing ceiling for {name} from {root} ...", flush=True)
            arm = run_arm(root, systems, allow_contaminated=args.allow_contaminated)
            corpus_arms[name] = {**arm, "role": role}
    except (ContaminatedArm, QuerySetMismatch, CeilingViolation) as exc:
        print(f"aborting: {exc}", file=sys.stderr)
        return 2

    result = {
        "result_version": RESULT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": (
            "Oracle routing ceiling: the best any query-characteristic router "
            "over these strategies could achieve, computed offline from the "
            "clean E1 arms. Offline and deterministic; no index, no API."
        ),
        "systems": systems,
        "metrics": list(ORACLE_METRICS),
        "latency_metric": LATENCY_METRIC,
        "pre_registered_gate": pre_registered_gate(),
        "deviations_from_plan": [
            (
                "The plan pointed detectability at routing.metadata.signals; the "
                "traces store them at routing.sufficiency.signals. The script "
                "reads the path the traces actually use."
            ),
            (
                "The frontier and detectability reference is the best-fixed "
                "strategy rather than a hard-coded 'dense'. The plan expects "
                "best-fixed to be dense; parameterising it stops the reference "
                "from silently disagreeing with the measured winner."
            ),
            (
                "Permutation tests run 10,000 seeded resamples each rather than "
                "being vectorised. Deterministic and offline; costs a few "
                "seconds per arm."
            ),
        ],
        "corpus_arms": corpus_arms,
        "contaminated_arms": compare_phase8_arms.CONTAMINATION.get("arms", []),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["contaminated_arms"]:
        print("CONTAMINATED arms included; the numbers below are NOT arm results:")
        for entry in result["contaminated_arms"]:
            print(
                f"  {entry['system']}: {entry['failed']} of {entry['total']} "
                f"traces failed, metrics over {entry['evaluated']}"
            )
        print()
    print(render(result))
    print(f"\nartifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
