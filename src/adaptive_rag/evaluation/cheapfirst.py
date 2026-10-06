"""Cheap-first need-prediction oracle and threshold policy for Phase 14.

Phase 13 closed the post-dense Dense→Hybrid routing track (Position C) and
selected a new, cheap-first track: after local BM25 retrieval, can
decision-time information available *before the Dense embedding call*
predict whether paying for Dense retrieval is worthwhile?

This module turns already-persisted BM25/dense per-query outcomes into a
deterministic need oracle, sweeps the six pre-registered single-feature
rules over it, and scores the resulting adaptive policy -- all as pure
functions over plain records, so every number is unit-testable without an
index, a provider, or a network.

Pre-routing integrity is structural, not conventional: signal extraction
serves only the six allowlisted pre-Dense scalars (`PRE_DENSE_SIGNALS`)
and refuses anything else, so no dense/hybrid/reranker-derived quantity
can enter a rule even by accident. Outcome fields (dense quality/latency)
exist on the records for *scoring* only and are never readable as signals.

The research contract lives in `docs/phases/phase-14.md`. The constants
below are that contract in code: the oracle rule, the candidate family,
the grid construction, the selection rule, and the random-ablation
protocol may not move after seeing a result.
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from adaptive_rag.evaluation.dispersion import EPSILON

#: Version stamped into every Phase 14 artifact built through this module.
NEED_ORACLE_VERSION = "phase14_cheapfirst_v1"

#: Rank depth of the BM25 single-arm signals. Frozen at the E1 arms'
#: evaluation depth; changing it would silently redefine the rule under test.
SIGNAL_K = 5

#: The six pre-registered candidate signals (docs/phases/phase-14.md §7).
#: `direction` is the primary (mechanism) direction; "high" escalates on
#: values at/above the threshold, "low" on values at/below it. The control
#: direction is the exact opposite and exists so the direction choice is
#: declared before results, not after them.
CANDIDATE_SIGNALS: tuple[dict[str, str], ...] = (
    {"name": "ddr_bm25", "source": "signal_table", "direction": "high"},
    {"name": "bm25_top1", "source": "bm25_trace", "direction": "low"},
    {"name": "bm25_gap", "source": "signal_table", "direction": "low"},
    {"name": "bm25_slope", "source": "signal_table", "direction": "high"},
    {"name": "complexity_score", "source": "adaptive_trace", "direction": "high"},
    {"name": "content_term_count", "source": "adaptive_trace", "direction": "high"},
)

#: The only signal names any Phase 14 rule may read. Anything else --
#: dense/hybrid/reranker-derived quantities, cross-arm agreement, labels --
#: is refused by `signal_value`, which makes the pre-routing ordering a
#: property of the code rather than of reviewer vigilance.
PRE_DENSE_SIGNALS: frozenset[str] = frozenset(
    spec["name"] for spec in CANDIDATE_SIGNALS
)

#: Escalation rates outside this band are degenerate policies, not routing
#: decisions. The selection rule may not pick them.
MIN_ESCALATION_RATE = 0.05
MAX_ESCALATION_RATE = 0.95

#: Random-escalation ablation protocol: fixed-count subsets at the adaptive
#: policy's own rate, this many draws, this seed (== Phase 10's seed, so the
#: ablation is comparable across tracks rather than re-rolled per phase).
RANDOM_ABLATION_DRAWS = 1000
RANDOM_ABLATION_SEED = 20250101


def oracle_label(delta_quality: float, epsilon: float = EPSILON) -> bool:
    """Whether paying for Dense was worth it for one query (frozen oracle).

    Strict improvement at or above EPSILON counts. Ties mean paying an
    embedding call for nothing; negative deltas mean paying for harm.
    Both are NO (stop at BM25).
    """
    return float(delta_quality) >= float(epsilon)


def signal_value(signals: Mapping[str, Any], name: str) -> float | None:
    """Read one pre-Dense signal for a query, refusing anything else.

    `signals` must contain exactly pre-Dense scalars; unknown names raise,
    and non-numeric stored values read as missing (None) rather than
    coercing. Missing values escalate nowhere: the rule maps them to STOP
    (no imputation -- a missing BM25 readout must not spend an embedding
    call by default).
    """
    if name not in PRE_DENSE_SIGNALS:
        raise ValueError(
            f"signal {name!r} is not a pre-Dense signal; Phase 14 rules may "
            f"only read {sorted(PRE_DENSE_SIGNALS)}"
        )
    raw = signals.get(name)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def apply_rule(
    value: float | None, threshold: float, direction: str
) -> bool:
    """The adaptive decision for one query: pay for Dense or stop at BM25."""
    if value is None:
        return False
    if direction == "high":
        return float(value) >= float(threshold)
    if direction == "low":
        return float(value) <= float(threshold)
    raise ValueError(f"unknown rule direction {direction!r}")


def decile_thresholds(values: Sequence[float]) -> list[float]:
    """Label-free threshold grid: deciles of a calibration signal sample.

    Deterministic and unsupervised -- it uses the signal distribution only,
    never the oracle labels, so the grid cannot encode label information.
    Returns at most nine increasing cut points (fewer under ties).
    """
    ordered = sorted(float(v) for v in values if v is not None)
    try:
        ordered = [v for v in ordered if math.isfinite(v)]
    except TypeError:
        return []
    n = len(ordered)
    if n == 0:
        return []
    cuts: list[float] = []
    for decile in range(1, 10):
        rank = decile * n / 10.0
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        lower = max(0, min(n - 1, lower))
        upper = max(0, min(n - 1, upper))
        cut = (ordered[lower] + ordered[upper]) / 2.0
        if not cuts or cut > cuts[-1]:
            cuts.append(float(cut))
    return cuts


def confusion_counts(
    decisions: Sequence[bool], labels: Sequence[bool]
) -> dict[str, int]:
    """TP/FP/FN/TN of a policy against the oracle, in decision terms."""
    if len(decisions) != len(labels):
        raise ValueError(
            f"decisions and labels must align, got {len(decisions)} and "
            f"{len(labels)}"
        )
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for decision, label in zip(decisions, labels):
        if decision and label:
            counts["tp"] += 1
        elif decision and not label:
            counts["fp"] += 1
        elif not decision and label:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def classify_decision(oracle: bool, decision: bool) -> str:
    """One query's policy outcome in the pre-registered four-way taxonomy."""
    if decision and oracle:
        return "true_escalation"
    if decision and not oracle:
        return "false_escalation"
    if not decision and oracle:
        return "missed_escalation"
    return "correct_stop"


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stderror(values: Sequence[float]) -> float | None:
    """Standard error of the mean, or None when it is undefined (n < 2)."""
    n = len(values)
    if n < 2:
        return None
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return float(math.sqrt(variance) / math.sqrt(n))


def sweep_rule(
    records: Sequence[dict[str, Any]],
    *,
    signal: str,
    thresholds: Sequence[float],
    direction: str,
    quality: str = "recall_at_5",
) -> list[dict[str, Any]]:
    """Score every candidate threshold of one rule on oracle records.

    Each record carries the pre-Dense `signals` dict, the oracle label
    (`oracle_escalate`), and both arms' quality under `bm25_{quality}` /
    `dense_{quality}` with their retrieval clocks (`bm25_latency_ms`,
    `dense_latency_ms`). Adaptive latency charges the dense clock only when
    the rule escalates -- queries stopped at BM25 pay BM25 alone. The caller
    owns split discipline: this function cannot tell calibration rows from
    test rows, which is why the analysis script only ever passes it
    calibration records.
    """
    if signal not in PRE_DENSE_SIGNALS:
        raise ValueError(
            f"signal {signal!r} is not a pre-Dense signal; refusing to sweep"
        )
    bm25_key = f"bm25_{quality}"
    dense_key = f"dense_{quality}"
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        decisions = [
            apply_rule(signal_value(r["signals"], signal), threshold, direction)
            for r in records
        ]
        qualities = [
            float(r[dense_key]) if d else float(r[bm25_key])
            for r, d in zip(records, decisions)
        ]
        latencies = [
            float(r["bm25_latency_ms"])
            + (float(r["dense_latency_ms"]) if d else 0.0)
            for r, d in zip(records, decisions)
        ]
        counts = confusion_counts(
            decisions, [bool(r["oracle_escalate"]) for r in records]
        )
        n = len(records)
        n_esc = sum(1 for d in decisions if d)
        results.append(
            {
                "signal": signal,
                "threshold": float(threshold),
                "direction": direction,
                "quality_metric": quality,
                "n": n,
                "n_escalated": n_esc,
                "escalation_rate": (n_esc / n) if n else 0.0,
                "mean_quality": _mean(qualities),
                "quality_stderror": _stderror(qualities),
                "mean_latency_ms": _mean(latencies),
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "tn": counts["tn"],
                "oracle_agreement": (
                    (counts["tp"] + counts["tn"]) / n if n else 0.0
                ),
            }
        )
    return results


def select_rule(
    sweep_results: Sequence[dict[str, Any]],
    *,
    min_rate: float = MIN_ESCALATION_RATE,
    max_rate: float = MAX_ESCALATION_RATE,
) -> dict[str, Any]:
    """The frozen selection rule: cheapest policy within 1 SE of the best.

    Eligible rules escalate a non-degenerate share of queries. Among those
    within one standard error of the family's maximum mean quality, the
    lowest escalation rate wins (quality first, cost second). When nothing
    is eligible the maximum-quality rule is returned flagged degenerate, so
    a failed selection is reported as a limitation rather than re-swept.
    """
    eligible = [
        r
        for r in sweep_results
        if min_rate <= float(r["escalation_rate"]) <= max_rate
    ]
    if not eligible:
        best = max(sweep_results, key=lambda r: float(r["mean_quality"]))
        return {
            "signal": best["signal"],
            "threshold": best["threshold"],
            "direction": best["direction"],
            "degenerate": True,
            "reason": (
                "no rule inside "
                f"[{min_rate}, {max_rate}] escalation rate; returning the "
                "maximum-quality rule and reporting the degeneracy"
            ),
        }
    ceiling = max(float(r["mean_quality"]) for r in eligible)
    within = [
        r
        for r in eligible
        if r["quality_stderror"] is not None
        and float(r["mean_quality"])
        >= ceiling - float(r["quality_stderror"])
    ]
    # A None standard error means n < 2, which cannot happen on a real split
    # but would otherwise silently drop a candidate; fall back to the maximum.
    pool = within or [
        r
        for r in eligible
        if float(r["mean_quality"]) >= ceiling
    ]
    winner = min(
        pool,
        key=lambda r: (float(r["escalation_rate"]), str(r["signal"]),
                       float(r["threshold"])),
    )
    return {
        "signal": winner["signal"],
        "threshold": winner["threshold"],
        "direction": winner["direction"],
        "degenerate": False,
        "escalation_rate": float(winner["escalation_rate"]),
        "mean_quality": float(winner["mean_quality"]),
        "reason": (
            f"lowest escalation rate ({winner['escalation_rate']:.4f}) among "
            f"{len(pool)} rule(s) within 1 SE of the family-maximum mean "
            f"quality ({ceiling:.4f})"
        ),
    }


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """ROC-AUC of a score against a binary label, via the Mann-Whitney U.

    Descriptive only under the Phase 14 protocol (≤6 calibration positives
    on the shipping arm): reported beside the operating numbers, never used
    for selection or verdicts. None when either class is absent.
    """
    pos = [float(s) for s, lab in zip(scores, labels) if lab]
    neg = [float(s) for s, lab in zip(scores, labels) if not lab]
    if not pos or not neg:
        return None
    pooled = sorted((float(s), lab) for s, lab in zip(scores, labels))
    ranks: list[float] = [0.0] * len(pooled)
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    rank_sum_pos = sum(r for (_, lab), r in zip(pooled, ranks) if lab)
    u_stat = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u_stat / (len(pos) * len(neg)))


def random_escalation(
    records: Sequence[dict[str, Any]],
    *,
    rate: float,
    n_draws: int = RANDOM_ABLATION_DRAWS,
    seed: int = RANDOM_ABLATION_SEED,
    quality: str = "recall_at_5",
) -> dict[str, Any]:
    """Ablation D: pay for Dense at the adaptive policy's rate, but at random.

    Fixed-count subsets (exactly round(rate * n) queries per draw) so every
    draw spends the same embedding calls the adaptive policy spent; only the
    *choice* of queries differs. Seeded and deterministic. If the adaptive
    policy's benefit comes from spending rather than selecting, the random
    draws match it; if it comes from selecting, they sit below it.
    """
    bm25_key = f"bm25_{quality}"
    dense_key = f"dense_{quality}"
    n = len(records)
    k = int(round(float(rate) * n))
    k = max(0, min(n, k))
    rng = random.Random(seed)
    mean_qualities: list[float] = []
    mean_latencies: list[float] = []
    for _ in range(n_draws):
        chosen = set(rng.sample(range(n), k)) if k else set()
        qualities = [
            float(records[i][dense_key]) if i in chosen
            else float(records[i][bm25_key])
            for i in range(n)
        ]
        latencies = [
            float(records[i]["bm25_latency_ms"])
            + (
                float(records[i]["dense_latency_ms"])
                if i in chosen
                else 0.0
            )
            for i in range(n)
        ]
        mean_qualities.append(_mean(qualities))
        mean_latencies.append(_mean(latencies))
    return {
        "n_draws": n_draws,
        "seed": seed,
        "n": n,
        "k_escalated_per_draw": k,
        "rate": (k / n) if n else 0.0,
        "mean_quality_values": mean_qualities,
        "mean_latency_ms_values": mean_latencies,
        "mean_quality": _mean(mean_qualities),
        "mean_latency_ms": _mean(mean_latencies),
    }
