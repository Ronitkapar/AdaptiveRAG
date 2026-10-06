"""Per-query escalation oracle and threshold policy for Phase 10.

Phase 9 established `distinct_doc_ratio@dense` as a post-dense
retrieval-dispersion signal (test rho = +0.410 against `T-disp3`) but ran no
intervention, so no per-query gain oracle exists to validate a policy against.
This module is that missing piece: it turns already-persisted dense/hybrid
per-query outcomes into a deterministic escalation oracle, sweeps the
pre-registered threshold families over it, and scores the resulting adaptive
policy -- all as pure functions over plain records, so every number is
unit-testable without an index, a provider, or a network.

The research contract lives in `docs/phases/phase-10.md`. The constants below
are that contract in code: the oracle rule, the threshold value set, the
non-degeneracy band, and the random-ablation protocol may not move after
seeing a result.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

from adaptive_rag.evaluation.agreement import distinct_doc_ratio
from adaptive_rag.evaluation.dispersion import EPSILON

#: Version stamped into every Phase 10 artifact built through this module.
ESCALATION_ORACLE_VERSION = "phase10_escalation_v1"

#: Rank depth of the Phase 9 signal. Frozen at the E1 arms' evaluation depth;
#: changing it would silently redefine the signal under test.
DDR_K = 5

#: The signal takes exactly these values (len(dedup(ids[:5])) / 5), so the
#: sweep enumerates the whole policy space rather than sampling it.
THRESHOLDS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

#: Pre-registered threshold families. "high" escalates on diffuse dense
#: evidence (the Phase 9 mechanism direction); "low" is the control family that
#: exists so the direction choice is declared before results, not after them.
DIRECTIONS: tuple[str, ...] = ("high", "low")

#: Escalation rates outside this band are degenerate policies (always escalate
#: at t = 0.2 by construction; near-never escalate at the top end), not
#: routing decisions. The selection rule may not pick them.
MIN_ESCALATION_RATE = 0.05
MAX_ESCALATION_RATE = 0.95

#: Random-escalation ablation protocol: fixed-count subsets at the adaptive
#: policy's own rate, this many draws, this seed (== statistics.DEFAULT_SEED).
RANDOM_ABLATION_DRAWS = 1000
RANDOM_ABLATION_SEED = 20250101


def recompute_ddr(document_ids: Sequence[str], k: int = DDR_K) -> float:
    """The Phase 9 signal, recomputed from a dense arm's ranked document ids.

    Used as an integrity check: the oracle build recomputes this from the
    dense rows and aborts unless it matches the frozen signal-table value, so
    the oracle provably joins the exact rows Phase 9 measured.
    """
    return distinct_doc_ratio(document_ids, k)


def oracle_label(delta_quality: float, epsilon: float = EPSILON) -> bool:
    """Whether escalation was worth it for one query (the frozen oracle rule).

    Strict improvement at or above EPSILON counts. Ties mean paying compute
    for nothing; negative deltas mean paying compute for harm. Both are NO.
    """
    return float(delta_quality) >= float(epsilon)


def incremental_latency_ms(
    *,
    bm25_latency_ms: float | None,
    fusion_latency_ms: float | None,
    search_latency_ms: float | None,
    fallback_latency_ms: float | None,
) -> tuple[float, str]:
    """Incremental cost of escalating after dense retrieval, in milliseconds.

    The query embedding is already paid at the dense stage and reused, so the
    honest incremental cost is the hybrid path's non-embedding stages: BM25
    retrieval, RRF fusion, and the vector re-search at candidate depth. When
    any stage clock is missing the full hybrid retrieval latency is used
    instead (conservative: it double-counts the embedding) and the method is
    reported so the fallback count travels with the artifact.
    """
    stages = [bm25_latency_ms, fusion_latency_ms, search_latency_ms]
    if all(
        isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
        for v in stages
    ):
        return float(sum(stages)), "stages"  # type: ignore[arg-type]
    if (
        isinstance(fallback_latency_ms, (int, float))
        and math.isfinite(fallback_latency_ms)
        and fallback_latency_ms >= 0
    ):
        return float(fallback_latency_ms), "full_fallback"
    raise ValueError(
        "no usable latency clock: all hybrid stages and the fallback are "
        "missing or non-finite"
    )


def apply_threshold(ddr: float, threshold: float, direction: str) -> bool:
    """The adaptive decision for one query: escalate or stop."""
    if direction == "high":
        return float(ddr) >= float(threshold)
    if direction == "low":
        return float(ddr) <= float(threshold)
    raise ValueError(f"unknown threshold direction {direction!r}")


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


def sweep_thresholds(
    records: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float] = THRESHOLDS,
    direction: str = "high",
    quality: str = "recall_at_5",
) -> list[dict[str, Any]]:
    """Score every candidate threshold of one family on oracle records.

    Each record carries the signal (`ddr_dense`), the oracle label
    (`oracle_escalate`), both arms' quality under `dense_{quality}` /
    `hybrid_{quality}`, and the latency model (`dense_latency_ms`,
    `incremental_latency_ms`). The caller is responsible for split discipline:
    this function cannot tell calibration rows from test rows, which is why
    the analysis script only ever passes it calibration records.
    """
    dense_key = f"dense_{quality}"
    hybrid_key = f"hybrid_{quality}"
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        decisions = [
            apply_threshold(r["ddr_dense"], threshold, direction)
            for r in records
        ]
        qualities = [
            float(r[hybrid_key]) if d else float(r[dense_key])
            for r, d in zip(records, decisions)
        ]
        latencies = [
            float(r["dense_latency_ms"])
            + (float(r["incremental_latency_ms"]) if d else 0.0)
            for r, d in zip(records, decisions)
        ]
        counts = confusion_counts(
            decisions, [bool(r["oracle_escalate"]) for r in records]
        )
        n = len(records)
        n_esc = sum(1 for d in decisions if d)
        results.append(
            {
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


def select_threshold(
    sweep_results: Sequence[dict[str, Any]],
    *,
    min_rate: float = MIN_ESCALATION_RATE,
    max_rate: float = MAX_ESCALATION_RATE,
) -> dict[str, Any]:
    """The frozen selection rule: cheapest policy within 1 SE of the best.

    Eligible thresholds escalate a non-degenerate share of queries. Among
    those within one standard error of the family's maximum mean quality, the
    lowest escalation rate wins (quality first, cost second). When nothing is
    eligible the maximum-quality threshold is returned flagged degenerate, so
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
            "threshold": best["threshold"],
            "direction": best["direction"],
            "degenerate": True,
            "reason": (
                "no threshold inside "
                f"[{min_rate}, {max_rate}] escalation rate; returning the "
                "maximum-quality threshold and reporting the degeneracy"
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
    winner = min(pool, key=lambda r: (float(r["escalation_rate"]), float(r["threshold"])))
    return {
        "threshold": winner["threshold"],
        "direction": winner["direction"],
        "degenerate": False,
        "reason": (
            f"lowest escalation rate ({winner['escalation_rate']:.4f}) among "
            f"{len(pool)} threshold(s) within 1 SE of the family-maximum "
            f"mean quality ({ceiling:.4f})"
        ),
    }


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """ROC-AUC of a score against a binary label, via the Mann-Whitney U.

    Implemented directly (mid-rank ties) so the analysis needs no modelling
    dependency for one number. Returns None when either class is absent --
    ranking the empty class is undefined, not 0.5.
    """
    pos = [float(s) for s, lab in zip(scores, labels) if lab]
    neg = [float(s) for s, lab in zip(scores, labels) if not lab]
    if not pos or not neg:
        return None
    # Mid-ranks over the pooled sample; the mean positive rank gives U.
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
    rank_sum_pos = sum(
        r for (_, lab), r in zip(pooled, ranks) if lab
    )
    u_stat = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u_stat / (len(pos) * len(neg)))


def average_precision(
    scores: Sequence[float], labels: Sequence[bool]
) -> float | None:
    """PR-AUC as average precision over the score ranking (ties: mean rank).

    Reported beside ROC-AUC because the oracle-positive class is expected to
    be small, where ROC-AUC flatters and precision-at-operating-rate is the
    honest reading. None when the positive class is absent.
    """
    order = sorted(
        range(len(scores)), key=lambda i: float(scores[i]), reverse=True
    )
    n_pos = sum(1 for lab in labels if lab)
    if n_pos == 0:
        return None
    num = 0.0
    seen_pos = 0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            seen_pos += 1
            num += seen_pos / rank
    return float(num / n_pos)


def random_escalation(
    records: Sequence[dict[str, Any]],
    *,
    rate: float,
    n_draws: int = RANDOM_ABLATION_DRAWS,
    seed: int = RANDOM_ABLATION_SEED,
    quality: str = "recall_at_5",
) -> dict[str, Any]:
    """Ablation D: escalate at the adaptive policy's rate, but at random.

    Fixed-count subsets (exactly round(rate * n) queries per draw) so every
    draw spends the same compute the adaptive policy spent; only the *choice*
    of queries differs. Seeded and deterministic. If the adaptive policy's
    benefit comes from spending rather than selecting, the random draws match
    it; if it comes from selecting, they sit below it.
    """
    dense_key = f"dense_{quality}"
    hybrid_key = f"hybrid_{quality}"
    n = len(records)
    k = int(round(float(rate) * n))
    k = max(0, min(n, k))
    rng = random.Random(seed)
    mean_qualities: list[float] = []
    mean_latencies: list[float] = []
    for _ in range(n_draws):
        chosen = set(rng.sample(range(n), k)) if k else set()
        qualities = [
            float(records[i][hybrid_key]) if i in chosen
            else float(records[i][dense_key])
            for i in range(n)
        ]
        latencies = [
            float(records[i]["dense_latency_ms"])
            + (
                float(records[i]["incremental_latency_ms"])
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
