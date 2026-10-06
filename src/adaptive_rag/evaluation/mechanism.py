"""Mechanism analysis for Phase 11: why escalation helps vs harms.

Phase 10 showed `distinct_doc_ratio@dense` tracks retrieval dispersion while
dispersion is dominated by dense succeeding where fusion fails -- disagreement
without escalation value. This module systematizes that failure: it assigns
each query to its frozen escalation group, attributes exactly what changed
between the dense and hybrid rankings, and exposes the dense-confidence and
disagreement features that already exist -- all as pure functions over plain
records, so the analysis is unit-testable without an index or a provider.

Frozen contract (`docs/phases/phase-11.md`): groups reuse the Phase 10 oracle
rule unchanged (helps == oracle YES, asserted, never redefined); relevance
labels may characterize *what happened* but never enter a candidate signal;
no threshold is fitted anywhere in Phase 11.
"""

from __future__ import annotations

from typing import Any, Sequence

from adaptive_rag.evaluation.agreement import (
    rank_correlation,
    score_decay_slope,
    score_gap_top1_top2,
)
from adaptive_rag.evaluation.dispersion import EPSILON

#: Version stamped into every Phase 11 artifact built through this module.
MECHANISM_VERSION = "phase11_mechanism_v1"

#: Frozen escalation groups. `helps` is exactly Phase 10's oracle YES class;
#: the builder asserts the equivalence so the outcome cannot drift.
GROUPS: tuple[str, ...] = ("helps", "ties", "harms")

#: Rank depth of the top-k outcome under study (matches the oracle metric).
TOP_K = 5

#: Source-attribution depth. Traces persist the final top-10 per arm, not the
#: candidate-depth (11-20) pools, so "neither" means absent from both top-10
#: lists -- not absent from retrieval altogether. A stated boundary, not a gap.
ATTRIBUTION_DEPTH = 10


def assign_group(delta_quality: float, epsilon: float = EPSILON) -> str:
    """Frozen escalation group for one query's dense->hybrid quality delta."""
    delta = float(delta_quality)
    if delta >= float(epsilon):
        return "helps"
    if delta <= -float(epsilon):
        return "harms"
    return "ties"


def hybrid_change(
    *,
    dense_ids: Sequence[str],
    hybrid_ids: Sequence[str],
    bm25_ids: Sequence[str],
    relevant_docs: Sequence[str],
    k: int = TOP_K,
    depth: int = ATTRIBUTION_DEPTH,
) -> dict[str, Any]:
    """Exactly what changed between the dense and hybrid top-k rankings.

    `entered`/`left` are document sets crossing the top-k boundary in either
    direction; `relevant_entered`/`relevant_left` restrict to labelled
    relevant documents (mechanism evidence only); `entered_source` attributes
    each entered document to the arm(s) holding it inside the top-`depth`
    pool each trace persists. Documents from neither pool -- e.g. promoted
    from unpersisted candidate depth -- are labelled `neither`, honestly.
    """
    dense_top = list(dense_ids[:k])
    hybrid_top = list(hybrid_ids[:k])
    entered = sorted(set(hybrid_top) - set(dense_top))
    left = sorted(set(dense_top) - set(hybrid_top))
    relevant = set(relevant_docs)
    dense_pool = set(dense_ids[:depth])
    bm25_pool = set(bm25_ids[:depth])
    sources: dict[str, str] = {}
    for doc in entered:
        in_dense = doc in dense_pool
        in_bm25 = doc in bm25_pool
        if in_dense and in_bm25:
            sources[doc] = "both"
        elif in_dense:
            sources[doc] = "dense_only"
        elif in_bm25:
            sources[doc] = "bm25_only"
        else:
            sources[doc] = "neither"
    return {
        "dense_top_k": dense_top,
        "hybrid_top_k": hybrid_top,
        "entered_docs": entered,
        "left_docs": left,
        "relevant_entered": sorted(relevant & set(entered)),
        "relevant_left": sorted(relevant & set(left)),
        "entered_source": sources,
    }


def dense_confidence(scores: Sequence[float | None]) -> dict[str, Any]:
    """Within-query dense strength readouts from one arm's top-k scores.

    Only margins and shapes are reported, never absolute cross-query score
    levels: dense cosine magnitudes are not calibrated across queries, so a
    "low top-1 score" threshold would compare uncomparable numbers. The gap
    and slope reuse the frozen Phase 9 closed forms (`agreement.py`).
    """
    finite = [float(s) for s in scores[:TOP_K] if s is not None]
    if not finite:
        return {"top1_score": None, "score_gap": None, "score_slope": None}
    return {
        "top1_score": finite[0],
        "score_gap": score_gap_top1_top2(finite),
        "score_slope": score_decay_slope(finite),
    }


def _trace_docs_and_scores(trace: dict[str, Any]) -> tuple[list[str], list[Any]]:
    results = (trace.get("retrieval") or {}).get("results") or []
    docs = [r["metadata"]["document_id"] for r in results]
    scores = [r.get("score") for r in results]
    return docs, scores


def build_mechanism_record(
    *,
    oracle_row: dict[str, Any],
    signal_row: dict[str, Any],
    dense_trace: dict[str, Any],
    bm25_trace: dict[str, Any],
    hybrid_trace: dict[str, Any],
    relevant_docs: Sequence[str],
) -> dict[str, Any]:
    """One query's full mechanism record (group, features, change, state).

    `signal_row` is the frozen Phase 9 signal-table row (agreement + score
    features are read off it, not recomputed, so the analysis provably uses
    Phase 9's values). `relevant_docs` labels the change attribution only.
    """
    group = assign_group(oracle_row["delta_recall_at_5"])
    if (group == "helps") != bool(oracle_row["oracle_escalate"]):
        raise ValueError(
            f"{oracle_row['query_id']}: group {group!r} disagrees with the "
            "frozen Phase 10 oracle label; the outcome definition moved"
        )
    dense_ids, dense_scores = _trace_docs_and_scores(dense_trace)
    bm25_ids, _ = _trace_docs_and_scores(bm25_trace)
    hybrid_ids, _ = _trace_docs_and_scores(hybrid_trace)
    change = hybrid_change(
        dense_ids=dense_ids,
        hybrid_ids=hybrid_ids,
        bm25_ids=bm25_ids,
        relevant_docs=relevant_docs,
    )
    confidence = dense_confidence(dense_scores)
    # The three conceptual states (§13 of the brief), operationalized: State 2
    # is helps; State 3 is harms where dense already held relevant evidence
    # in its top-5; State 1 is every other NO (sufficient-or-empty → stop).
    if group == "helps":
        state = "state2_uncertain_helpable"
    elif group == "harms" and float(oracle_row["dense_recall_at_5"]) > 0:
        state = "state3_disagree_but_correct"
    else:
        state = "state1_sufficient_or_empty"
    return {
        "query_id": oracle_row["query_id"],
        "corpus_arm": oracle_row["corpus_arm"],
        "split": oracle_row["split"],
        "category": oracle_row["category"],
        "group": group,
        "state": state,
        "oracle_escalate": bool(oracle_row["oracle_escalate"]),
        "dense_recall_at_5": float(oracle_row["dense_recall_at_5"]),
        "hybrid_recall_at_5": float(oracle_row["hybrid_recall_at_5"]),
        "delta_recall_at_5": float(oracle_row["delta_recall_at_5"]),
        "dense_mrr": float(oracle_row["dense_mrr"]),
        "hybrid_mrr": float(oracle_row["hybrid_mrr"]),
        "ddr_dense": float(signal_row["distinct_doc_ratio"]["dense"]),
        "dense_top1_score": confidence["top1_score"],
        "dense_score_gap": confidence["score_gap"],
        "dense_score_slope": confidence["score_slope"],
        "jaccard_bm25_dense": signal_row["jaccard"]["bm25|dense"],
        "union_concentration": signal_row["union_concentration"],
        "top1_agreement_bm25_dense": signal_row["top1_agreement"][
            "bm25|dense"
        ],
        "rank_correlation_bm25_dense": signal_row["rank_correlation"][
            "bm25|dense"
        ],
        **change,
    }


def separation_check(
    records: Sequence[dict[str, Any]],
    features: Sequence[str],
) -> list[dict[str, Any]]:
    """Bounded falsification check, not a search: for each pre-declared
    feature, does any single direction perfectly separate calibration helps
    from calibration harms? Reported to show that perfect separation on ~5
    points is uninformative (expected by chance), never to select a signal.
    `None` feature values are skipped pairwise -- a missing clock is not a
    zero, per the Phase 9 missing-data rule.
    """
    helps = [r for r in records if r["group"] == "helps"]
    harms = [r for r in records if r["group"] == "harms"]
    report: list[dict[str, Any]] = []
    for feature in features:
        h_vals = [r[feature] for r in helps if r[feature] is not None]
        m_vals = [r[feature] for r in harms if r[feature] is not None]
        if not h_vals or not m_vals:
            report.append(
                {"feature": feature, "separates": False,
                 "reason": "empty class after None-skipping"}
            )
            continue
        lo_h, hi_h = min(h_vals), max(h_vals)
        lo_m, hi_m = min(m_vals), max(m_vals)
        if hi_h < lo_m:
            direction: str | None = "helps_below_harms"
        elif hi_m < lo_h:
            direction = "helps_above_harms"
        else:
            direction = None
        report.append(
            {
                "feature": feature,
                "separates": direction is not None,
                "direction": direction,
                "helps_values": sorted(h_vals),
                "harms_values": sorted(m_vals),
            }
        )
    return report


def rank_correlation_defined(
    dense_ids: Sequence[str], bm25_ids: Sequence[str], k: int = TOP_K
) -> float | None:
    """Rank agreement of two arms' top-k, None when <2 docs are shared."""
    return rank_correlation(dense_ids, bm25_ids, k)
