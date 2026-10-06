"""Powered-confirmation utilities for Phase 15.

Phase 14 closed with INSUFFICIENT EVIDENCE on 11 test positives per arm.
Phase 15 re-tests the exact frozen Phase 14 rule on 110 freshly curated
queries (all test; nothing is fitted) and combines the two corpus arms
into one powered statement about selection quality.

This module holds only pure functions, so every number is unit-testable
without an index, a provider, or a network:

* curation helpers: normalized duplicate screening and quota checking for
  the blind Phase 15 benchmark expansion (text only -- these helpers never
  see retrieval outcomes, and the curation scripts that call them never
  call retrieval);
* the combined randomization test: the observed two-arm mean adaptive
  quality against its own randomization null (independent per-arm
  fixed-count draws at the same draw index, averaged), which is valid
  under query pairing because it assumes no arm independence;
* the cluster bootstrap: resample query ids with replacement, carrying
  both arms together, for the combined adaptive-minus-BM25 confidence
  interval.

The research contract lives in `docs/phases/phase-15.md`. The constants
below are that contract in code: the duplicate threshold, the quotas, the
draw counts, and the seeds may not move after seeing a powered result.
"""

from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

#: Resampling seed for every Phase 15 bootstrap/cluster interval. Distinct
#: from Phase 7 (20250103), Phase 9 (20250109), and the ablation seed, so a
#: Phase 15 artifact can never be mistaken for a re-run of an earlier one.
POWERED_BOOTSTRAP_SEED = 20250115

#: Draws for the combined randomization test (matches the per-arm ablation
#: draw count so the two nulls are comparable).
POWERED_N_DRAWS = 1000

#: Normalized token-set Jaccard at or above which a candidate query is
#: flagged as a duplicate of an existing query (docs/phases/phase-15.md
#: §15.7). Flagged pairs are rewritten or dropped before any retrieval.
DUPLICATE_JACCARD_THRESHOLD = 0.6

#: Powered benchmark size (docs/phases/phase-15.md §15.16).
POWERED_N_QUERIES = 110

#: Category quotas mirroring the v1 proportions (§15.7).
CATEGORY_QUOTAS: dict[str, int] = {
    "factual": 27,
    "conceptual": 30,
    "terminology": 21,
    "fine_grained": 17,
    "comparative": 8,
    "multi_document": 7,
}

#: Per-paper coverage minimums (§15.7). A multi-document query counts
#: toward each paper it lists.
PAPER_MINIMUMS: dict[str, int] = {
    "rag_lewis_2020": 18,
    "bm25_robertson_2009": 17,
    "retro_borgeaud_2022": 14,
    "splade_v2_formal_2021": 9,
    "rrf_cormack_2009": 7,
    "contriever_izacard_2022": 9,
    "colbert_khattab_2020": 8,
    "monobert_nogueira_2019": 8,
    "self_rag_asai_2023": 9,
    "crag_yan_2024": 6,
    "adaptive_rag_jeong_2024": 6,
    "dpr_karpukhin_2020": 9,
    "realm_guu_2020": 9,
    "pyserini_lin_2021": 4,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_query(text: str) -> frozenset[str]:
    """Token set of a query for duplicate screening (text only)."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def token_jaccard(first: str, second: str) -> float:
    """Jaccard similarity of two queries' normalized token sets."""
    set_a = normalize_query(first)
    set_b = normalize_query(second)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def screen_duplicates(
    candidates: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DUPLICATE_JACCARD_THRESHOLD,
) -> list[dict[str, Any]]:
    """Flag candidate queries too close to any reference (or each other).

    Each record carries `example_id` and `query`. Every candidate is
    compared against all references and all previously accepted candidates;
    the first comparison at/above threshold flags it. Order-dependent by
    construction: curation order is the acceptance order, fixed before any
    retrieval runs.
    """
    accepted: list[Mapping[str, Any]] = list(references)
    flags: list[dict[str, Any]] = []
    for candidate in candidates:
        hit: dict[str, Any] | None = None
        for other in accepted:
            score = token_jaccard(str(candidate["query"]), str(other["query"]))
            if score >= threshold:
                hit = {
                    "example_id": str(candidate["example_id"]),
                    "matches": str(other["example_id"]),
                    "jaccard": round(float(score), 4),
                }
                break
        if hit is None:
            accepted.append(candidate)
        else:
            flags.append(hit)
    return flags


def check_quotas(
    queries: Sequence[Mapping[str, Any]],
    *,
    category_quotas: Mapping[str, int] = CATEGORY_QUOTAS,
    paper_minimums: Mapping[str, int] = PAPER_MINIMUMS,
    n_queries: int = POWERED_N_QUERIES,
) -> dict[str, Any]:
    """Check a curated set against the frozen quotas (text fields only)."""
    category_counts: dict[str, int] = {}
    paper_counts: dict[str, int] = {}
    for query in queries:
        category = str(query["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        docs = query.get("relevant_documents") or []
        for doc in docs:
            paper_counts[str(doc)] = paper_counts.get(str(doc), 0) + 1
    category_gaps = {
        name: {"expected": want, "got": category_counts.get(name, 0)}
        for name, want in category_quotas.items()
        if category_counts.get(name, 0) != want
    }
    paper_gaps = {
        name: {"minimum": want, "got": paper_counts.get(name, 0)}
        for name, want in paper_minimums.items()
        if paper_counts.get(name, 0) < want
    }
    unexpected_categories = sorted(set(category_counts) - set(category_quotas))
    satisfied = (
        len(queries) == n_queries
        and not category_gaps
        and not paper_gaps
        and not unexpected_categories
    )
    return {
        "n_queries": len(queries),
        "expected_n": n_queries,
        "category_counts": category_counts,
        "paper_counts": paper_counts,
        "category_gaps": category_gaps,
        "paper_gaps": paper_gaps,
        "unexpected_categories": unexpected_categories,
        "satisfied": bool(satisfied),
    }


def _fixed_count_draw(n: int, k: int, rng: random.Random) -> set[int]:
    if k <= 0:
        return set()
    return set(rng.sample(range(n), min(k, n)))


def combined_randomization_test(
    *,
    adaptive_after: Sequence[float],
    adaptive_before: Sequence[float],
    bm25_after: Sequence[float],
    bm25_before: Sequence[float],
    dense_after: Sequence[float],
    dense_before: Sequence[float],
    k_after: int,
    k_before: int,
    n_draws: int = POWERED_N_DRAWS,
    seed: int = 20250101,
) -> dict[str, Any]:
    """Two-arm randomization test of selection quality (§15.11).

    Observed statistic M = mean adaptive quality over both arms (2n
    observations). Null draw i = (random_after_i + random_before_i) / 2,
    where each arm's draw is a fixed-count random subset at that arm's
    escalation count (the same spending the rule used, but unselected).
    P_comb = fraction of null draws at/above M. Deterministic in the seed.
    """
    after = [list(map(float, v)) for v in (adaptive_after, bm25_after, dense_after)]
    before = [list(map(float, v)) for v in (adaptive_before, bm25_before, dense_before)]
    adapt_a, b_a, d_a = after
    adapt_b, b_b, d_b = before
    n_a, n_b = len(adapt_a), len(adapt_b)
    if not (len(b_a) == len(d_a) == n_a and len(b_b) == len(d_b) == n_b):
        raise ValueError("arm vectors must align within each arm")
    observed = (sum(adapt_a) + sum(adapt_b)) / (n_a + n_b)
    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(n_draws):
        chosen_a = _fixed_count_draw(n_a, k_after, rng)
        chosen_b = _fixed_count_draw(n_b, k_before, rng)
        rand_a = (
            sum(d_a[i] if i in chosen_a else b_a[i] for i in range(n_a)) / n_a
        )
        rand_b = (
            sum(d_b[i] if i in chosen_b else b_b[i] for i in range(n_b)) / n_b
        )
        null_values.append((rand_a + rand_b) / 2.0)
    p_comb = sum(1 for v in null_values if v >= observed) / n_draws
    ordered = sorted(null_values)
    return {
        "n_draws": n_draws,
        "seed": seed,
        "n_after": n_a,
        "n_before": n_b,
        "k_after": k_after,
        "k_before": k_before,
        "observed_combined_mean": float(observed),
        "null_mean": float(sum(null_values) / len(null_values)),
        "null_p95": float(ordered[min(n_draws - 1, int(0.95 * n_draws))]),
        "p_comb": float(p_comb),
    }


def cluster_bootstrap_ci(
    *,
    adaptive_after: Sequence[float],
    adaptive_before: Sequence[float],
    bm25_after: Sequence[float],
    bm25_before: Sequence[float],
    n_boot: int = 10000,
    seed: int = POWERED_BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Cluster-bootstrap CI on the combined adaptive-minus-BM25 mean.

    Resamples query positions with replacement, carrying both arms of each
    sampled query together, so the paired-arm dependence is preserved rather
    than assumed away. Percentile interval over the resampled combined
    means. Deterministic in the seed.
    """
    adapt_a = list(map(float, adaptive_after))
    adapt_b = list(map(float, adaptive_before))
    b_a = list(map(float, bm25_after))
    b_b = list(map(float, bm25_before))
    n_a, n_b = len(adapt_a), len(adapt_b)
    if not (len(b_a) == n_a and len(b_b) == n_b):
        raise ValueError("arm vectors must align within each arm")
    if n_a != n_b:
        raise ValueError(
            f"cluster bootstrap needs paired arms, got {n_a} and {n_b}"
        )
    n = n_a
    if n == 0:
        raise ValueError("no queries to resample")
    rng = random.Random(seed)
    diffs_a = [t - b for t, b in zip(adapt_a, b_a)]
    diffs_b = [t - b for t, b in zip(adapt_b, b_b)]
    means: list[float] = []
    for _ in range(n_boot):
        picks = [rng.randrange(n) for _ in range(n)]
        total = sum(diffs_a[i] + diffs_b[i] for i in picks)
        means.append(total / (2 * n))
    ordered = sorted(means)
    tail = (1.0 - ci) / 2.0
    lower = ordered[max(0, min(n_boot - 1, int(tail * n_boot)))]
    upper = ordered[max(0, min(n_boot - 1, int((1.0 - tail) * n_boot)))]
    point = (sum(diffs_a) + sum(diffs_b)) / (2 * n)
    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_queries": n,
        "point_estimate": float(point),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "ci_level": ci,
    }
