"""Gate 9.2 -- mechanism screening.

Spearman rho (mid-ranks) of each of the 12 registered Phase 9 candidate signals
against `T-disp3`, on both corpus arms, with bootstrap CIs, permutation
p-values, Holm correction over the family of 12, and the preregistered
three-condition survival rule.

Reads persisted Phase 8 artifacts only. Runs no retrieval, calls no provider,
and writes nothing outside the one JSON report. Every constant is the frozen
value from docs/phases/phase-9.md 2.6; none is tuned from a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adaptive_rag.evaluation.agreement import DEFAULT_K
from adaptive_rag.evaluation.dispersion import EPSILON, SOUND_STRATEGIES
from adaptive_rag.evaluation.signals import (
    REGISTERED_CANDIDATES,
    SignalRow,
    defined_observations,
    read_signal_table,
)

# --- frozen constants (docs/phases/phase-9.md 2.6) ---------------------------
BOOTSTRAP_SEED = 20250109
PERMUTATION_SEED = 20250109
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000
ALPHA = 0.05
MIN_ABS_RHO = 0.30

SHIPPING_ARM = "phase8_after"
REPLICATION_ARM = "phase8_before"
SIGNAL_DIR = Path("experiments/phase9")
REPORT = SIGNAL_DIR / "gate9_mechanism_screening.json"


# --- statistics --------------------------------------------------------------


def midrank(values: Sequence[float]) -> np.ndarray:
    """Ranks with ties resolved to their mid-rank.

    Mandatory, not stylistic: `recall@5` takes only four distinct values on this
    benchmark (92 of 107 queries carry one relevant document), so ties are
    pervasive and an untie-broken rank correlation would be wrong.
    """
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Mid-rank Spearman rho, or None when undefined.

    Undefined when either input is constant (zero rank variance), which a
    bootstrap resample can produce by drawing one repeated value. Never 0.0.
    """
    rx, ry = midrank(x), midrank(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[float | None, float | None, int]:
    """Stratified-free percentile CI over the signal's DEFINED observations.

    Resamples query-level pairs with replacement -- never all 107 rows with the
    signal reconstructed. Resamples that are degenerate (a constant arm, so rho
    undefined) are skipped and counted rather than coerced.
    """
    n = len(x)
    draws: list[float] = []
    skipped = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        rho = spearman(x[idx], y[idx])
        if rho is None:
            skipped += 1
            continue
        draws.append(rho)
    if not draws:
        return None, None, skipped
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), skipped


def permutation_p(
    x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator
) -> float:
    """Two-sided permutation p on the SAME defined observations.

    Labels are permuted across the eligible queries only; NULL entries are never
    imputed and never permuted.
    """
    n = len(x)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(y)
        rho = spearman(x, shuffled)
        if rho is not None and abs(rho) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (N_PERMUTATIONS + 1)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, min(1.0, value))
        adjusted[idx] = running
    return adjusted


# --- data --------------------------------------------------------------------


def load_rows(arm: str) -> list[SignalRow]:
    records = read_signal_table(SIGNAL_DIR / f"signal_table_{arm}.jsonl")
    return [
        SignalRow(
            query_id=r["query_id"],
            dispersion=r["target_dispersion"],
            dispersion_mrr=r["target_dispersion_mrr"],
            jaccard=r["jaccard"],
            overlap_at_k=r["overlap_at_k"],
            top1_agreement=r["top1_agreement"],
            rank_correlation=r["rank_correlation"],
            union_concentration=r["union_concentration"],
            distinct_doc_ratio=r["distinct_doc_ratio"],
            score_gap_top1_top2=r["score_gap_top1_top2"],
            score_decay_slope=r["score_decay_slope"],
            query_features=r["query_features"],
        )
        for r in records
    ]


def evaluate(rows: list[SignalRow]) -> dict[str, dict[str, Any]]:
    """Every registered candidate on one arm."""
    targets = np.array([r.dispersion for r in rows], dtype=float)
    out: dict[str, dict[str, Any]] = {}
    for candidate in REGISTERED_CANDIDATES:
        ids, values = defined_observations(rows, candidate.name)
        idx = {r.query_id: i for i, r in enumerate(rows)}
        sel = np.array([idx[q] for q in ids], dtype=int)
        x = np.asarray(values, dtype=float)
        y = targets[sel]
        rho = spearman(x, y)
        rng_b = np.random.default_rng(BOOTSTRAP_SEED)
        lo, hi, skipped = bootstrap_ci(x, y, rng_b)
        p = permutation_p(x, y, rho, np.random.default_rng(PERMUTATION_SEED))
        out[candidate.name] = {
            "arm": "after" if False else None,
            "family": candidate.family,
            "expected_direction": candidate.expected_direction,
            "n_defined": len(ids),
            "query_ids": list(ids),
            "rho": rho,
            "bootstrap_ci_95": [lo, hi],
            "bootstrap_degenerate_resamples": skipped,
            "permutation_p": p,
            "direction_matches_expectation": (
                None
                if candidate.expected_direction == "either" or rho is None
                else (rho > 0) == (candidate.expected_direction == "positive")
            ),
            "abs_rho": None if rho is None else abs(rho),
        }
    return out


def main() -> dict[str, Any]:
    arms = {a: evaluate(load_rows(a)) for a in (SHIPPING_ARM, REPLICATION_ARM)}
    for arm, results in arms.items():
        for entry in results.values():
            entry["arm"] = "after" if arm == SHIPPING_ARM else "before"

    # Holm is applied once, on the shipping arm, over the family of 12.
    shipping_names = [c.name for c in REGISTERED_CANDIDATES]
    adjusted = holm([arms[SHIPPING_ARM][n]["permutation_p"] for n in shipping_names])
    for name, value in zip(shipping_names, adjusted):
        arms[SHIPPING_ARM][name]["p_holm"] = value

    # Condition 3 (section 2.7.1): same sign AND |rho| >= MIN_ABS_RHO on the
    # replication arm. Holm is deliberately NOT re-derived there.
    survivors: list[str] = []
    detail: list[dict[str, Any]] = []
    for candidate in REGISTERED_CANDIDATES:
        ship = arms[SHIPPING_ARM][candidate.name]
        repl = arms[REPLICATION_ARM][candidate.name]
        c1 = ship["abs_rho"] is not None and ship["abs_rho"] >= MIN_ABS_RHO
        c2 = ship["p_holm"] < ALPHA
        c3 = (
            repl["rho"] is not None
            and ship["rho"] is not None
            and (repl["rho"] > 0) == (ship["rho"] > 0)
            and repl["abs_rho"] >= MIN_ABS_RHO
        )
        survives = bool(c1 and c2 and c3)
        if survives:
            survivors.append(candidate.name)
        detail.append(
            {
                "signal": candidate.name,
                "family": candidate.family,
                "expected_direction": candidate.expected_direction,
                "shipping": {
                    k: ship[k]
                    for k in (
                        "n_defined", "rho", "abs_rho", "bootstrap_ci_95",
                        "permutation_p", "p_holm",
                        "direction_matches_expectation",
                    )
                },
                "replication": {
                    k: repl[k]
                    for k in ("n_defined", "rho", "abs_rho", "bootstrap_ci_95",
                              "permutation_p", "direction_matches_expectation")
                },
                "condition_1_abs_rho": c1,
                "condition_2_p_holm": c2,
                "condition_3_replication": c3,
                "survives": survives,
                "same_eligible_queries": (
                    set(ship["query_ids"]) == set(repl["query_ids"])
                ),
                "eligible_overlap": len(
                    set(ship["query_ids"]) & set(repl["query_ids"])
                ),
            }
        )

    verdict = (
        "mechanism_signal_identified" if survivors else "no_signal_meets_the_bar"
    )
    report = {
        "gate": "9.2",
        "target": "T-disp3 = max-min recall@5 over {bm25, dense, hybrid}",
        "primary_statistic": "Spearman rho with mid-ranks",
        "strategies": sorted(SOUND_STRATEGIES),
        "k": DEFAULT_K,
        "epsilon": EPSILON,
        "family_size": len(REGISTERED_CANDIDATES),
        "holm_denominator": len(REGISTERED_CANDIDATES),
        "multiplicity": "Holm-Bonferroni, applied once on the shipping arm",
        "alpha": ALPHA,
        "min_abs_rho": MIN_ABS_RHO,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_seed": PERMUTATION_SEED,
        "missing_data": (
            "pairwise exclusion per signal over its defined observations; no "
            "imputation; NULL never converted to 0"
        ),
        "shipping_arm": SHIPPING_ARM,
        "replication_arm": REPLICATION_ARM,
        "per_signal": detail,
        "survivors": survivors,
        "verdict": verdict,
        "interpretation": (
            "A surviving signal is an observable association with T-disp3 that "
            "is large enough, survives Holm, and holds across both corpus arms. "
            "It is not a mechanism and not a cause: Gate 9.4 characterises "
            "mechanism and no causal claim is made here."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps(rep["per_signal"], indent=2, sort_keys=True))
    print("VERDICT:", rep["verdict"], "SURVIVORS:", rep["survivors"])
