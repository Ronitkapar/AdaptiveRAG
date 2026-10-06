#!/usr/bin/env python3
"""
gate3_signal_exhaustion.py
--------------------------
Gate 3: **full observable signal exhaustion.**

Gate 2 established that routing headroom *exists* over the four selectable
strategies: the per-query oracle beats the best fixed strategy by +0.0218
recall@5 (`after`) and +0.0343 (`before`), and a paired bootstrap CI on that delta
excludes zero. Headroom existing is only half the question. A ceiling computed
with a signal that does not exist bounds routing; it does not evidence it.

This script asks the second half, out of sample:

    Can the information a router can actually observe tell it *which* queries
    are the few where a different retrieval strategy is actually beneficial?

**Scope discipline.** This is an *identification* study, not a router.

* It does **not** train or deploy a learned router. The model fitted here is a
  measuring instrument -- a ridge logistic regression used to ask whether a
  linear boundary in observable features carries out-of-sample information about
  the label. No decision rule is derived from it, no router is configured from
  it, and the shipped router is untouched.
* It does **not** touch the corpus, the index, `strategy_cost_ms`, or ADR-028.
  The 42 spliced section paths stay out of the primary routing closure
  experiment by construction: nothing here reads section labels, and the labels
  used are per-query recall@5 over the already-measured E1 arms.

**The target.** For each query, `needed` if *nothing* else in the selectable set
comes within `GATE_EPSILON` recall@5 of the reference (best-fixed) strategy --
the reference is the only acceptable answer -- else `sufficient`. This is the
same definition Gate 2 used, so the two gates agree.

**The honest difficulty, stated up front.** `recall@5` is near-binary on this
benchmark (92 of 107 queries carry a single relevant document), so the strategies
tie at the maximum on most queries and only **8 queries are `needed`**. Fitting
on the 47-query calibration split leaves 2-3 positives; testing on the 60-query
test split leaves 5-6. That caps what *any* method can detect here. The analysis
is built to report the power it actually has rather than to maximise a
favourable p-value: effect sizes and intervals are primary, the permutation test
is of the *whole procedure*, and a minimum-detectable-effect calculation is
reported so a null can be read as "too small to detect" rather than "no signal".

**Pre-registration.** Every threshold is a module constant fixed before
execution and copied verbatim into the artifact. The decision rule is
`decision_criteria()` and is applied mechanically by `evaluate_gate()`.

Writes `experiments/phase8/gate3_signal_exhaustion.json`, including the
per-query evidence table so any individual query driving or diluting the result
can be inspected.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_routing_ceiling import (  # noqa: E402
    ADAPTIVE_SYSTEM,
    SELECTABLE_SYSTEMS,
    QuerySetMismatch,
    attach_latency,
    best_fixed,
    build_panel,
    fixed_summaries,
    load_arm,
    needed_split,
    panel_latencies,
)

RESULT_VERSION = "gate3_signal_exhaustion_v1"

#: Same epsilon as the Gate 2 ceiling, so `needed` means the same thing in both.
GATE_EPSILON = 0.01

#: Corpus arms. Both are analysed; the gate requires agreement.
CORPUS_ARMS = {
    "phase8_after": REPO_ROOT / "experiments" / "phase8" / "combined" / "p8a_e1",
    "phase8_before": REPO_ROOT / "experiments" / "phase8" / "combined" / "p8b_e1",
}

#: Label classes. `needed` is the positive.
POSITIVE_LABEL = "needed"
NEGATIVE_LABEL = "sufficient"

#: Frozen benchmark splits. Fit on `calibration`; every reported out-of-sample
#: number comes from `test`.
FIT_SPLIT = "calibration"
EVAL_SPLIT = "test"

# --- Statistical decision criteria, fixed before execution -------------------
ALPHA = 0.05
#: A router that cannot do better than this is not worth deploying, so the gate
#: demands the out-of-sample AUC clear it, not merely exceed chance.
AUC_FLOOR = 0.65
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 20250103
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20250103

#: Ridge penalty for the measuring model. Fixed, not tuned: with 2-3 positives in
#: the fit split a tuned penalty would be selected on the data it is scored
#: against. Strong regularisation is the honest choice at this sample size.
RIDGE_LAMBDA = 1.0
MAX_IRLS_ITERATIONS = 100
IRLS_TOLERANCE = 1e-10
CLASS_WEIGHT = "balanced"

#: Multiple-comparison family for the univariate screen: every feature.
MULTIPLICITY_CORRECTION = "holm_bonferroni"
# ---------------------------------------------------------------------------
# Feature set -- every signal already observable to the shipped router.
# No new features are introduced.
# ---------------------------------------------------------------------------

#: (a) Query features: cheap, deterministic, from the raw query string alone
#: (`QueryFeatures`, analyzer_v1). Continuous.
QUERY_FEATURES: tuple[str, ...] = (
    "query_length_words",
    "query_length_chars",
    "content_term_count",
    "lexical_density",
    "entity_indicator_count",
    "entity_ratio",
    "technical_term_count",
    "technical_ratio",
    "semantic_indicator_count",
    "semantic_ratio",
    "concept_count",
    "comparison_indicator_count",
    "complexity_score",
)

#: (b) Retrieval-feedback features: what the first stage actually returned
#: (`SufficiencyDecision`). Observable at query time, and only *after* paying for
#: stage one -- which is how a real router would use them.
FEEDBACK_FEATURES: tuple[str, ...] = (
    "result_count",
    "coverage",
    "top1_coverage",
    "sufficiency_score",
    "initial_result_count",
)

#: (c) Existing routing signals: the shipped decision's own recorded output.
ROUTING_FEATURES: tuple[str, ...] = (
    "routing_confidence",
    "score_hybrid",
    "score_bm25",
    "score_dense",
    "score_hybrid_rerank",
    "evidence_margin",
)

#: (d) Categorical features, one-hot encoded.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "initial_strategy",
    "question_type",
    "multi_concept",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "query": QUERY_FEATURES,
    "retrieval_feedback": FEEDBACK_FEATURES,
    "routing": ROUTING_FEATURES,
}

#: One-hot levels, fixed from the observed vocabulary so the design matrix is
#: identical on both arms and in every permutation.
CATEGORICAL_LEVELS: dict[str, ...] = {
    "initial_strategy": tuple(SELECTABLE_SYSTEMS),
    "question_type": (
        "factual",
        "conceptual",
        "definition",
        "comparison",
        "procedural",
        "numeric",
        "other",
    ),
    "multi_concept": ("false", "true"),
}

#: Column names fed to the model, in a fixed order.
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    list(QUERY_FEATURES)
    + list(FEEDBACK_FEATURES)
    + list(ROUTING_FEATURES)
    + [
        f"{name}={level}"
        for name, levels in CATEGORICAL_LEVELS.items()
        for level in levels
    ]
)

#: Numeric columns screened one at a time in the univariate family.
UNIVARIATE_COLUMNS: tuple[str, ...] = (
    tuple(QUERY_FEATURES) + tuple(FEEDBACK_FEATURES) + tuple(ROUTING_FEATURES)
)


def decision_criteria() -> dict[str, Any]:
    """The pre-registered rule, stated as data and applied by `evaluate_gate`."""
    return {
        "registered_before_execution": True,
        "primary_metric": "out-of-sample AUC (test split), ridge logistic",
        "primary_test": (
            "permutation test of the whole procedure (fit on calibration, score "
            "test), labels permuted across all queries, "
            f"{N_PERMUTATIONS} resamples, seed {PERMUTATION_SEED}"
        ),
        "gate_passes_only_if_all_of": [
            f"permutation p < {ALPHA}",
            f"test AUC >= {AUC_FLOOR} (deployability floor, not chance)",
            "bootstrap 95% CI for test AUC has lower bound > 0.50",
            "at least one univariate feature survives Holm-Bonferroni at "
            f"{ALPHA} within the arm, oriented towards the positive class",
            "the same four conditions hold on BOTH corpus arms (replication)",
        ],
        "auc_floor": AUC_FLOOR,
        "alpha": ALPHA,
        "n_permutations": N_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "ridge_lambda": RIDGE_LAMBDA,
        "class_weight": CLASS_WEIGHT,
        "multiplicity_correction": MULTIPLICITY_CORRECTION,
        "fit_split": FIT_SPLIT,
        "eval_split": EVAL_SPLIT,
        "epsilon": GATE_EPSILON,
        "interpretation_rule": (
            "A failure is reported as 'no reliable out-of-sample signal at this "
            "sample size', not as 'no signal exists'. The minimum detectable "
            "effect is reported so the two can be told apart."
        ),
    }


# ---------------------------------------------------------------------------
# Observable features, read from the routed arm's traces
# ---------------------------------------------------------------------------


def read_observable_features(traces_path: Path) -> dict[str, dict[str, Any]]:
    """Per-query observable features, from the `adaptive` arm's traces only.

    Every field here is something the shipped router already computed and
    recorded at query time: the analyzer's `QueryFeatures`, the sufficiency
    checker's scalars, and the routing decision's own scores and confidence. The
    E1 arms for the other strategies are never read here -- using them would
    import per-strategy outcomes into the *feature* side of the analysis, which
    is exactly the oracle leakage this gate exists to avoid.

    The one derived field is `evidence_margin`, the gap between the best and
    second-best strategy score in the recorded decision. It is arithmetic on
    numbers the router already produced, not a new feature.
    """
    if not traces_path.is_file():
        raise FileNotFoundError(f"no traces at {traces_path}")

    out: dict[str, dict[str, Any]] = {}
    with open(traces_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            routing = trace.get("routing") or {}
            features = routing.get("features") or {}
            sufficiency = routing.get("sufficiency") or {}
            decision = routing.get("decision") or {}

            record: dict[str, Any] = {
                name: features.get(name) for name in QUERY_FEATURES
            }
            record["result_count"] = sufficiency.get("result_count")
            record["coverage"] = sufficiency.get("coverage")
            record["top1_coverage"] = sufficiency.get("top1_coverage")
            record["sufficiency_score"] = sufficiency.get("score")
            record["initial_result_count"] = routing.get("initial_result_count")
            record["routing_confidence"] = decision.get("confidence")

            scores = {
                str(item.get("strategy")): item.get("score")
                for item in (decision.get("evidence") or [])
            }
            for strategy in SELECTABLE_SYSTEMS:
                record[f"score_{strategy}"] = scores.get(strategy)
            ordered = sorted(
                (v for v in scores.values() if v is not None), reverse=True
            )
            record["evidence_margin"] = (
                ordered[0] - ordered[1] if len(ordered) >= 2 else None
            )

            record["initial_strategy"] = routing.get("initial_strategy")
            record["question_type"] = features.get("question_type")
            record["multi_concept"] = (
                "true" if features.get("multi_concept") else "false"
            )

            query_id = str(trace.get("example_id") or trace.get("trace_id"))
            out[query_id] = record
    return out


def design_matrix(
    feature_rows: Mapping[str, Mapping[str, Any]],
    query_ids: Sequence[str],
    *,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Build the numeric design matrix for `query_ids`, in `columns` order.

    Missing values become the column median computed over `query_ids`, so a
    missing signal is imputed from the same rows being analysed rather than
    from the other split. A signal that is absent for every query in the
    analysis is reported rather than silently zero-filled.
    """
    raw: dict[str, list[float]] = {}
    for column in columns:
        values: list[float] = []
        for query_id in query_ids:
            value = feature_rows[query_id].get(column)
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
            else:
                values.append(float("nan"))
        raw[column] = values

    matrix = np.empty((len(query_ids), len(columns)), dtype=float)
    for index, column in enumerate(columns):
        values = np.asarray(raw[column], dtype=float)
        finite = values[np.isfinite(values)]
        fill = float(np.median(finite)) if finite.size else 0.0
        column_values = np.where(np.isfinite(values), values, fill)
        matrix[:, index] = column_values
    return matrix


def constant_columns(matrix: np.ndarray, columns: Sequence[str]) -> list[str]:
    """Columns with no variance, which carry no information but do add a degree
    of freedom the model would otherwise spend on noise."""
    return [
        name
        for index, name in enumerate(columns)
        if float(np.std(matrix[:, index])) == 0.0
    ]


# ---------------------------------------------------------------------------
# The measuring model: ridge logistic regression, in numpy.
#
# sklearn is deliberately not used. The project declares no ML dependency
# (AGENTS.md: do not introduce unnecessary dependencies), and with 2-3 positives
# in the fit split a 25-parameter logistic regression needs exactly one thing
# from a library: a solver that is deterministic and cannot silently fail. IRLS
# is that, and it is 30 lines.
# ---------------------------------------------------------------------------


class RidgeLogistic:
    """L2-regularised logistic regression fitted by IRLS.

    Standardisation and class weighting are fitted on the training rows only.
    `class_weight="balanced"` reweights each observation's loss by
    n/(2*n_class), which stops 44 negatives from deciding the boundary when the
    positive class has 2 members -- without it the fit collapses to the
    intercept and every test AUC is 0.5 by construction.
    """

    def __init__(
        self,
        *,
        ridge_lambda: float = RIDGE_LAMBDA,
        max_iterations: int = MAX_IRLS_ITERATIONS,
        tolerance: float = IRLS_TOLERANCE,
        class_weight: str = CLASS_WEIGHT,
    ) -> None:
        self.ridge_lambda = ridge_lambda
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.class_weight = class_weight
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.converged_: bool = False
        self.iterations_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeLogistic":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n_rows, n_cols = x.shape

        # Standardise on the training rows only.
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        z = (x - self.mean_) / self.scale_

        weights = np.ones(n_rows, dtype=float)
        if self.class_weight == "balanced":
            positives = float(np.sum(y == 1))
            negatives = float(np.sum(y == 0))
            if positives > 0 and negatives > 0:
                weights[y == 1] = n_rows / (2.0 * positives)
                weights[y == 0] = n_rows / (2.0 * negatives)

        design = np.hstack([np.ones((n_rows, 1)), z])
        beta = np.zeros(n_cols + 1, dtype=float)
        # The penalty applies to slopes only; the intercept is left unpenalised so
        # a class-imbalanced fit is not dragged toward the base rate.
        penalty = np.eye(n_cols + 1) * self.ridge_lambda
        penalty[0, 0] = 0.0

        for iteration in range(self.max_iterations):
            eta = np.clip(design @ beta, -30.0, 30.0)
            mu = 1.0 / (1.0 + np.exp(-eta))
            variance = np.clip(mu * (1.0 - mu), 1e-9, None)
            gradient = design.T @ (weights * (y - mu)) - penalty @ beta
            hessian = design.T @ (design * (weights * variance)[:, None]) + penalty
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            beta = beta + step
            self.iterations_ = iteration + 1
            if float(np.max(np.abs(step))) < self.tolerance:
                self.converged_ = True
                break

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model is not fitted")
        z = (np.asarray(x, dtype=float) - self.mean_) / self.scale_
        return z @ self.coef_ + self.intercept_


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUC with ties counted as 0.5 (Mann-Whitney U / (n+ * n-))."""
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores_arr, kind="mergesort")
    ranks = np.empty(len(scores_arr), dtype=float)
    sorted_scores = scores_arr[order]
    index = 0
    while index < len(sorted_scores):
        stop = index
        while (
            stop + 1 < len(sorted_scores)
            and sorted_scores[stop + 1] == sorted_scores[index]
        ):
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        ranks[order[index : stop + 1]] = average_rank
        index = stop + 1
    rank_sum = float(np.sum(ranks[labels_arr == 1]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Stratified percentile bootstrap CI for AUC.

    Stratified because the class is rare: an unstratified resample of 60 rows
    containing 6 positives occasionally draws zero, and those draws would be
    dropped, biasing the interval optimistically.
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    pos_idx = np.flatnonzero(labels_arr == 1)
    neg_idx = np.flatnonzero(labels_arr == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {"lower": None, "upper": None, "n_resamples": 0}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        draws[i] = auc_score(labels_arr[idx], scores_arr[idx])
    lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lower": round(float(lower), 6),
        "upper": round(float(upper), 6),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def permutation_p_value(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    observed_auc: float,
    *,
    n_resamples: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Permutation test of the **whole procedure**, not of a single coefficient.

    Each resample permutes the label vector across all queries, then re-runs
    everything that produced `observed_auc`: standardise on the fit rows, fit
    ridge logistic on the fit rows, score the eval rows, and take the AUC
    against the *permuted* eval labels. This is the right null for the reported
    statistic. Permuting labels after the model is fitted would leave the fit
    untouched and test a much weaker null than the one the claim rests on.

    The number of permutations is read off the observed AUC, one-sided, since a
    larger AUC in the null distribution is evidence *against* the null.
    """
    rng = np.random.default_rng(seed)
    y = labels.astype(float)
    extreme = 0
    observed_dist: list[float] = []
    for _ in range(n_resamples):
        permuted = rng.permutation(y)
        model = RidgeLogistic().fit(matrix[fit_idx], permuted[fit_idx])
        scores = model.decision_function(matrix[eval_idx])
        draw = auc_score(permuted[eval_idx].astype(int), scores)
        if np.isfinite(draw):
            observed_dist.append(float(draw))
            if draw >= observed_auc:
                extreme += 1
    total = len(observed_dist)
    return {
        "p_value": round((extreme + 1) / (total + 1), 6),
        "n_effective_resamples": total,
        "n_resamples": n_resamples,
        "seed": seed,
        "observed_auc": round(float(observed_auc), 6),
        "null_auc_p95": (
            round(float(np.percentile(observed_dist, 95)), 6) if total else None
        ),
        "one_sided": "greater",
    }


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of the AUC, for a *true* AUC.

    Closed form, so the power sweep below is exact and costs nothing rather than
    nesting a bootstrap inside a power calculation.
    """
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    numerator = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    )
    return float(np.sqrt(max(numerator / (n_pos * n_neg), 0.0)))


def minimum_detectable_auc(n_pos: int, n_neg: int) -> dict[str, Any]:
    """How strong would a true association have to be for the CI to exclude 0.5?

    The counterfactual the gate needs. With 6 positives and 54 negatives the
    sampling noise in AUC is so large that only a strong true effect can push
    the interval clear of chance, so a null result must be reported as a power
    statement rather than as proof of absence.

    Power for a true AUC `theta` is `P(theta - 1.96*SE > 0.5)`, evaluated in
    closed form on the standard normal.
    """
    grid: list[dict[str, Any]] = []
    detectable: list[float] = []
    for true_auc in np.arange(0.55, 1.0001, 0.025):
        se = hanley_mcneil_se(float(true_auc), n_pos, n_neg)
        if se <= 0:
            power = 1.0
        else:
            z = (float(true_auc) - 1.96 * se - 0.5) / se
            power = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        if power >= 0.80:
            detectable.append(round(float(true_auc), 3))
        grid.append(
            {
                "true_auc": round(float(true_auc), 3),
                "standard_error": round(se, 6),
                "power_ci_excludes_0_5": round(float(power), 4),
            }
        )
    return {
        "method": "Hanley-McNeil analytic power",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "grid": grid,
        "min_detectable_auc_at_80pct_power": (
            round(min(detectable), 3) if detectable else None
        ),
        "note": (
            "smallest true AUC for which a 95% CI excludes 0.50 in >=80% of "
            "samples at this class balance"
        ),
    }


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------
def holm_bonferroni(p_values: Sequence[float], alpha: float = ALPHA) -> list[dict[str, Any]]:
    """Holm-Bonferroni step-down. Returns per-test adjusted p and a reject flag."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted: list[float] = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, min(candidate, 1.0))
        adjusted[index] = running
    rank_of = {index: position + 1 for position, index in enumerate(order)}
    return [
        {
            "p_holm": round(adjusted[i], 6),
            "significant_holm": bool(adjusted[i] < alpha),
            "rank": rank_of[i],
        }
        for i in range(m)
    ]


def univariate_screen(
    matrix: np.ndarray,
    labels: np.ndarray,
    columns: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    *,
    n_resamples: int = 2000,
    seed: int = PERMUTATION_SEED,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """One feature at a time: AUC, bootstrap CI, permutation p, Holm correction.

    Effect size first. AUC is reported for every feature *whether or not* it is
    significant, because with a handful of positives the honest output is the
    size of each association, not a filtered list of survivors.
    """
    rng = np.random.default_rng(seed)
    owner = {
        name: group for group, names in groups.items() for name in names
    }
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(columns):
        scores = matrix[:, index]
        observed = auc_score(labels.astype(int), scores)
        if not np.isfinite(observed):
            continue
        # Two-sided, against 0.5. A one-sided-positive screen would hide an
        # inverse predictor: a feature at AUC 0.323 separates just as well as
        # one at 0.677 once its sign is flipped, and the shipped router would
        # be free to use it that way. The direction is reported separately.
        observed_distance = abs(observed - 0.5)
        extreme = 0
        for _ in range(n_resamples):
            draw = auc_score(labels.astype(int), rng.permutation(scores))
            if np.isfinite(draw) and abs(draw - 0.5) >= observed_distance:
                extreme += 1
        p_raw = (extreme + 1) / (n_resamples + 1)
        ci = bootstrap_auc_ci(labels.astype(int), scores, n_resamples=1000, seed=seed)
        rows.append(
            {
                "feature": name,
                "group": owner.get(name, "unknown"),
                "auc": round(float(observed), 6),
                "direction": "positive" if observed >= 0.5 else "negative",
                "oriented_auc": round(
                    float(max(observed, 1.0 - observed)), 6
                ),
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
                "p_raw": round(float(p_raw), 6),
                "mean_positive": round(
                    float(np.mean(scores[labels == 1])), 6
                ),
                "mean_negative": round(
                    float(np.mean(scores[labels == 0])), 6
                ),
            }
        )

    for row, correction in zip(rows, holm_bonferroni([r["p_raw"] for r in rows], alpha)):
        row.update(correction)

    rows.sort(key=lambda r: (-r["oriented_auc"], r["feature"]))
    return {
        "family_size": len(rows),
        "correction": MULTIPLICITY_CORRECTION,
        "alpha": alpha,
        "test": "two-sided permutation on |AUC - 0.5|",
        "n_permutations_per_feature": n_resamples,
        "n_significant_holm": sum(1 for r in rows if r["significant_holm"]),
        "features": rows,
    }


def analyse_arm(
    arm_name: str,
    suite_root: Path,
    benchmark: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Full Gate 3 analysis for one corpus arm.

    Reuses the Gate 2 arm loader, so the contamination guard and the query-set
    equality assertion that caught the Phase 8 incident apply here without being
    restated. Labels come from all selectable arms (that is the target); features
    come only from the `adaptive` arm's traces (that is what a router sees).
    """
    systems = list(SELECTABLE_SYSTEMS)
    arm_rows = load_arm(suite_root, systems)
    panel = build_panel(arm_rows, systems)
    latencies = panel_latencies(arm_rows, systems)
    per_query = attach_latency(panel, latencies)

    reference = best_fixed(fixed_summaries(per_query, latencies, systems))
    labels_map = needed_split(per_query, systems, reference, GATE_EPSILON)

    features = read_observable_features(
        suite_root / f"{suite_root.name}__E1_baseline_comparison__{ADAPTIVE_SYSTEM}"
        / "traces.jsonl"
    )

    query_ids = sorted(
        q
        for q in per_query
        if q in features and q in benchmark
    )
    missing = sorted(set(per_query) - set(query_ids))
    if missing:
        raise QuerySetMismatch(
            f"{len(missing)} of {len(per_query)} queries lack either observable "
            f"features or a benchmark split (e.g. {missing[:5]}); the analysis "
            "would silently run on a subset"
        )

    matrix = design_matrix(features, query_ids)
    degenerate = constant_columns(matrix, FEATURE_COLUMNS)
    y = np.array(
        [1 if labels_map[q] == POSITIVE_LABEL else 0 for q in query_ids], dtype=int
    )
    split_of = np.array([str(benchmark[q]["split"]) for q in query_ids])
    fit_idx = np.flatnonzero(split_of == FIT_SPLIT)
    eval_idx = np.flatnonzero(split_of == EVAL_SPLIT)

    model = RidgeLogistic().fit(matrix[fit_idx], y[fit_idx])
    eval_scores = model.decision_function(matrix[eval_idx])
    observed_auc = auc_score(y[eval_idx], eval_scores)
    eval_ci = bootstrap_auc_ci(y[eval_idx], eval_scores)

    permutation = permutation_p_value(
        matrix, y, fit_idx, eval_idx, observed_auc
    )

    fit_auc = auc_score(y[fit_idx], model.decision_function(matrix[fit_idx]))
    full_scores = model.decision_function(matrix)
    in_sample_auc = auc_score(y, full_scores)

    univariate = univariate_screen(matrix, y, UNIVARIATE_COLUMNS, FEATURE_GROUPS)
    power = minimum_detectable_auc(
        int(np.sum(y[eval_idx] == 1)), int(np.sum(y[eval_idx] == 0))
    )

    coefficients = sorted(
        (
            {
                "feature": name,
                "coefficient": round(float(value), 6),
            }
            for name, value in zip(FEATURE_COLUMNS, model.coef_)
        ),
        key=lambda c: -abs(c["coefficient"]),
    )

    return {
        "arm": arm_name,
        "suite_root": str(suite_root),
        "reference_strategy": reference,
        "epsilon": GATE_EPSILON,
        "n_queries": len(query_ids),
        "constant_features": degenerate,
        "counts": {
            "n_positive_needed": int(np.sum(y == 1)),
            "n_negative_sufficient": int(np.sum(y == 0)),
            "base_rate": round(float(np.mean(y == 1)), 6),
            "by_split": {
                FIT_SPLIT: {
                    "n": int(np.sum(split_of == FIT_SPLIT)),
                    "n_positive": int(np.sum(y[fit_idx] == 1)),
                    "n_negative": int(np.sum(y[fit_idx] == 0)),
                },
                EVAL_SPLIT: {
                    "n": int(np.sum(split_of == EVAL_SPLIT)),
                    "n_positive": int(np.sum(y[eval_idx] == 1)),
                    "n_negative": int(np.sum(y[eval_idx] == 0)),
                },
            },
        },
        "model": {
            "kind": "ridge_logistic_irls",
            "ridge_lambda": RIDGE_LAMBDA,
            "class_weight": CLASS_WEIGHT,
            "n_features": len(FEATURE_COLUMNS),
            "n_fit_rows": int(fit_idx.size),
            "converged": bool(model.converged_),
            "iterations": int(model.iterations_),
            "fit_auc_in_sample_split": round(float(fit_auc), 6),
            "full_sample_auc": round(float(in_sample_auc), 6),
            "top_coefficients": coefficients[:10],
        },
        "primary_out_of_sample": {
            "metric": "AUC on the held-out test split",
            "auc": round(float(observed_auc), 6),
            "ci_lower": eval_ci["lower"],
            "ci_upper": eval_ci["upper"],
            "ci_excludes_0_5": bool(
                eval_ci["lower"] is not None and eval_ci["lower"] > 0.5
            ),
            "permutation": permutation,
        },
        "univariate_screen": univariate,
        "power": power,
        "per_query_evidence": build_evidence(
            query_ids, benchmark, features, labels_map, per_query, reference,
            systems, y, split_of, full_scores,
        ),
    }


def build_evidence(
    query_ids: Sequence[str],
    benchmark: Mapping[str, Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]],
    labels_map: Mapping[str, str],
    per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reference: str,
    systems: Sequence[str],
    y: np.ndarray,
    split_of: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-query evidence: label, split, every observable feature, oracle gain.

    Requirement: any query driving or diluting the result must be inspectable.
    This also records *which* strategy actually wins on each positive, because
    "needed" says the reference is the only answer but not which arm the oracle
    credited, and those two are not the same thing.
    """
    rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(query_ids):
        arms = per_query[query_id]
        best_recall = max(float(arms[s]["recall_at_5"]) for s in systems)
        winners = [s for s in systems if float(arms[s]["recall_at_5"]) == best_recall]
        gain = best_recall - float(arms[reference]["recall_at_5"])
        record = features[query_id]
        rows.append(
            {
                "query_id": query_id,
                "split": str(split_of[index]),
                "category": benchmark[query_id].get("category"),
                "label": labels_map[query_id],
                "y": int(y[index]),
                "reference_recall_at_5": round(
                    float(arms[reference]["recall_at_5"]), 6
                ),
                "best_recall_at_5": round(best_recall, 6),
                "oracle_gain": round(gain, 6),
                "oracle_winner": (
                    winners[0] if len(winners) == 1 else sorted(winners)
                ),
                "n_tied_at_max": len(winners),
                "model_score": round(float(scores[index]), 6),
                "features": {
                    name: record.get(name) for name in FEATURE_COLUMNS
                },
            }
        )
    rows.sort(key=lambda r: (r["y"] * -1, -r["oracle_gain"], r["query_id"]))
    return rows


def evaluate_arm_gate(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered rule to one arm. No judgement calls."""
    primary = arm["primary_out_of_sample"]
    conditions = {
        "permutation_p_below_alpha": {
            "observed": primary["permutation"]["p_value"],
            "threshold": ALPHA,
            "pass": bool(primary["permutation"]["p_value"] < ALPHA),
        },
        "auc_at_or_above_floor": {
            "observed": primary["auc"],
            "threshold": AUC_FLOOR,
            "pass": bool(primary["auc"] >= AUC_FLOOR),
        },
        "ci_lower_above_chance": {
            "observed": primary["ci_lower"],
            "threshold": 0.5,
            "pass": bool(primary["ci_excludes_0_5"]),
        },
        "a_feature_survives_holm": {
            "observed": arm["univariate_screen"]["n_significant_holm"],
            "threshold": 1,
            "pass": bool(arm["univariate_screen"]["n_significant_holm"] >= 1),
        },
    }
    passed = all(c["pass"] for c in conditions.values())
    return {
        "verdict": "signal_detected" if passed else "no_out_of_sample_signal",
        "passed": passed,
        "conditions": conditions,
    }


def evaluate_gate(arms: Mapping[str, Any]) -> dict[str, Any]:
    """The gate requires agreement across both corpus arms."""
    per_arm = {name: arm["gate"] for name, arm in arms.items()}
    replication = len({g["passed"] for g in per_arm.values()}) == 1 and all(
        g["passed"] for g in per_arm.values()
    )
    return {
        "verdict": (
            "signal_detected" if replication else "no_out_of_sample_signal"
        ),
        "replication_required": True,
        "replication_holds": bool(replication),
        "per_arm": per_arm,
        "permits": (
            "Gate 4: test whether the deterministic router can exploit the signal "
            "under corrected ladder/threshold configurations."
            if replication
            else "Nothing further. Oracle headroom exists (Gate 2) but is not "
            "identifiable from the currently available observable signals. A "
            "learned router must not be attempted merely because the rule-based "
            "router failed."
        ),
    }
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        return f"{float(value):.{digits}f}"
    return str(value)


def render(result: Mapping[str, Any]) -> str:
    lines = [
        "Gate 3 - full observable signal exhaustion",
        "=" * 72,
        "",
        "Question: can anything a router can observe, out of sample, tell it",
        "which queries need a different strategy?",
        "",
        f"Features: {len(FEATURE_COLUMNS)} columns from "
        f"{len(FEATURE_GROUPS) + 1} groups (query, retrieval-feedback, routing,",
        "initial strategy), all already recorded by the shipped router.",
        f"Split: fit on {FIT_SPLIT}, evaluate on {EVAL_SPLIT}.",
        f"Decision rule (pre-registered): p < {ALPHA}, AUC >= {AUC_FLOOR},",
        "CI lower > 0.50, >=1 feature surviving Holm, on BOTH arms.",
        "",
    ]

    for name, arm in result["corpus_arms"].items():
        counts = arm["counts"]
        primary = arm["primary_out_of_sample"]
        lines.append(f"[{name}] reference={arm['reference_strategy']}  "
                     f"n={arm['n_queries']}")
        lines.append(
            f"  positives (needed)={counts['n_positive_needed']}  "
            f"negatives={counts['n_negative_sufficient']}  "
            f"base rate={_fmt(counts['base_rate'] * 100, 1)}%"
        )
        lines.append(
            f"  {counts['by_split'][FIT_SPLIT]['n_positive']} positive in "
            f"{FIT_SPLIT} (n={counts['by_split'][FIT_SPLIT]['n']}), "
            f"{counts['by_split'][EVAL_SPLIT]['n_positive']} positive in "
            f"{EVAL_SPLIT} (n={counts['by_split'][EVAL_SPLIT]['n']})"
        )
        lines.append(
            f"  OUT-OF-SAMPLE AUC={_fmt(primary['auc'])}  "
            f"CI[{_fmt(primary['ci_lower'])}, {_fmt(primary['ci_upper'])}]  "
            f"permutation p={_fmt(primary['permutation']['p_value'])}"
        )
        lines.append(
            f"  in-sample AUC (full)={_fmt(arm['model']['full_sample_auc'])}  "
            f"fit-split AUC={_fmt(arm['model']['fit_auc_in_sample_split'])}"
        )
        lines.append("")
        lines.append("  univariate screen (Holm-Bonferroni, "
                     f"family of {arm['univariate_screen']['family_size']}):")
        lines.append(
            f"    {'feature':<22}{'group':<20}{'AUC':>8}{'95% CI':>20}"
            f"{'p':>9}{'p_holm':>9}"
        )
        for row in arm["univariate_screen"]["features"][:8]:
            ci = f"[{_fmt(row['ci_lower'], 2)}, {_fmt(row['ci_upper'], 2)}]"
            star = " *" if row["significant_holm"] else ""
            lines.append(
                f"    {row['feature']:<22}{row['group']:<20}"
                f"{_fmt(row['auc'], 3):>8}{ci:>20}"
                f"{_fmt(row['p_raw'], 3):>9}{_fmt(row['p_holm'], 3):>9}{star}"
            )
        lines.append(
            f"    surviving Holm at alpha={ALPHA}: "
            f"{arm['univariate_screen']['n_significant_holm']}"
            f" of {arm['univariate_screen']['family_size']}"
        )
        lines.append("")
        detectable = arm["power"]["min_detectable_auc_at_80pct_power"]
        lines.append(
            f"  power: a true AUC >= {_fmt(detectable, 2)} would be needed for "
            f"the bootstrap CI to exclude 0.50 in >=80% of resamples at this "
            f"class balance ({arm['power']['n_pos']}+ / {arm['power']['n_neg']}-)."
        )
        lines.append("")

    gate = result["gate_verdict"]
    lines.append(f"GATE 3 VERDICT: {gate['verdict']}")
    for name, arm_gate in gate["per_arm"].items():
        failed = [k for k, c in arm_gate["conditions"].items() if not c["pass"]]
        lines.append(f"  {name}: {arm_gate['verdict']}"
                     + (f"  (failing: {', '.join(failed)})" if failed else ""))
    lines.append("")
    lines.append(gate["permits"])
    lines.append("")
    lines.append("Scope: no learned router was trained or deployed, the shipped")
    lines.append("router is unmodified, and the corpus was not rebuilt.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate 3: test out of sample whether observable router signals "
            "identify the queries where a different strategy is beneficial. "
            "Identification study only - trains no router, changes nothing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "phase8" / "gate3_signal_exhaustion.json",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=REPO_ROOT / "data" / "evaluation" / "phase7_eval_v1.jsonl",
        help="Frozen benchmark; supplies the calibration/test split.",
    )
    args = parser.parse_args(argv)

    benchmark: dict[str, dict[str, Any]] = {}
    with open(args.benchmark, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                benchmark[str(record["example_id"])] = record

    corpus_arms: dict[str, Any] = {}
    try:
        for name, root in CORPUS_ARMS.items():
            print(f"analysing {name} from {root} ...", flush=True)
            arm = analyse_arm(name, root, benchmark)
            arm["gate"] = evaluate_arm_gate(arm)
            corpus_arms[name] = arm
    except (QuerySetMismatch, FileNotFoundError) as exc:
        print(f"aborting: {exc}", file=sys.stderr)
        return 2

    result = {
        "result_version": RESULT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate": "Gate 3 - full observable signal exhaustion",
        "question": (
            "Can observable information available to the router distinguish the "
            "small subset of queries for which a different retrieval strategy "
            "is actually beneficial, out of sample?"
        ),
        "study": (
            "Identification study. A ridge logistic regression is used as a "
            "measuring instrument only: no decision rule is derived from it, no "
            "router is configured from it, and src/adaptive_rag/routing/ is "
            "untouched. Labels come from per-query recall@5 across the "
            "selectable E1 arms; features come only from the adaptive arm's "
            "traces, so no oracle outcome enters the feature side."
        ),
        "selectable_systems": list(SELECTABLE_SYSTEMS),
        "features": {
            "groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
            "categorical": list(CATEGORICAL_FEATURES),
            "columns": list(FEATURE_COLUMNS),
            "n_columns": len(FEATURE_COLUMNS),
            "no_new_features_introduced": True,
        },
        "splits": {"fit": FIT_SPLIT, "evaluate": EVAL_SPLIT},
        "pre_registered_criteria": decision_criteria(),
        "corpus_arms": corpus_arms,
    }
    result["gate_verdict"] = evaluate_gate(corpus_arms)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(render(result))
    print(f"\nartifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
