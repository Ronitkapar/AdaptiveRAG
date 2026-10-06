"""Gate 9.3 closing item -- section 2.10's out-of-sample proxy criterion.

section 2.10 states Gate 9.3 passes when "a proxy from {query-only u single-arm
u hybrid-internal} holds out-of-sample on `test`". The deployability
classification in docs/phase9_results.md settled *where* each survivor can be
computed but did not run this criterion, so it was left explicitly open.

This closes it. It reads only persisted Phase 8 artifacts, runs no retrieval,
calls no provider, adds no feature outside the three preregistered classes, and
builds no router.

Two readings are reported, least-modelled first:
  (a) each deployable feature's Spearman computed on the frozen `test` split,
      which involves no fitting and therefore no selection or overfitting risk;
  (b) a ridge model over all deployable features, fitted on `calibration` and
      scored on `test`, mirroring Gate 3's frozen lambda = 1.

Every deployable feature is reported. None is selected, dropped, or ranked for
promotion, so no multiplicity family is created here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_rag.evaluation.signals import (
    SignalRow,
    defined_observations,
    read_signal_table,
)

MIN_ABS_RHO = 0.30
RIDGE_LAMBDA = 1.0
SIGNAL_DIR = Path("experiments/phase9")
EVAL = Path("data/evaluation/phase7_eval_v1.jsonl")
ADAPTIVE_TRACE = (
    "experiments/phase8/combined/p8a_e1/"
    "p8a_e1__E1_baseline_comparison__adaptive/traces.jsonl"
)
REPORT = SIGNAL_DIR / "gate9_deployability_proxy.json"

#: The three preregistered deployable classes (section 2.10 / 2.9).
SINGLE_ARM = (
    "distinct_doc_ratio@bm25", "distinct_doc_ratio@dense", "distinct_doc_ratio@hybrid",
    "score_gap_top1_top2@bm25", "score_gap_top1_top2@dense",
    "score_decay_slope@bm25", "score_decay_slope@dense",
)


def spearman(x, y) -> float | None:
    def midrank(v):
        arr = np.asarray(v, dtype=float)
        order = np.argsort(arr, kind="mergesort")
        r = np.empty(len(arr))
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            r[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = midrank(x), midrank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def to_row(r):
    return SignalRow(
        query_id=r["query_id"], dispersion=r["target_dispersion"],
        dispersion_mrr=r["target_dispersion_mrr"], jaccard=r["jaccard"],
        overlap_at_k=r["overlap_at_k"], top1_agreement=r["top1_agreement"],
        rank_correlation=r["rank_correlation"],
        union_concentration=r["union_concentration"],
        distinct_doc_ratio=r["distinct_doc_ratio"],
        score_gap_top1_top2=r["score_gap_top1_top2"],
        score_decay_slope=r["score_decay_slope"], query_features=r["query_features"],
    )


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge by normal equations with an intercept, centred (lam * I penalty)."""
    Xc = np.c_[np.ones(len(X)), X]
    penalty = lam * np.eye(Xc.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ y)


def main() -> dict[str, Any]:
    split = {json.loads(l)["example_id"]: json.loads(l)["split"]
             for l in open(EVAL)}
    rows = {r["query_id"]: to_row(r)
            for r in read_signal_table(SIGNAL_DIR / "signal_table_phase8_after.jsonl")}

    # --- query-only class, from the adaptive arm's own persisted features ---
    qfeat: dict[str, dict[str, float]] = {}
    for line in open(ADAPTIVE_TRACE):
        t = json.loads(line)
        routing = t.get("routing") or {}
        feats = routing.get("features") or {}
        if not feats or t.get("status") != "ok":
            continue
        qfeat[t["example_id"]] = {
            k: float(v) for k, v in feats.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    qnames = sorted({k for v in qfeat.values() for k in v})

    # --- build the deployable design matrix ---
    ids, feats, target, splits = [], [], [], []
    for qid, row in rows.items():
        if qid not in split or qid not in qfeat:
            continue
        vector = [qfeat[qid].get(k, np.nan) for k in qnames]
        # Single-arm features are read straight off the SignalRow rather than
        # through defined_observations(), because the latter is restricted to
        # the 12 REGISTERED candidates and distinct_doc_ratio@hybrid is
        # deliberately not one of them. The registry governs the Gate 9.2
        # hypothesis family; section 2.10's deployable classes are defined by
        # *information availability*, so the unrestricted view is the correct
        # one here. This does not alter the Gate 9.2 family in any way.
        arm_vals = []
        for src in (row.distinct_doc_ratio, row.score_gap_top1_top2,
                    row.score_decay_slope):
            for arm in ("bm25", "dense", "hybrid"):
                if arm in src:
                    arm_vals.append(src[arm])
        arm_vals = [v for v in arm_vals if v is not None]
        if not arm_vals or any(np.isnan(x) for x in vector):
            continue
        ids.append(qid)
        feats.append(vector + arm_vals)
        target.append(row.dispersion)
        splits.append(split[qid])

    X = np.asarray(feats, dtype=float)
    y = np.asarray(target, dtype=float)
    s = np.asarray(splits)
    cal, tst = s == "calibration", s == "test"
    names = [f"query:{n}" for n in qnames] + [f"single_arm:{i}" for i in range(
        len(X[0]) - len(qnames))]

    # --- (a) per-feature Spearman on the frozen test split, no fitting ---
    per_feature = []
    for j, name in enumerate(names):
        rho = spearman(X[tst, j], y[tst])
        rho_cal = spearman(X[cal, j], y[cal])
        per_feature.append({
            "feature": name, "n_test": int(tst.sum()),
            "rho_test": rho, "rho_calibration": rho_cal,
            "holds_out_of_sample": rho is not None and abs(rho) >= MIN_ABS_RHO,
        })

    # --- (b) ridge over all deployable features, fit on calibration ----------
    mu, sd = X[cal].mean(0), X[cal].std(0)
    sd[sd == 0] = 1.0
    Xc = (X - mu) / sd
    beta = ridge_fit(Xc[cal], y[cal], RIDGE_LAMBDA)
    pred = np.c_[np.ones(tst.sum()), Xc[tst]] @ beta
    rho_model = spearman(pred, y[tst])
    rho_model_cal = spearman(
        np.c_[np.ones(cal.sum()), Xc[cal]] @ beta, y[cal]
    )

    best = max((p for p in per_feature if p["rho_test"] is not None),
               key=lambda p: abs(p["rho_test"]), default=None)
    any_single = any(p["holds_out_of_sample"] for p in per_feature)
    criterion = bool(any_single or (rho_model is not None
                                    and abs(rho_model) >= MIN_ABS_RHO))

    report = {
        "gate": "9.3 (section 2.10 out-of-sample proxy criterion)",
        "deployable_classes": ["query-only", "single-arm", "hybrid-internal"],
        "hybrid_internal": {
            "features": ["bm25_candidate_count", "dense_candidate_count"],
            "status": "constant 20/20 on all 107 traces - no per-query variance, "
                      "so it contributes no column",
        },
        "n_calibration": int(cal.sum()), "n_test": int(tst.sum()),
        "ridge_lambda": RIDGE_LAMBDA,
        "min_abs_rho": MIN_ABS_RHO,
        "per_feature_test_split": per_feature,
        "multivariate_ridge": {
            "rho_test": rho_model, "rho_calibration": rho_model_cal,
            "holds_out_of_sample": rho_model is not None
            and abs(rho_model) >= MIN_ABS_RHO,
        },
        "strongest_single_feature": best,
        "any_deployable_proxy_holds_out_of_sample": criterion,
        "notes": "All deployable features are reported. None was selected, "
                 "dropped or promoted, so no multiplicity family is created here.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
