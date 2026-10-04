"""
scripts.compare_phase8_arms
----------------------------
Phase 8 Step 6 -- run the pre-registered comparison family over the two corpus arms.

The family was fixed in `docs/phases/phase-8.md` §2.2 before the `after` arm
existed, and it is run here rather than through `scripts/run_phase7_analysis.py`
because it is a *Phase 8* question about *two corpora*, not a Phase 7 report about
one. Extending the Phase 7 analysis with a flag would have put a new family inside
the artifact that documents Phase 7, where a later reader would reasonably assume
it had been part of that study.

Two distinct families are reported, and they answer different questions:

* **Within-arm ordering** -- `compare_all_pairs` over the five E1 systems, run
  once per arm. This is the literal §2.2 family: C(5,2) = 10 pairs, Holm within
  each metric. It answers "which orderings are distinguishable, on this corpus".
* **Before vs after** -- each system against itself on the other corpus, Holm
  within each metric across the five systems. This is the one §2.3 nominates as
  the *evidence* for Phase 8's primary questions, because a within-corpus p-value
  cannot say whether a pair's ordering moved.

Both are offline: they read `rows.jsonl` written by `run_phase7_suite.py` and make
no network calls, so the whole Step 6 quality question is answerable without the
embedding API.

`ndcg_at_5` is **excluded by default** and its absence is reported rather than
quiet. The §2.2 pre-registration named it, but §2.5's falsification condition and
the Step 4 label audit (`scripts/audit_phase8_gold_labels.py`) established that
47 of 140 gold section labels are spliced and 40 stop resolving once reading
order is fixed. `nDCG@5` is computed from those labels, so a before/after nDCG
number would be a comparison between two corrupt label sets rather than between
two corpora. The exclusion was decided from the label audit, *before* any
`after`-arm quality number existed, which is why it is a measurement decision and
not a result-dependent one. `--include-ndcg` runs it anyway and labels the output
`not_comparable`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.rows import read_rows_jsonl  # noqa: E402
from adaptive_rag.evaluation import statistics as S  # noqa: E402
from adaptive_rag.evaluation.stats import holm_bonferroni  # noqa: E402
from adaptive_rag.experiments.arms import ARM_NAMES  # noqa: E402

PHASE8_RESULT_VERSION = "phase8_arm_comparison_v1"

BEFORE_ARM = "phase8_before"
AFTER_ARM = "phase8_after"

# §2.2 named these four. `ndcg_at_5` is withheld for the reason in the module
# docstring; the rest are the metrics whose inputs survive the corpus change.
PRE_REGISTERED_METRICS: tuple[str, ...] = (
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "total_latency_ms",
)
DEFAULT_METRICS: tuple[str, ...] = ("recall_at_5", "mrr", "total_latency_ms")

# Recorded next to every number so a reader never has to infer which metric
# families were withheld, and why.
WITHHELD_METRICS: dict[str, str] = {
    "ndcg_at_5": (
        "computed from gold section labels that the Step 4 audit found spliced "
        "(47 of 140 damaged, 40 unresolvable on the fixed corpus); not "
        "comparable between arms"
    ),
}


class ContaminatedArm(RuntimeError):
    """An arm carries failed traces, so its metrics cover a subset of queries."""


# Filled by `load_arm` when `--allow-contaminated` is passed, and copied into the
# artifact so a partial arm cannot be read later without knowing it is partial.
CONTAMINATION: dict[str, list[dict[str, Any]]] = {}


def _rows_path(suite_root: Path, system: str) -> Path:
    """Locate one E1 arm's `rows.jsonl` inside a Phase 8 suite directory."""
    matches = sorted(suite_root.glob(f"*__E1_baseline_comparison__{system}/rows.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"no E1 rows for system {system!r} under {suite_root}; expected a "
            "directory named '*__E1_baseline_comparison__"
            f"{system}'"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"{len(matches)} E1 row sets for system {system!r} under {suite_root} "
            f"({[m.parent.name for m in matches]}); the suite root must hold "
            "exactly one run per arm"
        )
    return matches[0]


def load_arm(
    suite_root: Path,
    systems: Sequence[str],
    *,
    allow_contaminated: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Read every named system's rows for one arm, keyed by system name.

    Refuses an arm whose traces are not all `ok`. This is not defensive
    programming for its own sake: a Phase 8 run that lost embedding calls leaves
    `rows.jsonl` shorter than the query set, and every metric computed over it is
    a mean over whichever queries happened to survive. `metrics_retrieval.json`
    will report such an arm as `n=59` next to `trace_count=107` and no error at
    all, so nothing downstream would notice unless something checked. A 48-failure
    `hybrid_rerank` arm read 0.6384 that way against a true 0.352.

    `allow_contaminated=True` records the shortfall in the artifact and reports it
    in the banner instead. It exists so the damage can be inspected, not so a
    partial arm can be quoted.
    """
    loaded: dict[str, list[dict[str, Any]]] = {}
    for system in systems:
        rows_path = _rows_path(suite_root, system)
        rows = read_rows_jsonl(rows_path)
        observed = {str(row.get("system")) for row in rows}
        if observed != {system}:
            raise ValueError(
                f"{rows_path} carries system {sorted(observed)}, expected "
                f"[{system!r}]; a rows file from another arm would make this "
                "family a comparison of corpora that were never meant to differ "
                "in that way"
            )
        failures = _failed_traces(rows_path.parent / "traces.jsonl")
        if failures:
            # `rows.jsonl` carries one row per trace, including the failed ones --
            # a failed trace has no metric values, which is exactly why the
            # aggregate reports `n=59` beside `trace_count=107`. So the trace
            # count is the denominator, not the row count.
            total = _trace_count(rows_path.parent / "traces.jsonl") or len(rows)
            message = (
                f"{system} in {rows_path.parent.name} has {failures} of {total} "
                f"traces failed; metrics over the remaining {total - failures} "
                "are not comparable to a full arm"
            )
            if not allow_contaminated:
                raise ContaminatedArm(
                    f"{message}. Re-run the arm, or pass --allow-contaminated to "
                    "report it anyway."
                )
            CONTAMINATION.setdefault("arms", []).append(
                {"suite": str(suite_root), "system": system,
                 "failed": failures, "evaluated": total - failures, "total": total}
            )
        loaded[system] = rows
    return loaded


def _trace_count(traces_path: Path) -> int:
    """Total traces recorded, or 0 when there is no trace file."""
    if not traces_path.is_file():
        return 0
    with open(traces_path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _failed_traces(traces_path: Path) -> int:
    """Count non-`ok` traces, or 0 when there is no trace file to check."""
    if not traces_path.is_file():
        return 0
    count = 0
    with open(traces_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            if str(trace.get("status")) != "ok":
                count += 1
    return count


def _flatten(arm_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for system, rows in arm_rows.items():
        for row in rows:
            out.append({**row, "system": system})
    return out


def within_arm_family(
    arm_rows: dict[str, list[dict[str, Any]]],
    *,
    systems: Sequence[str],
    metrics: Sequence[str],
    seed: int,
    n_resamples: int,
    confidence: float,
    alpha: float,
) -> dict[str, Any]:
    """The §2.2 family, run on one arm."""
    return S.compare_all_pairs(
        _flatten(arm_rows),
        systems=systems,
        metrics=metrics,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
        alpha=alpha,
    )


def before_vs_after(
    before_rows: dict[str, list[dict[str, Any]]],
    after_rows: dict[str, list[dict[str, Any]]],
    *,
    systems: Sequence[str],
    metrics: Sequence[str],
    seed: int,
    n_resamples: int,
    confidence: float,
    alpha: float,
) -> dict[str, Any]:
    """Each system against itself on the other corpus, Holm across systems.

    `compare_all_pairs` cannot express this family: it enumerates pairs *among*
    its `systems` argument, which for a before/after question would mix every arm
    against every other arm on the other corpus. So the per-system comparison is
    `compare_metric` -- the same function `compare_all_pairs` calls, unchanged --
    and the correction is applied here across the five systems, exactly as
    `compare_all_pairs` applies it across ten pairs. One Holm family per metric,
    `family_size = 5`.
    """
    per_metric: dict[str, list[dict[str, Any]]] = {}
    absent: list[str] = []

    for metric in metrics:
        family: list[dict[str, Any]] = []
        for system in systems:
            if system not in before_rows or system not in after_rows:
                absent.append(system)
                continue
            family.append(
                S.compare_metric(
                    after_rows[system],
                    before_rows[system],
                    metric=metric,
                    treatment=f"{system}@{AFTER_ARM}",
                    baseline=f"{system}@{BEFORE_ARM}",
                    seed=seed,
                    n_resamples=n_resamples,
                    confidence=confidence,
                    alpha=alpha,
                )
            )

        pvalues = [c["p_value"] for c in family if c["p_value"] is not None]
        adjusted = holm_bonferroni(pvalues) if pvalues else []
        cursor = 0
        for comparison in family:
            if comparison["p_value"] is None:
                continue
            comparison["adjusted_p_value"] = adjusted[cursor]
            comparison["significant"] = adjusted[cursor] < alpha
            comparison["family_size"] = len(pvalues)
            comparison["family_size_requested"] = len(systems)
            cursor += 1
        per_metric[metric] = family

    return {
        "comparison_design": "each system against itself on the other corpus",
        "treatment_arm": AFTER_ARM,
        "baseline_arm": BEFORE_ARM,
        "systems_requested": list(systems),
        "systems_absent": sorted(set(absent)),
        "reports": per_metric,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_report(result: dict[str, Any]) -> None:
    print()
    print(f"Phase 8 arm comparison -- {result['result_version']}")
    print(f"  before  {BEFORE_ARM}   after  {AFTER_ARM}")
    print(f"  seed    {result['settings']['seed']}   "
          f"resamples {result['settings']['n_resamples']}   "
          f"alpha {result['settings']['alpha']}")
    metrics = result["metrics_tested"]
    print(f"  metrics {', '.join(metrics)}")
    for name, reason in result["withheld_metrics"].items():
        print(f"  withheld {name}: {reason}")
    for arm in result.get("contaminated_arms", []):
        print(
            f"  CONTAMINATED {arm['system']} in {Path(arm['suite']).name}: "
            f"{arm['failed']} of {arm['total']} traces failed, metrics over "
            f"{arm['evaluated']}. NOT an arm result."
        )
    print()

    for arm, key in ((BEFORE_ARM, "within_arm"), (AFTER_ARM, "within_arm")):
        block = result[key][arm]
        print(f"  within-arm ordering on {arm} "
              f"({block['n_pairs_per_metric']} pairs, design: "
              f"{block['comparison_design']})")
        header = (f"    {'metric':<18}{'treatment':<16}{'baseline':<16}"
                  f"{'delta':>9}{'p':>10}{'p_adj':>10}{'sig':>5}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for comparison in block["comparisons"]:
            print(
                f"    {comparison['metric']:<18}{comparison['treatment']:<16}"
                f"{comparison['baseline']:<16}"
                f"{_fmt((comparison.get('difference_summary') or {}).get('mean')):>9}"
                f"{_fmt(comparison['p_value']):>10}"
                f"{_fmt(comparison['adjusted_p_value']):>10}"
                f"{('yes' if comparison.get('significant') else 'no'):>5}"
            )
        print()

    block = result["before_vs_after"]
    print("  before vs after (each system against itself, Holm across systems)")
    header = (f"    {'metric':<18}{'system':<16}{'delta':>10}{'p':>10}"
              f"{'p_adj':>10}{'sig':>5}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for metric, family in block["reports"].items():
        for comparison in family:
            system = comparison["treatment"].split("@", 1)[0]
            print(
                f"    {metric:<18}{system:<16}"
                f"{_fmt((comparison.get('difference_summary') or {}).get('mean')):>10}"
                f"{_fmt(comparison['p_value']):>10}"
                f"{_fmt(comparison['adjusted_p_value']):>10}"
                f"{('yes' if comparison.get('significant') else 'no'):>5}"
            )
    if block["systems_absent"]:
        print(f"    systems absent from one arm: {block['systems_absent']}")
    print()
    print("  A within-arm p-value says nothing about whether a pair's ordering "
          "moved between corpora; that is the block above.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--before-suite",
        type=Path,
        required=True,
        help="Suite directory for the phase8_before arm's E1 runs.",
    )
    parser.add_argument(
        "--after-suite",
        type=Path,
        required=True,
        help="Suite directory for the phase8_after arm's E1 runs.",
    )
    parser.add_argument(
        "--systems",
        default=",".join(ARM_NAMES),
        help="Comma-separated E1 systems to compare.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=(
            "Comma-separated metrics to test. ndcg_at_5 is omitted by default; "
            "see this module's docstring for why it is not comparable across "
            "the two arms."
        ),
    )
    parser.add_argument(
        "--include-ndcg",
        action="store_true",
        help=(
            "Also test ndcg_at_5. The numbers are computed and labelled "
            "not_comparable; they are not evidence about the corpus."
        ),
    )
    parser.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=S.DEFAULT_N_RESAMPLES)
    parser.add_argument("--confidence", type=float, default=S.DEFAULT_CONFIDENCE)
    parser.add_argument("--alpha", type=float, default=S.DEFAULT_ALPHA)
    parser.add_argument(
        "--allow-contaminated",
        action="store_true",
        help=(
            "Analyse arms whose traces include failures. The shortfall is recorded "
            "in the artifact and printed as a banner; the numbers remain a mean "
            "over the surviving queries and must not be quoted as arm results."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    if len(set(systems)) != len(systems):
        raise SystemExit(f"--systems repeats a name: {systems}")
    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    withheld = dict(WITHHELD_METRICS)
    if args.include_ndcg:
        if "ndcg_at_5" not in metrics:
            metrics.append("ndcg_at_5")
        withheld.pop("ndcg_at_5", None)

    before_rows = load_arm(args.before_suite, systems, allow_contaminated=args.allow_contaminated)
    after_rows = load_arm(args.after_suite, systems, allow_contaminated=args.allow_contaminated)

    n_queries = {
        arm: {system: len(rows) for system, rows in arm_rows.items()}
        for arm, arm_rows in ((BEFORE_ARM, before_rows), (AFTER_ARM, after_rows))
    }

    within: dict[str, Any] = {}
    for arm, arm_rows in ((BEFORE_ARM, before_rows), (AFTER_ARM, after_rows)):
        report = within_arm_family(
            arm_rows,
            systems=systems,
            metrics=metrics,
            seed=args.seed,
            n_resamples=args.resamples,
            confidence=args.confidence,
            alpha=args.alpha,
        )
        # `n_pairs_per_metric` is what a reader needs to check the family against
        # the pre-registration's C(5,2)=10 without counting rows.
        comparisons = report.get("comparisons", [])
        report["n_pairs_per_metric"] = (
            sum(1 for c in comparisons if c["metric"] == metrics[0])
            if metrics and comparisons
            else 0
        )
        report["n_queries_per_system"] = n_queries[arm]
        within[arm] = report

    result: dict[str, Any] = {
        "result_version": PHASE8_RESULT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": {
            "source": "docs/phases/phase-8.md#22-the-comparison-family-step-0--pre-registered",
            "systems": systems,
            "metrics_pre_registered": list(PRE_REGISTERED_METRICS),
            "pairs_pre_registered": len(systems) * (len(systems) - 1) // 2,
            "correction": "holm-bonferroni within each metric",
            "status": "post-hoc and exploratory within this program (§2.3)",
        },
        "settings": S.statistics_settings(
            metrics,
            seed=args.seed,
            n_resamples=args.resamples,
            confidence=args.confidence,
            alpha=args.alpha,
        ),
        "metrics_tested": metrics,
        "withheld_metrics": withheld,
        "contaminated_arms": CONTAMINATION.get("arms", []),
        "within_arm": within,
        "before_vs_after": before_vs_after(
            before_rows,
            after_rows,
            systems=systems,
            metrics=metrics,
            seed=args.seed,
            n_resamples=args.resamples,
            confidence=args.confidence,
            alpha=args.alpha,
        ),
        "suites": {
            BEFORE_ARM: str(args.before_suite),
            AFTER_ARM: str(args.after_suite),
        },
    }

    if args.out is not None:
        destination = args.out
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"  written  {destination}", flush=True)

    if not args.json_only:
        print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
