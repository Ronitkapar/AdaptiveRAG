#!/usr/bin/env python
"""
scripts.run_phase7_analysis
---------------------------
Phase 7 *analysis* driver: turn the row exports the suite produced into the E1-E8
results, the paired statistics, and every mandated table.

This script runs no retrieval. It reads what `scripts/run_phase7_suite.py` wrote
-- a registry entry per arm, each pointing at a run directory holding
`rows.jsonl` -- and analyses those rows. That separation is the point: the suite
is the only thing that touches a corpus, an index or a provider, so the analysis
can be re-run, re-seeded and re-checked on a machine with none of them.

    python scripts/run_phase7_analysis.py --dataset data/evaluation/phase7_eval_v1.jsonl
    python scripts/run_phase7_analysis.py --dataset data/evaluation/phase7_eval_v1.jsonl --split test --baseline hybrid
    python scripts/run_phase7_analysis.py --dataset data/evaluation/phase7_eval_v1.jsonl --experiments E1_baseline_comparison
    python scripts/run_phase7_analysis.py --dataset data/evaluation/phase7_eval_v1.jsonl --seed 1 --resamples 2000 --alpha 0.01

One flag is worth knowing about before a long analysis:

* `--dataset` has **no default**, on purpose, and it is the same contract
  `scripts/run_phase7_suite.py` enforces. Phase 7 is defined over the frozen
  107-record `phase7_eval_v1.jsonl`; the Phase 2-6 20-record `dense_eval_v1.jsonl`
  also lives in that directory and shares no `example_id` with it, so a default
  pointing there would hand E8 a gold relevance map keyed by queries the Phase 7
  rows do not contain. Naming the dataset is the cheapest possible provenance, so
  it is required rather than defaulted, and the legacy file is refused unless
  `--allow-legacy-dataset` says it was deliberate.

Outputs, all under `experiments/phase7/analysis/`:

    analysis.json      every analyzer's result, keyed by study id
    statistics.json    every paired comparison, with the test-selection rule
    provenance.json    what was analysed, and under what conditions
    tables/*.md|csv|json   one file per table, plus analysis_report.md

Provenance is written the way `scripts/measure_strategy_cost.py` writes it, from
the same gate helpers, and it records the things a reader cannot recover from the
numbers: the git commit, the dataset version *and* its sha256, the config hash,
the corpus version, the environment, the exact bootstrap seed and resample count,
and the row-selection rule. Two runs that disagree on any of those are not two
readings of one measurement.

What this script will not do: invent a study. A study with no registered runs is
reported as absent, its table is not written, and the provenance says so. E6,
E7 and E8 are analyses over the runs E1-E3 produced, so a registry holding only
E2 arms legitimately yields no E6 number.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation import analysis as A  # noqa: E402
from adaptive_rag.evaluation import statistics as S  # noqa: E402
from adaptive_rag.evaluation.statistics import _group_rows  # noqa: E402
from adaptive_rag.evaluation import tables as T  # noqa: E402
from adaptive_rag.evaluation.dataset import (  # noqa: E402
    LEGACY_DENSE_DATASET_PATH,
    PHASE7_DATASET_PATH,
    VALID_SPLITS,
    is_legacy_dataset,
    load_evaluation_dataset,
)
from adaptive_rag.evaluation.rows import read_rows_jsonl  # noqa: E402
from adaptive_rag.evaluation.suite import (  # noqa: E402
    ANALYSIS_ONLY_EXPERIMENT_IDS,
    DEFAULT_SUITE_ROOT,
)
from adaptive_rag.errors import AdaptiveRAGError  # noqa: E402
from adaptive_rag.experiments.arms import ARM_NAMES  # noqa: E402
from adaptive_rag.experiments.registry import (  # noqa: E402
    DEFAULT_REGISTRY_NAME,
    EXPERIMENT_IDS,
    ExperimentRegistry,
    RegistryEntry,
    normalize_experiment_id,
)

ANALYSIS_ARTIFACT_VERSION = "phase7_analysis_artifact_v1"

DEFAULT_OUT = DEFAULT_SUITE_ROOT / "analysis"

# Which study's rows feed the analysis-only studies. E6 measures the adaptive
# arm's routing clock, E7 breaks that arm down by category, and E8 reads its
# escalation record; none of them needs a run of its own.
DEFAULT_ANALYSIS_SOURCE = "E1_baseline_comparison"

# The system E7's per-category breakdown is computed over. E7 is a statement
# about how the *router* treats each query category, so it is the adaptive arm's
# rows and not the union of all five arms -- a union would average the router's
# behaviour with four fixed strategies that never route at all.
DEFAULT_E7_SYSTEM = "adaptive"

DEFAULT_METRICS: tuple[str, ...] = (
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "total_latency_ms",
    "estimated_cost_usd",
)

# Which variant each ablation study is measured against.
# E2 is a component-addition ablation (A -> B -> C). The scientifically
# meaningful pairs are A-vs-B (cost of measuring sufficiency) and B-vs-C
# (value of acting on it), NOT all-vs-one-baseline. This dict names the
# *floor* variant for each study; build_statistics handles E2's chained
# comparisons explicitly. See ablation.py:escalation_variants docstring.
ABLATION_BASELINES: dict[str, str] = {
    "E2_escalation_ablation": "A_no_sufficiency_no_escalation",
    "E3_feature_ablation": "full",
}


def _load_gate_module() -> Any:
    """Import the sibling gate script so provenance is written one way.

    `scripts/` has no `__init__.py`, so this cannot be a normal import. Loaded
    by path rather than duplicated, exactly as `run_phase7_suite.py` and
    `measure_strategy_cost.py` do: three copies of `_snapshot` would drift, and
    an analysis artifact describing a different machine than the gate that
    preceded it is hard to trust.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "validate_environment.py"
    spec = importlib.util.spec_from_file_location("_phase7_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate_module()


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------


def _rows_path(entry: RegistryEntry) -> Path:
    return Path(entry.run_dir) / "rows.jsonl"


def load_entries(
    registry_path: Path, *, experiments: Sequence[str] | None = None
) -> tuple[list[RegistryEntry], list[str]]:
    """Registered runs, plus the study ids that had none.

    A registered arm whose `rows.jsonl` is missing is a *finding*, not a
    silent skip: the registry says the arm ran, and a reader of the analysis
    needs to know its rows are unavailable rather than assume it was never run.
    """
    registry = ExperimentRegistry.load(registry_path)
    wanted = (
        {normalize_experiment_id(eid) for eid in experiments}
        if experiments
        else set(EXPERIMENT_IDS)
    )
    entries: list[RegistryEntry] = []
    missing: list[str] = []
    for entry in registry.entries:
        if entry.experiment_id not in wanted:
            continue
        if not _rows_path(entry).is_file():
            missing.append(f"{entry.experiment_id}/{entry.variant}")
            continue
        entries.append(entry)
    # E6/E7/E8 are analyses over the runs E1-E3 produce, so they have no
    # registered runs by construction and are not "absent" -- reporting them as
    # missing would tell a reader their routing-overhead study did not happen
    # when in fact it ran over the E1 rows just below.
    absent = sorted(
        (wanted - set(ANALYSIS_ONLY_EXPERIMENT_IDS))
        - {entry.experiment_id for entry in entries}
        - {mishap.split("/", 1)[0] for mishap in missing}
    )
    return entries, absent


def rows_by_experiment(
    entries: Sequence[RegistryEntry], *, split: str | None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Every registered arm's rows, grouped by study, optionally split-filtered.

    The split filter is applied here, once, and recorded in the provenance: a
    report over calibration queries labelled as a test result is the exact
    failure the split exists to prevent, and the only defence is that the filter
    is stated in the artifact next to the numbers.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    kept = 0
    dropped = 0
    for entry in entries:
        rows = read_rows_jsonl(_rows_path(entry))
        if split:
            rows = [row for row in rows if str(row.get("split")) == split]
        dropped += len(read_rows_jsonl(_rows_path(entry))) - len(rows) if split else 0
        kept += len(rows)
        grouped.setdefault(entry.experiment_id, []).extend(rows)
    return grouped, {"rows_kept": kept, "rows_dropped_by_split": dropped}


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------


def _source_rows(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]], experiment_id: str
) -> list[dict[str, Any]]:
    return [dict(row) for row in grouped.get(experiment_id, [])]


def build_analyses(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    min_cell: int,
    systems: Sequence[str],
    source_study: str = DEFAULT_ANALYSIS_SOURCE,
    source_system: str = DEFAULT_E7_SYSTEM,
    gold_chunk_ids: Mapping[str, set[str]] | None = None,
    gold_document_ids: Mapping[str, set[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every analyzer whose input rows exist.

    A study with no rows produces no result and is listed by the caller as
    absent. E6/E7/E8 read the E1 rows (or whichever study `--analysis-source`
    names), because those three are analyses over runs rather than runs.
    """
    analyses: dict[str, dict[str, Any]] = {}

    e1_rows = _source_rows(grouped, "E1_baseline_comparison")
    if e1_rows:
        analyses["E1"] = A.analyze_main_comparison(
            e1_rows, systems=systems, min_cell=min_cell
        )
    # E6/E7/E8 measure the routing arm rather than running their own: the clock,
    # the category breakdown and the escalation record are all properties of the
    # adaptive run that E1 already produced.
    routed = _filter_system(_source_rows(grouped, source_study), source_system)
    if routed:
        analyses["E6"] = A.analyze_routing_overhead(routed)
        analyses["E7"] = A.analyze_query_type(routed, min_cell=min_cell)
        analyses["E8"] = A.analyze_escalation_transitions(
            routed,
            gold_chunk_ids=gold_chunk_ids,
            gold_document_ids=gold_document_ids,
            min_cell=min_cell,
        )

    for study, builder in (
        ("E2_escalation_ablation", A.analyze_escalation_ablation),
        ("E3_feature_ablation", A.analyze_feature_ablation),
        ("E4_threshold_calibration", A.analyze_threshold_sweep),
        ("E5_cost_weight", A.analyze_cost_weight),
    ):
        rows = _source_rows(grouped, study)
        if rows:
            analyses[study.split("_", 1)[0]] = builder(rows, min_cell=min_cell)

    return analyses


def _filter_system(rows: Sequence[Mapping[str, Any]], system: str) -> list[dict[str, Any]]:
    """Rows of one system, or every row when that system is not present.

    The fallback is deliberate and reported: a registry whose E1 entry is named
    differently from the conventional arm name should still yield an analysis,
    and the analyzer's own `n` makes the substitution visible.
    """
    selected = [dict(row) for row in rows if str(row.get("system")) == system]
    if selected:
        return selected
    present = sorted({str(row.get("system")) for row in rows})
    if system in present or not present:
        return []
    return [dict(row) for row in rows]


def build_statistics(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    systems: Sequence[str],
    baseline: str,
    metrics: Sequence[str],
    seed: int,
    n_resamples: int,
    confidence: float,
    alpha: float,
) -> dict[str, Any]:
    """Paired comparisons for E1 and every ablation study that has a baseline.

    Each study gets its own report, with its own Holm families. A study whose
    baseline arm is not registered is recorded with its reason and no numbers.
    """
    out: dict[str, Any] = {
        "statistics_version": S.STATISTICS_VERSION,
        "settings": S.statistics_settings(
            metrics,
            seed=seed,
            n_resamples=n_resamples,
            confidence=confidence,
            alpha=alpha,
        ),
        "reports": {},
        "absent": [],
    }

    e1_rows = _source_rows(grouped, "E1_baseline_comparison")
    if e1_rows and baseline in {str(r.get("system")) for r in e1_rows}:
        out["reports"]["E1"] = S.compare_systems(
            e1_rows,
            systems=systems,
            baseline=baseline,
            metrics=metrics,
            seed=seed,
            n_resamples=n_resamples,
            confidence=confidence,
            alpha=alpha,
        )
    elif e1_rows:
        out["absent"].append(
            f"E1: baseline {baseline!r} has no registered rows; "
            f"present systems: {sorted({str(r.get('system')) for r in e1_rows})}"
        )

    for study, study_baseline in ABLATION_BASELINES.items():
        rows = _source_rows(grouped, study)
        if not rows:
            continue
        present = {str(r.get("system")) for r in rows}
        if study_baseline not in present:
            out["absent"].append(
                f"{study}: baseline {study_baseline!r} has no registered rows; "
                f"present variants: {sorted(present)}"
            )
            continue

        if study == "E2_escalation_ablation":
            # E2 is a component-addition ablation: A -> B -> C.
            # The meaningful pairs are A-vs-B and B-vs-C (not all-vs-A).
            # Run explicit paired comparisons for each intended pair.
            escalation_variants = A.DEFAULT_ESCALATION_VARIANTS
            # Verify all three variants are present
            for v in escalation_variants:
                if v not in present:
                    out["absent"].append(
                        f"{study}: variant {v!r} has no registered rows; "
                        f"present variants: {sorted(present)}"
                    )
                    break
            else:
                # A-vs-B: cost of measuring sufficiency
                # B-vs-C: value of acting on it (bounded escalation)
                pairs = [
                    ("A_no_sufficiency_no_escalation", "B_sufficiency_no_escalation"),
                    ("B_sufficiency_no_escalation", "C_sufficiency_bounded_escalation"),
                ]
                comparisons = []
                by_metric: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
                grouped_rows = _group_rows(rows, "system")
                for baseline_name, treatment_name in pairs:
                    for metric in metrics:
                        comparison = S.compare_metric(
                            grouped_rows[treatment_name],
                            grouped_rows[baseline_name],
                            metric=metric,
                            treatment=treatment_name,
                            baseline=baseline_name,
                            seed=seed,
                            n_resamples=n_resamples,
                            confidence=confidence,
                            alpha=alpha,
                        )
                        by_metric[metric].append(comparison)
                        comparisons.append(comparison)
                # Apply Holm-Bonferroni within each metric
                for metric, family in by_metric.items():
                    pvalues = [c["p_value"] for c in family if c["p_value"] is not None]
                    adjusted = S.holm_bonferroni(pvalues) if pvalues else []
                    cursor = 0
                    for comparison in family:
                        if comparison["p_value"] is None:
                            continue
                        comparison["adjusted_p_value"] = adjusted[cursor]
                        comparison["significant"] = adjusted[cursor] < alpha
                        comparison["family_size"] = len(pvalues)
                        cursor += 1
                out["reports"]["E2"] = {
                    "statistics_version": S.STATISTICS_VERSION,
                    "group_column": "system",
                    "baseline": "chained (A-vs-B, B-vs-C)",
                    "groups_requested": list(escalation_variants),
                    "groups_present": list(escalation_variants),
                    "groups_absent": [],
                    "metrics": list(metrics),
                    "alpha": alpha,
                    "bootstrap_seed": seed,
                    "bootstrap_resamples": n_resamples,
                    "bootstrap_confidence": confidence,
                    "n_rows": len(rows),
                    "test_selection_rule": S.TEST_SELECTION_RULE,
                    "family_correction_note": (
                        "Holm-Bonferroni is applied within each metric across the two "
                        "chained comparisons (A-vs-B, B-vs-C); the family size is 2."
                    ),
                    "by_metric": by_metric,
                    "comparisons": comparisons,
                }
        else:
            # E3, E4, E5: standard all-vs-baseline (E3 uses 'full' as baseline)
            variants = (
                A.DEFAULT_FEATURE_VARIANTS
                if study.startswith("E3")
                else A.DEFAULT_ESCALATION_VARIANTS
            )
            out["reports"][study.split("_", 1)[0]] = S.compare_variants(
                rows,
                variants=variants,
                baseline=study_baseline,
                metrics=metrics,
                seed=seed,
                n_resamples=n_resamples,
                confidence=confidence,
                alpha=alpha,
            )
    return out


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def build_provenance(
    *,
    registry_path: Path,
    entries: Sequence[RegistryEntry],
    absent_studies: Sequence[str],
    selection: Mapping[str, Any],
    split: str | None,
    min_cell: int,
    systems: Sequence[str],
    baseline: str,
    metrics: Sequence[str],
    seed: int,
    n_resamples: int,
    confidence: float,
    alpha: float,
    gold: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Everything needed to say what was analysed, and to repeat it.

    `dataset_version` / `dataset_sha256` / `config_hash` / `corpus_version` are
    *sets*, not single values, because an analysis that spans several runs can
    legitimately span several config hashes (E3 turns a routing knob per arm).
    Collapsing them to one would either pick a winner silently or refuse to
    analyse a multi-arm study; reporting the set makes the spread visible and
    lets a reader check whether an arm drifted out of the family.
    """
    def distinct(field: str) -> list[str]:
        return sorted({str(getattr(entry, field)) for entry in entries if getattr(entry, field, None)})

    return {
        "artifact_version": ANALYSIS_ARTIFACT_VERSION,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "git_commit": _GATE._git_commit(),
        "environment": dict(environment if environment is not None else _GATE._snapshot()),
        "registry": {
            "path": str(registry_path),
            "n_entries_analysed": len(entries),
            "entries": [
                {
                    "experiment_id": entry.experiment_id,
                    "variant": entry.variant,
                    "kind": entry.kind,
                    "run_dir": entry.run_dir,
                    "trace_count": entry.trace_count,
                    "git_commit": entry.git_commit,
                    "dataset_version": entry.dataset_version,
                    "dataset_sha256": entry.dataset_sha256,
                    "config_hash": entry.config_hash,
                    "corpus_version": entry.corpus_version,
                    "rows_path": str(_rows_path(entry)),
                }
                for entry in entries
            ],
        },
        "dataset_versions": distinct("dataset_version"),
        "dataset_sha256s": distinct("dataset_sha256"),
        "config_hashes": distinct("config_hash"),
        "corpus_versions": distinct("corpus_version"),
        "row_selection": {
            "split": split,
            "valid_splits": list(VALID_SPLITS),
            "min_cell": min_cell,
            "systems": list(systems),
            "statistics_baseline": baseline,
            "analysis_source_study": DEFAULT_ANALYSIS_SOURCE,
            "analysis_source_system": DEFAULT_E7_SYSTEM,
            **dict(selection),
        },
        "statistics": {
            "version": S.STATISTICS_VERSION,
            "bootstrap_seed": seed,
            "bootstrap_resamples": n_resamples,
            "bootstrap_confidence": confidence,
            "alpha": alpha,
            "bootstrap_statistic": "median of the paired differences",
            "metrics": list(metrics),
            "test_selection_rule": S.TEST_SELECTION_RULE,
        },
        "relevance_labels": dict(gold),
        "studies_absent": sorted(absent_studies),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _print_report(
    analyses: Mapping[str, Any],
    statistics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    written: Mapping[str, Any],
) -> None:
    print()
    print(f"Phase 7 analysis -- {ANALYSIS_ARTIFACT_VERSION}")
    print(f"  commit    {provenance['git_commit']}")
    print(f"  split     {provenance['row_selection']['split'] or '(all)'}")
    print(
        f"  rows      kept={provenance['row_selection']['rows_kept']} "
        f"dropped_by_split={provenance['row_selection']['rows_dropped_by_split']}"
    )
    print(
        f"  datasets  {', '.join(provenance['dataset_versions']) or '(none)'} "
        f"sha256={', '.join(provenance['dataset_sha256s']) or '(none)'}"
    )
    print(
        f"  stats     seed={provenance['statistics']['bootstrap_seed']} "
        f"resamples={provenance['statistics']['bootstrap_resamples']} "
        f"alpha={provenance['statistics']['alpha']}"
    )
    if provenance["studies_absent"]:
        print(f"  ABSENT    {', '.join(provenance['studies_absent'])}")
    print()
    for key in sorted(analyses):
        print(f"  [{key}] {analyses[key]['experiment_id']}  n_rows={analyses[key].get('n_rows')}")
    print()
    for study, report in sorted(statistics.get("reports", {}).items()):
        significant = [
            f"{c['treatment']}/{c['metric']}"
            for c in report.get("comparisons", [])
            if c.get("significant")
        ]
        print(
            f"  {study}: {len(report.get('comparisons', []))} comparisons, "
            f"{len(significant)} significant after Holm"
        )
        for line in significant:
            print(f"      {line}")
    for note in statistics.get("absent", []):
        print(f"  statistics skipped: {note}")
    print()
    for name, paths in sorted(written.items()):
        print(f"  {name}: {', '.join(f'{k}={v}' for k, v in sorted(paths.items()))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_SUITE_ROOT / DEFAULT_REGISTRY_NAME,
        help="Registry written by scripts/run_phase7_suite.py.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--experiments",
        default=None,
        help=(
            "Comma-separated subset of studies to analyse. Default: every study "
            f"in the registry ({','.join(EXPERIMENT_IDS)})."
        ),
    )
    parser.add_argument(
        "--split",
        choices=[*VALID_SPLITS, "all"],
        default="all",
        help=(
            "Which dataset split's rows to analyse. 'all' mixes them and says so "
            "in the provenance; prefer naming one."
        ),
    )
    parser.add_argument(
        "--systems",
        default=",".join(ARM_NAMES),
        help="Comma-separated E1 systems to compare.",
    )
    parser.add_argument(
        "--baseline",
        default="bm25",
        help="E1's paired-comparison baseline. Must have registered rows.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=(
            "Comma-separated metrics to test. Each is classified as quality, "
            "latency or cost by evaluation.statistics.metric_kind, which decides "
            "the test."
        ),
    )
    parser.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=S.DEFAULT_N_RESAMPLES)
    parser.add_argument("--confidence", type=float, default=S.DEFAULT_CONFIDENCE)
    parser.add_argument("--alpha", type=float, default=S.DEFAULT_ALPHA)
    parser.add_argument(
        "--min-cell",
        type=int,
        default=A.MIN_CELL,
        help="Per-cell minimum below which `insufficient_data` is set.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Dataset whose relevance labels E8's quality-before needs, as an explicit "
            "path. There is no default: the frozen Phase 7 set is "
            f"{PHASE7_DATASET_PATH.relative_to(REPO_ROOT)} and the Phase 2-6 set is a "
            "different 20 records sharing no example_id with it, so an omitted flag "
            "would join labels to rows that do not correspond."
        ),
    )
    parser.add_argument(
        "--allow-legacy-dataset",
        action="store_true",
        help=(
            "Permit taking relevance labels from the Phase 2-6 20-record set "
            f"({LEGACY_DENSE_DATASET_PATH.relative_to(REPO_ROOT)}). Refused by default "
            "because its labels cannot be matched to the frozen Phase 7 rows; pass "
            "this only for a deliberate re-analysis of a legacy run."
        ),
    )
    parser.add_argument(
        "--no-gold",
        action="store_true",
        help=(
            "Skip relevance-label loading. E8 then reports its transitions, "
            "latency and cost with quality-before left empty rather than zero."
        ),
    )
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    # Exactly the guard scripts/run_phase7_suite.py applies: identity of the legacy
    # *file* plus an explicit opt-in, and nothing more. Path identity is deliberately
    # the whole policy here, not a record-count or split-composition check. This
    # analysis consumes the dataset for exactly one thing -- the gold relevance labels
    # keyed by example_id that E8 joins onto the routed rows -- so the only question
    # worth refusing is "is this the file those rows were measured against?". Split
    # composition is the suite's business, because it is what runs the queries, and
    # re-deriving a 107-record / 47-60 split here would only be a second, drifting
    # definition of the frozen set.
    if is_legacy_dataset(args.dataset) and not args.allow_legacy_dataset:
        raise AdaptiveRAGError(
            f"{args.dataset} is the Phase 2-6 20-record set, not the frozen Phase 7 "
            f"benchmark ({PHASE7_DATASET_PATH}). Analysing against it would give E8 a "
            "gold relevance map keyed by example_ids the Phase 7 rows do not contain. "
            "Pass the Phase 7 path, or pass --allow-legacy-dataset if the 20-record "
            "run is deliberate."
        )

    experiment_ids = (
        [eid.strip() for eid in args.experiments.split(",") if eid.strip()]
        if args.experiments
        else None
    )
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for metric in metrics:
        S.metric_kind(metric)  # raises on an unknown column, before anything runs
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    if not args.registry.is_file():
        print(
            f"no registry at {args.registry}; run scripts/run_phase7_suite.py first, "
            "or point --registry at an existing one",
            file=sys.stderr,
        )
        return 2
    entries, absent = load_entries(args.registry, experiments=experiment_ids)

    split = None if args.split == "all" else args.split
    grouped, selection = rows_by_experiment(entries, split=split)

    gold: dict[str, Any] = {
        "chunk_ids_available": False,
        "document_ids_available": False,
        "dataset_path": str(args.dataset),
        "reason": "--no-gold was passed",
        "dataset_version": None,
        "dataset_sha256": None,
    }
    gold_chunk_ids: dict[str, set[str]] | None = None
    gold_document_ids: dict[str, set[str]] | None = None
    if not args.no_gold:
        try:
            examples = load_evaluation_dataset(args.dataset)
        except Exception as exc:  # noqa: BLE001 -- reported, never faked
            gold["reason"] = f"dataset could not be loaded: {type(exc).__name__}: {exc}"
        else:
            gold_document_ids = A.gold_document_map(examples)
            gold_chunk_ids = A.gold_chunk_map(examples)
            gold = {
                "chunk_ids_available": bool(gold_chunk_ids),
                "document_ids_available": bool(gold_document_ids),
                "dataset_path": str(args.dataset),
                "dataset_sha256": _GATE._file_sha256(args.dataset),
                "dataset_version": examples[0].dataset_version if examples else None,
                "reason": (
                    "loaded"
                    if gold_chunk_ids
                    else "dataset loaded but the corpus is absent, so no chunk-level "
                    "relevance set could be enumerated; E8 reports document-level "
                    "quality-before only"
                ),
            }

    analyses = build_analyses(
        grouped,
        min_cell=args.min_cell,
        systems=systems,
        gold_chunk_ids=gold_chunk_ids,
        gold_document_ids=gold_document_ids,
    )
    statistics = build_statistics(
        grouped,
        systems=systems,
        baseline=args.baseline,
        metrics=metrics,
        seed=args.seed,
        n_resamples=args.resamples,
        confidence=args.confidence,
        alpha=args.alpha,
    )

    provenance = build_provenance(
        registry_path=args.registry,
        entries=entries,
        absent_studies=absent,
        selection=selection,
        split=split,
        min_cell=args.min_cell,
        systems=systems,
        baseline=args.baseline,
        metrics=metrics,
        seed=args.seed,
        n_resamples=args.resamples,
        confidence=args.confidence,
        alpha=args.alpha,
        gold=gold,
    )

    built = T.build_tables(analyses)
    args.out.mkdir(parents=True, exist_ok=True)
    written = T.write_tables(built, args.out / "tables")
    (args.out / "analysis.json").write_text(
        json.dumps(analyses, indent=2, sort_keys=True, default=list), encoding="utf-8"
    )
    (args.out / "statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance["artifacts"] = {
        "analysis": str(args.out / "analysis.json"),
        "statistics": str(args.out / "statistics.json"),
        "provenance": str(args.out / "provenance.json"),
        "tables": written,
    }
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    if args.json_only:
        print(json.dumps({"analyses": analyses, "statistics": statistics}, indent=2, sort_keys=True))
    else:
        _print_report(analyses, statistics, provenance, written)

    return 0 if analyses else 1


if __name__ == "__main__":
    raise SystemExit(main())
