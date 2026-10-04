#!/usr/bin/env python
"""
scripts.run_phase7_suite
-----------------------
Phase 7 suite runner: execute one or more of the E1-E5 studies over the
evaluation dataset and register every run.

`experiments.runner.ExperimentRunner` runs one configuration. This script runs a
*study*: it builds every arm the requested experiment ids imply, checks the arms
still agree on the frozen shared conditions, executes each one, exports
query-level rows for each, and appends every run to the phase 7 registry.

    python scripts/run_phase7_suite.py --dataset data/evaluation/phase7_eval_v1.jsonl
    # one study, the frozen Phase 7 set, both splits
    python scripts/run_phase7_suite.py --dataset data/evaluation/phase7_eval_v1.jsonl --split all
    # two studies
    python scripts/run_phase7_suite.py --dataset data/evaluation/phase7_eval_v1.jsonl --experiments E1_baseline_comparison,E2_escalation_ablation
    # held-out split, three arms, no generation stage
    python scripts/run_phase7_suite.py --dataset data/evaluation/phase7_eval_v1.jsonl --split test --arms bm25,dense,adaptive --retrieval-only
    # resume a partially written ablation
    python scripts/run_phase7_suite.py --dataset data/evaluation/phase7_eval_v1.jsonl --experiments E3_feature_ablation --pace 0 --resume

Five flags are worth knowing about before a long run:

* `--dataset` has **no default**, on purpose. Phase 7 measures the frozen 107-record
  `phase7_eval_v1.jsonl`; the Phase 2-6 20-record `dense_eval_v1.jsonl` also exists in
  this directory, and a default pointing at it means an omitted flag benchmarks the
  wrong set and writes an artifact nobody can use. Naming the dataset is the cheapest
  possible provenance, so it is required rather than defaulted, and passing the legacy
  file is refused unless `--allow-legacy-dataset` says it was deliberate.
* `--retrieval-only` runs the Phase 7 protocol: no generator is called, so a run needs
  neither a `GROQ_API_KEY` nor network. Traces are `status="ok"` with their retrieval
  populated and `generation=None`; `total_latency_ms` still reports the retrieval work
  that was performed. Off by default, so an omitted flag keeps the generate
  behaviour it always had.
* `--pace` (default 0.75s) sleeps between queries on arms that call the embedding
  provider. Three of the four costed strategies embed every query remotely, and
  the 7.0b cost sweep died on HTTP 429 until this was added. The sleep sits
  outside the retriever's clock, so it changes request spacing and nothing else;
  it is recorded in `suite.json` and in each registry entry.
* `--strict` aborts on the first failing arm, after persisting what already ran.
  Without it a failing arm is recorded as a failure in `suite.json` and the suite
  continues, because a partial study is worth more than none.
* `--no-registration` runs the study without touching `registry.json`.

E6, E7 and E8 are analysis studies over the runs E1-E3 produce, so they have no
arm of their own; asking for them is reported, not silently ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.config.paths import REPO_ROOT  # noqa: E402
from adaptive_rag.evaluation.dataset import (  # noqa: E402
    LEGACY_DENSE_DATASET_PATH,
    PHASE7_DATASET_PATH,
    SPLIT_CALIBRATION,
    SPLIT_TEST,
    is_legacy_dataset,
    load_evaluation_dataset,
    partition_by_split,
    validate_split_separation,
)
from adaptive_rag.evaluation.suite import (  # noqa: E402
    ANALYSIS_ONLY_EXPERIMENT_IDS,
    DEFAULT_PACE_SECONDS,
    DEFAULT_SUITE_ROOT,
    RUNNABLE_EXPERIMENT_IDS,
    SuiteExecutionError,
    run_suite,
)
from adaptive_rag.errors import AdaptiveRAGError  # noqa: E402
from adaptive_rag.experiments.arms import ARM_NAMES, build_arms  # noqa: E402
from adaptive_rag.experiments.config import build_experiment_config  # noqa: E402
from adaptive_rag.schemas import PHASE8_INDEX_NAMESPACES, IndexConfig  # noqa: E402

SPLIT_CHOICES = (SPLIT_CALIBRATION, SPLIT_TEST, "all")


def _load_gate_module() -> Any:
    """Import the sibling gate script so provenance is written one way.

    `scripts/` has no `__init__.py`, so this cannot be a normal import. Loaded by
    path rather than duplicated because two copies of `_snapshot` would drift, and
    a suite artifact that describes a different machine than the gate it followed
    is hard to trust. `scripts/measure_strategy_cost.py` takes the same route.
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


def select_split(examples: list[Any], split: str) -> list[Any]:
    """Select the requested split, failing loudly rather than reusing the other.

    Asking for `test` on a dataset with no test records is an error, not a
    fallback: the split exists precisely so calibration and test queries are not
    mixed, and quietly returning the calibration queries would report a
    calibrated-on-the-test-set number under a test label.
    """
    if split == "all":
        return list(examples)
    calibration, test = partition_by_split(examples)
    selected = calibration if split == SPLIT_CALIBRATION else test
    if not selected:
        raise AdaptiveRAGError(
            f"split {split!r} is empty: this dataset has "
            f"{len(calibration)} calibration and {len(test)} test records. "
            "A held-out split has to be labelled by hand; it is never inferred."
        )
    return selected


def _print_report(result: Any) -> None:
    print()
    print(f"Phase 7 suite -- {result.suite_version}")
    plan = result.plan
    print(f"  studies    {', '.join(plan.experiment_ids) or '(none)'}")
    print(f"  arms       {len(plan.arms)}: {', '.join(plan.arms)}")
    if plan.analysis_only:
        print(f"  analysis   {', '.join(plan.analysis_only)} (no run of their own)")
    print(f"  output     {result.output_root}")
    print(f"  pace       {plan.pace_seconds}s on provider-calling arms")
    print()
    header = f"  {'arm':<38}{'status':<9}{'traces':>7}  rows"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for entry in result.results:
        rows = entry.rows_written.get("csv", "")
        print(
            f"  {entry.experiment_id + '/' + entry.variant:<38}{entry.status:<9}"
            f"{entry.trace_count:>7}  {rows}"
        )
        if entry.failure:
            print(
                f"      FAILED at {entry.failure.stage}: "
                f"{entry.failure.error_type}: {entry.failure.message}"
            )
    print()
    print(f"  {result.n_succeeded}/{result.n_arms} arms produced a run")
    if result.failed:
        print(f"  FAILED: {', '.join(result.failed)}")
    if result.artifact_path:
        print(f"  artifact  {result.artifact_path}")
    if result.registry_path:
        print(f"  registry  {result.registry_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Evaluation dataset to benchmark, as an explicit path. There is no "
            f"default: the frozen Phase 7 set is {PHASE7_DATASET_PATH.relative_to(REPO_ROOT)} "
            "and the Phase 2-6 set is a different 20 records, so the choice has to be "
            "made by whoever runs the study."
        ),
    )
    parser.add_argument(
        "--allow-legacy-dataset",
        action="store_true",
        help=(
            "Permit benchmarking the Phase 2-6 20-record set "
            f"({LEGACY_DENSE_DATASET_PATH.relative_to(REPO_ROOT)}). Refused by "
            "default because its numbers are not comparable with the frozen Phase 7 "
            "set; pass this only for a deliberate smoke run."
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Measure retrieval only: no generator is constructed or called, so the "
            "arm needs no generation credentials and no network. Traces are "
            "status=ok with retrieval populated, generation is null, and "
            "total_latency_ms still reports the retrieval work performed."
        ),
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default=SPLIT_CALIBRATION,
        help=(
            "Which dataset split to run. Defaults to calibration: the shipped "
            "dataset's records predate the split and are calibration material."
        ),
    )
    parser.add_argument(
        "--experiments",
        default="E1_baseline_comparison",
        help=(
            "Comma-separated experiment ids to run. Runnable: "
            f"{','.join(RUNNABLE_EXPERIMENT_IDS)}. Analysis-only (no run of their "
            f"own): {','.join(ANALYSIS_ONLY_EXPERIMENT_IDS)}."
        ),
    )
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "Comma-separated subset of the five benchmark arms "
            f"({','.join(ARM_NAMES)}). Ignored for the ablation studies, whose "
            "arm set is the variant grid."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE_SECONDS,
        help=(
            "Seconds to sleep between queries on provider-calling arms, to avoid "
            "the embedding API rate limit. 0 disables pacing. Applied outside the "
            "retriever's clock, so it cannot enter any latency_ms."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first failing arm, after persisting what already ran.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a partially written run directory instead of starting over. "
            "Requires the same --run-prefix as the original invocation."
        ),
    )
    parser.add_argument(
        "--no-registration",
        action="store_true",
        help="Run the study without appending to registry.json.",
    )
    parser.add_argument(
        "--allow-reentry",
        action="store_true",
        help=(
            "Let a new run of an already-registered (study, variant) replace its "
            "registration. Off by default so a re-run cannot silently overwrite "
            "an earlier measurement."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Prefix for run directory names. Run directories are named "
            "deterministically without one, which is what makes --resume able to "
            "find them; set a prefix when running the same study twice."
        ),
    )
    parser.add_argument(
        "--corpus-arm",
        choices=sorted(PHASE8_INDEX_NAMESPACES),
        default=IndexConfig().corpus_arm,
        help=(
            "Which corpus arm to retrieve from: 'phase8_before' (two-column "
            "reading order left broken) or 'phase8_after' (the column fix). The "
            "arms share a corpus version, so this selects the collection and the "
            "lexical index together and is checked on load. Use "
            "--run-prefix when running the same study on both arms."
        ),
    )
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    if is_legacy_dataset(args.dataset) and not args.allow_legacy_dataset:
        raise AdaptiveRAGError(
            f"{args.dataset} is the Phase 2-6 20-record set, not the frozen Phase 7 "
            f"benchmark ({PHASE7_DATASET_PATH}). Benchmarking it would produce an "
            "artifact whose numbers cannot be compared with the rest of Phase 7. "
            "Pass the Phase 7 path, or pass --allow-legacy-dataset if the 20-record "
            "run is deliberate."
        )

    examples = load_evaluation_dataset(args.dataset)
    validate_split_separation(examples)
    selected = select_split(examples, args.split)

    experiment_ids = [eid.strip() for eid in args.experiments.split(",") if eid.strip()]
    arm_names = (
        [name.strip() for name in args.arms.split(",") if name.strip()]
        if args.arms
        else None
    )

    common = build_experiment_config(
        name="phase7_suite_common",
        index=IndexConfig(corpus_arm=args.corpus_arm),
    )
    arms = None
    if arm_names:
        built = build_arms(common)
        known = {arm.name for arm in built}
        unknown = sorted(set(arm_names) - known)
        if unknown:
            raise AdaptiveRAGError(
                f"unknown arms {unknown}; expected a subset of {sorted(known)}"
            )
        arms = [arm for arm in built if arm.name in set(arm_names)]

    print(
        f"running {len(arms) if arms else 'all'} arm(s) over {len(selected)} "
        f"{args.split} queries from {args.dataset}"
        f"{' (retrieval-only)' if args.retrieval_only else ''} -> {args.out}",
        flush=True,
    )

    try:
        result = run_suite(
            common=common,
            dataset=selected,
            output_root=args.out,
            experiment_ids=experiment_ids,
            arms=arms,
            pace_seconds=args.pace,
            retrieval_only=args.retrieval_only,
            strict=args.strict,
            resume=args.resume,
            register=not args.no_registration,
            allow_reentry=args.allow_reentry,
            dataset_path=args.dataset,
            environment=_GATE._snapshot(),
            run_prefix=args.run_prefix,
        )
    except SuiteExecutionError as exc:
        print(f"strict mode: {exc}", file=sys.stderr)
        return 1

    if args.json_only:
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        _print_report(result)
    return 1 if result.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

