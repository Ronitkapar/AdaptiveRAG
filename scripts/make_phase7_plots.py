#!/usr/bin/env python
"""
scripts.make_phase7_plots
-------------------------
Draw the Phase 7 figures from the row exports the suite produced, and record what
each one was drawn from.

This script runs no retrieval and re-derives no metric. It reads `rows.jsonl`
exports -- the same flat rows `scripts/run_phase7_analysis.py` reads -- hands
them to `evaluation.plots`, and writes six PNGs plus a manifest. That separation
is the point: the suite is the only thing that touches a corpus, an index or a
provider, so the figures can be re-drawn, re-seeded and re-checked on a machine
with none of them.

    python scripts/make_phase7_plots.py --rows experiments/phase7/runs/adaptive/rows.jsonl
    python scripts/make_phase7_plots.py --rows a/rows.jsonl b/rows.jsonl --out /tmp/figs
    python scripts/make_phase7_plots.py --rows ... --experiments quality_vs_latency,feature_ablation
    python scripts/make_phase7_plots.py --rows ... --json-only

Every figure carries its own provenance line -- source artifact, dataset version,
dataset sha256, git commit -- printed under the axes, because a PNG detached from
the rows that produced it is an unsourced claim. The manifest repeats it in
machine-readable form and adds each PNG's own sha256, so a figure can be traced
back to the exact rows and the exact code that drew it.

A figure whose rows cannot support it is *recorded as skipped with its reason*,
not drawn as an empty chart and not fatal to the other five. The trade-off plots
need latency and cost columns, the distribution and transition figures need a
routing decision, the ablations need a leave-one-out arm: when a run did not
produce one of those, saying so in the manifest is the honest outcome, and
`evaluation.plots` raises rather than inventing the number.

matplotlib is an optional extra. Without it the script reports the missing
dependency and exits 2 instead of writing five files and silently losing the
sixth.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_rag.evaluation import analysis as A  # noqa: E402
from adaptive_rag.evaluation.plots import (  # noqa: E402
    DEFAULT_COST_COLUMN,
    DEFAULT_LATENCY_COLUMN,
    DEFAULT_QUALITY_METRIC,
    DEFAULT_SEED,
    FIGURE_NAMES,
    PLOTS_VERSION,
    PlotDataError,
    PlotDependencyError,
    PlotProvenance,
    plot_escalation_transitions,
    plot_feature_ablation,
    plot_quality_vs_cost,
    plot_quality_vs_latency,
    plot_strategy_distribution,
    plot_threshold_sensitivity,
    sha256_file,
)
from adaptive_rag.evaluation.rows import read_rows_jsonl  # noqa: E402
from adaptive_rag.evaluation.suite import DEFAULT_SUITE_ROOT  # noqa: E402

MANIFEST_VERSION = "phase7_figures_manifest_v1"
MANIFEST_NAME = "figures_manifest.json"

DEFAULT_OUT = DEFAULT_SUITE_ROOT / "figures"

# The arm the router-behaviour figures are read from. Fixed rather than inferred
# from whichever arm happens to be present: a strategy distribution drawn from a
# fixed-strategy arm would be a distribution of nothing.
ROUTED_SYSTEM = "adaptive"

# The question each figure exists to answer, recorded in the manifest so a
# directory of PNGs can be read without opening them.
FIGURE_QUESTIONS: dict[str, str] = {
    "quality_vs_latency": "Does adaptive routing earn its extra latency?",
    "quality_vs_cost": "Is the retrieval quality gain worth what it costs?",
    "strategy_distribution": "What does the router actually select?",
    "escalation_transitions": "When the router escalates, where does it go?",
    "threshold_sensitivity": (
        "What does a stricter sufficiency threshold buy, and what does it spend?"
    ),
    "feature_ablation": "Which router feature group is carrying the routing?",
}


def _load_gate_module() -> Any:
    """Import the sibling gate script so provenance is written one way.

    `scripts/` has no `__init__.py`, so this cannot be a normal import. Loaded by
    path rather than duplicated, exactly as `run_phase7_analysis.py` and
    `measure_strategy_cost.py` do: three copies of the commit and environment
    helpers would drift, and a figure that names a different machine than the gate
    that preceded it is hard to trust.
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


def load_rows(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every row from every export, plus a per-file record of what was read.

    Each export's own path and sha256 are kept so a figure over the union of
    several arms can name all of its inputs; a manifest that listed one path for
    three arms' rows would be describing a file that does not exist.
    """
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"no rows export at {path}")
        loaded = read_rows_jsonl(path)
        rows.extend(loaded)
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "n_rows": len(loaded),
                "systems": sorted({str(row.get("system")) for row in loaded}),
            }
        )
    return rows, inputs


def routed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The adaptive arm's rows, or an empty list when no arm routed.

    No fallback to "all rows": the distribution and transition figures describe
    the router, and a router's distribution computed over four fixed-strategy
    arms is a distribution of `None`. The caller records the absence instead.
    """
    return [dict(row) for row in rows if str(row.get("system")) == ROUTED_SYSTEM]


def inherited_provenance(path: Path | None) -> dict[str, Any]:
    """Provenance fields taken from an existing analysis provenance artifact.

    The row export carries no dataset version and no commit -- those live in the
    provenance artifact the suite and the analysis driver wrote. Reading them from
    there rather than asking the caller to retype them means a figure's caption
    states the same dataset version as the table beside it.
    """
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_reason": f"could not read {path}: {type(exc).__name__}: {exc}"}
    if not isinstance(payload, Mapping):
        return {"_reason": f"{path} is not a provenance artifact"}
    versions = payload.get("dataset_versions")
    sha256s = payload.get("dataset_sha256s")
    hashes = payload.get("config_hashes")
    corpora = payload.get("corpus_versions")
    entries = (payload.get("registry") or {}).get("entries") or []
    return {
        "dataset_version": (versions[0] if isinstance(versions, list) and versions else None),
        "dataset_sha256": (sha256s[0] if isinstance(sha256s, list) and sha256s else None),
        "config_hash": (hashes[0] if isinstance(hashes, list) and hashes else None),
        "corpus_version": (corpora[0] if isinstance(corpora, list) and corpora else None),
        "registry_entries": len(entries),
    }


def build_provenance(
    *,
    inputs: Sequence[Mapping[str, Any]],
    inherited: Mapping[str, Any],
    dataset_path: Path | None,
    seed: int,
    generated_at: str,
) -> PlotProvenance:
    """The caption every figure carries, built from the inputs and the gate."""
    source = ", ".join(str(entry["path"]) for entry in inputs)
    return PlotProvenance(
        source_artifact=source,
        dataset_version=inherited.get("dataset_version"),
        dataset_sha256=inherited.get("dataset_sha256")
        or (sha256_file(dataset_path) if dataset_path and dataset_path.is_file() else None),
        git_commit=_GATE._git_commit(),
        config_hash=inherited.get("config_hash"),
        corpus_version=inherited.get("corpus_version"),
        generated_at=generated_at,
        notes=(
            f"rows={sum(int(entry['n_rows']) for entry in inputs)}",
            f"plots={PLOTS_VERSION}",
            f"seed={seed}",
        ),
    )


def figure_callables(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    metric: str,
    latency_column: str,
    cost_column: str,
) -> dict[str, Callable[..., Path]]:
    """Name to drawing function, each closed over the rows it should be given.

    The routing figures are handed only the adaptive arm's rows; the others get
    the whole export, because each analyzer groups by `system` and ignores the
    arms it was not asked about.
    """
    routed = routed_rows(rows)
    all_rows = [dict(row) for row in rows]
    return {
        "quality_vs_latency": lambda out, prov: plot_quality_vs_latency(
            all_rows, out, provenance=prov, metric=metric,
            latency_column=latency_column, seed=seed,
        ),
        "quality_vs_cost": lambda out, prov: plot_quality_vs_cost(
            all_rows, out, provenance=prov, metric=metric, cost_column=cost_column, seed=seed,
        ),
        "strategy_distribution": lambda out, prov: plot_strategy_distribution(
            routed, out, provenance=prov, seed=seed
        ),
        "escalation_transitions": lambda out, prov: plot_escalation_transitions(
            routed, out, provenance=prov, metric=metric, seed=seed
        ),
        "threshold_sensitivity": lambda out, prov: plot_threshold_sensitivity(
            all_rows, out, provenance=prov, metric=metric,
            latency_column=latency_column, cost_column=cost_column, seed=seed,
        ),
        "feature_ablation": lambda out, prov: plot_feature_ablation(
            all_rows, out, provenance=prov, metric=metric, seed=seed
        ),
    }


def draw(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    provenance: PlotProvenance,
    selected: Sequence[str],
    inputs: Sequence[Mapping[str, Any]],
    seed: int,
    metric: str,
    latency_column: str,
    cost_column: str,
    environment: Mapping[str, Any],
    generated_at: str,
    inherited: Mapping[str, Any],
) -> dict[str, Any]:
    """Draw every selected figure, then write the manifest.

    A figure that cannot be drawn is recorded in `skipped` with its exception
    type and message. The run still succeeds as long as one figure was written,
    and the exit code says which of those two things happened.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    callables = figure_callables(
        rows, seed=seed, metric=metric, latency_column=latency_column, cost_column=cost_column
    )
    figures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name in FIGURE_NAMES:
        if name not in selected:
            continue
        try:
            path = callables[name](out_dir, provenance)
        except PlotDataError as exc:
            skipped.append({"figure": name, "error": type(exc).__name__, "reason": str(exc)})
            continue
        figures.append(
            {
                "figure": name,
                "question": FIGURE_QUESTIONS[name],
                "path": str(path),
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "provenance": provenance.model_dump(mode="json"),
                "seed": seed,
            }
        )

    manifest = {
        "artifact_version": MANIFEST_VERSION,
        "plots_version": PLOTS_VERSION,
        "analysis_version": A.ANALYSIS_VERSION,
        "timestamp": generated_at,
        "git_commit": provenance.git_commit,
        "environment": dict(environment),
        "inputs": {
            "rows_exports": [dict(entry) for entry in inputs],
            "n_rows": sum(int(entry["n_rows"]) for entry in inputs),
            "systems": sorted({str(row.get("system")) for row in rows}),
            "routed_system": ROUTED_SYSTEM,
            "n_routed_rows": len(routed_rows(rows)),
        },
        "selection": {
            "figures_requested": list(selected),
            "figures_written": [entry["figure"] for entry in figures],
            "figures_skipped": [entry["figure"] for entry in skipped],
            "quality_metric": metric,
            "latency_column": latency_column,
            "cost_column": cost_column,
            "seed": seed,
        },
        "inherited_provenance": dict(inherited),
        "composite_score_policy": A.NO_COMPOSITE_POLICY,
        "figures": figures,
        "skipped": skipped,
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _print_report(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    print()
    print(f"Phase 7 figures -- {manifest['artifact_version']} (plots {manifest['plots_version']})")
    print(f"  commit    {manifest['git_commit']}")
    print(f"  rows      {manifest['inputs']['n_rows']} over "
          f"{len(manifest['inputs']['rows_exports'])} export(s)")
    print(f"  dataset   {manifest['figures'][0]['provenance']['dataset_version'] if manifest['figures'] else '(unknown)'}")
    print(f"  out       {manifest_path.parent}")
    print()
    for entry in manifest["figures"]:
        print(f"  [drawn]   {entry['figure']}: {entry['filename']} ({entry['bytes']} bytes)")
        print(f"            {entry['question']}")
    for entry in manifest["skipped"]:
        print(f"  [skipped] {entry['figure']}: {entry['reason']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows",
        type=Path,
        nargs="+",
        required=True,
        help="One or more rows.jsonl exports written by scripts/run_phase7_suite.py.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--experiments",
        default=None,
        help=(
            "Comma-separated subset of figures to draw. Default: all of "
            f"{','.join(FIGURE_NAMES)}."
        ),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help=(
            "provenance.json written by scripts/run_phase7_analysis.py. The dataset "
            "version, sha256, config hash and corpus version are read from it so a "
            "figure names the same dataset as the table beside it."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset file to hash for the caption when --provenance is absent.",
    )
    parser.add_argument("--metric", default=DEFAULT_QUALITY_METRIC)
    parser.add_argument("--latency-column", default=DEFAULT_LATENCY_COLUMN)
    parser.add_argument("--cost-column", default=DEFAULT_COST_COLUMN)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the coincident-point offsets. Recorded on every figure.",
    )
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    selected = (
        [name.strip() for name in args.experiments.split(",") if name.strip()]
        if args.experiments
        else list(FIGURE_NAMES)
    )
    unknown = [name for name in selected if name not in FIGURE_NAMES]
    if unknown:
        print(
            f"unknown figure(s): {', '.join(unknown)}; choose from "
            f"{', '.join(FIGURE_NAMES)}",
            file=sys.stderr,
        )
        return 2

    try:
        rows, inputs = load_rows(args.rows)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not rows:
        print(
            f"no rows in {', '.join(str(path) for path in args.rows)}; there is "
            "nothing to draw and a blank figure is not a result",
            file=sys.stderr,
        )
        return 2

    inherited = inherited_provenance(args.provenance)
    generated_at = datetime.now(timezone.utc).isoformat()
    provenance = build_provenance(
        inputs=inputs,
        inherited=inherited,
        dataset_path=args.dataset,
        seed=args.seed,
        generated_at=generated_at,
    )
    try:
        manifest = draw(
            rows,
            args.out,
            provenance=provenance,
            selected=selected,
            inputs=inputs,
            seed=args.seed,
            metric=args.metric,
            latency_column=args.latency_column,
            cost_column=args.cost_column,
            environment=_GATE._snapshot(),
            generated_at=generated_at,
            inherited=inherited,
        )
    except PlotDependencyError as exc:
        # No figure at all is possible without the extra, and half a report is
        # worse than none: say what is missing and leave the exit code saying so.
        print(str(exc), file=sys.stderr)
        return 2

    manifest_path = args.out / MANIFEST_NAME
    if args.json_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        _print_report(manifest, manifest_path)
    return 0 if manifest["figures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
