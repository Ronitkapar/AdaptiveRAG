"""
evaluation.plots
----------------
The six Phase 7 figures, and only the figures that answer a question.

Every chart here exists because a Phase 7 question cannot be read off a table.
The quality-versus-latency scatter is the trade-off itself; the transition chart
is the only way to see which escalations the ladder actually produced; the
threshold panels turn a one-parameter sweep into a curve. Nothing decorative is
drawn: there is no pie of a constant, no bar chart of a value already printed in
a table, and no chart whose only content is that the run finished.

Three rules shape the module, and each exists because the failure it prevents is
a published number nobody can defend:

1. **A figure is a view of `evaluation.analysis`, never a second analysis.**
   Every function here either takes an analyzer's result or calls the analyzer
   itself. Quality, latency, cost, p95, the strategy distribution and the
   transition list all come from `analysis.py`, which in turn defers to
   `RetrievalEvaluator` for every metric and to `evaluation.base.percentile`
   (nearest-rank, no interpolation) for every percentile. A figure that
   re-averaged the rows could disagree with the table beside it, and two numbers
   for one quantity is how a report stops being reproducible.

2. **Insufficient data raises; it is never drawn as a zero.** No rows, no
   routing decision, no observed escalation and no ablation arm are each an
   absence of evidence, and a chart renders an absence as a fact the moment it
   draws an empty bar. Those cases raise `PlotDataError` -- a typed
   `EvaluationError` -- naming what was missing, and a figure is written only
   when every number in it was measured. Panels that are genuinely optional (the
   cost axis of a sweep whose rows carry no cost) say so in text.

3. **Only observed transitions are plotted.** `plot_escalation_transitions`
   draws the pairs in `analyze_escalation_transitions`'s `transitions` list and
   nothing else. A zero-height bar for a pair that never occurred would be
   indistinguishable from a pair that occurred and scored nothing, and the first
   is a claim about the ladder while the second is a claim about the router.

matplotlib is an optional extra (`pyproject.toml`'s `plots` extra) and is
imported *lazily*, inside the function that draws, exactly as
`reranking/onnx_backend.py` defers its `onnxruntime` import to first use. Two
consequences follow and both are tested: importing this module never requires
matplotlib, and no display is ever opened -- the `Agg` backend is forced before
`pyplot` is imported, so every figure renders headless.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from adaptive_rag.errors import AdaptiveRAGError, ConfigurationError, EvaluationError
from adaptive_rag.evaluation import analysis as A
from adaptive_rag.evaluation.analysis import RowLike
from adaptive_rag.schemas.config import FEATURE_GROUPS
from adaptive_rag.schemas.routing import STRATEGY_ORDER

PLOTS_VERSION = "phase7_plots_v1"

# The seed every figure that offsets coincident points is drawn with. Recorded in
# the caption of each such figure, so a re-render that disagrees is visibly a
# different draw rather than a mysterious one.
DEFAULT_SEED = 0

FIGURE_DPI = 150

# The quality axis: recall@5, because the retrieval runs report at k=5. Named
# here only as the default; `analysis.PRIMARY_QUALITY_METRIC` is the definition.
DEFAULT_QUALITY_METRIC = A.PRIMARY_QUALITY_METRIC
DEFAULT_LATENCY_COLUMN = "total_latency_ms"
DEFAULT_COST_COLUMN = "estimated_cost_usd"

# The strategies the router may select, in the order `STRATEGY_ORDER` declares
# them, with the display names a table would print. An unrecognised strategy
# string -- a future fifth strategy -- is printed verbatim rather than dropped.
STRATEGY_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid+Reranker",
}

# The figure names the CLI selects with `--experiments`, in drawing order.
FIGURE_NAMES: tuple[str, ...] = (
    "quality_vs_latency",
    "quality_vs_cost",
    "strategy_distribution",
    "escalation_transitions",
    "threshold_sensitivity",
    "feature_ablation",
)

# Placeholder for a provenance field the caller could not supply. Printed rather
# than omitted: "sha256=unknown" is a gap a reader can see, and a caption with no
# sha256 at all is a gap nobody notices.
UNKNOWN = "unknown"


class PlotError(AdaptiveRAGError):
    """Base class for figure-rendering failures."""


class PlotDataError(PlotError, EvaluationError):
    """Raised when the rows cannot support the figure that was requested.

    An `EvaluationError` as well as a `PlotError`, so a caller that only handles
    the evaluation hierarchy still catches a figure that refused to be drawn for
    want of data -- the same failure mode a metric computation has.
    """


class PlotDependencyError(PlotError, ConfigurationError):
    """Raised when the optional `plots` extra is not installed."""


class PlotProvenance(BaseModel):
    """What a figure was drawn from, printed on the figure itself.

    The caption is part of the artifact, not a side note: a PNG detached from the
    rows that produced it is an unsourced claim, so the source artifact, the
    dataset version, the dataset sha256 and the git commit travel with the image.
    A field the caller could not supply prints `unknown` rather than being
    dropped, because a visible gap can be chased and a silent one cannot.
    """

    model_config = ConfigDict(extra="forbid")

    source_artifact: str
    dataset_version: str | None = None
    dataset_sha256: str | None = None
    git_commit: str | None = None
    config_hash: str | None = None
    corpus_version: str | None = None
    generated_at: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_paths(
        cls,
        source_artifact: str | Path,
        *,
        dataset_path: str | Path | None = None,
        dataset_version: str | None = None,
        git_commit: str | None = None,
        config_hash: str | None = None,
        corpus_version: str | None = None,
        generated_at: str | None = None,
        notes: Sequence[str] = (),
    ) -> "PlotProvenance":
        """Build a provenance block, hashing the dataset file when it is present.

        A dataset path that does not exist leaves the sha256 unknown instead of
        raising: the caller is describing an artifact it already has, and a
        missing dataset is a fact about that artifact worth carrying rather than
        an error in describing it.
        """
        sha = None
        if dataset_path is not None and Path(dataset_path).is_file():
            sha = sha256_file(dataset_path)
        return cls(
            source_artifact=str(source_artifact),
            dataset_version=dataset_version,
            dataset_sha256=sha,
            git_commit=git_commit,
            config_hash=config_hash,
            corpus_version=corpus_version,
            generated_at=generated_at,
            notes=tuple(notes),
        )

    def line(self, *extra: str) -> str:
        """The one-line caption drawn under the axes of every figure."""
        parts = [
            f"source={self.source_artifact}",
            f"dataset={self.dataset_version or UNKNOWN}",
            f"sha256={(self.dataset_sha256 or UNKNOWN)[:16]}",
            f"commit={self.git_commit or UNKNOWN}",
        ]
        for note in (*self.notes, *extra):
            if note:
                parts.append(note)
        return " | ".join(parts)


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file, read in chunks so a large dataset is not slurped."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# rendering internals
# --------------------------------------------------------------------------


def _pyplot() -> tuple[Any, Any]:
    """Import matplotlib with the `Agg` backend forced, and return it with pyplot.

    The import is deferred to first draw so that importing this module does not
    require the optional dependency, and `Agg` is selected before `pyplot` exists
    so no toolkit is ever initialised and no display is ever opened.
    """
    try:
        import matplotlib
    except ImportError as exc:
        raise PlotDependencyError(
            "matplotlib is required to render figures but is not installed; "
            "install the optional extra with `pip install -e .[plots]`"
        ) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save(
    fig: Any, plt: Any, out_dir: str | Path, filename: str
) -> Path:
    """Write one figure, close it, and confirm a non-empty PNG landed on disk.

    The size check is not paranoia about the filesystem: a figure whose artists
    all failed can still produce a valid, tiny, empty PNG, and an empty figure
    written where a measurement should be is the one outcome this module exists to
    prevent.
    """
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=FIGURE_DPI,
        format="png",
        metadata={"Software": f"adaptive_rag {PLOTS_VERSION}"},
    )
    plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise PlotError(f"figure was not written: {path}")
    return path


def _finish(
    fig: Any,
    plt: Any,
    *,
    out_dir: str | Path,
    filename: str,
    provenance: PlotProvenance,
    notes: Sequence[str] = (),
) -> Path:
    """Stamp the provenance caption, lay the figure out, and write it."""
    caption = provenance.line(*notes)
    fig.text(
        0.01,
        0.012,
        caption,
        ha="left",
        va="bottom",
        fontsize=5.5,
        family="monospace",
        wrap=True,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return _save(fig, plt, out_dir, filename)


def _require_rows(rows: Sequence[RowLike], figure: str) -> None:
    if not rows:
        raise PlotDataError(
            f"{figure} needs rows; got none. A figure drawn from an empty row set "
            "would be an empty chart, which reads as a measured result rather than "
            "as the absence of one."
        )


def _require_block(result: Mapping[str, Any], key: str, figure: str) -> Mapping[str, Any]:
    block = result.get(key)
    if not isinstance(block, Mapping) or not block:
        raise PlotDataError(
            f"{figure} was given an analysis result with no {key!r} block; "
            f"available keys: {sorted(result)}"
        )
    return block


def _strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def _strategy_order(name: str) -> tuple[int, str]:
    """Sort key placing the known strategies in ladder order, unknowns last."""
    return (STRATEGY_ORDER.index(name) if name in STRATEGY_ORDER else len(STRATEGY_ORDER), name)


def _axis_unit(column: str) -> str:
    """The unit an axis label states for a latency or cost column.

    Spelled once because "USD per query" appearing on one figure and a bare column
    name on another is exactly the kind of small inconsistency that makes a reader
    unsure whether the two axes are even comparable.
    """
    if column.endswith("_usd"):
        return "USD per query"
    if column.endswith("_ms"):
        return "ms"
    return f"{column} per query"


def _cell_mean(
    block: Mapping[str, Any], group: str, column: str
) -> float | None:
    """`(group, column)` mean out of a summary or variant report, or None."""
    section = block.get(group)
    if not isinstance(section, Mapping):
        return None
    entry = section.get(column)
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _p95(block: Mapping[str, Any], group: str, column: str) -> float | None:
    """Nearest-rank p95 as `analysis.distribution` computed it, or None."""
    section = block.get(group)
    if not isinstance(section, Mapping):
        return None
    entry = section.get(column)
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("p95")
    return float(value) if isinstance(value, (int, float)) else None


def _jitter_offsets(values: Sequence[float], *, seed: int, span: float) -> list[float]:
    """Deterministic x-offsets for points that share a coordinate.

    Two arms with identical means plot as one point otherwise, and the reader
    cannot tell whether one system was measured or two. Only *coincident* points
    move, so a figure with no ties is identical to one drawn without jitter, and
    the offsets come from a seeded generator, so two runs of the same rows draw
    the same picture.
    """
    rng = np.random.default_rng(seed)
    offsets = [0.0] * len(values)
    seen: dict[float, int] = {}
    for index, value in enumerate(values):
        count = seen.get(value, 0)
        seen[value] = count + 1
        if count:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            offsets[index] = sign * span * count * (0.5 + 0.5 * float(rng.random()))
    return offsets


def _span(values: Sequence[float]) -> float:
    """A jitter span in data units: 2% of the observed range, never zero."""
    if not values:
        return 0.0
    low, high = min(values), max(values)
    return max((high - low) * 0.02, abs(high) * 0.001, 1e-9)


def _columns_with_data(
    blocks: Mapping[str, Mapping[str, Any]], group: str, columns: Sequence[str]
) -> list[str]:
    """Which of `columns` hold a value for at least one block.

    Used only to name the alternatives in an error message: a caller who asked for
    a cost axis the rows do not carry is told which cost axes the rows do carry,
    rather than being left to guess or -- worse -- handed a substitute axis they
    did not ask for.
    """
    return [
        column
        for column in columns
        if any(_cell_mean(block, group, column) is not None for block in blocks.values())
    ]


# --------------------------------------------------------------------------
# 1. quality versus latency
# --------------------------------------------------------------------------


def plot_quality_vs_latency(
    rows: Sequence[RowLike],
    out_dir: str | Path,
    *,
    provenance: PlotProvenance,
    analysis: Mapping[str, Any] | None = None,
    systems: Sequence[str] = A.DEFAULT_SYSTEMS,
    metric: str = DEFAULT_QUALITY_METRIC,
    latency_column: str = DEFAULT_LATENCY_COLUMN,
    seed: int = DEFAULT_SEED,
    filename: str | None = None,
) -> Path:
    """The central trade-off: what each system pays in latency for its quality.

    One labelled point per system, x the mean per-query latency, y the mean
    quality, and a rightward whisker to the nearest-rank p95 latency so a system
    whose mean flatters its tail is visible as such. A cell below
    `analysis.MIN_CELL` is drawn as an open marker and labelled with its `n`,
    because an under-powered mean is still a measurement and hiding it would be
    the dishonest option, while drawing it identically to a well-powered one
    would be the other.

    Question answered: *does adaptive routing earn its extra latency?*
    """
    _require_rows(rows, "quality_vs_latency")
    result = analysis if analysis is not None else A.analyze_main_comparison(rows, systems=systems)
    per_system = _require_block(result, "per_system", "quality_vs_latency")

    points: list[tuple[str, float, float, float | None, int, bool]] = []
    unplottable: list[str] = []
    for name, block in per_system.items():
        if not block.get("n"):
            unplottable.append(f"{name} (no rows)")
            continue
        quality = _cell_mean(block, "quality", metric)
        latency = _cell_mean(block, "latency", latency_column)
        if quality is None or latency is None:
            absent = "quality" if quality is None else latency_column
            unplottable.append(f"{name} (no {absent})")
            continue
        points.append(
            (
                name,
                latency,
                quality,
                _p95(block, "latency", latency_column),
                int(block.get("n") or 0),
                bool(block.get("insufficient_data")),
            )
        )
    if not points:
        raise PlotDataError(
            "quality_vs_latency has no plottable system: no arm carries both "
            f"{metric!r} and {latency_column!r}. Rows: "
            f"{', '.join(unplottable) or '(none)'}."
        )

    _, plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    xs = [p[1] for p in points]
    offsets = _jitter_offsets(xs, seed=seed, span=_span(xs))
    for (name, latency, quality, p95, n, thin), offset in zip(points, offsets):
        marker = "o" if not thin else "^"
        fill = None if thin else "tab:blue"
        ax.errorbar(
            latency + offset,
            quality,
            xerr=[[0.0], [max(0.0, (p95 or latency) - latency)]],
            fmt=marker,
            color=fill,
            markerfacecolor=fill,
            markeredgecolor="black",
            markersize=9,
            capsize=4,
            elinewidth=1.0,
            label=name,
        )
        ax.annotate(
            f"{name}\nn={n}" + (" (n<%d)" % A.MIN_CELL if thin else ""),
            (latency + offset, quality),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
        )

    ax.set_xlabel(f"Mean {latency_column} per query (ms)")
    ax.set_ylabel(f"Mean {metric} (0-1)")
    ax.set_title(
        "Quality versus latency: what does each system pay in latency for its recall?",
        fontsize=10,
    )
    if unplottable:
        ax.text(
            0.0,
            -0.16,
            "not plotted (no measured value): " + "; ".join(unplottable),
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
        )
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    return _finish(
        fig,
        plt,
        out_dir=out_dir,
        filename=filename or "e1_quality_vs_latency.png",
        provenance=provenance,
        notes=[
            f"systems={len(points)}",
            f"jitter seed={seed}",
            "whisker=nearest-rank p95 latency",
            f"analysis={A.ANALYSIS_VERSION}",
        ],
    )


# --------------------------------------------------------------------------
# 2. quality versus cost
# --------------------------------------------------------------------------


def plot_quality_vs_cost(
    rows: Sequence[RowLike],
    out_dir: str | Path,
    *,
    provenance: PlotProvenance,
    analysis: Mapping[str, Any] | None = None,
    systems: Sequence[str] = A.DEFAULT_SYSTEMS,
    metric: str = DEFAULT_QUALITY_METRIC,
    cost_column: str = DEFAULT_COST_COLUMN,
    seed: int = DEFAULT_SEED,
    filename: str | None = None,
) -> Path:
    """The second trade-off axis: quality against measured cost or resources.

    Same construction as the latency figure with `cost_column` on x. The column
    is a parameter rather than a fixed choice because the rows carry several
    measured resource columns (`analysis.COST_COLUMNS`: context tokens, input and
    output tokens, estimated USD) and which one is meaningful depends on whether
    the run recorded money at all. A column that holds no value raises rather
    than silently falling back to another: a figure whose axis is not the axis
    the reader asked for is a different measurement wearing the same filename, and
    the error message names the cost columns that do carry data.

    Question answered: *is the quality gain worth what it costs?*
    """
    _require_rows(rows, "quality_vs_cost")
    result = analysis if analysis is not None else A.analyze_main_comparison(rows, systems=systems)
    per_system = _require_block(result, "per_system", "quality_vs_cost")

    points: list[tuple[str, float, float, float | None, int, bool]] = []
    unplottable: list[str] = []
    for name, block in per_system.items():
        if not block.get("n"):
            unplottable.append(f"{name} (no rows)")
            continue
        quality = _cell_mean(block, "quality", metric)
        cost = _cell_mean(block, "cost", cost_column)
        if quality is None or cost is None:
            absent = "quality" if quality is None else cost_column
            unplottable.append(f"{name} (no {absent})")
            continue
        points.append(
            (
                name,
                cost,
                quality,
                _p95(block, "cost", cost_column),
                int(block.get("n") or 0),
                bool(block.get("insufficient_data")),
            )
        )
    if not points:
        available = _columns_with_data(per_system, "cost", A.COST_COLUMNS)
        raise PlotDataError(
            f"quality_vs_cost has no plottable system: no arm carries both "
            f"{metric!r} and {cost_column!r}. Cost columns that do carry data: "
            f"{available or 'none'}."
        )

    _, plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    xs = [p[1] for p in points]
    offsets = _jitter_offsets(xs, seed=seed, span=_span(xs))
    for (name, cost, quality, p95, n, thin), offset in zip(points, offsets):
        marker = "o" if not thin else "^"
        fill = None if thin else "tab:green"
        ax.errorbar(
            cost + offset,
            quality,
            xerr=[[0.0], [max(0.0, (p95 or cost) - cost)]],
            fmt=marker,
            color=fill,
            markerfacecolor=fill,
            markeredgecolor="black",
            markersize=9,
            capsize=4,
            elinewidth=1.0,
            label=name,
        )
        ax.annotate(
            f"{name}\nn={n}" + (" (n<%d)" % A.MIN_CELL if thin else ""),
            (cost + offset, quality),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
        )

    ax.set_xlabel(f"Mean {cost_column} ({_axis_unit(cost_column)})")
    ax.set_ylabel(f"Mean {metric} (0-1)")
    ax.set_title(
        "Quality versus cost: is the retrieval gain worth what it spends?",
        fontsize=10,
    )
    if unplottable:
        ax.text(
            0.0,
            -0.16,
            "not plotted (no measured value): " + "; ".join(unplottable),
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
        )
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    return _finish(
        fig,
        plt,
        out_dir=out_dir,
        filename=filename or "e1_quality_vs_cost.png",
        provenance=provenance,
        notes=[
            f"systems={len(points)}",
            f"jitter seed={seed}",
            "whisker=nearest-rank p95 cost",
            f"analysis={A.ANALYSIS_VERSION}",
        ],
    )


# --------------------------------------------------------------------------
# 3. adaptive strategy distribution
# --------------------------------------------------------------------------


def strategy_distribution(
    rows: Sequence[RowLike],
    *,
    system: str = "adaptive",
    field: str = "final_strategy",
) -> dict[str, Any]:
    """The router's strategy selection, as `analysis` computed it.

    Exposed separately from the drawing so the counts can be asserted without
    reading pixels, and so the CLI can record them in the figure manifest.
    """
    result = A.analyze_main_comparison(rows, systems=(system,))
    block = result["per_system"].get(system) or {}
    distribution = block.get("strategy_distribution") or {}
    return distribution.get(field) or {"observed": 0, "counts": {}, "shares": {}}


def plot_strategy_distribution(
    rows: Sequence[RowLike],
    out_dir: str | Path,
    *,
    provenance: PlotProvenance,
    system: str = "adaptive",
    field: str = "final_strategy",
    seed: int = DEFAULT_SEED,
    filename: str | None = None,
) -> Path:
    """Which of the four strategies the router actually selects, and how often.

    Only observed strategies get a bar. A strategy the router never chose is
    absent from the chart rather than drawn at zero, and the caption says so,
    because the denominator is the queries that recorded a routing decision: a
    fixed-strategy arm records none, and dividing by the cell size would report a
    0% distribution for a selection that is simply undefined.

    `field` chooses the routing column: `initial_strategy` is the router's first
    pick, `final_strategy` is what answered the query after any escalation.

    Question answered: *what does the router do with the queries it is given?*
    """
    _require_rows(rows, "strategy_distribution")
    if field not in ("initial_strategy", "final_strategy"):
        raise PlotDataError(
            f"field must be 'initial_strategy' or 'final_strategy'; got {field!r}"
        )
    selected = strategy_distribution(rows, system=system, field=field)
    counts: dict[str, int] = {str(k): int(v) for k, v in (selected.get("counts") or {}).items()}
    observed = int(selected.get("observed") or 0)
    if not counts or observed == 0:
        raise PlotDataError(
            f"no {field} was recorded for system {system!r}: the rows carry no "
            "routing decision, so there is no distribution to draw. A zero bar "
            "here would claim the router chose nothing."
        )

    shares: dict[str, float | None] = dict(selected.get("shares") or {})
    names = sorted(counts, key=_strategy_order)

    _, plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(names), dtype=float)
    heights = [counts[name] for name in names]
    ax.bar(positions, heights, color="tab:blue", width=0.6)
    for position, name, height in zip(positions, names, heights):
        share = shares.get(name)
        ax.annotate(
            f"{counts[name]}\n" + ("n/a" if share is None else f"{share * 100:.1f}%"),
            (position, height),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_strategy_label(name) for name in names])
    ax.set_ylabel(f"Queries selecting this strategy (n), of {observed} routed")
    ax.set_xlabel("Strategy selected by the router")
    ax.set_ylim(0, max(heights) * 1.2 if max(heights) else 1.0)
    ax.set_title(
        f"Strategy distribution: what does the router select ({field})?",
        fontsize=10,
    )
    ax.text(
        0.0,
        -0.2,
        "strategies the router never selected are absent from this chart, not zero",
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
    )
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    return _finish(
        fig,
        plt,
        out_dir=out_dir,
        filename=filename or "e1_strategy_distribution.png",
        provenance=provenance,
        notes=[
            f"system={system}",
            f"field={field}",
            f"seed={seed}",
            "jitter=none",
            f"observed={observed}",
            f"analysis={A.ANALYSIS_VERSION}",
        ],
    )


# --------------------------------------------------------------------------
# 4. escalation transitions (observed only)
# --------------------------------------------------------------------------


def observed_transition_labels(result: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The `(initial -> final)` pairs `analyze_escalation_transitions` observed.

    Read straight off the analyzer's `transitions` list, which contains one entry
    per pair that actually occurred. The plotting function builds its axis from
    exactly this, and the test asserts against exactly this, so "never plot an
    unobserved transition" is a property of one list rather than of two
    independent filters that could drift apart.
    """
    transitions = result.get("transitions")
    if not isinstance(transitions, Sequence):
        return ()
    return tuple(
        (str(entry["from"]), str(entry["to"]))
        for entry in transitions
        if isinstance(entry, Mapping) and entry.get("from") and entry.get("to")
    )


def plot_escalation_transitions(
    rows: Sequence[RowLike],
    out_dir: str | Path,
    *,
    provenance: PlotProvenance,
    analysis: Mapping[str, Any] | None = None,
    metric: str = DEFAULT_QUALITY_METRIC,
    seed: int = DEFAULT_SEED,
    filename: str | None = None,
) -> Path:
    """Which escalations the ladder actually produced, and how often each occurred.

    One bar per observed `(initial -> final)` pair, annotated with its share of
    all escalated queries. Pairs that never occurred are not drawn at all: the
    question this figure answers is what the router did, and a zero-height bar
    for a pair that never happened answers a different one -- whether the ladder
    offers the rung -- while looking exactly the same.

    Question answered: *when the router escalates, where does it go?*
    """
    _require_rows(rows, "escalation_transitions")
    result = (
        analysis
        if analysis is not None
        else A.analyze_escalation_transitions(rows, metric=metric)
    )
    transitions = result.get("transitions") or []
    if not transitions:
        raise PlotDataError(
            f"no escalation transition was observed in the {len(rows)} row(s) "
            "supplied, so there is nothing to plot. That is a statement about this "
            "row set and not about the router: these rows carry one adaptive "
            "configuration, and another configuration of the same router may well "
            "escalate. An all-zero transition chart would instead claim the router "
            "declined to escalate, which these rows cannot establish either way."
        )

    _, plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.6, 0.6 * len(transitions) + 2.4))
    labels = [
        f"{_strategy_label(str(entry['from']))} -> {_strategy_label(str(entry['to']))}"
        for entry in transitions
    ]
    occurrences = [int(entry.get("occurrences") or 0) for entry in transitions]
    positions = np.arange(len(labels), dtype=float)
    ax.barh(positions, occurrences, color="tab:orange", height=0.6)
    for position, entry, height in zip(positions, transitions, occurrences):
        share = entry.get("share_of_escalated")
        ax.annotate(
            f"n={height}" + ("" if share is None else f", {share * 100:.1f}% of escalations"),
            (height, position),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=7.5,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Observed escalations (n)")
    ax.set_ylabel("Escalation transition (initial -> final)")
    ax.set_title(
        "Escalation transitions: where does the router go when it escalates?",
        fontsize=10,
    )
    ax.text(
        0.0,
        -0.18,
        "only observed transitions are shown; an absent pair never happened",
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
    )
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    return _finish(
        fig,
        plt,
        out_dir=out_dir,
        filename=filename or "e8_escalation_transitions.png",
        provenance=provenance,
        notes=[
            f"transitions={len(transitions)}",
            f"n_escalated={result.get('n_escalated')}",
            f"seed={seed}",
            "jitter=none",
            f"analysis={A.ANALYSIS_VERSION}",
        ],
    )


# --------------------------------------------------------------------------
# 5. threshold sensitivity
# --------------------------------------------------------------------------


def _threshold_of(name: str, column: str) -> float:
    """The threshold a sweep variant name carries, e.g. `sufficiency_threshold=0.5`."""
    prefix = f"{column}="
    if not name.startswith(prefix):
        raise PlotDataError(
            f"variant {name!r} does not name a {column} value; the threshold sweep "
            f"labels its arms '{column}=<value>', so the x-axis cannot be read. "
            "Pass an analysis built with evaluation.ablation.threshold_sweep."
        )
    try:
        return float(name[len(prefix):])
    except ValueError as exc:
        raise PlotDataError(
            f"variant {name!r} carries a non-numeric {column} value"
        ) from exc


def _panel(
    ax: Any,
    xs: Sequence[float],
    ys: Sequence[float | None],
    *,
    ylabel: str,
    column: str,
) -> None:
    """One sweep panel: a line where measured, an explicit note where not."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        ax.text(
            0.5,
            0.5,
            f"no measured values for {column}\n(absence of evidence, not zero)",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
        )
        ax.set_xlabel("Sufficiency threshold")
        ax.set_ylabel(ylabel)
        return
    ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", color="tab:blue")
    for x, y in pairs:
        ax.annotate(f"{y:.4g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Sufficiency threshold")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def plot_threshold_sensitivity(
    rows: Sequence[RowLike],
    out_dir: str | Path,
    *,
    provenance: PlotProvenance,
    analysis: Mapping[str, Any] | None = None,
    metric: str = DEFAULT_QUALITY_METRIC,
    latency_column: str = DEFAULT_LATENCY_COLUMN,
    cost_column: str = DEFAULT_COST_COLUMN,
    threshold_column: str = "sufficiency_threshold",
    seed: int = DEFAULT_SEED,
    filename: str | None = None,
) -> Path:
    """One parameter, three consequences: threshold against quality, latency, cost.

    Three panels sharing the threshold axis, because the sweep's answer is a
    trade-off and a single panel would show one side of it. A threshold that
    escalates more often is only interesting next to what those escalations cost
    and bought; a panel whose column was never measured says so in words instead
    of drawing a flat line at zero.

    The x coordinate is the value `evaluation.ablation.threshold_sweep` put in
    the variant name, not a number inferred from the rows, and a variant whose
    name does not carry one raises rather than being placed by guesswork.

    Question answered: *what does a stricter sufficiency threshold buy, and what
    does it spend?*
    """
    _require_rows(rows, "threshold_sensitivity")
    result = analysis if analysis is not None else A.analyze_threshold_sweep(rows)
    per_variant = _require_block(result, "per_variant", "threshold_sensitivity")

    xs: list[float] = []
    quality: list[float | None] = []
    latency: list[float | None] = []
    cost: list[float | None] = []
    empty_variants: list[str] = []
    for name, block in per_variant.items():
        value = _cell_mean(block, "quality", metric)
        if not block.get("n") or value is None:
            empty_variants.append(str(name))
            continue
        xs.append(_threshold_of(str(name), threshold_column))
        quality.append(value)
        latency.append(_cell_mean(block, "latency", latency_column))
        cost.append(_cell_mean(block, "cost", cost_column))
    if not xs:
        raise PlotDataError(
            f"no threshold variant carries a measured {metric!r}; nothing to sweep. "
            f"Variants with no data: {empty_variants or '(none)'}."
        )

    _, plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    _panel(axes[0], xs, quality, ylabel=f"Mean {metric} (0-1)", column=metric)
    _panel(
        axes[1],
        xs,
        latency,
        ylabel=f"Mean {latency_column} ({_axis_unit(latency_column)})",
        column=latency_column,
    )
    _panel(
        axes[2],
        xs,
        cost,
        ylabel=f"Mean {cost_column} ({_axis_unit(cost_column)})",
        column=cost_column,
    )
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    fig.suptitle(
        f"Threshold sensitivity: what does {threshold_column} buy, and what does it cost?",
        fontsize=10,
    )
    notes = [
        f"variants={len(xs)}",
        f"analysis={A.ANALYSIS_VERSION}",
        f"seed={seed}",
        "jitter=none",
    ]
    if empty_variants:
        notes.append("no data: " + ",".join(empty_variants))
    return _finish(
        fig,
        plt,
        out_dir=out_dir,
        filename=filename or "e4_threshold_sensitivity.png",
        provenance=provenance,
        notes=notes,
    )


# --------------------------------------------------------------------------
# 6. feature ablation
# --------------------------------------------------------------------------


def ablated_feature_groups(
    result: Mapping[str, Any],
    *,
    baseline: str = "full",
    groups: Sequence[str] = FEATURE_GROUPS,
) -> tuple[str, ...]:
    """The feature groups this ablation actually removed, in `FEATURE_GROUPS` order.

    All six real groups -- including `entity`, the one a hand-written sweep most
    often drops -- come from `schemas.config.FEATURE_GROUPS`, and only the ones
    present in the result with rows are returned. A group whose variant is absent
    was not ablated, and plotting it would be a claim about a run that did not
    happen.
    """
    per_variant = result.get("per_variant")
    if not isinstance(per_variant, Mapping):
        return ()
    ablated = {
        str(name)[len("without_"):]
        for name, block in per_variant.items()
        if str(name) != baseline
        and str(name).startswith("without_")
        and block.get("n")
    }
    return tuple(group for group in groups if group in ablated)


def plot_feature_ablation(
    rows: Sequence[RowLike],
    out_dir: str | Path,
    *,
    provenance: PlotProvenance,
    analysis: Mapping[str, Any] | None = None,
    baseline: str = "full",
    metric: str = DEFAULT_QUALITY_METRIC,
    seed: int = DEFAULT_SEED,
    filename: str | None = None,
) -> Path:
    """The full router against one leave-one-out arm per feature group.

    A bar per ablated group, with the full router drawn as the reference line it
    is compared against and the signed difference printed on each bar, because
    "removing `entity` cost 0.02 recall" is the finding and the absolute bar
    heights are context. Only groups that were really ablated appear; a group
    already disabled in the base config is skipped by
    `evaluation.ablation.feature_group_variants` and is absent here too, since
    dropping it again would report a phantom effect.

    Question answered: *which router signal is actually carrying the routing?*
    """
    _require_rows(rows, "feature_ablation")
    result = analysis if analysis is not None else A.analyze_feature_ablation(rows)
    per_variant = _require_block(result, "per_variant", "feature_ablation")

    base_block = per_variant.get(baseline)
    base_value = _cell_mean(base_block or {}, "quality", metric)
    if base_value is None:
        raise PlotDataError(
            f"the feature-ablation baseline {baseline!r} has no measured {metric!r}; "
            f"variants present: {sorted(per_variant)}. A comparison against a "
            "missing baseline would be a difference from nothing."
        )

    groups = ablated_feature_groups(result, baseline=baseline)
    if not groups:
        raise PlotDataError(
            "no leave-one-out feature group was ablated in these rows, so there is "
            f"nothing to compare against {baseline!r}."
        )

    values: list[float] = []
    for group in groups:
        value = _cell_mean(per_variant[f"without_{group}"], "quality", metric)
        if value is None:
            raise PlotDataError(
                f"the {group!r} ablation arm has no measured {metric!r}; a bar for "
                "it would be a guess about a run that produced no number."
            )
        values.append(value)

    _, plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    positions = np.arange(len(groups), dtype=float)
    ax.bar(positions, values, color="tab:purple", width=0.6)
    for position, group, value in zip(positions, groups, values):
        ax.annotate(
            f"{value:.4f}\n{value - base_value:+.4f} vs full",
            (position, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
        )
    ax.axhline(
        base_value,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"full router ({base_value:.4f})",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"without\n{group}" for group in groups], fontsize=8)
    ax.set_xlabel("Router feature group disabled (leave-one-out)")
    ax.set_ylabel(f"Mean {metric} (0-1)")
    ax.set_title(
        "Feature ablation: which router signal is carrying the routing?",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    return _finish(
        fig,
        plt,
        out_dir=out_dir,
        filename=filename or "e3_feature_ablation.png",
        provenance=provenance,
        notes=[
            f"groups={len(groups)}:{','.join(groups)}",
            f"baseline={baseline}",
            f"seed={seed}",
            "jitter=none",
            f"analysis={A.ANALYSIS_VERSION}",
        ],
    )


__all__ = [
    "DEFAULT_COST_COLUMN",
    "DEFAULT_LATENCY_COLUMN",
    "DEFAULT_QUALITY_METRIC",
    "DEFAULT_SEED",
    "FIGURE_DPI",
    "FIGURE_NAMES",
    "FEATURE_GROUPS",
    "PLOTS_VERSION",
    "PlotDataError",
    "PlotDependencyError",
    "PlotError",
    "PlotProvenance",
    "STRATEGY_LABELS",
    "ablated_feature_groups",
    "observed_transition_labels",
    "plot_escalation_transitions",
    "plot_feature_ablation",
    "plot_quality_vs_cost",
    "plot_quality_vs_latency",
    "plot_strategy_distribution",
    "plot_threshold_sensitivity",
    "sha256_file",
    "strategy_distribution",
]
