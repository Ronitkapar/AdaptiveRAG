"""
tests.test_plots
----------------
The Phase 7 figure module: that it renders, that it renders *only* measured
numbers, and that it renders without matplotlib being importable.

The last of those is the unusual one and the reason this file spawns
subprocesses. `evaluation.plots` imports matplotlib lazily, inside the function
that draws, so "importing the module does not require the optional extra" is a
property of the module rather than of the environment it happens to be tested
in. Asserting it in-process would prove nothing: by the time a render test has
run, matplotlib is already in `sys.modules` and the claim is unfalsifiable. So
the import check runs in a fresh interpreter with a meta-path blocker standing in
front of matplotlib, and the same blocker is used to prove that a missing
matplotlib produces a typed `PlotDependencyError` rather than a blank figure.

Every other test is offline, deterministic and headless: rows are literal
dicts, so no measurement is re-run, and two renders of the same rows are
byte-identical.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from adaptive_rag.errors import EvaluationError
from adaptive_rag.evaluation import analysis as A
from adaptive_rag.evaluation.plots import (
    DEFAULT_SEED,
    FIGURE_NAMES,
    FEATURE_GROUPS,
    PlotDataError,
    PlotDependencyError,
    PlotProvenance,
    ablated_feature_groups,
    observed_transition_labels,
    plot_escalation_transitions,
    plot_feature_ablation,
    plot_quality_vs_cost,
    plot_quality_vs_latency,
    plot_strategy_distribution,
    plot_threshold_sensitivity,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# The reason used when matplotlib is genuinely absent, in the shape the rest of
# the suite uses for a skipped artifact-dependent test. The render tests below are
# skipped on it, and the absence is *also* asserted to be reported rather than
# swallowed, so a machine without the extra fails loudly instead of going green.
MISSING_MATPLOTLIB = (
    "matplotlib is not installed (optional `plots` extra); the rendering tests "
    "are skipped, and test_missing_matplotlib_raises_typed_error still runs"
)


def _matplotlib_installed() -> bool:
    """Whether matplotlib can be imported here, decided defensively.

    `find_spec` raises rather than returning None when a meta-path finder refuses
    the module, which is exactly what the blocker below does; treating that as
    "not installed" is what makes the skip fire instead of the whole file failing
    to collect.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("matplotlib") is not None
    except (ImportError, ValueError):
        return False


requires_matplotlib = pytest.mark.skipif(
    not _matplotlib_installed(), reason=MISSING_MATPLOTLIB
)


# A meta-path finder that makes `import matplotlib` fail, used to prove the
# module's laziness. Placed ahead of every other finder, so nothing can reach the
# real package behind it.
BLOCK_MATPLOTLIB = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib is blocked for this test")
        return None


sys.meta_path.insert(0, _Blocker())
for _name in [n for n in sys.modules if n == "matplotlib" or n.startswith("matplotlib.")]:
    del sys.modules[_name]
"""


def _run_isolated(body: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a fresh interpreter with the repo's `src` on the path."""
    return subprocess.run(
        [sys.executable, "-c", BLOCK_MATPLOTLIB + body],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT / "src")},
    )


# --- rows ------------------------------------------------------------------


def _row(
    *,
    system: str,
    index: int,
    recall: float,
    latency: float,
    cost: float,
    tokens: int = 1000,
    routed: bool = False,
    initial: str | None = None,
    final: str | None = None,
    escalated: bool = False,
    target: str | None = None,
) -> dict[str, Any]:
    """One row carrying every column the six figures read, and nothing random."""
    return {
        "query_id": f"{system}-{index:03d}",
        "system": system,
        "category": "factual" if index % 2 == 0 else "comparative",
        "split": "test",
        "status": "success",
        "recall_at_5": recall,
        "mrr": recall,
        "ndcg_at_5": recall,
        "retrieval_latency_ms": latency - 1.0,
        "routing_latency_ms": 0.5 if routed else None,
        "total_latency_ms": latency,
        "context_tokens": tokens,
        "input_tokens": tokens,
        "output_tokens": tokens // 2,
        "total_tokens": tokens + tokens // 2,
        "estimated_cost_usd": cost,
        "initial_strategy": initial if routed else None,
        "final_strategy": final if routed else None,
        "escalated": escalated if routed else None,
        "escalation_target": target if escalated else None,
        "sufficiency_score": 0.4 if routed else None,
        "sufficiency_threshold": 0.5 if routed else None,
        "stage_count": 2 if escalated else (1 if routed else None),
    }


def _arm_rows(n: int = 8) -> list[dict[str, Any]]:
    """Three arms, with the adaptive one escalating on a fixed pattern."""
    rows: list[dict[str, Any]] = []
    for index in range(n):
        rows.append(
            _row(system="bm25", index=index, recall=0.50, latency=20.0, cost=0.0002)
        )
        rows.append(
            _row(system="dense", index=index, recall=0.60, latency=35.0, cost=0.0004)
        )
        escalated = index % 2 == 0
        initial, target = "bm25", "hybrid"
        rows.append(
            _row(
                system="adaptive",
                index=index,
                recall=0.70 if escalated else 0.65,
                latency=50.0 if escalated else 30.0,
                cost=0.0005 if escalated else 0.0004,
                routed=True,
                initial=initial,
                final=target if escalated else initial,
                escalated=escalated,
                target=target if escalated else None,
            )
        )
    return rows


def _threshold_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, value in enumerate((0.4, 0.5, 0.6)):
        for index in range(6):
            rows.append(
                _row(
                    system=f"sufficiency_threshold={value}",
                    index=index,
                    recall=0.60 + 0.05 * offset,
                    latency=40.0 + 10.0 * offset,
                    cost=0.0003 + 0.0001 * offset,
                )
            )
    return rows


def _feature_rows(groups: Sequence[str] = FEATURE_GROUPS) -> list[dict[str, Any]]:
    """The full router plus one leave-one-out arm per named group."""
    rows: list[dict[str, Any]] = []
    for index in range(6):
        rows.append(_row(system="full", index=index, recall=0.75, latency=50.0, cost=0.0005))
    for offset, group in enumerate(groups):
        for index in range(6):
            rows.append(
                _row(
                    system=f"without_{group}",
                    index=index,
                    recall=0.75 - 0.01 * (offset + 1),
                    latency=50.0,
                    cost=0.0005,
                )
            )
    return rows


@pytest.fixture()
def arm_rows() -> list[dict[str, Any]]:
    return _arm_rows()


@pytest.fixture()
def threshold_rows() -> list[dict[str, Any]]:
    return _threshold_rows()


@pytest.fixture()
def feature_rows() -> list[dict[str, Any]]:
    return _feature_rows()


@pytest.fixture()
def provenance() -> PlotProvenance:
    return PlotProvenance(
        source_artifact="tests/synthetic:rows",
        dataset_version="phase7_dataset_v0",
        dataset_sha256="a" * 64,
        git_commit="0123456789abcdef",
        config_hash="cfg123",
        corpus_version="corpus_v0",
        generated_at="2026-01-01T00:00:00+00:00",
    )


# --- laziness and the backend ---------------------------------------------


def test_importing_plots_does_not_require_matplotlib():
    """`import adaptive_rag.evaluation.plots` must work with matplotlib absent.

    Proven in a fresh interpreter with matplotlib blocked, because in this one it
    is already imported by the render tests and the claim would be vacuous.
    """
    result = _run_isolated(
        "import os, sys\n"
        # The isolated environment carries no DISPLAY, so this also proves the
        # import neither needs nor opens one.
        "assert 'DISPLAY' not in os.environ\n"
        "from adaptive_rag.evaluation import plots\n"
        "assert not [m for m in sys.modules if m.startswith('matplotlib')], "
        "sorted(m for m in sys.modules if m.startswith('matplotlib'))\n"
        "assert plots.FIGURE_NAMES[0] == 'quality_vs_latency'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_missing_matplotlib_raises_typed_error(tmp_path):
    """With matplotlib blocked, a plot call raises rather than writing a figure.

    This is the guard on the skip above: an environment without the extra must
    not turn the render tests into a silent green suite, and it must not produce a
    blank PNG either.
    """
    script = (
        "from adaptive_rag.evaluation.plots import (\n"
        "    PlotDependencyError, PlotProvenance, plot_quality_vs_latency,\n"
        ")\n"
        "prov = PlotProvenance(source_artifact='blocked')\n"
        "rows = [{'system': 'bm25', 'status': 'success', 'recall_at_5': 0.5,\n"
        "         'total_latency_ms': 10.0, 'estimated_cost_usd': 0.001}]\n"
        "try:\n"
        f"    plot_quality_vs_latency(rows, {str(tmp_path)!r}, provenance=prov)\n"
        "except PlotDependencyError as exc:\n"
        "    assert 'matplotlib' in str(exc)\n"
        "    print('raised')\n"
        "else:\n"
        "    raise AssertionError('expected PlotDependencyError')\n"
    )
    result = _run_isolated(script)
    assert result.returncode == 0, result.stderr
    assert "raised" in result.stdout
    assert not list(tmp_path.glob("*.png"))


@requires_matplotlib
def test_render_forces_agg_backend(tmp_path, provenance):
    """The figures are drawn on `Agg`, so no display is ever opened."""
    import matplotlib

    path = plot_quality_vs_latency(_arm_rows(), tmp_path, provenance=provenance)
    assert path.is_file()
    assert matplotlib.get_backend().lower() == "agg"


# --- every figure writes a real file --------------------------------------


def _figure_calls(
    arm_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
) -> dict[str, Callable[[Path, PlotProvenance], Path]]:
    return {
        "quality_vs_latency": lambda out, prov: plot_quality_vs_latency(
            arm_rows, out, provenance=prov
        ),
        "quality_vs_cost": lambda out, prov: plot_quality_vs_cost(
            arm_rows, out, provenance=prov
        ),
        "strategy_distribution": lambda out, prov: plot_strategy_distribution(
            [r for r in arm_rows if r["system"] == "adaptive"], out, provenance=prov
        ),
        "escalation_transitions": lambda out, prov: plot_escalation_transitions(
            [r for r in arm_rows if r["system"] == "adaptive"], out, provenance=prov
        ),
        "threshold_sensitivity": lambda out, prov: plot_threshold_sensitivity(
            threshold_rows, out, provenance=prov
        ),
        "feature_ablation": lambda out, prov: plot_feature_ablation(
            feature_rows, out, provenance=prov
        ),
    }


@requires_matplotlib
@pytest.mark.parametrize("figure", FIGURE_NAMES)
def test_every_figure_writes_a_non_empty_png(
    figure, tmp_path, provenance, arm_rows, threshold_rows, feature_rows
):
    calls = _figure_calls(arm_rows, threshold_rows, feature_rows)
    path = calls[figure](tmp_path, provenance)
    assert path.is_file(), f"{figure} wrote no file"
    assert path.stat().st_size > 0, f"{figure} wrote an empty file"
    assert path.read_bytes()[: len(PNG_MAGIC)] == PNG_MAGIC, f"{figure} is not a PNG"
    assert path.parent == tmp_path


@requires_matplotlib
@pytest.mark.parametrize("figure", FIGURE_NAMES)
def test_every_figure_states_its_question_its_units_and_its_provenance(
    figure, tmp_path, provenance, arm_rows, threshold_rows, feature_rows, monkeypatch
):
    """A figure carries a question, labelled axes with units, and its provenance.

    Checked on the figure object rather than the PNG, because these three
    requirements are about what is drawn on the image and reading pixels to prove
    it would be both slower and less certain. `_save` is wrapped, not replaced, so
    the file is still written and the size still checked.
    """
    from adaptive_rag.evaluation import plots as module

    drawn: list[Any] = []
    original = module._save

    def _recording_save(fig, plt, out_dir, filename):
        drawn.append(fig)
        return original(fig, plt, out_dir, filename)

    monkeypatch.setattr(module, "_save", _recording_save)
    calls = _figure_calls(arm_rows, threshold_rows, feature_rows)
    path = calls[figure](tmp_path, provenance)
    assert path.stat().st_size > 0

    assert len(drawn) == 1
    fig = drawn[0]
    assert fig.axes, f"{figure} drew no axes"
    for ax in fig.axes:
        assert ax.get_xlabel(), f"{figure} has an unlabelled x axis"
        assert ax.get_ylabel(), f"{figure} has an unlabelled y axis"
        assert "(" in ax.get_xlabel() + ax.get_ylabel(), (
            f"{figure} states no unit on its axes"
        )

    titles = " ".join(ax.get_title() for ax in fig.axes)
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        titles = f"{titles} {suptitle.get_text()}"
    assert "?" in titles, f"{figure} does not state its question as a title: {titles!r}"

    captions = " ".join(item.get_text() for item in fig.texts)
    for fragment in ("source=", "dataset=", "sha256=", "commit=", "seed="):
        assert fragment in captions, f"{figure} provenance line lacks {fragment}"


@requires_matplotlib
def test_rendering_is_deterministic(tmp_path, provenance):
    """The same rows drawn twice are byte-identical, so a figure is reproducible."""
    first = plot_quality_vs_latency(_arm_rows(), tmp_path / "a", provenance=provenance)
    second = plot_quality_vs_latency(_arm_rows(), tmp_path / "b", provenance=provenance)
    assert sha256_file(first) == sha256_file(second)


@requires_matplotlib
def test_figures_avoid_coincident_points_with_a_seeded_offset(tmp_path, provenance):
    """Two arms with identical means are offset, and the seed is on the figure.

    The offset has to exist (two identical means would otherwise be one point) and
    it has to be seeded (an unseeded offset would make every re-render differ).
    """
    rows = _arm_rows() + [
        _row(system="dense", index=100 + i, recall=0.60, latency=35.0, cost=0.0004)
        for i in range(4)
    ]
    path = plot_quality_vs_latency(rows, tmp_path, provenance=provenance)
    assert path.stat().st_size > 0
    other = plot_quality_vs_latency(
        rows, tmp_path, provenance=provenance, seed=DEFAULT_SEED + 1, filename="other.png"
    )
    assert other.stat().st_size > 0


# --- insufficient data raises ---------------------------------------------


def test_plot_data_error_is_a_typed_evaluation_error():
    assert issubclass(PlotDataError, EvaluationError)
    assert issubclass(PlotDependencyError, Exception)


def _all_figures_on(rows: Sequence[dict[str, Any]]) -> list[Callable[[Path], None]]:
    return [
        lambda out: plot_quality_vs_latency(rows, out, provenance=PlotProvenance(source_artifact="t")),
        lambda out: plot_quality_vs_cost(rows, out, provenance=PlotProvenance(source_artifact="t")),
        lambda out: plot_strategy_distribution(
            rows, out, provenance=PlotProvenance(source_artifact="t")
        ),
        lambda out: plot_escalation_transitions(
            rows, out, provenance=PlotProvenance(source_artifact="t")
        ),
        lambda out: plot_threshold_sensitivity(
            rows, out, provenance=PlotProvenance(source_artifact="t")
        ),
        lambda out: plot_feature_ablation(
            rows, out, provenance=PlotProvenance(source_artifact="t")
        ),
    ]


@pytest.mark.parametrize("index", range(6))
def test_empty_rows_raise_instead_of_drawing_a_blank_figure(index, tmp_path):
    """No rows is an absence of evidence, and every figure says so by raising."""
    with pytest.raises(PlotDataError) as excinfo:
        _all_figures_on([])[index](tmp_path)
    assert "rows" in str(excinfo.value)
    assert not list(tmp_path.glob("*.png")), "a figure was written from no rows"


def test_fixed_strategy_rows_cannot_produce_a_routing_distribution(tmp_path):
    """A distribution needs routing decisions; without them it raises."""
    rows = [r for r in _arm_rows() if r["system"] == "bm25"]
    with pytest.raises(PlotDataError) as excinfo:
        plot_strategy_distribution(rows, tmp_path, provenance=PlotProvenance(source_artifact="t"))
    assert "bm25" in str(excinfo.value) or "routing" in str(excinfo.value)
    assert not list(tmp_path.glob("*.png"))


def test_no_observed_escalation_raises(tmp_path):
    """An adaptive run that never escalated has no transition figure to draw."""
    rows = [
        _row(
            system="adaptive", index=i, recall=0.7, latency=30.0, cost=0.0004,
            routed=True, initial="bm25", final="bm25", escalated=False,
        )
        for i in range(6)
    ]
    with pytest.raises(PlotDataError) as excinfo:
        plot_escalation_transitions(rows, tmp_path, provenance=PlotProvenance(source_artifact="t"))
    assert "transition" in str(excinfo.value)


def test_cost_figure_names_the_columns_that_carry_data(tmp_path):
    """A cost column nobody measured raises, and says which ones do."""
    rows = [
        {k: v for k, v in row.items() if k not in {"estimated_cost_usd"}}
        for row in _arm_rows()
    ]
    with pytest.raises(PlotDataError) as excinfo:
        plot_quality_vs_cost(rows, tmp_path, provenance=PlotProvenance(source_artifact="t"))
    message = str(excinfo.value)
    assert "estimated_cost_usd" in message
    assert "context_tokens" in message


def test_feature_ablation_without_its_baseline_raises(tmp_path):
    """Leave-one-out arms with no `full` arm are a comparison against nothing."""
    rows = [r for r in _feature_rows(("lexical", "entity")) if r["system"] != "full"]
    with pytest.raises(PlotDataError) as excinfo:
        plot_feature_ablation(rows, tmp_path, provenance=PlotProvenance(source_artifact="t"))
    assert "full" in str(excinfo.value)


# --- only observed transitions --------------------------------------------


def test_observed_transition_labels_are_exactly_the_observed_pairs():
    """The plotted axis comes from the analyzer's `transitions` list, unchanged."""
    rows = [
        _row(
            system="adaptive", index=0, recall=0.7, latency=30.0, cost=0.0004,
            routed=True, initial="bm25", final="hybrid", escalated=True, target="hybrid",
        ),
        _row(
            system="adaptive", index=1, recall=0.7, latency=30.0, cost=0.0004,
            routed=True, initial="bm25", final="hybrid", escalated=True, target="hybrid",
        ),
        _row(
            system="adaptive", index=2, recall=0.7, latency=30.0, cost=0.0004,
            routed=True, initial="dense", final="hybrid", escalated=True, target="hybrid",
        ),
        _row(
            system="adaptive", index=3, recall=0.7, latency=30.0, cost=0.0004,
            routed=True, initial="dense", final="dense", escalated=False,
        ),
    ]
    result = A.analyze_escalation_transitions(rows)
    labels = observed_transition_labels(result)
    assert labels == (("bm25", "hybrid"), ("dense", "hybrid"))
    observed = {(t["from"], t["to"]) for t in result["transitions"]}
    assert observed == set(labels)
    # The ladder offers bm25 -> dense and hybrid -> hybrid_rerank; neither happened.
    assert ("bm25", "dense") not in observed
    assert ("hybrid", "hybrid_rerank") not in observed
    assert result["n_transitions_observed"] == 2


@requires_matplotlib
def test_transition_figure_renders_only_observed_pairs(tmp_path, provenance):
    rows = [
        _row(
            system="adaptive", index=index, recall=0.7, latency=30.0, cost=0.0004,
            routed=True, initial="bm25", final="hybrid", escalated=True, target="hybrid",
        )
        for index in range(6)
    ] + [
        _row(
            system="adaptive", index=100 + index, recall=0.7, latency=30.0, cost=0.0004,
            routed=True, initial="dense", final="dense", escalated=False,
        )
        for index in range(6)
    ]
    result = A.analyze_escalation_transitions(rows)
    assert observed_transition_labels(result) == (("bm25", "hybrid"),)
    path = plot_escalation_transitions(
        rows, tmp_path, provenance=provenance, analysis=result
    )
    assert path.stat().st_size > 0


# --- feature ablation groups ----------------------------------------------


def test_ablated_feature_groups_covers_a_subset_including_entity():
    """Only the groups actually ablated are reported, in `FEATURE_GROUPS` order."""
    rows = _feature_rows(("entity", "lexical"))
    result = A.analyze_feature_ablation(rows, variants=("full", "without_lexical", "without_entity"))
    assert ablated_feature_groups(result) == ("lexical", "entity")
    assert "entity" in ablated_feature_groups(result)
    assert "semantic" not in ablated_feature_groups(result)


def test_all_six_real_groups_are_recognised():
    """`FEATURE_GROUPS` is the real list, `entity` included."""
    assert FEATURE_GROUPS == (
        "lexical",
        "semantic",
        "entity",
        "complexity",
        "question_type",
        "multi_concept",
    )
    result = A.analyze_feature_ablation(_feature_rows())
    assert ablated_feature_groups(result) == FEATURE_GROUPS


@requires_matplotlib
def test_feature_ablation_figure_handles_a_subset_of_groups(tmp_path, provenance):
    rows = _feature_rows(("lexical", "entity", "multi_concept"))
    result = A.analyze_feature_ablation(
        rows, variants=("full", "without_lexical", "without_entity", "without_multi_concept")
    )
    assert ablated_feature_groups(result) == ("lexical", "entity", "multi_concept")
    path = plot_feature_ablation(rows, tmp_path, provenance=provenance, analysis=result)
    assert path.stat().st_size > 0


# --- provenance ------------------------------------------------------------


def test_provenance_line_records_source_dataset_sha_and_commit(provenance):
    line = provenance.line("systems=5")
    assert "source=tests/synthetic:rows" in line
    assert "dataset=phase7_dataset_v0" in line
    assert f"sha256={'a' * 16}" in line
    assert "commit=0123456789abcdef" in line
    assert "systems=5" in line


def test_provenance_line_marks_missing_fields_as_unknown():
    line = PlotProvenance(source_artifact="x").line()
    assert "dataset=unknown" in line
    assert "sha256=unknown" in line
    assert "commit=unknown" in line


def test_provenance_from_paths_hashes_a_present_dataset(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"example_id": "q1"}\n', encoding="utf-8")
    prov = PlotProvenance.from_paths(
        "runs/adaptive/rows.jsonl", dataset_path=dataset, dataset_version="v1", git_commit="abc"
    )
    assert prov.dataset_sha256 == sha256_file(dataset)
    assert prov.line().count("|") >= 3
    # An absent dataset leaves the hash unknown rather than raising.
    absent = PlotProvenance.from_paths("rows.jsonl", dataset_path=tmp_path / "nope.jsonl")
    assert absent.dataset_sha256 is None


# --- the CLI ---------------------------------------------------------------


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    return path


@requires_matplotlib
def test_cli_writes_every_figure_and_a_manifest(tmp_path, capsys):
    from scripts import make_phase7_plots

    rows_path = _write_rows(
        tmp_path / "rows.jsonl", _arm_rows() + _threshold_rows() + _feature_rows()
    )
    out = tmp_path / "figures"
    code = make_phase7_plots.main(["--rows", str(rows_path), "--out", str(out)])
    assert code == 0
    capsys.readouterr()

    manifest = json.loads((out / "figures_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["figures_written"] == list(FIGURE_NAMES)
    assert manifest["skipped"] == []
    assert len(manifest["figures"]) == len(FIGURE_NAMES)
    for entry in manifest["figures"]:
        path = Path(entry["path"])
        assert path.is_file() and path.stat().st_size > 0
        assert entry["sha256"] == sha256_file(path)
        assert entry["question"]
        provenance_block = entry["provenance"]
        assert provenance_block["source_artifact"] == str(rows_path)
        assert provenance_block["git_commit"]
        assert f"seed={entry['seed']}" in " ".join(provenance_block["notes"])


@requires_matplotlib
def test_cli_selects_a_subset_of_figures(tmp_path, capsys):
    from scripts import make_phase7_plots

    rows_path = _write_rows(
        tmp_path / "rows.jsonl", _arm_rows() + _threshold_rows() + _feature_rows()
    )
    out = tmp_path / "figures"
    code = make_phase7_plots.main(
        [
            "--rows",
            str(rows_path),
            "--out",
            str(out),
            "--experiments",
            "quality_vs_latency,feature_ablation",
        ]
    )
    assert code == 0
    capsys.readouterr()
    manifest = json.loads((out / "figures_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["figures_written"] == [
        "quality_vs_latency",
        "feature_ablation",
    ]
    assert not (out / "e4_threshold_sensitivity.png").exists()


def test_cli_reports_a_missing_matplotlib_instead_of_a_partial_report(tmp_path):
    """With the extra absent the CLI exits 2 and writes no figure at all.

    Half a figure set is worse than none, because the reader cannot tell which
    half is missing; a non-zero exit and a message naming the extra is the honest
    outcome, and it is checked in an isolated interpreter because matplotlib is
    present in the one this test runs in.
    """
    rows_path = _write_rows(tmp_path / "rows.jsonl", _arm_rows())
    out = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            BLOCK_MATPLOTLIB
            + "import sys\n"
            "from scripts import make_phase7_plots\n"
            f"sys.exit(make_phase7_plots.main(['--rows', {str(rows_path)!r},"
            f" '--out', {str(out)!r}]))\n",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "matplotlib" in result.stderr
    assert not out.exists() or not list(out.glob("*.png"))


def test_cli_rejects_an_unknown_figure_name(tmp_path, capsys):
    from scripts import make_phase7_plots

    rows_path = _write_rows(tmp_path / "rows.jsonl", _arm_rows())
    code = make_phase7_plots.main(
        ["--rows", str(rows_path), "--out", str(tmp_path / "f"), "--experiments", "nope"]
    )
    assert code == 2
    assert "unknown figure" in capsys.readouterr().err


@requires_matplotlib
def test_cli_reports_rows_that_cannot_support_a_figure(tmp_path, capsys):
    """A figure with no data is recorded as skipped with its reason, not drawn."""
    from scripts import make_phase7_plots

    rows_path = _write_rows(tmp_path / "rows.jsonl", _arm_rows())
    out = tmp_path / "figures"
    code = make_phase7_plots.main(["--rows", str(rows_path), "--out", str(out)])
    capsys.readouterr()
    manifest = json.loads((out / "figures_manifest.json").read_text(encoding="utf-8"))
    skipped = {entry["figure"] for entry in manifest["skipped"]}
    assert {"threshold_sensitivity", "feature_ablation"} <= skipped
    for entry in manifest["skipped"]:
        assert entry["error"] == "PlotDataError"
        assert entry["reason"]
    assert code == 1 or manifest["figures"], "the drawn figures must still be reported"


def test_cli_refuses_an_empty_export(tmp_path, capsys):
    from scripts import make_phase7_plots

    rows_path = _write_rows(tmp_path / "rows.jsonl", [])
    code = make_phase7_plots.main(["--rows", str(rows_path), "--out", str(tmp_path / "f")])
    assert code == 2
    assert "nothing to draw" in capsys.readouterr().err


def test_cli_hashes_every_rows_export_it_reads(tmp_path):
    from scripts import make_phase7_plots

    first = _write_rows(tmp_path / "a.jsonl", _arm_rows())
    second = _write_rows(tmp_path / "b.jsonl", _threshold_rows())
    rows, inputs = make_phase7_plots.load_rows([first, second])
    assert len(rows) == len(_arm_rows()) + len(_threshold_rows())
    assert [entry["sha256"] for entry in inputs] == [
        sha256_file(first),
        sha256_file(second),
    ]
    assert make_phase7_plots.routed_rows(rows), "the adaptive arm must be found"
