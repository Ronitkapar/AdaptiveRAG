"""
tests.test_phase7_harness
-------------------------
Safety properties of the Phase 7 measurement harness itself.

Phase 7 is retrieval-only and benchmarks the frozen 107-record dataset. Both of
those facts used to be optional: the suite had no way to say "do not generate"
(passing `generator=None` fell through to the arm's default generator, so every
retrieval-only run came back `generation_failed`), and `--dataset` defaulted to the
Phase 2-6 20-record set, so an omitted flag silently benchmarked the wrong dataset.

These are tests of the harness contract rather than of any measurement:
`--dataset` is required and the legacy set is refused unless asked for by name, the
frozen set is the documented target, the Phase 2-6 set still loads unchanged, and
`--retrieval-only` produces `ok` traces with no generation stage. The dataset
contract is checked on *both* Phase 7 entry points -- the suite runner and the
analysis driver -- because the analysis reads the same file to build E8's gold
relevance map, and a defaulted 20-record set there joins labels to rows that do not
correspond rather than failing.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from adaptive_rag.config.paths import REPO_ROOT
from adaptive_rag.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    LEGACY_DENSE_DATASET_PATH,
    PHASE7_DATASET_PATH,
    is_legacy_dataset,
    load_evaluation_dataset,
    partition_by_split,
    validate_split_separation,
)
from adaptive_rag.errors import AdaptiveRAGError
from adaptive_rag.indexing.bm25 import BM25_INDEX_PATH

FROZEN_RECORD_COUNT = 107
LEGACY_RECORD_COUNT = 20


def _cli():
    """Load `scripts/run_phase7_suite.py` by path; `scripts/` has no package."""
    spec = importlib.util.spec_from_file_location(
        "_phase7_suite_cli", REPO_ROOT / "scripts" / "run_phase7_suite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analysis_cli():
    """Load `scripts/run_phase7_analysis.py` by path; `scripts/` has no package."""
    spec = importlib.util.spec_from_file_location(
        "_phase7_analysis_cli", REPO_ROOT / "scripts" / "run_phase7_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the dataset contract -----------------------------------------------------


def test_the_frozen_phase7_dataset_is_the_documented_benchmark_target():
    assert PHASE7_DATASET_PATH.is_file()
    assert PHASE7_DATASET_PATH != LEGACY_DENSE_DATASET_PATH
    assert is_legacy_dataset(PHASE7_DATASET_PATH) is False

    examples = load_evaluation_dataset(PHASE7_DATASET_PATH)
    validate_split_separation(examples)
    assert len(examples) == FROZEN_RECORD_COUNT
    calibration, test = partition_by_split(examples)
    assert calibration and test


def test_the_phase2_6_dataset_still_loads_unchanged():
    """Phases 2-6 reproducibility: the legacy 20-record set must keep working."""
    assert DEFAULT_DATASET_PATH == LEGACY_DENSE_DATASET_PATH
    assert LEGACY_DENSE_DATASET_PATH.is_file()
    assert len(load_evaluation_dataset()) == LEGACY_RECORD_COUNT
    assert len(load_evaluation_dataset(LEGACY_DENSE_DATASET_PATH)) == LEGACY_RECORD_COUNT


def test_is_legacy_dataset_recognises_a_relative_spelling(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert is_legacy_dataset(Path("data/evaluation/dense_eval_v1.jsonl")) is True
    assert is_legacy_dataset(Path("data/evaluation/phase7_eval_v1.jsonl")) is False


# --- the CLI cannot silently pick the wrong dataset ---------------------------


def test_omitting_the_dataset_flag_is_an_error_not_a_default(tmp_path: Path):
    """An omitted flag must not resolve to *some* dataset; it must resolve to none."""
    with pytest.raises(SystemExit) as excinfo:
        _cli().main(["--arms", "bm25", "--retrieval-only", "--out", str(tmp_path)])
    assert excinfo.value.code == 2


def test_the_legacy_dataset_is_refused_unless_the_run_says_so(tmp_path: Path):
    with pytest.raises(AdaptiveRAGError, match="allow-legacy-dataset"):
        _cli().main(
            [
                "--dataset",
                str(LEGACY_DENSE_DATASET_PATH),
                "--arms",
                "bm25",
                "--retrieval-only",
                "--out",
                str(tmp_path),
            ]
        )
    assert not list(tmp_path.glob("**/traces.jsonl"))


def test_the_escape_hatch_gets_past_the_dataset_guard(tmp_path: Path):
    """`--allow-legacy-dataset` proceeds; the *next* guard is what then complains.

    The legacy set carries no `split` field, so asking for the held-out split fails
    downstream. Reaching that error is the proof the dataset guard was passed
    deliberately rather than silently.
    """
    with pytest.raises(AdaptiveRAGError, match="split"):
        _cli().main(
            [
                "--dataset",
                str(LEGACY_DENSE_DATASET_PATH),
                "--allow-legacy-dataset",
                "--split",
                "test",
                "--arms",
                "bm25",
                "--retrieval-only",
                "--out",
                str(tmp_path),
            ]
        )


def test_the_analysis_cli_cannot_silently_pick_a_dataset(capsys):
    """The E8 gold join is per-example_id, so a defaulted dataset joins nothing."""
    with pytest.raises(SystemExit) as excinfo:
        _analysis_cli().main([])
    assert excinfo.value.code == 2
    assert "--dataset" in capsys.readouterr().err


def test_the_analysis_cli_refuses_the_legacy_dataset_unless_the_run_says_so(tmp_path: Path):
    with pytest.raises(AdaptiveRAGError, match="allow-legacy-dataset"):
        _analysis_cli().main(
            ["--dataset", str(LEGACY_DENSE_DATASET_PATH), "--out", str(tmp_path)]
        )
    assert not list(tmp_path.iterdir())


def test_the_analysis_cli_escape_hatch_gets_past_the_dataset_guard(
    tmp_path: Path, capsys
):
    """`--allow-legacy-dataset` proceeds; the *next* gate is what then complains.

    Reaching the registry check is the proof the dataset guard was passed
    deliberately rather than silently, and that nothing was written on the way.
    """
    exit_code = _analysis_cli().main(
        [
            "--dataset",
            str(LEGACY_DENSE_DATASET_PATH),
            "--allow-legacy-dataset",
            "--registry",
            str(tmp_path / "absent.json"),
            "--out",
            str(tmp_path / "analysis"),
        ]
    )
    assert exit_code == 2
    assert "no registry" in capsys.readouterr().err
    assert not (tmp_path / "analysis").exists()


def test_the_phase7_dataset_needs_no_opt_in_to_be_analysed(tmp_path: Path, capsys):
    """The frozen set is what the guard points at, so it passes unchallenged."""
    exit_code = _analysis_cli().main(
        [
            "--dataset",
            str(PHASE7_DATASET_PATH),
            "--registry",
            str(tmp_path / "absent.json"),
            "--out",
            str(tmp_path / "analysis"),
        ]
    )
    assert exit_code == 2
    assert "no registry" in capsys.readouterr().err
    assert not (tmp_path / "analysis").exists()


# --- --retrieval-only reaches the whole pipeline ------------------------------


@pytest.mark.skipif(
    not BM25_INDEX_PATH.is_file(),
    reason="needs the prebuilt Phase 1 BM25 index; build it before running this",
)
def test_the_retrieval_only_cli_run_is_ok_end_to_end(tmp_path: Path):
    """The protocol the Phase 7 study is defined over, through the real CLI."""
    dataset = tmp_path / "two_records.jsonl"
    source = [
        line
        for line in PHASE7_DATASET_PATH.read_text(encoding="utf-8").splitlines()[:2]
        if line.strip()
    ]
    dataset.write_text("\n".join(source) + "\n", encoding="utf-8")

    exit_code = _cli().main(
        [
            "--dataset",
            str(dataset),
            "--split",
            "all",
            "--arms",
            "bm25",
            "--retrieval-only",
            "--pace",
            "0",
            "--out",
            str(tmp_path / "suite"),
            "--no-registration",
        ]
    )

    assert exit_code == 0
    run_dir = tmp_path / "suite" / "E1_baseline_comparison__bm25"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trace_count"] == 2
    assert manifest["status_counts"]["ok"] == 2
    assert manifest["status_counts"]["generation_failed"] == 0
    assert manifest["error_types"] == []

    traces = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(traces) == 2
    for trace in traces:
        assert trace["status"] == "ok"
        assert trace["retrieval"] is not None
        assert trace["generation"] is None
        assert trace["generation_latency_ms"] is None
        assert trace["total_latency_ms"] is not None
        assert trace["total_latency_ms"] > 0
        assert trace["total_latency_ms"] == trace["retrieval_latency_ms"]
