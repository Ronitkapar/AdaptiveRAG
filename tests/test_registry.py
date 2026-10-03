"""
tests.test_registry
-------------------
Offline checks for the Phase 7 experiment registry.

The registry is the spine of the reporting layer, and its one interesting
property is that it refuses. Everything else -- round-tripping, ordering, reading
provenance back out of a run directory -- is plumbing that has to work, but the
tests here mostly exist to prove that a measurement cannot be silently replaced
by a later one, and that a study outside the mandated vocabulary cannot be
invented by a typo.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_rag.errors import ExperimentRunError
from adaptive_rag.experiments.registry import (
    DEFAULT_REGISTRY_NAME,
    EXPERIMENT_IDS,
    REGISTRY_VERSION,
    ExperimentRegistry,
    RegistryEntry,
    RegistryError,
    normalize_experiment_id,
    varied_routing_fields,
)
from adaptive_rag.schemas.config import RoutingConfig

CORPUS_VERSION = "corpus_test_fixture_v1"
GIT_COMMIT = "0" * 40
DATASET_SHA = "f" * 64


def make_entry(
    experiment_id: str = "E1_baseline_comparison",
    variant: str = "bm25",
    *,
    run_dir: str = "/tmp/runs/one",
    trace_count: int = 20,
    kind: str = "baseline",
    **overrides,
) -> RegistryEntry:
    values = {
        "experiment_id": experiment_id,
        "variant": variant,
        "kind": kind,
        "run_dir": run_dir,
        "trace_count": trace_count,
        "git_commit": GIT_COMMIT,
        "dataset_version": "dense_eval_v1",
        "dataset_sha256": DATASET_SHA,
        "config_hash": "abc123",
        "corpus_version": CORPUS_VERSION,
        "timestamp": "2026-10-02T14:00:00+00:00",
        "environment": {"platform": "linux"},
    }
    values.update(overrides)
    return RegistryEntry(**values)


def write_run_dir(
    run_dir: Path,
    *,
    trace_count: int = 20,
    routing: RoutingConfig | None = None,
    config_name: str = "phase7_bm25",
) -> Path:
    """A run directory shaped like the ones `ExperimentRunner` writes.

    Only the two files the registry reads are written. Duplicating the runner's
    real output here would test the fixture rather than the reader.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "config_name": config_name,
                "config_hash": "deadbeef",
                "corpus_version": CORPUS_VERSION,
                "retrieval_method": "bm25",
                "trace_count": trace_count,
                "status_counts": {"ok": trace_count},
                "git_commit": GIT_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": config_name,
                "routing": (routing or RoutingConfig()).model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


# --- vocabulary ---------------------------------------------------------------


def test_the_registry_covers_every_mandated_study():
    assert EXPERIMENT_IDS == (
        "E1_baseline_comparison",
        "E2_escalation_ablation",
        "E3_feature_ablation",
        "E4_threshold_calibration",
        "E5_cost_weight",
        "E6_routing_overhead",
        "E7_query_type",
        "E8_escalation_analysis",
    )


def test_the_registry_declares_a_version():
    assert REGISTRY_VERSION == "phase7_registry_v1"


def test_an_unknown_experiment_id_is_rejected():
    """A typo must not invent a ninth study; an unreportable id is worse than none."""
    with pytest.raises(RegistryError, match="unknown experiment_id"):
        normalize_experiment_id("E9_made_up")


def test_the_cost_weight_alias_maps_onto_the_mandated_id():
    """`ablation.cost_weight_sweep` stamps the pre-registry spelling; the registry
    normalises it in one documented place rather than editing the ablation module."""
    assert normalize_experiment_id("E5_cost_weight_ablation") == "E5_cost_weight"


def test_an_entry_is_normalised_on_construction():
    entry = make_entry(experiment_id="E5_cost_weight_ablation", variant="cost_weight=0.5")

    assert entry.experiment_id == "E5_cost_weight"
    assert entry.key == ("E5_cost_weight", "cost_weight=0.5")


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValidationError):
        make_entry(kind="sneaky")


def test_extra_fields_are_forbidden():
    with pytest.raises(Exception):
        make_entry(unexpected_field="x")


# --- append-only --------------------------------------------------------------


def test_registering_the_same_entry_twice_raises():
    registry = ExperimentRegistry()
    registry.register(make_entry())

    with pytest.raises(RegistryError, match="already registered"):
        registry.register(make_entry())


def test_the_refusal_names_the_run_directory_already_on_record():
    """The message has to point at what is being protected, not just say no."""
    registry = ExperimentRegistry()
    registry.register(make_entry(run_dir="/tmp/runs/first"))

    with pytest.raises(RegistryError, match="/tmp/runs/first"):
        registry.register(make_entry(run_dir="/tmp/runs/second"))


def test_allow_reentry_replaces_rather_than_duplicating():
    """The one sanctioned overwrite, and it must not leave two rows behind."""
    registry = ExperimentRegistry()
    registry.register(make_entry(run_dir="/tmp/runs/first"))

    registry.register(make_entry(run_dir="/tmp/runs/second"), allow_reentry=True)

    assert len(registry) == 1
    assert registry.get("E1_baseline_comparison", "bm25").run_dir == "/tmp/runs/second"


def test_a_different_variant_joins_the_same_study_without_clobbering():
    """This is the E3 shape: one study, one arm per disabled feature group."""
    registry = ExperimentRegistry()
    registry.register(make_entry(experiment_id="E3_feature_ablation", variant="full"))
    registry.register(
        make_entry(experiment_id="E3_feature_ablation", variant="without_technical")
    )
    registry.register(
        make_entry(experiment_id="E3_feature_ablation", variant="without_semantic")
    )

    assert len(registry) == 3
    assert registry.variants("E3_feature_ablation") == [
        "full",
        "without_semantic",
        "without_technical",
    ]
    # The first registration is still the one on record for its own variant.
    assert (
        registry.get("E3_feature_ablation", "full").run_dir == "/tmp/runs/one"
    )


def test_a_different_study_may_reuse_a_variant_name():
    """`full` in E3 and `adaptive` in E1 are different measurements."""
    registry = ExperimentRegistry()
    registry.register(make_entry(experiment_id="E3_feature_ablation", variant="full"))
    registry.register(make_entry(experiment_id="E1_baseline_comparison", variant="adaptive"))

    assert len(registry) == 2


def test_extend_stops_at_the_first_refusal():
    registry = ExperimentRegistry([make_entry()])
    with pytest.raises(RegistryError):
        registry.extend([make_entry(variant="dense"), make_entry()])


def test_lookups_of_an_unregistered_entry_return_none():
    registry = ExperimentRegistry()

    assert registry.get("E1_baseline_comparison", "bm25") is None


# --- provenance read back from a run -------------------------------------------


def test_an_entry_is_built_from_the_run_directorys_own_artifacts(tmp_path: Path):
    """Provenance is read, not passed in, so an entry cannot disagree with its run."""
    run_dir = write_run_dir(tmp_path / "20261002T140000Z-phase7_bm25", trace_count=20)

    entry = RegistryEntry.from_run_dir(
        run_dir,
        experiment_id="E1_baseline_comparison",
        variant="bm25",
        kind="baseline",
        dataset_version="dense_eval_v1",
        dataset_sha256=DATASET_SHA,
        base_routing=RoutingConfig(),
    )

    assert entry.trace_count == 20
    assert entry.git_commit == GIT_COMMIT
    assert entry.config_hash == "deadbeef"
    assert entry.corpus_version == CORPUS_VERSION
    assert entry.routing["available_strategies"] == list(RoutingConfig().available_strategies)
    assert entry.varied_routing_fields == {}


def test_the_timestamp_is_parsed_out_of_a_timestamped_run_id(tmp_path: Path):
    run_dir = write_run_dir(tmp_path / "20261002T140000Z-phase7_bm25")

    entry = RegistryEntry.from_run_dir(
        run_dir,
        experiment_id="E1_baseline_comparison",
        variant="bm25",
        kind="baseline",
        dataset_version="dense_eval_v1",
        dataset_sha256=DATASET_SHA,
    )

    assert entry.timestamp == "2026-10-02T14:00:00+00:00"


def test_an_untimestamped_run_id_asks_for_a_timestamp_rather_than_inventing_one(
    tmp_path: Path,
):
    """The suite names run directories deterministically, so it passes one in."""
    run_dir = write_run_dir(tmp_path / "E1_baseline_comparison__bm25")

    with pytest.raises(RegistryError, match="timestamp"):
        RegistryEntry.from_run_dir(
            run_dir,
            experiment_id="E1_baseline_comparison",
            variant="bm25",
            kind="baseline",
            dataset_version="dense_eval_v1",
            dataset_sha256=DATASET_SHA,
        )

    explicit = RegistryEntry.from_run_dir(
        run_dir,
        experiment_id="E1_baseline_comparison",
        variant="bm25",
        kind="baseline",
        dataset_version="dense_eval_v1",
        dataset_sha256=DATASET_SHA,
        timestamp="2026-10-02T15:00:00+00:00",
    )
    assert explicit.timestamp == "2026-10-02T15:00:00+00:00"


def test_varied_routing_fields_are_the_diff_against_the_base_config():
    """The small set of knobs a study turned, next to the full config that ran."""
    varied = varied_routing_fields(
        RoutingConfig(cost_weight=0.75, sufficiency_threshold=0.4),
        RoutingConfig(),
    )

    assert varied == {"cost_weight": 0.75, "sufficiency_threshold": 0.4}


def test_an_unvaried_config_reports_an_empty_diff():
    assert varied_routing_fields(RoutingConfig(), RoutingConfig()) == {}
    assert varied_routing_fields(None, RoutingConfig()) == {}


def test_pacing_is_recorded_on_the_entry(tmp_path: Path):
    run_dir = write_run_dir(tmp_path / "20261002T140000Z-phase7_dense")

    entry = RegistryEntry.from_run_dir(
        run_dir,
        experiment_id="E1_baseline_comparison",
        variant="dense",
        kind="baseline",
        dataset_version="dense_eval_v1",
        dataset_sha256=DATASET_SHA,
        pace_seconds=0.75,
    )

    assert entry.pace_seconds == 0.75


def test_a_missing_manifest_is_an_error_not_an_empty_entry(tmp_path: Path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(RegistryError, match="manifest.json"):
        RegistryEntry.from_run_dir(
            tmp_path / "empty",
            experiment_id="E1_baseline_comparison",
            variant="bm25",
            kind="baseline",
            dataset_version="dense_eval_v1",
            dataset_sha256=DATASET_SHA,
        )


def test_a_registry_error_is_an_experiment_error():
    """Callers already catching experiment failures must catch this one."""
    assert issubclass(RegistryError, ExperimentRunError)


# --- persistence --------------------------------------------------------------


def test_a_registry_round_trips_through_disk(tmp_path: Path):
    registry = ExperimentRegistry()
    registry.register(make_entry())
    registry.register(make_entry(experiment_id="E3_feature_ablation", variant="full"))
    path = registry.save(tmp_path / DEFAULT_REGISTRY_NAME)

    loaded = ExperimentRegistry.load(path)

    assert [e.key for e in loaded.entries] == [e.key for e in registry.entries]
    assert loaded.entries[0].model_dump() == registry.entries[0].model_dump()


def test_saving_twice_is_byte_identical(tmp_path: Path):
    """A stable artifact is what lets two re-runs be diffed."""
    registry = ExperimentRegistry()
    registry.register(make_entry(experiment_id="E2_escalation_ablation", variant="C_full"))
    registry.register(make_entry(experiment_id="E1_baseline_comparison", variant="bm25"))

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    registry.save(first)
    registry.save(second)

    assert first.read_bytes() == second.read_bytes()


def test_entries_are_ordered_by_study_then_variant(tmp_path: Path):
    """Ordering is by the mandated study order, not by insertion or by name."""
    registry = ExperimentRegistry()
    registry.register(make_entry(experiment_id="E5_cost_weight", variant="cost_weight=1.0"))
    registry.register(make_entry(experiment_id="E1_baseline_comparison", variant="hybrid"))
    registry.register(make_entry(experiment_id="E1_baseline_comparison", variant="bm25"))

    path = registry.save(tmp_path / "registry.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert [e["experiment_id"] for e in payload["entries"]] == [
        "E1_baseline_comparison",
        "E1_baseline_comparison",
        "E5_cost_weight",
    ]
    assert [e["variant"] for e in payload["entries"]] == ["bm25", "hybrid", "cost_weight=1.0"]


def test_the_saved_payload_declares_its_version_and_vocabulary(tmp_path: Path):
    path = ExperimentRegistry().save(tmp_path / "registry.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["registry_version"] == REGISTRY_VERSION
    assert payload["experiment_ids"] == list(EXPERIMENT_IDS)
    assert payload["n_entries"] == 0


def test_loading_a_missing_registry_raises(tmp_path: Path):
    with pytest.raises(RegistryError, match="no registry"):
        ExperimentRegistry.load(tmp_path / "absent.json")


def test_load_or_empty_starts_a_registry_when_there_is_no_file(tmp_path: Path):
    registry = ExperimentRegistry.load_or_empty(tmp_path / "absent.json")

    assert len(registry) == 0


def test_load_or_empty_extends_an_existing_registration(tmp_path: Path):
    """A resumed suite must join the previous registry, not replace it."""
    path = tmp_path / DEFAULT_REGISTRY_NAME
    ExperimentRegistry([make_entry()]).save(path)

    registry = ExperimentRegistry.load_or_empty(path)
    registry.register(make_entry(variant="dense"))

    assert len(registry) == 2


def test_a_seeded_registry_obeys_the_same_duplicate_rule():
    with pytest.raises(RegistryError, match="already registered"):
        ExperimentRegistry([make_entry(), make_entry()])


def test_merging_reports_the_conflict_rather_than_overwriting():
    left = ExperimentRegistry([make_entry(run_dir="/tmp/runs/left")])
    right = ExperimentRegistry([make_entry(run_dir="/tmp/runs/right")])

    with pytest.raises(RegistryError, match="already registered"):
        left.merge(right)

    assert left.get("E1_baseline_comparison", "bm25").run_dir == "/tmp/runs/left"

    left.merge(right, allow_reentry=True)
    assert left.get("E1_baseline_comparison", "bm25").run_dir == "/tmp/runs/right"


def test_the_summary_reports_an_unrun_study_as_empty_rather_than_omitting_it():
    summary = ExperimentRegistry([make_entry()]).summary()

    assert summary["n_entries"] == 1
    assert summary["by_experiment"]["E1_baseline_comparison"] == ["bm25"]
    assert summary["by_experiment"]["E8_escalation_analysis"] == []


def test_the_entries_property_hands_out_a_copy():
    registry = ExperimentRegistry([make_entry()])
    registry.entries.clear()

    assert len(registry) == 1

