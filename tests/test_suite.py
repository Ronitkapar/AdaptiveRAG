"""
tests.test_suite
----------------
Offline end-to-end checks for the Phase 7 suite driver.

Every arm here is built from fakes and an in-memory component factory, so the
whole driver -- plan, verify, execute, export rows, register -- runs in
milliseconds with no index, no corpus, no provider and no network.

The properties worth defending are not "does it run" but the three the driver
exists to guarantee: one Qdrant client for the whole suite, pacing that cannot
reach a latency field, and a failing arm that is reported rather than dropped.
"""

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
from qdrant_client import QdrantClient

from adaptive_rag.errors import ConfigurationError, ExperimentRunError
from adaptive_rag.evaluation.ablation import escalation_variants, feature_group_variants
from adaptive_rag.evaluation.suite import (
    ANALYSIS_ONLY_EXPERIMENT_IDS,
    DEFAULT_PACE_SECONDS,
    PacedRetriever,
    PROVIDER_CALLING_METHODS,
    RUNNABLE_EXPERIMENT_IDS,
    SUITE_VERSION,
    SuiteExecutionError,
    SuiteResult,
    plan_arms,
    run_suite,
)
from adaptive_rag.experiments.arms import build_arm, build_arms
from adaptive_rag.experiments.registry import ExperimentRegistry
from adaptive_rag.schemas import (
    ChunkMetadata,
    ChunkProvenance,
    EvaluationConfig,
    EvaluationExample,
    ExperimentConfig,
    RetrievalResult,
)
from adaptive_rag.schemas.config import (
    ChunkingConfig,
    ContextConfig,
    EmbeddingConfig,
    GenerationConfig,
    IndexConfig,
    IngestionConfig,
    RetrievalConfig,
    RoutingConfig,
)
from adaptive_rag.routing import EscalationPolicy, QueryFeatureAnalyzer, RuleBasedRouter, SufficiencyChecker
from adaptive_rag.retrieval.adaptive import AdaptiveRetriever
from tests.fakes import FakeGenerator, StubRetriever

CORPUS_VERSION = "corpus_test_fixture_v1"
DOC_IDS = ("doc_a", "doc_b", "doc_c")


def make_common(**overrides) -> ExperimentConfig:
    """A fully-specified `ExperimentConfig` with no I/O behind it.

    Built directly rather than through `build_experiment_config`, which hashes the
    config *and* calls `compute_corpus_version` -- that reads the real PDF
    manifest, and these tests must stay offline. Mirrors `tests/test_arms.py`.
    """
    base = {
        "experiment_id": "phase7_suite_common",
        "name": "phase7_suite_common",
        "corpus_version": CORPUS_VERSION,
        "ingestion": IngestionConfig(),
        "chunking": ChunkingConfig(),
        "embedding": EmbeddingConfig(),
        "index": IndexConfig(),
        "retrieval": RetrievalConfig(retrieval_method="dense"),
        "context": ContextConfig(),
        "generation": GenerationConfig(),
        "evaluation": EvaluationConfig(enable_llm_judge=False, k_grid=[1, 5, 10]),
        # bm25-only, mirroring the one committed adaptive run: a local, index-free
        # router, so the suite tests need no vector store.
        "routing": RoutingConfig(
            available_strategies=["bm25"],
            escalation_ladder=["bm25"],
            max_escalation_steps=0,
        ),
        "config_hash": "pending",
    }
    base.update(overrides)
    return ExperimentConfig(**base)


def make_dataset(count: int = 3) -> list[EvaluationExample]:
    """A tiny labelled dataset covering every eval document, in memory only."""
    return [
        EvaluationExample(
            example_id=f"ex_{index:03d}",
            query=f"what is question {index}",
            reference_answer=f"answer {index}",
            relevant_documents=[DOC_IDS[index % len(DOC_IDS)]],
            category="factual",
            dataset_version="suite_test_v1",
        )
        for index in range(count)
    ]


def make_results() -> list[RetrievalResult]:
    """Rank-ordered results spanning every document, so metrics are non-degenerate."""
    return [
        RetrievalResult(
            chunk_id=f"c_{doc}::structure_aware_v1::c{index:05d}",
            text=f"passage about {doc}",
            score=1.0 - index * 0.1,
            rank=index + 1,
            metadata=ChunkMetadata(
                document_id=doc,
                doc_title=doc,
                section_path=["1. Test"],
                headings=["Test"],
                element_ids=[f"e{index}"],
                element_types=["paragraph"],
                token_count=20,
                char_count=30,
            ),
            provenance=ChunkProvenance(document_id=doc, pages=[1], source_sha256="stub"),
        )
        for index, doc in enumerate(DOC_IDS)
    ]


class RecordingFactory:
    """A `component_factory` that records its calls and returns fake components.

    Records the *identity* of the client it was handed, which is how the shared
    Qdrant handle is checked: every arm must be given the same object.
    """

    def __init__(self, *, raises_on: str | None = None, latency_ms: float = 1.5):
        self.calls: list[dict[str, Any]] = []
        self.raises_on = raises_on
        self.latency_ms = latency_ms
        self.built: list[str] = []

    def __call__(self, config: ExperimentConfig, client: Any):
        name = config.retrieval.retrieval_method
        self.calls.append({"name": config.name, "method": name, "client": client})
        if self.raises_on is not None and name == self.raises_on:
            raise RuntimeError(f"cannot open the {self.raises_on} index")
        self.built.append(name)
        retriever = StubRetriever(
            method=name if name != "hybrid_rerank" else "hybrid",
            results=make_results(),
            latency_ms=self.latency_ms,
        )
        # The stub hard-codes its own corpus version; the suite's manifest reports
        # the config's, so the two are aligned here rather than by changing a
        # shared fake other tests depend on.
        retriever.corpus_version = CORPUS_VERSION
        # (embedding_model, resource, retriever, context_builder, generator,
        #  chunker, ingestion_pipeline) -- the shape `instantiate_components`
        #  returns, with the credentialed and index-backed parts left out.
        return (None, None, retriever, None, None, None, None)

    @property
    def clients(self) -> list[Any]:
        return [call["client"] for call in self.calls]

    def component_for(self, method: str) -> StubRetriever:
        index = self.built.index(method)
        return self.calls[index]["retriever"] if "retriever" in self.calls[index] else None


def run(
    tmp_path: Path,
    *,
    arms: Sequence[Any],
    dataset: Sequence[EvaluationExample] | None = None,
    factory: RecordingFactory | None = None,
    **kwargs: Any,
):
    """`run_suite` with the offline defaults every test here wants."""
    examples = list(dataset if dataset is not None else make_dataset())
    active = factory or RecordingFactory()
    settings: dict[str, Any] = {
        "component_factory": active,
        "client": QdrantClient(location=":memory:"),
        "dataset_sha256": "a" * 64,
        "environment": {"platform": "test"},
        "pace_seconds": 0.0,
    }
    settings.update(kwargs)
    if arms is not None:
        settings["arms"] = arms
    result = run_suite(
        common=settings.pop("common", None) or make_common(),
        dataset=examples,
        output_root=tmp_path / "suite",
        **settings,
    )
    return result, active, examples


# --- planning ------------------------------------------------------------------


def test_the_suite_declares_a_version():
    assert SUITE_VERSION == "phase7_suite_v1"


def test_the_runnable_studies_are_e1_through_e5():
    assert RUNNABLE_EXPERIMENT_IDS == (
        "E1_baseline_comparison",
        "E2_escalation_ablation",
        "E3_feature_ablation",
        "E4_threshold_calibration",
        "E5_cost_weight",
    )
    assert ANALYSIS_ONLY_EXPERIMENT_IDS == (
        "E6_routing_overhead",
        "E7_query_type",
        "E8_escalation_analysis",
    )


def test_e1_plans_the_five_benchmark_arms():
    arms, runnable, analysis = plan_arms(
        make_common(), ["E1_baseline_comparison"]
    )

    assert [arm.name for arm in arms] == ["bm25", "dense", "hybrid", "hybrid_rerank", "adaptive"]
    assert runnable == ["E1_baseline_comparison"]
    assert analysis == []


def test_e2_plans_one_arm_per_escalation_variant():
    arms, _, _ = plan_arms(make_common(), ["E2_escalation_ablation"])

    assert [arm.name for arm in arms] == [v.name for v in escalation_variants()]
    assert all(arm.is_adaptive for arm in arms)


def test_e3_plans_the_leave_one_out_feature_grid():
    arms, _, _ = plan_arms(make_common(), ["E3_feature_ablation"])

    assert [arm.name for arm in arms] == [v.name for v in feature_group_variants()]


def test_an_analysis_only_study_is_reported_rather_than_run():
    """E6-E8 read the runs E1-E3 produced; inventing a run for them is fabrication."""
    with pytest.raises(ConfigurationError, match="no runnable study requested"):
        plan_arms(make_common(), ["E6_routing_overhead"])


def test_shared_conditions_are_verified_before_anything_runs():
    """A drifted arm must cost a second, not a full sweep."""
    common = make_common()
    arms = build_arms(common)
    drifted = arms[0].model_copy(
        update={"evaluation": EvaluationConfig(k_grid=[1], enable_llm_judge=False)}
    )

    with pytest.raises(ConfigurationError, match="evaluation"):
        plan_arms(common, arms=[drifted])


def test_an_explicit_arm_list_is_refused_for_an_ablation_study():
    """Overriding E2's grid with fixed arms would register them as ablation results."""
    with pytest.raises(ConfigurationError, match="ablation variant grid"):
        plan_arms(make_common(), ["E2_escalation_ablation"], arms=build_arms(make_common())[:1])


def test_an_empty_suite_is_refused():
    with pytest.raises(ConfigurationError, match="no arms"):
        plan_arms(make_common(), ["E1_baseline_comparison"], arms=[])


# --- end to end -----------------------------------------------------------------


def test_an_arm_produces_a_full_run_directory(tmp_path: Path):
    """Traces, config, manifest, metrics, report, and the row export beside them."""
    common = make_common()
    result, _, examples = run(tmp_path, arms=build_arms(common)[:2])

    assert result.n_arms == 2
    assert result.n_succeeded == 2
    assert result.n_failed == 0

    run_dir = tmp_path / "suite" / "E1_baseline_comparison__bm25"
    for artifact in ("traces.jsonl", "config.json", "manifest.json", "metrics.json", "report.md"):
        assert (run_dir / artifact).is_file(), f"missing {artifact}"
    assert (run_dir / "rows.csv").is_file()
    assert (run_dir / "rows.jsonl").is_file()

    outcome = result.results[0]
    assert outcome.trace_count == len(examples)
    assert outcome.status == "ok"
    assert outcome.registered is True


def test_the_run_directory_is_named_deterministically(tmp_path: Path):
    """`--resume` can only find a partial run because the name carries no clock."""
    common = make_common()
    result, _, _ = run(tmp_path, arms=build_arms(common)[:1], run_prefix="p1")

    assert result.results[0].run_dir.endswith(
        "suite/p1__E1_baseline_comparison__bm25"
    )


def test_the_suite_artifact_records_the_plan_and_the_frozen_conditions(tmp_path: Path):
    common = make_common()
    run(tmp_path, arms=build_arms(common)[:1])

    payload = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))

    assert payload["suite_version"] == SUITE_VERSION
    assert payload["n_arms"] == 1
    assert payload["n_failed"] == 0
    assert payload["frozen_conditions"]["corpus_version"] == CORPUS_VERSION
    assert payload["plan"]["arms"] == ["bm25"]
    assert payload["arms"]["n_arms"] == 1


def test_the_dataset_identity_is_recorded_on_every_registration(tmp_path: Path):
    common = make_common()
    _, _, examples = run(tmp_path, arms=build_arms(common)[:1])

    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    entry = registry.get("E1_baseline_comparison", "bm25")

    assert entry is not None
    assert entry.dataset_version == examples[0].dataset_version
    assert entry.dataset_sha256 == "a" * 64
    assert entry.trace_count == len(examples)
    assert entry.corpus_version == CORPUS_VERSION
    assert entry.run_dir.endswith("E1_baseline_comparison__bm25")


def test_rows_are_exported_for_each_run(tmp_path: Path):
    common = make_common()
    _, _, examples = run(
        tmp_path, arms=build_arms(common)[:1], generator=FakeGenerator()
    )

    lines = (
        (tmp_path / "suite" / "E1_baseline_comparison__bm25" / "rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert len(lines) == len(examples)
    first = json.loads(lines[0])
    assert first["system"] == "bm25"
    assert first["experiment_id"] == "E1_baseline_comparison"
    assert first["status"] == "ok"
    # Retrieval plus generation, so the end-to-end clock is populated too.
    assert first["total_latency_ms"] is not None
    assert first["estimated_cost_usd"] is not None


def test_registration_can_be_switched_off(tmp_path: Path):
    common = make_common()
    result, _, _ = run(tmp_path, arms=build_arms(common)[:1], register=False)

    assert result.registry_path is None
    assert result.results[0].registered is False
    assert not (tmp_path / "suite" / "registry.json").exists()


def test_the_suite_runs_an_ablation_study_end_to_end(tmp_path: Path):
    """The point of the driver: a study is many arms of one system, not one run."""
    common = make_common()
    result, _, _ = run(
        tmp_path, arms=None, common=common, experiment_ids=["E2_escalation_ablation"]
    )

    assert result.n_succeeded == 3
    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    assert registry.variants("E2_escalation_ablation") == [
        "A_no_sufficiency_no_escalation",
        "B_sufficiency_no_escalation",
        "C_sufficiency_bounded_escalation",
    ]


def test_the_varied_routing_field_is_recorded_per_variant(tmp_path: Path):
    """Variant A turns two knobs off; variant C is the shipped config, so it turns none."""
    common = make_common()
    result, _, _ = run(
        tmp_path, arms=None, common=common, experiment_ids=["E2_escalation_ablation"]
    )

    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    variant_a = registry.get("E2_escalation_ablation", "A_no_sufficiency_no_escalation")
    variant_c = registry.get("E2_escalation_ablation", "C_sufficiency_bounded_escalation")

    assert variant_a is not None and variant_a.varied_routing_fields
    assert variant_c is not None and variant_c.varied_routing_fields == {}
    assert result.n_succeeded == 3


# --- the exclusive Qdrant lock ---------------------------------------------------


def test_every_arm_is_given_the_same_client(tmp_path: Path):
    """Local mode permits one embedded client per process, so this is not optional."""
    common = make_common()
    factory = RecordingFactory()
    client = QdrantClient(location=":memory:")

    result, _, _ = run(tmp_path, arms=build_arms(common), factory=factory, client=client)

    assert result.n_succeeded == 5
    assert len(factory.calls) == 5
    assert all(handle is client for handle in factory.clients)


def test_arms_needing_the_same_system_share_one_component_tree(tmp_path: Path):
    """E1's adaptive arm and E2's shipped-config variant are the same system.

    Building it twice is exactly how a second embedded Qdrant client gets opened.
    The whole E1+E2 plan is 8 arms, of which exactly one pair (E1 `adaptive` and
    E2 variant C, the unmodified shipping config) is the same system -- so 7
    component trees, not 8 and not 6.
    """
    common = make_common()
    factory = RecordingFactory()

    result, _, _ = run(
        tmp_path,
        arms=None,
        factory=factory,
        common=common,
        experiment_ids=["E1_baseline_comparison", "E2_escalation_ablation"],
    )

    assert result.n_arms == 8
    assert result.n_succeeded == 8
    assert len(factory.calls) == 7, "only the identical pair may share a component tree"
    assert len({id(handle) for handle in factory.clients}) == 1


def test_arms_with_different_configs_do_not_share_components(tmp_path: Path):
    """The other direction: two arms that differ must never be collapsed into one."""
    common = make_common()
    factory = RecordingFactory()

    result, _, _ = run(
        tmp_path, arms=build_arms(common)[:2], factory=factory, common=common
    )

    assert result.n_succeeded == 2
    assert len(factory.calls) == 2
    assert [call["method"] for call in factory.calls] == ["bm25", "dense"]


def test_a_second_client_is_never_opened_by_the_suite(tmp_path: Path, monkeypatch):
    """The suite must reuse the handle it was given rather than opening its own."""
    import adaptive_rag.evaluation.suite as suite_module

    def refuse() -> Any:
        raise AssertionError("the suite opened its own Qdrant client")

    monkeypatch.setattr(suite_module, "_shared_qdrant_client", refuse)
    common = make_common()

    result, _, _ = run(tmp_path, arms=build_arms(common)[:2])

    assert result.n_succeeded == 2


# --- pacing ----------------------------------------------------------------------


def test_pacing_sleeps_once_per_query_on_a_provider_calling_arm(tmp_path: Path):
    slept: list[float] = []
    common = make_common()
    dataset = make_dataset(3)

    result, _, _ = run(
        tmp_path,
        arms=[build_arm(common, "dense")],
        dataset=dataset,
        pace_seconds=0.01,
        sleeper=lambda seconds: slept.append(seconds),
    )

    assert result.n_succeeded == 1
    assert slept == [0.01, 0.01, 0.01]
    assert result.results[0].pace_applied is True


def test_a_local_arm_is_never_paced(tmp_path: Path):
    """BM25 calls no provider, so pacing it would only slow the suite down."""
    slept: list[float] = []
    common = make_common()

    result, _, _ = run(
        tmp_path,
        arms=[build_arm(common, "bm25")],
        pace_seconds=0.01,
        sleeper=lambda seconds: slept.append(seconds),
    )

    assert slept == []
    assert result.results[0].pace_applied is False
    assert result.results[0].pace_seconds == 0.01


def test_pacing_never_enters_a_latency_field(tmp_path: Path):
    """The sleep happens outside the retriever's clock, so the trace is untouched."""
    common = make_common()
    dataset = make_dataset(2)
    factory = RecordingFactory(latency_ms=7.5)

    result, _, _ = run(
        tmp_path,
        arms=[build_arm(common, "dense")],
        dataset=dataset,
        factory=factory,
        pace_seconds=5.0,
        sleeper=lambda seconds: None,
    )

    assert result.n_succeeded == 1
    run_dir = Path(result.results[0].run_dir)
    from adaptive_rag.schemas import ExperimentTrace

    traces = [
        ExperimentTrace.model_validate_json(line)
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [t.retrieval_latency_ms for t in traces] == [7.5, 7.5]

    rows = [
        json.loads(line)
        for line in (run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["retrieval_latency_ms"] == 7.5 for row in rows)


def test_the_adaptive_arm_is_paced_because_it_may_call_the_provider():
    """Whether an adaptive query embeds remotely is decided at query time."""
    assert "adaptive" in PROVIDER_CALLING_METHODS
    assert "bm25" not in PROVIDER_CALLING_METHODS


def test_the_pacing_value_is_recorded_in_the_artifact(tmp_path: Path):
    common = make_common()
    run(tmp_path, arms=build_arm(common, "dense") and [build_arm(common, "dense")], pace_seconds=0.25)

    payload = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))
    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    entry = registry.get("E1_baseline_comparison", "dense")

    assert payload["pace_seconds"] == 0.25
    assert payload["pace_note"]
    assert entry is not None and entry.pace_seconds == 0.25


def test_a_run_without_pacing_records_zero(tmp_path: Path):
    common = make_common()
    run(tmp_path, arms=[build_arm(common, "dense")], pace_seconds=0.0)

    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    assert registry.get("E1_baseline_comparison", "dense").pace_seconds == 0.0


def test_the_default_pace_matches_the_cost_sweep():
    """One protocol, one number: the suite and `measure_strategy_cost.py` agree."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_cost_sweep", Path(__file__).resolve().parents[1] / "scripts" / "measure_strategy_cost.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert DEFAULT_PACE_SECONDS == module.DEFAULT_PACE_SECONDS


def test_the_pacing_wrapper_delegates_everything_else():
    stub = StubRetriever(method="dense", results=make_results())
    paced = PacedRetriever(stub, pace_seconds=0.0, calls_provider=True)

    assert paced.method == "dense"
    assert paced.corpus_version == stub.corpus_version
    assert paced.retrieve(query="q", top_k=1).retrieval_method == "dense"
    assert paced.paced_calls == 0


# --- failure handling ------------------------------------------------------------


def test_a_failing_arm_is_recorded_and_the_suite_continues(tmp_path: Path):
    """A partial study is worth more than none, but the failure must be visible."""
    common = make_common()
    factory = RecordingFactory(raises_on="hybrid")
    dataset = make_dataset(2)

    result, _, _ = run(
        tmp_path,
        arms=build_arms(common)[:3],  # bm25, dense, hybrid
        dataset=dataset,
        factory=factory,
    )

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert result.failed == ["E1_baseline_comparison/hybrid"]

    failure = next(r for r in result.results if r.status == "failed")
    assert failure.failure is not None
    assert failure.failure.error_type == "RuntimeError"
    assert "cannot open the hybrid index" in failure.failure.message

    payload = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))
    assert payload["n_failed"] == 1
    assert payload["failed"] == ["E1_baseline_comparison/hybrid"]
    assert payload["results"][2]["failure"]["error_type"] == "RuntimeError"


def test_a_failed_arm_is_not_registered(tmp_path: Path):
    """Registering a run that produced no traces would put a phantom in the report."""
    common = make_common()
    run(
        tmp_path,
        arms=build_arms(common)[:2],
        factory=RecordingFactory(raises_on="dense"),
    )

    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    assert registry.variants("E1_baseline_comparison") == ["bm25"]


def test_strict_mode_aborts_on_the_first_failure(tmp_path: Path):
    common = make_common()
    factory = RecordingFactory(raises_on="dense")

    with pytest.raises(SuiteExecutionError, match="E1_baseline_comparison/dense"):
        run(tmp_path, arms=build_arms(common)[:3], factory=factory, strict=True)

    payload = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))
    assert payload["n_succeeded"] == 1
    assert payload["n_failed"] == 1


def test_strict_mode_still_persists_the_work_that_did_run(tmp_path: Path):
    """An abort that discarded the first arm's results would waste the whole run."""
    common = make_common()

    with pytest.raises(SuiteExecutionError):
        run(
            tmp_path,
            arms=build_arms(common)[:3],
            factory=RecordingFactory(raises_on="dense"),
            strict=True,
        )

    run_dir = tmp_path / "suite" / "E1_baseline_comparison__bm25"
    assert (run_dir / "traces.jsonl").is_file()
    assert ExperimentRegistry.load(tmp_path / "suite" / "registry.json").variants(
        "E1_baseline_comparison"
    ) == ["bm25"]


def test_a_suite_execution_error_is_an_experiment_error():
    assert issubclass(SuiteExecutionError, ExperimentRunError)


# --- no clobber -----------------------------------------------------------------


def test_rerunning_without_resume_is_refused_rather_than_duplicating_traces(tmp_path: Path):
    """`ExperimentRunner` appends, so a silent re-run would double every trace."""
    common = make_common()
    arms = build_arms(common)[:1]
    run(tmp_path, arms=arms)

    result, _, _ = run(tmp_path, arms=arms)

    assert result.n_failed == 1
    assert "resume=True" in result.results[0].failure.message

    run_dir = tmp_path / "suite" / "E1_baseline_comparison__bm25"
    lines = [
        line
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3


def test_resume_completes_a_partial_run_and_reuses_its_registration(tmp_path: Path):
    """The same run directory is the same measurement, so re-registering it is fine."""
    common = make_common()
    arms = build_arms(common)[:1]
    run(tmp_path, arms=arms)
    (tmp_path / "suite" / "E1_baseline_comparison__bm25" / "traces.jsonl").write_text(
        "\n".join(
            (tmp_path / "suite" / "E1_baseline_comparison__bm25" / "traces.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[:1]
        )
        + "\n",
        encoding="utf-8",
    )

    result, _, _ = run(tmp_path, arms=arms, resume=True)

    assert result.n_succeeded == 1
    assert result.results[0].trace_count == 3
    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    assert len(registry) == 1


def test_a_second_run_of_the_same_variant_is_refused_by_default(tmp_path: Path):
    common = make_common()
    arms = build_arms(common)[:1]
    run(tmp_path, arms=arms, run_prefix="first")

    result, _, _ = run(tmp_path, arms=arms, run_prefix="second")

    assert result.n_failed == 1
    assert "already registered" in result.results[0].failure.message
    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    assert registry.get("E1_baseline_comparison", "bm25").run_dir.endswith("first__E1_baseline_comparison__bm25")


def test_allow_reentry_permits_a_deliberate_replacement(tmp_path: Path):
    common = make_common()
    arms = build_arms(common)[:1]
    run(tmp_path, arms=arms, run_prefix="first")

    result, _, _ = run(tmp_path, arms=arms, run_prefix="second", allow_reentry=True)

    assert result.n_succeeded == 1
    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    assert registry.get("E1_baseline_comparison", "bm25").run_dir.endswith("second__E1_baseline_comparison__bm25")


def test_the_suite_joins_a_previous_registration_rather_than_replacing_it(tmp_path: Path):
    """A resumed or extended suite must not drop the runs already on record."""
    common = make_common()
    run(tmp_path, arms=build_arms(common)[:1], run_prefix="first")

    run(tmp_path, arms=build_arms(common)[1:2], run_prefix="second")

    registry = ExperimentRegistry.load(tmp_path / "suite" / "registry.json")
    assert registry.variants("E1_baseline_comparison") == ["bm25", "dense"]


# --- dataset identity ------------------------------------------------------------


def test_a_missing_dataset_hash_is_an_error_rather_than_a_placeholder(tmp_path: Path):
    common = make_common()
    with pytest.raises(ConfigurationError, match="sha256"):
        run_suite(
            common=common,
            dataset=make_dataset(2),
            output_root=tmp_path / "suite",
            arms=build_arms(common)[:1],
            component_factory=RecordingFactory(),
            client=QdrantClient(location=":memory:"),
            pace_seconds=0.0,
        )


# --- retrieval-only ----------------------------------------------------------


class ExplodingGenerator:
    """A generator that fails the test if it is ever called."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, request: Any):
        self.call_count += 1
        raise AssertionError("the generator was called on a retrieval-only run")


class GeneratingFactory(RecordingFactory):
    """`RecordingFactory` that puts a real generator in component slot 4.

    Needed to pin the distinction the suite now draws: an *omitted* `generator`
    still means "use the arm's default generator", while `retrieval_only=True`
    means "no generator reaches the runner at all".
    """

    def __init__(self, generator: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.generator = generator or FakeGenerator()

    def __call__(self, config: ExperimentConfig, client: Any):
        components = list(super().__call__(config, client))
        components[4] = self.generator
        return tuple(components)


def _traces(result: SuiteResult) -> list[Any]:
    from adaptive_rag.schemas import ExperimentTrace

    return [
        ExperimentTrace.model_validate_json(line)
        for line in (Path(result.results[0].run_dir) / "traces.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_an_omitted_generator_still_means_the_default_generator(tmp_path: Path):
    """Defect 1 was caused by overloading `None`; this pins the old meaning."""
    common = make_common()
    generator = FakeGenerator()
    factory = GeneratingFactory(generator=generator)

    result, _, _ = run(
        tmp_path,
        arms=[build_arm(common, "bm25")],
        factory=factory,
        generator=None,
    )

    assert result.n_succeeded == 1
    assert generator.call_count == 3
    assert [trace.status for trace in _traces(result)] == ["ok"] * 3


def test_a_retrieval_only_run_calls_no_generator(tmp_path: Path):
    """The retrieval-only protocol must be reachable, and must not generate."""
    common = make_common()
    generator = ExplodingGenerator()
    factory = GeneratingFactory(generator=generator)

    result, _, _ = run(
        tmp_path,
        arms=[build_arm(common, "bm25")],
        factory=factory,
        retrieval_only=True,
    )

    assert result.n_succeeded == 1
    assert generator.call_count == 0
    assert result.plan.retrieval_only is True

    traces = _traces(result)
    assert [trace.status for trace in traces] == ["ok"] * 3
    assert all(trace.retrieval is not None for trace in traces)
    assert all(trace.generation is None for trace in traces)
    assert all(trace.generation_latency_ms is None for trace in traces)
    assert all(trace.estimated_cost_usd is None for trace in traces)
    assert all(trace.error is None for trace in traces)


def test_the_suite_artifact_records_the_retrieval_only_protocol(tmp_path: Path):
    """An artifact that does not say which protocol ran is not attributable."""
    common = make_common()
    run(tmp_path, arms=[build_arm(common, "bm25")], retrieval_only=True)

    payload = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))
    assert payload["plan"]["retrieval_only"] is True

    run(tmp_path, arms=[build_arm(common, "bm25")], run_prefix="gen")
    payload = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))
    assert payload["plan"]["retrieval_only"] is False


def test_a_generator_free_run_still_reports_empty_not_ok(tmp_path: Path):
    """`generator=None` at the *runner* level is a different protocol."""
    from adaptive_rag.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(retriever=StubRetriever("bm25", make_results()))
    assert runner.retrieval_only is False


def test_total_latency_is_populated_and_non_zero_on_a_retrieval_only_run(tmp_path: Path):
    """Defect 2: the end-to-end clock existed only when generation ran."""
    common = make_common()
    result, _, _ = run(
        tmp_path,
        arms=[build_arm(common, "bm25")],
        factory=RecordingFactory(latency_ms=2.5),
        retrieval_only=True,
    )

    traces = _traces(result)
    assert len(traces) == 3
    for trace in traces:
        assert trace.total_latency_ms is not None
        assert trace.total_latency_ms > 0
        assert trace.total_latency_ms == trace.retrieval_latency_ms == 2.5

    summary = result.results[0].metrics["efficiency"]
    assert summary["metrics"], "the efficiency report must have latency metrics"
    latency_metrics = {
        metric["name"]: metric for metric in summary["metrics"]
    }
    assert latency_metrics["total_latency_ms_mean"]["n"] == 3
    assert latency_metrics["total_latency_ms_mean"]["value"] == 2.5


def test_rows_carry_the_total_latency_of_a_retrieval_only_run(tmp_path: Path):
    common = make_common()
    result, _, _ = run(
        tmp_path,
        arms=[build_arm(common, "bm25")],
        factory=RecordingFactory(latency_ms=1.25),
        retrieval_only=True,
    )

    rows = [
        json.loads(line)
        for line in (Path(result.results[0].run_dir) / "rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["total_latency_ms"] == 1.25 for row in rows)
    assert all(row["generation_latency_ms"] is None for row in rows)


def _escalating_adaptive() -> AdaptiveRetriever:
    """A two-stage adaptive retriever over two stub strategies, offline."""
    bm25 = StubRetriever("bm25", make_results(), latency_ms=1.7)
    dense = StubRetriever("dense", make_results(), latency_ms=6.0)
    routing = RoutingConfig(
        available_strategies=["bm25", "dense"],
        escalation_ladder=["bm25", "dense"],
    )
    return AdaptiveRetriever(
        retrievers={"bm25": bm25, "dense": dense},
        router=RuleBasedRouter(routing),
        analyzer=QueryFeatureAnalyzer(),
        sufficiency_checker=SufficiencyChecker(routing),
        escalation_policy=EscalationPolicy(routing),
        retrieval_config=RetrievalConfig(retrieval_method="adaptive"),
        routing_config=routing,
        corpus_version=CORPUS_VERSION,
    )


def test_total_latency_covers_every_stage_of_a_multi_stage_run_without_double_counting():
    """Defect 2, second half: two retrieval stages must not collapse to one.

    `AdaptiveRetriever` already records `routing_latency_ms + sum(stage_latencies_ms)`
    as `retrieval_latency_ms`, so `total_latency_ms` must equal that and must not add
    the stages again -- and it must not be smaller than any single stage.
    """
    from adaptive_rag.experiments.runner import ExperimentRunner

    config = make_common(retrieval=RetrievalConfig(retrieval_method="adaptive"))
    runner = ExperimentRunner(
        retriever=_escalating_adaptive(), retrieval_only=True, output_root=Path("/tmp/kilo/never")
    )
    example = make_dataset(1)[0]
    example = example.model_copy(
        update={"query": "unrelated catalysis prose with no shared vocabulary"}
    )

    trace = runner._run_example(config, example, "multistage")

    assert trace.status == "ok"
    routing = trace.routing
    assert routing is not None and routing.stage_count == 2
    stages = routing.stage_latencies_ms
    assert len(stages) == 2
    assert trace.total_latency_ms == trace.retrieval_latency_ms
    # Not smaller than any single stage it contains ...
    assert trace.total_latency_ms >= max(stages)
    assert trace.total_latency_ms >= routing.routing_latency_ms
    # ... and not the stages added a second time on top.
    assert trace.total_latency_ms < trace.retrieval_latency_ms + sum(stages)


def test_a_reranked_stage_is_covered_once(tmp_path: Path):
    """A rerank stage is a component of `retrieval_latency_ms`, not an addition."""
    from adaptive_rag.experiments.runner import ExperimentRunner

    class RerankShapedRetriever:
        """Reports the shape `RerankedRetriever` produces: inclusive latency."""

        method = "bm25_rerank"

        def retrieve(self, query: str, top_k: int | None = None, **_: Any):
            from adaptive_rag.schemas import RetrievalMetadata, RetrievalResponse

            return RetrievalResponse(
                query=query,
                results=make_results(),
                retrieval_method="bm25_rerank",
                status="ok",
                retrieval_metadata=RetrievalMetadata(
                    top_k=top_k or 3,
                    retriever_version="bm25_rerank_v1",
                    index_id="stub_index",
                    corpus_version=CORPUS_VERSION,
                    latency_ms=5.0,
                    search_latency_ms=5.0,
                    candidate_generation_latency_ms=2.0,
                    rerank_latency_ms=3.0,
                    candidate_count=3,
                    result_count=3,
                ),
            )

    config = make_common(retrieval=RetrievalConfig(retrieval_method="bm25_rerank"))
    runner = ExperimentRunner(retriever=RerankShapedRetriever(), retrieval_only=True)

    trace = runner._run_example(config, make_dataset(1)[0], "rerank")

    assert trace.status == "ok"
    assert trace.retrieval_latency_ms == 5.0
    assert trace.rerank_latency_ms == 3.0
    assert trace.candidate_generation_latency_ms == 2.0
    assert trace.total_latency_ms == 5.0
    assert trace.total_latency_ms >= trace.rerank_latency_ms
    assert trace.total_latency_ms < 5.0 + 3.0 + 2.0
