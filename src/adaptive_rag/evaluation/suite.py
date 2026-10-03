"""
evaluation.suite
----------------
The Phase 7 suite driver: build every arm, verify the shared conditions, execute
each arm, and register the result.

Everything a Phase 7 study needs that is not a single run lives here. The runner
(`experiments.runner.ExperimentRunner`) runs one configuration over one dataset
and writes its artifacts; this module is what turns "run E3" into eleven runs of
the same system with one routing knob turned each, and what makes the resulting
set of runs a report rather than a pile of directories.

Three decisions are load-bearing and are defended in place below:

* **One Qdrant client for the whole suite.** Local mode takes an exclusive lock
  on the storage folder, so a second embedded client fails outright. A suite
  opens every arm inside one `try` and passes the same handle to
  `instantiate_components` each time, which is also what the 7.0b cost sweep
  does. `experiments.config.instantiate_components` honours an injected client
  in its adaptive branch (that was the `AlreadyLocked` defect 7.0b fixed), so
  the suite never has to open one itself per arm.
* **Pacing between timed retrievals, never inside one.** Three of the four
  costed strategies embed every query remotely, and the 7.0b sweep died on HTTP
  429 until queries were paced. `PacedRetriever` sleeps *before* delegating, so
  the sleep is outside the retriever's own clock and cannot reach any latency
  field.
* **Retrieval-only is an explicit switch, not an absent generator.** Phase 7
  measures retrieval. `run_suite(retrieval_only=True)` therefore hands the runner
  no generator at all, so no generation call can happen; `generator=None` still
  means "use this arm's default generator" exactly as before, and both spellings
  keep their old meaning on every existing call path. `experiments.config`
  builds a `GroqGenerator` into component slot 4 eagerly, but that object is lazy
  (no client, no credentials) and is dropped here rather than merely ignored, so
  a retrieval-only run cannot fail because a provider was unreachable.
* **A failing arm is recorded, not dropped.** An arm that cannot be built or run
  produces a failure record in `suite.json` and in the returned result; the
  suite continues so the rest of the study still happens, unless `strict=True`,
  which aborts on the first failure after persisting what did run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.config.hashing import compute_file_sha256
from adaptive_rag.config.paths import REPO_ROOT
from adaptive_rag.errors import ConfigurationError, ExperimentRunError
from adaptive_rag.evaluation.ablation import (
    Variant,
    cost_weight_sweep,
    escalation_step_sweep,
    escalation_variants,
    feature_group_variants,
    threshold_sweep,
)
from adaptive_rag.evaluation.efficiency import EfficiencyEvaluator
from adaptive_rag.evaluation.retrieval import RetrievalEvaluator
from adaptive_rag.evaluation.routing import RoutingEvaluator
from adaptive_rag.evaluation.rows import build_rows, write_rows
from adaptive_rag.experiments.arms import (
    Arm,
    arm_summary,
    build_ablation_arms,
    build_arms,
    frozen_conditions,
    verify_arm_configs,
    verify_shared_conditions,
)
from adaptive_rag.experiments.config import _shared_qdrant_client, instantiate_components
from adaptive_rag.experiments.runner import ExperimentRunner
from adaptive_rag.experiments.registry import (
    DEFAULT_REGISTRY_NAME,
    EXPERIMENT_IDS,
    ExperimentRegistry,
    RegistryEntry,
    normalize_experiment_id,
)
from adaptive_rag.schemas import EvaluationExample, ExperimentConfig, ExperimentTrace

SUITE_VERSION = "phase7_suite_v1"

# Run output root. `experiments/phase7/` already holds the 7.0a gate artifact and
# the 7.0b cost table, so the suite's runs join them rather than starting a
# second output tree.
DEFAULT_SUITE_ROOT = REPO_ROOT / "experiments" / "phase7"

# Studies the suite can execute. E6, E7 and E8 are analyses over the runs E1-E3
# already produced -- routing overhead, per-query-type breakdown, and the
# escalation story -- so they have no arm of their own. Requesting them is
# answered with that explanation rather than a run, because inventing a run for an
# analysis study is exactly the fabrication this phase must not commit.
RUNNABLE_EXPERIMENT_IDS: tuple[str, ...] = (
    "E1_baseline_comparison",
    "E2_escalation_ablation",
    "E3_feature_ablation",
    "E4_threshold_calibration",
    "E5_cost_weight",
)
ANALYSIS_ONLY_EXPERIMENT_IDS: tuple[str, ...] = (
    "E6_routing_overhead",
    "E7_query_type",
    "E8_escalation_analysis",
)

# Seconds between queries on arms that call the embedding provider. Same default
# as `scripts/measure_strategy_cost.py`, for the same reason.
DEFAULT_PACE_SECONDS = 0.75

# Retrieval methods that issue a live query-embedding call per query. BM25 is
# local and needs no pacing. `adaptive` is included because whether it calls the
# provider is decided at query time, exactly as in the 7.0b sweep -- pacing it
# unconditionally is the conservative choice, since the alternative is a paced
# run that looks unpaced in the artifact and a 429 that does not.
PROVIDER_CALLING_METHODS: frozenset[str] = frozenset(
    {"dense", "hybrid", "hybrid_rerank", "adaptive"}
)


class SuiteExecutionError(ExperimentRunError):
    """Raised in `strict` mode when an arm fails, carrying its failure record."""


class PacedRetriever:
    """Wraps a retriever and sleeps before delegating on provider-calling arms.

    The sleep is deliberately *before* the call rather than after: the retriever's
    own `latency_ms` clock starts inside `retrieve`, so a sleep taken here is
    outside every timed region and cannot enter a latency field. Pacing is
    observable in exactly one place -- the artifact's `pace_seconds` -- and in
    none of the numbers.

    Attribute access falls through to the wrapped retriever, so the suite's
    cleanup path can still reach a vector store and close it.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        pace_seconds: float = 0.0,
        calls_provider: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self._retriever = retriever
        self.pace_seconds = float(pace_seconds)
        self.calls_provider = calls_provider
        self._sleeper = sleeper
        self.paced_calls = 0

    @property
    def wrapped(self) -> Any:
        return self._retriever

    @property
    def method(self) -> str:
        return getattr(self._retriever, "method", "unknown")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ):
        if self.pace_seconds > 0 and self.calls_provider:
            self._sleeper(self.pace_seconds)
            self.paced_calls += 1
        return self._retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._retriever, name)


class ArmFailure(BaseModel):
    """Why an arm did not produce a run, in the shape `ErrorInfo` uses."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    error_type: str
    message: str


class ArmRunResult(BaseModel):
    """The outcome of one arm, successful or not."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    variant: str
    kind: str
    status: str  # "ok" | "failed"
    run_dir: str | None = None
    trace_count: int = 0
    rows_written: dict[str, str] = Field(default_factory=dict)
    pace_seconds: float = 0.0
    pace_applied: bool = False
    failure: ArmFailure | None = None
    registered: bool = False
    metrics: dict[str, Any] | None = None


class SuitePlan(BaseModel):
    """What the suite decided to run, before anything runs."""

    model_config = ConfigDict(extra="forbid")

    suite_version: str
    experiment_ids: list[str]
    arms: list[str]
    analysis_only: list[str]
    output_root: str
    run_prefix: str | None = None
    pace_seconds: float
    strict: bool
    resume: bool
    # Which protocol produced these runs. Recorded in `suite.json` because a
    # retrieval-only number and a retrieve-then-generate number are not the same
    # measurement, and an artifact that does not say which is not attributable.
    retrieval_only: bool
    # Named `registers_runs` rather than `register`, which pydantic treats as a
    # shadowed BaseModel attribute.
    registers_runs: bool
    notes: list[str] = Field(default_factory=list)


class SuiteResult(BaseModel):
    """The outcome of a whole suite run, and the artifact that describes it."""

    model_config = ConfigDict(extra="forbid")

    suite_version: str
    plan: SuitePlan
    output_root: str
    timestamp: str
    n_arms: int
    n_succeeded: int
    n_failed: int
    results: list[ArmRunResult]
    failed: list[str] = Field(default_factory=list)
    registry_path: str | None = None
    registry_summary: dict[str, Any] | None = None
    frozen_conditions: dict[str, Any] = Field(default_factory=dict)
    arm_summary: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None

    @property
    def succeeded(self) -> list[ArmRunResult]:
        return [r for r in self.results if r.status == "ok"]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _variants_for(experiment_id: str, common: ExperimentConfig) -> list[Variant]:
    """The ablation variants a study runs, as configuration and nothing else.

    Every branch is a call into `evaluation.ablation`, so a study's grid is
    defined in one place and the suite cannot drift from it. The base config is
    `common.routing`: an ablation turns a knob on the system's own shipping
    configuration, not on a fresh default.
    """
    base = common.routing
    if experiment_id == "E2_escalation_ablation":
        return escalation_variants(base)
    if experiment_id == "E3_feature_ablation":
        return feature_group_variants(base)
    if experiment_id == "E4_threshold_calibration":
        return threshold_sweep(base) + escalation_step_sweep(base)
    if experiment_id == "E5_cost_weight":
        return cost_weight_sweep(base)
    raise ConfigurationError(
        f"{experiment_id} has no variant grid; runnable studies are "
        f"{list(RUNNABLE_EXPERIMENT_IDS)}"
    )


def _kind_for(experiment_id: str, arm: Arm) -> str:
    """A run's purpose, as distinct from its study."""
    if experiment_id != "E1_baseline_comparison":
        return "ablation"
    return "adaptive" if arm.is_adaptive else "baseline"


def plan_arms(
    common: ExperimentConfig,
    experiment_ids: Sequence[str] = ("E1_baseline_comparison",),
    *,
    arms: Sequence[Arm] | None = None,
) -> tuple[list[Arm], list[str], list[str]]:
    """Resolve requested studies into the concrete arms to run.

    Returns `(arms, resolved_ids, analysis_only)`. An explicit `arms` list
    overrides the study mapping entirely -- that is the escape hatch for a
    caller that has already built exactly the arms it wants, and it is also how
    the offline tests drive the driver without a corpus.

    The shared conditions are verified *here*, before anything runs and before
    any component is opened, so a drift between arms costs a second rather than
    a full sweep.
    """
    resolved = [normalize_experiment_id(eid) for eid in experiment_ids]
    analysis_only = [eid for eid in resolved if eid in ANALYSIS_ONLY_EXPERIMENT_IDS]
    runnable = [eid for eid in resolved if eid in RUNNABLE_EXPERIMENT_IDS]

    if arms is not None:
        foreign = [eid for eid in runnable if eid != "E1_baseline_comparison"]
        if foreign:
            raise ConfigurationError(
                f"an explicit arm list is only meaningful for "
                f"E1_baseline_comparison; {foreign} derive their arm set from the "
                "ablation variant grid, and overriding it would register a fixed "
                "arm as an ablation result"
            )
        selected = list(arms)
    else:
        if not runnable:
            raise ConfigurationError(
                f"no runnable study requested: {resolved}. "
                f"{list(ANALYSIS_ONLY_EXPERIMENT_IDS)} are analyses over the runs "
                "E1-E3 produce, so they produce no run of their own."
            )
        selected = []
        for experiment_id in runnable:
            if experiment_id == "E1_baseline_comparison":
                selected.extend(
                    build_arms(common, experiment_id=experiment_id)
                )
            else:
                selected.extend(
                    build_ablation_arms(common, _variants_for(experiment_id, common))
                )

    if not selected:
        raise ConfigurationError("the suite has no arms to run")
    # Both guards, before the first component is built. `verify_arm_configs` also
    # catches the ablation grid's own invariants (unique names, reranking only
    # where it belongs), which `build_ablation_arms` runs internally as well.
    verify_shared_conditions(common, selected)
    verify_arm_configs(selected)
    return selected, runnable, analysis_only


def _component_key(config: ExperimentConfig) -> str:
    """The identity of the component tree a config needs.

    Two arms whose configs differ only in `name`, `experiment_id` or
    `config_hash` need the same retriever, and building it twice would mean
    opening a second Qdrant client -- the `AlreadyLocked` failure. Anything that
    can change behaviour is included in the key, so sharing is only ever allowed
    when the two arms genuinely run the same system. That is the E1 adaptive arm
    and the E2 "C" (shipped config) variant, which is exactly the case that
    should not pay to open an index twice.
    """
    payload = config.model_dump(mode="json", exclude={"name", "experiment_id", "config_hash"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _calls_provider(config: ExperimentConfig) -> bool:
    return config.retrieval.retrieval_method in PROVIDER_CALLING_METHODS


def _load_traces(run_dir: Path) -> list[ExperimentTrace]:
    """Read back a run's persisted traces.

    The runner owns writing them; reading them back rather than returning them
    is deliberate, because the row export must describe exactly what was
    persisted, including any traces a previous `--resume` pass had already
    written.
    """
    traces_path = run_dir / "traces.jsonl"
    if not traces_path.is_file():
        raise ExperimentRunError(f"no traces.jsonl in {run_dir}")
    traces: list[ExperimentTrace] = []
    with open(traces_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(ExperimentTrace.model_validate_json(line))
    return traces


def _close_quietly(handle: Any) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001, S110 -- cleanup is best-effort
            pass


def _dataset_identity(
    dataset: Sequence[EvaluationExample],
    dataset_path: Path | None,
    dataset_version: str | None,
    dataset_sha256: str | None,
) -> tuple[str, str]:
    """Resolve the dataset's version and content hash for the registry.

    The version comes from the records themselves (`EvaluationExample.dataset_version`),
    which is the authoritative answer and needs no path. The hash cannot be
    derived from loaded records -- it is a property of the file -- so it must be
    supplied or computed from `dataset_path`. When neither is available this
    raises rather than registering a placeholder: an unattributable measurement
    is not a measurement anyone can defend.
    """
    version = dataset_version
    if version is None:
        if not dataset:
            raise ConfigurationError("cannot resolve dataset_version from an empty dataset")
        version = dataset[0].dataset_version
    digest = dataset_sha256
    if digest is None:
        if dataset_path is None:
            raise ConfigurationError(
                "a dataset sha256 is required to register runs: pass dataset_sha256 "
                "or dataset_path"
            )
        if not dataset_path.is_file():
            raise ConfigurationError(f"dataset not found: {dataset_path}")
        digest = compute_file_sha256(dataset_path)
    return version, digest


def run_suite(
    *,
    common: ExperimentConfig,
    dataset: Sequence[EvaluationExample],
    output_root: Path | str = DEFAULT_SUITE_ROOT,
    experiment_ids: Sequence[str] = ("E1_baseline_comparison",),
    arms: Sequence[Arm] | None = None,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
    strict: bool = False,
    resume: bool = False,
    retrieval_only: bool = False,
    register: bool = True,
    allow_reentry: bool = False,
    dataset_path: Path | None = None,
    dataset_version: str | None = None,
    dataset_sha256: str | None = None,
    environment: Mapping[str, Any] | None = None,
    run_prefix: str | None = None,
    timestamp: str | None = None,
    client: Any = None,
    own_client: bool = True,
    component_factory: Callable[[ExperimentConfig, Any], Any] | None = None,
    generator: Any = None,
    context_builder: Any = None,
    evaluators: Sequence[Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> SuiteResult:
    """Build, verify, execute and register every requested arm.

    `component_factory` defaults to `experiments.config.instantiate_components`
    and is injectable so the driver can be exercised offline against fakes. It
    must be called as `factory(config, client)`; the same `client` is passed to
    every arm, which is the whole point.

    `own_client=True` with no `client` opens one embedded Qdrant client for the
    suite and closes it at the end. Pass a `client` to reuse an existing handle
    (the offline tests pass an in-memory one), and the suite will not close it.

    `sleeper` is the same seam for pacing: it defaults to `time.sleep` and exists
    so a test can observe the pacing without waiting for it.

    `retrieval_only=True` runs the Phase 7 protocol: no generator is handed to the
    runner, so no generation call happens, and each trace is `status="ok"` with its
    retrieval populated, `generation=None`, no generation latency and no cost.
    It is a distinct flag from `generator=None`, which keeps its existing meaning
    of "fall back to this arm's default generator" -- overloading `None` would have
    silently changed what an existing call path measures.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or _timestamp()
    notes: list[str] = []

    selected, runnable, analysis_only = plan_arms(common, experiment_ids, arms=arms)
    if analysis_only:
        notes.append(
            f"{', '.join(analysis_only)} are analysis studies over the runs "
            f"{', '.join(RUNNABLE_EXPERIMENT_IDS)} produce; they have no arm of their "
            "own and no run was made for them."
        )
    plan = SuitePlan(
        suite_version=SUITE_VERSION,
        experiment_ids=list(runnable),
        arms=[arm.name for arm in selected],
        analysis_only=list(analysis_only),
        output_root=str(root),
        run_prefix=run_prefix,
        pace_seconds=float(pace_seconds),
        strict=strict,
        resume=resume,
        retrieval_only=bool(retrieval_only),
        registers_runs=register,
        notes=notes,
    )

    active_evaluators = list(evaluators) if evaluators is not None else [
        RetrievalEvaluator(),
        RoutingEvaluator(),
        EfficiencyEvaluator(),
    ]

    version, digest = _dataset_identity(dataset, dataset_path, dataset_version, dataset_sha256)
    registry_path = root / DEFAULT_REGISTRY_NAME
    registry = ExperimentRegistry.load_or_empty(registry_path) if register else ExperimentRegistry()

    # One client for every arm. Local mode takes an exclusive lock on the storage
    # folder, so this handle is the difference between a suite and a crash on the
    # second arm. `instantiate_components` honours it in the adaptive branch too.
    owns_client = client is None and own_client
    active_client = _shared_qdrant_client() if owns_client else client
    factory = component_factory or (lambda config, handle: instantiate_components(config, client=handle))

    # Component trees are cached by config identity so two arms that need the same
    # system (E1's adaptive arm and E2's shipped-config variant) share one
    # retriever, one index handle and one embedding model.
    component_cache: dict[str, Any] = {}
    results: list[ArmRunResult] = []
    try:
        for arm in selected:
            experiment_id = normalize_experiment_id(arm.experiment_id)
            kind = _kind_for(experiment_id, arm)
            run_id = f"{run_prefix}__{experiment_id}__{arm.name}" if run_prefix else f"{experiment_id}__{arm.name}"
            run_dir = root / run_id
            result = ArmRunResult(
                experiment_id=experiment_id,
                variant=arm.name,
                kind=kind,
                status="failed",
                run_dir=str(run_dir),
                pace_seconds=float(pace_seconds),
            )
            try:
                if not resume and (run_dir / "traces.jsonl").is_file():
                    # `ExperimentRunner` *appends* to traces.jsonl, so re-running an
                    # existing run directory without `resume` would silently
                    # duplicate every trace. Refusing here is the only safe answer.
                    raise ConfigurationError(
                        f"run directory {run_dir} already holds traces; pass resume=True "
                        f"to complete it, or choose a different run_prefix"
                    )

                config = arm.build_config(common)
                key = _component_key(config)
                components = component_cache.get(key)
                if components is None:
                    components = factory(config, active_client)
                    component_cache[key] = components

                retriever = components[2]
                provider_calling = _calls_provider(config)
                if float(pace_seconds) > 0 and provider_calling:
                    retriever = PacedRetriever(
                        retriever,
                        pace_seconds=float(pace_seconds),
                        calls_provider=True,
                        sleeper=sleeper,
                    )
                    result.pace_applied = True

                # `generator=None` keeps its meaning -- "use the arm's default
                # generator" -- so retrieval-only is asked for explicitly. The
                # eagerly-built GroqGenerator in slot 4 is lazy, so dropping it here
                # costs nothing and leaves no generation call reachable.
                active_generator = (
                    None
                    if retrieval_only
                    else (generator if generator is not None else components[4])
                )
                runner = ExperimentRunner(
                    retriever=retriever,
                    generator=active_generator,
                    context_builder=(
                        context_builder if context_builder is not None else components[3]
                    ),
                    evaluators=active_evaluators,
                    output_root=root,
                    retrieval_only=bool(retrieval_only),
                )
                summary = runner.run(config, dataset, resume=resume, run_id=run_id)
                result.trace_count = int(summary["trace_count"])
                result.metrics = summary["metrics"]

                rows = build_rows(
                    _load_traces(run_dir),
                    dataset,
                    system=arm.name,
                    experiment_id=experiment_id,
                )
                written = write_rows(rows, run_dir)
                result.rows_written = {key_: str(value) for key_, value in written.items()}

                if register:
                    entry = RegistryEntry.from_run_dir(
                        run_dir,
                        experiment_id=experiment_id,
                        variant=arm.name,
                        kind=kind,  # type: ignore[arg-type]
                        dataset_version=version,
                        dataset_sha256=digest,
                        base_routing=common.routing,
                        environment=environment,
                        pace_seconds=float(pace_seconds) if result.pace_applied else 0.0,
                        timestamp=_iso(stamp),
                    )
                    existing = registry.get(experiment_id, arm.name)
                    # Re-registering the *same* run directory is completing the
                    # same measurement, so it is allowed; a different run directory
                    # under the same (study, variant) is a genuine conflict and is
                    # refused unless the caller passed `allow_reentry` or
                    # `resume`.
                    same_run = existing is not None and existing.run_dir == str(run_dir)
                    registry.register(
                        entry, allow_reentry=allow_reentry or resume or same_run
                    )
                    result.registered = True

                result.status = "ok"
            except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised in strict mode
                result.status = "failed"
                result.failure = ArmFailure(
                    stage="arm",
                    error_type=type(exc).__name__,
                    message=str(exc)[:1000],
                )

            results.append(result)
            if register:
                registry.save(registry_path)
            if result.status == "failed" and strict:
                artifact = _write_suite_artifact(
                    root,
                    plan=plan,
                    results=results,
                    timestamp=stamp,
                    registry_path=registry_path if register else None,
                    registry=registry if register else None,
                    common=common,
                    arms=selected,
                )
                raise SuiteExecutionError(
                    f"arm {result.experiment_id}/{result.variant} failed: "
                    f"{result.failure.error_type}: {result.failure.message}"
                    + (f" (partial results in {artifact})" if artifact else "")
                )

        artifact = _write_suite_artifact(
            root,
            plan=plan,
            results=results,
            timestamp=stamp,
            registry_path=registry_path if register else None,
            registry=registry if register else None,
            common=common,
            arms=selected,
        )
    finally:
        for components in component_cache.values():
            _close_quietly(components[1])
        if owns_client and active_client is not None:
            _close_quietly(active_client)

    return SuiteResult(
        suite_version=SUITE_VERSION,
        plan=plan,
        output_root=str(root),
        timestamp=stamp,
        n_arms=len(results),
        n_succeeded=sum(1 for r in results if r.status == "ok"),
        n_failed=sum(1 for r in results if r.status == "failed"),
        results=results,
        failed=[f"{r.experiment_id}/{r.variant}" for r in results if r.status == "failed"],
        registry_path=str(registry_path) if register else None,
        registry_summary=registry.summary() if register else None,
        frozen_conditions=frozen_conditions(common),
        arm_summary=arm_summary(selected),
        artifact_path=str(artifact) if artifact else None,
    )


def _iso(stamp: str) -> str:
    """The suite's compact UTC stamp in ISO-8601, for registry entries.

    Every arm in one suite invocation shares it, which is true and is what makes
    a resumed run identifiable. The per-run directory name carries no timestamp
    of its own -- the suite names run directories deterministically so `--resume`
    can find them -- so this is the recorded time of the run.
    """
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()


def _write_suite_artifact(
    root: Path,
    *,
    plan: SuitePlan,
    results: Sequence[ArmRunResult],
    timestamp: str,
    registry_path: Path | None,
    registry: ExperimentRegistry | None,
    common: ExperimentConfig,
    arms: Sequence[Arm],
) -> Path:
    """Write `suite.json`: the plan, the frozen conditions, and every outcome.

    Written after the run and again on a strict-mode abort, so a partial suite is
    still an artifact rather than a stack trace. Failed arms appear in it with
    their error, which is what makes "the suite reported it, nobody hid it"
    checkable after the fact.
    """
    destination = root / "suite.json"
    payload = {
        "suite_version": SUITE_VERSION,
        "plan": plan.model_dump(mode="json"),
        "timestamp": timestamp,
        "frozen_conditions": frozen_conditions(common),
        "arms": arm_summary(arms),
        "results": [result.model_dump(mode="json") for result in results],
        "n_arms": len(results),
        "n_succeeded": sum(1 for r in results if r.status == "ok"),
        "n_failed": sum(1 for r in results if r.status == "failed"),
        "failed": [f"{r.experiment_id}/{r.variant}" for r in results if r.status == "failed"],
        "registry": {
            "path": str(registry_path) if registry_path else None,
            "summary": registry.summary() if registry else None,
        },
        "pace_seconds": plan.pace_seconds,
        "pace_note": (
            "Sleep before each retrieval on provider-calling arms, so the "
            "embedding provider is not called in a burst. Applied outside the "
            "retriever's own clock, so it cannot enter any latency_ms."
        ),
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


__all__ = [
    "ANALYSIS_ONLY_EXPERIMENT_IDS",
    "ArmFailure",
    "ArmRunResult",
    "DEFAULT_PACE_SECONDS",
    "DEFAULT_SUITE_ROOT",
    "EXPERIMENT_IDS",
    "PacedRetriever",
    "PROVIDER_CALLING_METHODS",
    "RUNNABLE_EXPERIMENT_IDS",
    "SUITE_VERSION",
    "SuiteExecutionError",
    "SuitePlan",
    "SuiteResult",
    "plan_arms",
    "run_suite",
]

