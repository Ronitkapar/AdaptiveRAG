"""
experiments.registry
--------------------
The append-only registry of Phase 7 runs.

Phase 7 is eight studies (`E1`..`E8`), and every one of them is executed many
times: E3 runs the shipped router plus one arm per disabled feature group, E4
sweeps a threshold, E5 sweeps a cost weight. The question the registry answers is
therefore never "what did you run?" but "which run, on which data, at which
commit, under which configuration?" -- and the answer has to survive the
`experiments/` directory being gitignored, which is why provenance travels
*inside* the registry entry rather than being left in the repository.

Three properties make it usable as the spine of the reporting layer:

* **Stable IDs.** `EXPERIMENT_IDS` is the mandated vocabulary. An entry whose
  `experiment_id` is not in it is rejected, so a typo cannot invent a ninth
  study, and an analysis stage can enumerate the studies without parsing
  directory names.
* **Append-only.** Re-registering an `(experiment_id, variant)` pair raises.
  A Phase 7 report that silently overwrote the full-system run with the
  second full-system run would be a report about whichever run happened to be
  last, which is precisely the failure this module exists to make impossible.
* **Provenance by reference.** `RegistryEntry.from_run_dir` reads the run's
  `manifest.json` and `config.json` rather than being handed a second copy of
  the same facts. If the runner's manifest gains a field, the entry follows;
  if the two disagreed, nothing would notice.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.errors import ExperimentRunError
from adaptive_rag.schemas.config import RoutingConfig

REGISTRY_VERSION = "phase7_registry_v1"

# The eight Phase 7 studies, in report order. This tuple is the registry's
# primary sort key, so a listing is grouped the way the write-up is.
EXPERIMENT_IDS: tuple[str, ...] = (
    "E1_baseline_comparison",
    "E2_escalation_ablation",
    "E3_feature_ablation",
    "E4_threshold_calibration",
    "E5_cost_weight",
    "E6_routing_overhead",
    "E7_query_type",
    "E8_escalation_analysis",
)

# Ids that appear in existing code but are not the mandated spelling. Kept as an
# explicit, tiny map rather than normalised by pattern-matching, so adding a new
# alias is a reviewed decision and not a string surgery.
#
# `E5_cost_weight_ablation` is stamped by `evaluation.ablation.cost_weight_sweep`,
# which predates this registry and is not modifiable from here. The Phase 7
# vocabulary calls the study `E5_cost_weight`; aliasing on the way in keeps the
# registry keyed by the mandated ids and leaves the ablation module, and any
# write-up that cites its literal, untouched.
EXPERIMENT_ID_ALIASES: dict[str, str] = {
    "E5_cost_weight_ablation": "E5_cost_weight",
}

# What a run is *for*, as distinct from which study it belongs to. E6/E7/E8
# measure the runs E1 already produced rather than new systems, and a report
# that mixed their rows into a quality comparison would be answering a
# different question.
ExperimentKind = Literal["baseline", "adaptive", "ablation", "analysis"]

EXPERIMENT_KINDS: tuple[str, ...] = ("baseline", "adaptive", "ablation", "analysis")

# The run id `experiments.runner.ExperimentRunner` generates when no `run_id` is
# passed: `<UTC timestamp>-<config name>`. Parsed so a registry entry read back
# from an existing run directory can recover when it ran without being handed a
# timestamp, and without inventing one.
_RUN_ID_TIMESTAMP = re.compile(r"^(\d{8}T\d{6}Z)-")

DEFAULT_REGISTRY_NAME = "registry.json"


class RegistryError(ExperimentRunError):
    """Raised when a registration would violate the registry's contract."""


def normalize_experiment_id(experiment_id: str) -> str:
    """Map a known alias onto its mandated `EXPERIMENT_IDS` spelling.

    Rejects anything outside the mandated vocabulary. A permissive registry is
    worse than none: an unrecognised id cannot be ordered, cannot be reported
    against, and would only be discovered as a missing row much later.
    """
    canonical = EXPERIMENT_ID_ALIASES.get(experiment_id, experiment_id)
    if canonical not in EXPERIMENT_IDS:
        raise RegistryError(
            f"unknown experiment_id {experiment_id!r}; expected one of "
            f"{list(EXPERIMENT_IDS)}"
        )
    return canonical


def varied_routing_fields(
    routing: RoutingConfig | Mapping[str, Any] | None,
    base: RoutingConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The routing fields that differ between `routing` and the base config.

    A registry entry records the whole routing config, because that is what
    reproduces the run. This is the other half: the small set of fields a study
    actually turned, which is what an ablation table is keyed by. Recording both
    means a reader can see the change without diffing two config dumps, and a
    study that varies nothing shows an empty diff rather than hiding in the dump.
    """
    if routing is None:
        return {}
    actual = routing.model_dump(mode="json") if isinstance(routing, BaseModel) else dict(routing)
    if base is None:
        return {}
    reference = base.model_dump(mode="json") if isinstance(base, BaseModel) else dict(base)
    return {
        field: actual[field]
        for field in sorted(actual)
        if field in reference and reference[field] != actual[field]
    }


class RegistryEntry(BaseModel):
    """One executed run, with everything needed to interpret and repeat it."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    # The arm or ablation variant name: "bm25", "adaptive", "without_technical",
    # "cost_weight=0.5". Together with `experiment_id` this is the entry's
    # identity, and two runs of the same variant are the same measurement
    # attempted twice rather than two measurements.
    variant: str
    kind: ExperimentKind
    run_dir: str
    trace_count: int
    git_commit: str | None = None
    dataset_version: str
    dataset_sha256: str
    config_hash: str
    corpus_version: str
    # ISO-8601 UTC. Kept as a string rather than a datetime so a round trip
    # through JSON cannot change its textual form.
    timestamp: str
    environment: dict[str, Any] = Field(default_factory=dict)
    # The full routing config the run used, verbatim. `None` is not used: a fixed
    # arm still carries the shared routing config, and blanking it would lose
    # the fact that the shared config participated in `config_hash`.
    routing: dict[str, Any] = Field(default_factory=dict)
    varied_routing_fields: dict[str, Any] = Field(default_factory=dict)
    pace_seconds: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        """The entry's identity: which study, which variant."""
        return (self.experiment_id, self.variant)

    def model_post_init(self, _context: Any) -> None:
        # Normalise the study id on the way in, so a hand-built entry cannot claim a
        # study outside the vocabulary and cannot smuggle the aliased spelling past
        # the ordering. `kind` needs no equivalent check: pydantic validates the
        # Literal before this hook runs.
        object.__setattr__(self, "experiment_id", normalize_experiment_id(self.experiment_id))

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path | str,
        *,
        experiment_id: str,
        variant: str,
        kind: ExperimentKind,
        dataset_version: str,
        dataset_sha256: str,
        base_routing: RoutingConfig | Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        pace_seconds: float = 0.0,
        timestamp: str | None = None,
    ) -> "RegistryEntry":
        """Build an entry by reading a run directory's own artifacts.

        `manifest.json` is the authority for `git_commit`, `config_hash`,
        `corpus_version` and `trace_count`; `config.json` is the authority for
        the routing config. Both are read rather than passed in, so an entry can
        never disagree with the run it describes. The two facts the manifest does
        not carry -- which dataset, and when -- are supplied by the caller, which
        is the only place that knows them.
        """
        directory = Path(run_dir)
        manifest_path = directory / "manifest.json"
        config_path = directory / "config.json"
        if not manifest_path.is_file():
            raise RegistryError(f"no manifest.json in run directory {directory}")
        if not config_path.is_file():
            raise RegistryError(f"no config.json in run directory {directory}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        routing = config.get("routing", {})

        return cls(
            experiment_id=experiment_id,
            variant=variant,
            kind=kind,
            run_dir=str(directory),
            trace_count=int(manifest.get("trace_count", 0)),
            git_commit=manifest.get("git_commit"),
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            config_hash=str(manifest.get("config_hash", "")),
            corpus_version=str(manifest.get("corpus_version", "")),
            timestamp=timestamp if timestamp is not None else _timestamp_from_run_id(directory),
            environment=dict(environment or {}),
            routing=routing,
            varied_routing_fields=varied_routing_fields(routing, base_routing),
            pace_seconds=float(pace_seconds),
        )


def _timestamp_from_run_id(run_dir: Path) -> str:
    """Recover a run's start time from the runner's timestamped run id.

    Purely a parse: the runner stamps `%Y%m%dT%H%M%SZ` into the run id it
    generates, and a run directory named anything else simply has no recorded
    start time. That raises rather than substituting `now()`, because a
    fabricated provenance field is worse than a missing one.
    """
    match = _RUN_ID_TIMESTAMP.match(run_dir.name)
    if not match:
        raise RegistryError(
            f"run directory {run_dir.name!r} does not carry a timestamped run id, "
            "so the entry's timestamp is unknown; pass timestamp= explicitly"
        )
    parsed = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _sort_key(entry: RegistryEntry) -> tuple[int, str, str]:
    return (EXPERIMENT_IDS.index(entry.experiment_id), entry.variant, entry.run_dir)


class ExperimentRegistry:
    """An append-only collection of `RegistryEntry`, persisted as one JSON file.

    The registry is held in memory and written whole, which is the right shape
    for a study that runs a few dozen arms. `register` is the only mutator and it
    never overwrites by accident; `save` is explicit so a caller can register a
    batch and persist once at the end.
    """

    def __init__(self, entries: Iterable[RegistryEntry] | None = None):
        self._entries: list[RegistryEntry] = []
        for entry in entries or []:
            # Constructed through `register` so a seed list obeys the same
            # duplicate rule as a live run does.
            self.register(entry)

    # --- access ------------------------------------------------------------

    @property
    def entries(self) -> list[RegistryEntry]:
        """Every entry, in report order. A copy: callers cannot mutate state."""
        return sorted(self._entries, key=_sort_key)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, experiment_id: str, variant: str) -> RegistryEntry | None:
        canonical = normalize_experiment_id(experiment_id)
        for entry in self._entries:
            if entry.key == (canonical, variant):
                return entry
        return None

    def variants(self, experiment_id: str) -> list[str]:
        """Registered variant names for one study, in report order."""
        canonical = normalize_experiment_id(experiment_id)
        return [entry.variant for entry in self.entries if entry.experiment_id == canonical]

    def by_experiment(self) -> dict[str, list[RegistryEntry]]:
        """Entries grouped by study, with every mandated study present.

        An empty study is reported as an empty list rather than omitted, so a
        summary cannot imply coverage that was never run.
        """
        grouped: dict[str, list[RegistryEntry]] = {eid: [] for eid in EXPERIMENT_IDS}
        for entry in self.entries:
            grouped[entry.experiment_id].append(entry)
        return grouped

    # --- mutation ----------------------------------------------------------

    def register(
        self,
        entry: RegistryEntry,
        *,
        allow_reentry: bool = False,
    ) -> RegistryEntry:
        """Add an entry, or refuse.

        Identity is `(experiment_id, variant)`. Re-registering an existing pair
        raises `RegistryError` unless `allow_reentry=True`, which replaces the
        earlier entry in place and is the one sanctioned way to correct a
        registration.

        Registering a *different* variant under an existing study is not
        reentry and is never refused: that is how E3 accumulates one arm per
        disabled feature group, and how a repeated run of the same study with a
        different routing config joins the same report.
        """
        existing = self.get(entry.experiment_id, entry.variant)
        if existing is not None and not allow_reentry:
            raise RegistryError(
                f"{entry.experiment_id}/{entry.variant} is already registered "
                f"(run_dir={existing.run_dir}); pass allow_reentry=True to replace it, "
                "or register under a different variant name"
            )
        if existing is not None:
            self._entries[self._entries.index(existing)] = entry
        else:
            self._entries.append(entry)
        return entry

    def extend(
        self,
        entries: Iterable[RegistryEntry],
        *,
        allow_reentry: bool = False,
    ) -> list[RegistryEntry]:
        """Register several entries, stopping at the first refusal."""
        return [self.register(entry, allow_reentry=allow_reentry) for entry in entries]

    # --- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_version": REGISTRY_VERSION,
            "experiment_ids": list(EXPERIMENT_IDS),
            "n_entries": len(self._entries),
            "entries": [entry.model_dump(mode="json") for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRegistry":
        entries = [RegistryEntry.model_validate(item) for item in payload.get("entries", [])]
        return cls(entries)

    def save(self, path: Path | str = DEFAULT_REGISTRY_NAME) -> Path:
        """Write the registry as deterministic, human-readable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: Path | str = DEFAULT_REGISTRY_NAME) -> "ExperimentRegistry":
        source = Path(path)
        if not source.is_file():
            raise RegistryError(f"no registry at {source}")
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    @classmethod
    def load_or_empty(cls, path: Path | str = DEFAULT_REGISTRY_NAME) -> "ExperimentRegistry":
        """Load an existing registry, or start one when there is no file yet.

        Used by the suite driver so a resumed run extends the previous
        registration instead of replacing it.
        """
        source = Path(path)
        if not source.is_file():
            return cls()
        return cls.load(source)

    def merge(
        self,
        other: "ExperimentRegistry",
        *,
        allow_reentry: bool = False,
    ) -> list[RegistryEntry]:
        """Register every entry of `other` into this registry."""
        return self.extend(other.entries, allow_reentry=allow_reentry)

    def summary(self) -> dict[str, Any]:
        """Compact per-study counts for `suite.json` and the CLI report."""
        grouped = self.by_experiment()
        return {
            "registry_version": REGISTRY_VERSION,
            "n_entries": len(self._entries),
            "by_experiment": {
                experiment_id: [entry.variant for entry in entries]
                for experiment_id, entries in grouped.items()
            },
        }

