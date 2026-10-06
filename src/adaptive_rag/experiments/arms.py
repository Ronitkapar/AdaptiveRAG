"""
experiments.arms
----------------
The five Phase 7 benchmark systems, built from one frozen configuration.

An *arm* is a named retrieval strategy under otherwise identical conditions.
The Phase 7 brief requires that the corpus, preprocessing, chunking, query set,
K values, embedding model, reranker, generation configuration, and hardware are
the same for every arm -- and that differences in computational cost between
strategies are the thing being measured rather than something to normalize away.

That is why an arm carries almost nothing. Every arm derives its
`RetrievalConfig` from a single `common` `ExperimentConfig`, overriding only
`retrieval_method` and `rerank_enabled`. There is no per-arm knob to forget to
keep in sync, because there is no per-arm knob.

The fifth arm, `adaptive`, additionally carries a `RoutingConfig`. It starts
from `common.routing`, and the ablation module (`evaluation.ablation`) turns it
into further arms by changing that config and nothing else.

This module builds configurations. It does not run anything, open an index, or
import a concrete retriever -- `experiments.config.instantiate_components`
already does that, and Phase 7 must not grow a second retrieval system.
"""

from typing import TYPE_CHECKING, Sequence

from pydantic import BaseModel, ConfigDict

from adaptive_rag.errors import ConfigurationError
from adaptive_rag.experiments.config import build_experiment_config
from adaptive_rag.schemas.config import EvaluationConfig, RetrievalConfig, RoutingConfig
from adaptive_rag.schemas.experiment import ExperimentConfig

if TYPE_CHECKING:
    # Imported for the type signature only. `evaluation.ablation` imports
    # RoutingConfig from schemas and nothing from experiments, so a real import
    # would work -- but keeping the dependency one-way makes it structurally
    # impossible for the ablation vocabulary to grow into the experiment layer.
    from adaptive_rag.evaluation.ablation import Variant

ARMS_VERSION = "phase7_arms_v1"

# The five systems the brief compares, in increasing cost order. The order is a
# display convention for tables, not a claim about quality -- Phase 7 must not
# rank its own arms in advance of measurement.
ARM_NAMES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank", "adaptive")

# ExperimentConfig fields that must be byte-identical across every arm. If one of
# these differs between arms, the comparison is no longer a comparison of
# retrieval strategies, and no amount of statistical care downstream rescues it.
SHARED_CONDITION_FIELDS: tuple[str, ...] = (
    "corpus_version",
    "ingestion",
    "chunking",
    "embedding",
    "index",
    "context",
    "generation",
    "evaluation",
)


class Arm(BaseModel):
    """One benchmark system: a name and the configs that distinguish it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    experiment_id: str
    retrieval: RetrievalConfig
    routing: RoutingConfig
    evaluation: EvaluationConfig
    description: str
    # True only for the adaptive arm, whose traces carry a routing story.
    is_adaptive: bool = False

    def build_config(self, common: ExperimentConfig) -> ExperimentConfig:
        """Materialise this arm as a full, hash-stamped `ExperimentConfig`.

        Returns a *new* config rather than mutating `common`, so arms can be built
        in any order and the shared config stays a trustworthy reference point.
        """
        return build_experiment_config(
            name=f"phase7_{self.name}",
            experiment_id=self.experiment_id,
            ingestion=common.ingestion,
            chunking=common.chunking,
            embedding=common.embedding,
            index=common.index,
            retrieval=self.retrieval,
            context=common.context,
            generation=common.generation,
            evaluation=self.evaluation,
            routing=self.routing,
            corpus_version=common.corpus_version,
        )


_DESCRIPTIONS: dict[str, str] = {
    "bm25": "Fixed BM25 for every query. The cheapest arm; no query understanding.",
    "dense": "Fixed dense retrieval for every query. Single-vector semantic match.",
    "hybrid": "Fixed RRF fusion of BM25 and dense. Both branches, always.",
    "hybrid_rerank": "Fixed hybrid plus a cross-encoder second stage. Most expensive arm.",
    "adaptive": (
        "Phase 6 adaptive routing: analyze, route, check sufficiency, escalate "
        "boundedly when the retrieved evidence is insufficient."
    ),
}


def _retrieval_config(base: RetrievalConfig, method: str, *, rerank: bool) -> RetrievalConfig:
    """Derive an arm's retrieval config from the shared one.

    Only `retrieval_method`, `rerank_enabled`, and the version string that
    belongs to that method change. The version is taken from a default instance
    rather than hand-written here: `RetrievalConfig` already auto-aligns a
    default version to the active method, but that alignment only fires when the
    incoming version is itself a default. Inheriting, say, `bm25_v1` from a
    BM25 common config and then switching to dense would otherwise keep the
    stale BM25 version and be recorded in the manifest as such.
    """
    default_version = RetrievalConfig(retrieval_method=method).retriever_version
    values = base.model_dump()
    values.update(
        retriever_version=default_version,
        retrieval_method=method,
        rerank_enabled=rerank,
    )
    if method == "hybrid_rerank":
        # The fused pool must be at least as deep as the rerank depth; the
        # validator rejects it otherwise, which is the correct place for that.
        values["candidate_k"] = max(base.candidate_k, base.rerank_candidate_k)
    return RetrievalConfig(**values)


def build_arm(
    common: ExperimentConfig,
    name: str,
    *,
    routing: RoutingConfig | None = None,
    experiment_id: str = "E1_baseline_comparison",
    label: str | None = None,
) -> Arm:
    """Build a single named arm from the shared configuration.

    `name` selects the strategy. `label` distinguishes arms that share a
    strategy -- the ablation variants all run `adaptive`, and without a label
    they would collide in the per-arm result storage and overwrite each other's
    rows. It defaults to `name`.
    """
    if name not in ARM_NAMES:
        raise ConfigurationError(
            f"Unknown arm {name!r}; expected one of {list(ARM_NAMES)}"
        )
    if name == "hybrid_rerank":
        retrieval = _retrieval_config(
            common.retrieval, "hybrid_rerank", rerank=True
        )
    else:
        retrieval = _retrieval_config(
            common.retrieval, name, rerank=False
        )
    return Arm(
        name=label or name,
        experiment_id=experiment_id,
        retrieval=retrieval,
        # A fixed arm has no router. It still carries `common.routing` because
        # `ExperimentConfig` requires the field and it participates in
        # `config_hash` -- but the retriever builder never reads it for a
        # non-adaptive method, so it cannot influence behaviour.
        routing=routing or common.routing,
        evaluation=common.evaluation,
        description=_DESCRIPTIONS[name],
        is_adaptive=(name == "adaptive"),
    )


def build_ablation_arms(
    common: ExperimentConfig,
    variants: Sequence["Variant"],
) -> list[Arm]:
    """Build one adaptive arm per ablation variant.

    Every variant runs the same strategy with a different `RoutingConfig`, so
    this is where the `label` matters: without it all of them would be named
    `adaptive`. The variant name is used verbatim as the arm name because that
    is the identifier the ablation tables are keyed by.
    """
    arms: list[Arm] = []
    for variant in variants:
        arm = build_arm(
            common,
            "adaptive",
            routing=variant.routing,
            experiment_id=variant.experiment_id,
            label=variant.name,
        )
        arms.append(arm.model_copy(update={"description": variant.description}))
    verify_arm_configs(arms)
    return arms


def build_arms(
    common: ExperimentConfig,
    *,
    routing: RoutingConfig | None = None,
    experiment_id: str = "E1_baseline_comparison",
) -> list[Arm]:
    """Build all five benchmark arms from one frozen configuration."""
    return [
        build_arm(common, name, routing=routing, experiment_id=experiment_id)
        for name in ARM_NAMES
    ]


def frozen_conditions(common: ExperimentConfig) -> dict[str, object]:
    """The conditions held constant across the suite.

    Written into `suite.json` so a reader can confirm what was held fixed
    without reconstructing it from the individual run manifests.
    """
    return {
        field: getattr(common, field).model_dump(mode="json")
        if hasattr(getattr(common, field), "model_dump")
        else getattr(common, field)
        for field in SHARED_CONDITION_FIELDS
    }


def verify_shared_conditions(
    common: ExperimentConfig, arms: Sequence[Arm]
) -> None:
    """Fail loudly if any arm drifted from the frozen shared conditions.

    This is checked rather than assumed because the arms are built from `common`
    by construction -- which means a future edit that gives an arm its own
    evaluation config or corpus version would pass review unnoticed and quietly
    invalidate the whole comparison. Cheap to check, expensive to miss.
    """
    expected = frozen_conditions(common)
    mismatches: list[str] = []
    for arm in arms:
        actual = arm.build_config(common)
        for field in SHARED_CONDITION_FIELDS:
            want = expected[field]
            got = (
                getattr(actual, field).model_dump(mode="json")
                if hasattr(getattr(actual, field), "model_dump")
                else getattr(actual, field)
            )
            if got != want:
                mismatches.append(f"{arm.name}.{field}")
    if mismatches:
        raise ConfigurationError(
            "arms drifted from the frozen shared conditions: "
            + ", ".join(sorted(mismatches))
        )


def verify_arm_configs(arms: Sequence[Arm]) -> None:
    """Check the invariants the benchmark depends on, on built `Arm` objects.

    Separate from `verify_shared_conditions` because it also covers properties
    that hold arm-to-arm rather than against `common` -- for instance, the
    reranked arm must be the only one with `rerank_enabled`, or the suite would
    be paying cross-encoder cost inside an arm advertised as cheap.
    """
    if not arms:
        raise ConfigurationError("no arms were built")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ConfigurationError(f"duplicate arm names: {names}")

    reranked = [arm.name for arm in arms if arm.retrieval.rerank_enabled]
    expected = ["hybrid_rerank"] if "hybrid_rerank" in names else []
    if sorted(reranked) != sorted(expected):
        raise ConfigurationError(
            f"reranking must be enabled only for the hybrid_rerank arm, got {reranked}"
        )

    for arm in arms:
        adaptive = arm.retrieval.retrieval_method == "adaptive"
        if adaptive != arm.is_adaptive:
            raise ConfigurationError(
                f"arm {arm.name!r} is_adaptive={arm.is_adaptive} but "
                f"retrieval_method={arm.retrieval.retrieval_method!r}"
            )


def arm_summary(arms: Sequence[Arm]) -> dict[str, object]:
    """Suite-level arm description for `suite.json`."""
    return {
        "arms_version": ARMS_VERSION,
        "n_arms": len(arms),
        "arms": [
            {
                "name": arm.name,
                "experiment_id": arm.experiment_id,
                "description": arm.description,
                "is_adaptive": arm.is_adaptive,
                "retrieval_method": arm.retrieval.retrieval_method,
                "rerank_enabled": arm.retrieval.rerank_enabled,
                "retrieval": arm.retrieval.model_dump(mode="json"),
                "routing": arm.routing.model_dump(mode="json"),
            }
            for arm in arms
        ],
    }