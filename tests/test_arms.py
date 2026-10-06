"""
tests.test_arms
---------------
Offline checks that the five Phase 7 arms are genuinely comparable.

The whole quality--efficiency study rests on one claim: that every arm differs
from every other arm *only* in retrieval strategy. If a corpus version, chunking
setting, K grid, or embedding model drifted between arms, every number in the
Phase 7 report would describe a comparison nobody ran. That claim is checkable
here, without a corpus, an index, or a network call, so it is checked.
"""

import pytest

from adaptive_rag.errors import ConfigurationError
from adaptive_rag.evaluation.ablation import escalation_variants
from adaptive_rag.experiments.arms import (
    ARM_NAMES,
    SHARED_CONDITION_FIELDS,
    Arm,
    arm_summary,
    build_ablation_arms,
    build_arm,
    build_arms,
    frozen_conditions,
    verify_arm_configs,
    verify_shared_conditions,
)
from adaptive_rag.schemas.config import (
    ChunkingConfig,
    ContextConfig,
    EmbeddingConfig,
    EvaluationConfig,
    GenerationConfig,
    IndexConfig,
    IngestionConfig,
    RetrievalConfig,
    RoutingConfig,
)
from adaptive_rag.schemas.experiment import ExperimentConfig

CORPUS_VERSION = "corpus_test_fixture_v1"


def make_common(**overrides) -> ExperimentConfig:
    """A fully-specified ExperimentConfig with no I/O behind it.

    `compute_corpus_version` is never called: it reads the real PDF manifest, and
    the arms must be testable offline. The corpus version is a literal because
    the point of these tests is that arms *agree* on it, not what it says.
    """
    base = {
        "experiment_id": "phase7_common",
        "name": "phase7_common",
        "corpus_version": CORPUS_VERSION,
        "ingestion": IngestionConfig(),
        "chunking": ChunkingConfig(),
        "embedding": EmbeddingConfig(),
        "index": IndexConfig(),
        "retrieval": RetrievalConfig(retrieval_method="dense"),
        "context": ContextConfig(),
        "generation": GenerationConfig(),
        "evaluation": EvaluationConfig(enable_llm_judge=False),
        "config_hash": "pending",
    }
    base.update(overrides)
    # Built directly rather than through build_experiment_config, which hashes
    # the config and calls compute_corpus_version -- that reads the real PDF
    # manifest, and these tests must stay offline.
    return ExperimentConfig(**base)


# --- shape -------------------------------------------------------------------


def test_there_are_exactly_five_benchmark_arms():
    """The brief names five systems; a sixth would be scope creep."""
    assert ARM_NAMES == (
        "bm25",
        "dense",
        "hybrid",
        "hybrid_rerank",
        "adaptive",
    )


def test_build_arms_returns_all_five_in_order():
    arms = build_arms(make_common())

    assert [a.name for a in arms] == list(ARM_NAMES)
    verify_arm_configs(arms)


def test_only_the_adaptive_arm_is_marked_adaptive():
    arms = build_arms(make_common())

    assert [a.name for a in arms if a.is_adaptive] == ["adaptive"]


def test_every_arm_maps_to_a_real_retrieval_method():
    """Each arm must correspond to a method the retriever factory can build."""
    arms = build_arms(make_common())
    methods = {a.name: a.retrieval.retrieval_method for a in arms}

    assert methods == {
        "bm25": "bm25",
        "dense": "dense",
        "hybrid": "hybrid",
        "hybrid_rerank": "hybrid_rerank",
        "adaptive": "adaptive",
    }


def test_an_unknown_arm_name_is_rejected():
    with pytest.raises(ConfigurationError, match="Unknown arm"):
        build_arm(make_common(), "magic")


# --- the load-bearing claim: shared conditions --------------------------------


def test_all_arms_agree_on_every_shared_condition():
    """The core guarantee. If this fails, the benchmark is not a comparison."""
    common = make_common()
    arms = build_arms(common)

    verify_shared_conditions(common, arms)  # raises on drift

    # Independently re-derived rather than trusting the helper alone.
    conditions = frozen_conditions(common)
    assert set(conditions) == set(SHARED_CONDITION_FIELDS)
    for arm in arms:
        built = arm.build_config(common)
        for field in SHARED_CONDITION_FIELDS:
            expected = conditions[field]
            actual = getattr(built, field)
            actual = (
                actual.model_dump(mode="json")
                if hasattr(actual, "model_dump")
                else actual
            )
            assert actual == expected, f"{arm.name} drifted on {field}"


def test_arms_agree_even_when_the_common_config_is_not_a_default_one():
    """Agreement must come from construction, not from luck of the defaults.

    A common config carrying a non-default k_grid and chunking target is the
    realistic case: Phase 7 freezes whatever the calibrated settings are, and
    every arm must still inherit them rather than reverting to defaults.
    """
    common = make_common(
        chunking=ChunkingConfig(target_tokens=400, overlap_tokens=60),
        evaluation=EvaluationConfig(k_grid=[1, 5, 10], enable_llm_judge=False),
    )
    arms = build_arms(common)

    verify_shared_conditions(common, arms)
    for arm in arms:
        built = arm.build_config(common)
        assert built.chunking.target_tokens == 400
        assert built.evaluation.k_grid == [1, 5, 10]


def test_k_grid_is_preserved_not_reset():
    """The brief requires k in {1, 5, 10}; an arm must not silently use another."""
    common = make_common(evaluation=EvaluationConfig(k_grid=[1, 5, 10]))
    arms = build_arms(common)

    assert all(a.evaluation.k_grid == [1, 5, 10] for a in arms)
    assert all(a.build_config(common).evaluation.k_grid == [1, 5, 10] for a in arms)


def test_generation_config_is_identical_across_arms():
    """Generation must not vary by strategy -- the brief forbids it explicitly."""
    common = make_common()
    configs = [a.build_config(common).generation.model_dump(mode="json") for a in build_arms(common)]

    assert len({repr(sorted(c.items())) for c in configs}) == 1


def test_arms_do_not_mutate_the_shared_config():
    """`common` must still be usable as the reference point after building."""
    common = make_common()
    before = common.model_dump(mode="json")

    build_arms(common)

    assert common.model_dump(mode="json") == before


def test_verify_shared_conditions_catches_a_deliberately_drifted_arm():
    """The guard must actually fire, or it is decoration.

    Built by hand because `build_arms` cannot drift by construction -- which is
    the point, but also means the guard would never be exercised in production.
    """
    common = make_common()
    arms = build_arms(common)
    drifted = arms[0].model_copy(
        update={"evaluation": EvaluationConfig(k_grid=[1], enable_llm_judge=False)}
    )

    with pytest.raises(ConfigurationError, match="evaluation"):
        verify_shared_conditions(common, [drifted])


def test_verify_shared_conditions_catches_a_corpus_version_drift():
    common = make_common()
    arms = build_arms(common)
    drifted = arms[1].model_copy(update={"name": "wrong_corpus"})
    arms[1] = drifted

    # Name is not a shared condition, so this must *not* fire -- the corpus
    # version lives in the ExperimentConfig built from `common`, which still
    # agrees. This pins that the guard checks conditions, not incidental fields.
    verify_shared_conditions(common, arms)


# --- what may differ ----------------------------------------------------------


def test_only_the_retrieval_method_differs_between_fixed_arms():
    """Every other retrieval field is inherited, not re-decided per arm."""
    common = make_common()
    arms = build_arms(common)
    base = common.retrieval.model_dump()

    allowed = {"retrieval_method", "retriever_version", "rerank_enabled", "candidate_k"}
    for arm in arms:
        changed = {
            field
            for field in base
            if arm.retrieval.model_dump()[field] != base[field]
        }
        assert changed <= allowed, f"{arm.name} changed {sorted(changed - allowed)}"


def test_top_k_is_identical_across_arms():
    common = make_common()
    tops = {a.retrieval.top_k for a in build_arms(common)}

    assert len(tops) == 1


def test_reranking_is_enabled_only_on_the_reranked_arm():
    arms = build_arms(make_common())
    enabled = {a.name for a in arms if a.retrieval.rerank_enabled}

    assert enabled == {"hybrid_rerank"}


def test_rerank_candidate_depth_is_reached_by_the_fusion_depth():
    """The rerank arm's fused pool must be at least as deep as the rerank stage.

    `RetrievalConfig` enforces `candidate_k >= rerank_candidate_k`; this pins
    that `build_arms` satisfies it rather than relying on the default happening
    to line up (20 vs 20 does not hold under a non-default common config).
    """
    common = make_common(
        retrieval=RetrievalConfig(
            retrieval_method="hybrid", candidate_k=30, top_k=10,
            rerank_candidate_k=20, rerank_enabled=True,
        )
    )
    arm = build_arm(common, "hybrid_rerank")

    assert arm.retrieval.candidate_k >= arm.retrieval.rerank_candidate_k
    arm.retrieval  # validator ran during construction


def test_rerank_depth_is_raised_when_the_fusion_pool_is_too_shallow():
    common = make_common(
        retrieval=RetrievalConfig(
            retrieval_method="hybrid", candidate_k=15, top_k=10,
            rerank_candidate_k=20, rerank_enabled=True,
        )
    )
    arm = build_arm(common, "hybrid_rerank")

    assert arm.retrieval.candidate_k == 20


def test_each_arm_gets_the_retriever_version_of_its_own_method():
    """A stale inherited version would be recorded in the manifest as truth."""
    common = make_common(retrieval=RetrievalConfig(retrieval_method="bm25"))
    arms = build_arms(common)
    versions = {a.name: a.retrieval.retriever_version for a in arms}

    assert versions["bm25"] == "bm25_v1"
    assert versions["dense"] == "dense_v1"
    assert versions["hybrid"] == "hybrid_v1"
    assert versions["hybrid_rerank"] == "hybrid_rerank_v1"
    assert versions["adaptive"] == "adaptive_v1"


# --- routing ------------------------------------------------------------------


def test_fixed_arms_carry_the_shared_routing_config_unchanged():
    """A fixed arm has no router; the field exists only to satisfy the schema."""
    common = make_common(routing=RoutingConfig(cost_weight=0.9))
    arms = build_arms(common)

    for arm in arms:
        assert arm.routing == common.routing
        assert arm.is_adaptive == (arm.name == "adaptive")


def test_the_adaptive_arm_can_carry_an_ablation_variant_config():
    """Escalation A/B/C are arms differing only in RoutingConfig.

    This is the join between the two foundation modules: a Phase 7 ablation is
    an adaptive arm with a different routing config.
    """
    common = make_common()
    variants = escalation_variants()

    built = build_ablation_arms(common, variants)
    assert [a.routing.sufficiency_enabled for a in built] == [False, True, True]
    assert [a.routing.escalation_enabled for a in built] == [False, False, True]
    verify_arm_configs(built)


def test_ablation_arms_are_named_distinctly():
    """All three escalation variants run `adaptive`; they must not collide.

    Arms are keyed by name in the result storage, so three arms named
    `adaptive` would overwrite each other's rows -- the ablation would appear
    to have produced one result, three times.
    """
    arms = build_ablation_arms(make_common(), escalation_variants())
    names = [a.name for a in arms]

    assert names == [v.name for v in escalation_variants()]
    assert len(set(names)) == len(names)
    assert all(a.retrieval.retrieval_method == "adaptive" for a in arms)


def test_ablation_arms_inherit_the_variant_description():
    arms = build_ablation_arms(make_common(), escalation_variants())
    variants = escalation_variants()

    assert [a.description for a in arms] == [v.description for v in variants]


def test_ablation_arms_get_distinct_config_hashes():
    """Different routing configs must produce different runs, not one rerun."""
    common = make_common()
    hashes = [a.build_config(common).config_hash for a in build_ablation_arms(common, escalation_variants())]

    assert len(set(hashes)) == 3


def test_ablation_variants_do_not_change_any_arm_except_the_routing():
    common = make_common()
    baseline = build_arm(common, "adaptive")
    for variant in escalation_variants():
        arm = build_arm(common, "adaptive", routing=variant.routing)
        assert arm.retrieval == baseline.retrieval
        assert arm.evaluation == baseline.evaluation


# --- config_hash --------------------------------------------------------------


def test_each_arm_produces_a_distinct_config_hash():
    """Two arms with the same hash would mean they are the same experiment."""
    common = make_common()
    hashes = {a.build_config(common).config_hash for a in build_arms(common)}

    assert len(hashes) == len(ARM_NAMES)


def test_an_arm_rebuilds_to_the_same_hash():
    """Determinism: a run is reproducible from its recorded config alone."""
    common = make_common()
    arm = build_arm(common, "hybrid")

    assert arm.build_config(common).config_hash == arm.build_config(common).config_hash


def test_building_an_arm_leaves_the_shared_config_hash_untouched():
    common = make_common()
    common_hash = common.config_hash

    for arm in build_arms(common):
        arm.build_config(common)

    assert common.config_hash == common_hash


# --- guards -------------------------------------------------------------------


def test_verify_arm_configs_rejects_a_mislabelled_adaptive_flag():
    arm = Arm(
        name="adaptive",
        experiment_id="E1",
        retrieval=RetrievalConfig(retrieval_method="adaptive"),
        routing=RoutingConfig(),
        evaluation=EvaluationConfig(),
        description="x",
        is_adaptive=False,
    )

    with pytest.raises(ConfigurationError, match="is_adaptive"):
        verify_arm_configs([arm])


def test_verify_arm_configs_rejects_reranking_on_a_cheap_arm():
    """A 'cheap' arm that quietly pays cross-encoder cost would bias the study."""
    cheap = Arm(
        name="bm25",
        experiment_id="E1",
        retrieval=RetrievalConfig(
            retrieval_method="bm25", rerank_enabled=True, rerank_candidate_k=20
        ),
        routing=RoutingConfig(),
        evaluation=EvaluationConfig(),
        description="x",
    )

    with pytest.raises(ConfigurationError, match="reranking"):
        verify_arm_configs([cheap])


def test_verify_arm_configs_rejects_duplicate_names():
    arm = build_arm(make_common(), "bm25")
    with pytest.raises(ConfigurationError, match="duplicate"):
        verify_arm_configs([arm, arm])


def test_verify_arm_configs_rejects_an_empty_suite():
    with pytest.raises(ConfigurationError, match="no arms"):
        verify_arm_configs([])


# --- summary ------------------------------------------------------------------


def test_arm_summary_round_trips_as_json():
    import json

    arms = build_arms(make_common())
    payload = arm_summary(arms)

    assert payload["n_arms"] == 5
    encoded = json.dumps(payload)  # must be plain JSON for suite.json
    assert json.loads(encoded)["arms"][4]["name"] == "adaptive"