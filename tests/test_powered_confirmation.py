"""
tests.test_powered_confirmation
-------------------------------
Offline tests for the Phase 15 powered-confirmation machinery
(`docs/phases/phase-15.md`).

* oracle equivalence with Phase 14 (same frozen oracle, same signal
  definitions: the Phase 15 builder reproduces the Phase 14 oracle labels
  and signal values on Phase 14 inputs);
* the frozen Phase 14 rule pinned (signal, threshold, direction);
* policy determinism (same inputs, same decisions);
* calibration/test separation (evaluation consumes split == test only;
  the builder rejects unexpected splits);
* random-baseline reproducibility (seeded draws are deterministic);
* combined randomization test and cluster bootstrap on handcrafted data;
* dataset-expansion integrity (duplicate screen, quota checker);
* metric aggregation (adaptive blend correctness);
* structural separation (powered code and Phase 15 scripts never import
  retrieval, embeddings, or provider machinery).

No test runs retrieval, touches the network, or requires credentials. The
equivalence test reads frozen committed-or-gitignored Phase 8/14
artifacts; it fits nothing and selects nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from adaptive_rag.evaluation import powered
from adaptive_rag.evaluation.cheapfirst import (
    PRE_DENSE_SIGNALS,
    apply_rule,
    classify_decision,
    confusion_counts,
    oracle_label,
)
from adaptive_rag.evaluation.powered import (
    CATEGORY_QUOTAS,
    DUPLICATE_JACCARD_THRESHOLD,
    PAPER_MINIMUMS,
    POWERED_N_QUERIES,
    check_quotas,
    cluster_bootstrap_ci,
    combined_randomization_test,
    normalize_query,
    screen_duplicates,
    token_jaccard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PHASE14_DIR = REPO_ROOT / "experiments" / "phase14"
PHASE8_DIR = REPO_ROOT / "experiments" / "phase8" / "combined"
PHASE7_DATASET = REPO_ROOT / "data" / "evaluation" / "phase7_eval_v1.jsonl"

SUITES = {"after": "p8a_e1", "before": "p8b_e1"}


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "phase15_build_oracle", SCRIPTS_DIR / "phase15_build_oracle.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase14_oracle_rows(arm: str) -> dict[str, dict]:
    path = PHASE14_DIR / f"oracle_phase8_{arm}.jsonl"
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record["query_id"]] = record
    return rows


# --- frozen rule ---------------------------------------------------------------


def test_frozen_rule_pinned_to_phase14_policy():
    """The Phase 15 candidate is the exact Phase 14 frozen rule (Option A)."""
    with open(PHASE14_DIR / "frozen_policy.json", encoding="utf-8") as handle:
        policy = json.load(handle)["frozen_rule"]
    assert policy["signal"] == "bm25_slope"
    assert policy["threshold"] == -0.9152305000000001
    assert policy["direction"] == "high"
    assert policy.get("degenerate") is False
    assert policy["signal"] in PRE_DENSE_SIGNALS


def test_policy_application_is_deterministic():
    signals = {"bm25_slope": -0.5, "bm25_gap": 0.1}
    first = apply_rule(signals["bm25_slope"], -0.9152305000000001, "high")
    second = apply_rule(signals["bm25_slope"], -0.9152305000000001, "high")
    assert first is True and second is True
    assert apply_rule(None, -0.9152305000000001, "high") is False


# --- oracle equivalence --------------------------------------------------------


@pytest.mark.skipif(
    not (PHASE8_DIR / "p8a_e1").exists() or not PHASE14_DIR.exists(),
    reason="needs frozen Phase 8 suites and Phase 14 oracle artifacts",
)
def test_oracle_equivalence_with_phase14():
    """Phase 15 builder reproduces Phase 14 labels+signals on old inputs."""
    builder = _load_builder()
    for arm, suite in SUITES.items():
        records, _ = builder.build_arm(
            arm,
            PHASE8_DIR / suite,
            PHASE7_DATASET,
            {"calibration", "test"},
        )
        frozen = _phase14_oracle_rows(arm)
        assert len(records) == len(frozen) == 107
        for record in records:
            old = frozen[record["query_id"]]
            assert record["oracle_escalate"] == old["oracle_escalate"]
            assert record["delta_recall_at_5"] == pytest.approx(
                old["delta_recall_at_5"]
            )
            for name, value in record["signals"].items():
                # Frozen Phase 14 scalars carry 6-decimal rounding; the
                # build's own integrity check compared at that precision.
                assert value == pytest.approx(old["signals"][name], abs=1e-6)


def test_oracle_rule_matches_frozen_definition():
    assert oracle_label(0.01) is True
    assert oracle_label(0.009999) is False
    assert oracle_label(0.0) is False
    assert oracle_label(-0.2) is False


# --- split discipline ------------------------------------------------------------


@pytest.mark.skipif(
    not (PHASE8_DIR / "p8a_e1").exists(),
    reason="needs frozen Phase 8 after-arm suite",
)
def test_builder_rejects_unexpected_splits():
    builder = _load_builder()
    with pytest.raises(ValueError, match="outside expected"):
        builder.build_arm(
            "after", PHASE8_DIR / "p8a_e1", PHASE7_DATASET, {"test"}
        )


# --- random baseline reproducibility -----------------------------------------------


def test_combined_randomization_is_deterministic():
    kwargs = {
        "adaptive_after": [1.0, 0.5, 0.0, 1.0],
        "adaptive_before": [0.5, 1.0, 0.5, 0.0],
        "bm25_after": [0.5, 0.5, 0.0, 0.5],
        "bm25_before": [0.5, 0.5, 0.5, 0.0],
        "dense_after": [1.0, 1.0, 0.5, 1.0],
        "dense_before": [1.0, 1.0, 0.5, 0.5],
        "k_after": 1,
        "k_before": 1,
    }
    first = combined_randomization_test(**kwargs)
    second = combined_randomization_test(**kwargs)
    assert first == second
    assert 0.0 <= first["p_comb"] <= 1.0
    assert first["n_draws"] == 1000 and first["seed"] == 20250101


def test_combined_randomization_detects_perfect_selection():
    """A rule that escalates exactly the positives beats the null."""
    bm25 = [0.0] * 10
    # Only the first 3 queries benefit from Dense; the rest tie.
    dense = [1.0, 1.0, 1.0] + [0.0] * 7
    adaptive = [1.0, 1.0, 1.0] + [0.0] * 7
    result = combined_randomization_test(
        adaptive_after=adaptive,
        adaptive_before=adaptive,
        bm25_after=bm25,
        bm25_before=bm25,
        dense_after=dense,
        dense_before=dense,
        k_after=3,
        k_before=3,
    )
    assert result["observed_combined_mean"] == pytest.approx(0.3)
    assert result["p_comb"] < 0.05


def test_combined_randomization_misaligned_vectors_rejected():
    with pytest.raises(ValueError, match="align"):
        combined_randomization_test(
            adaptive_after=[1.0],
            adaptive_before=[1.0],
            bm25_after=[0.5, 0.5],
            bm25_before=[0.5],
            dense_after=[1.0],
            dense_before=[1.0],
            k_after=1,
            k_before=0,
        )


def test_cluster_bootstrap_ci_constant_differences():
    """Identical per-query gains give a degenerate interval at the gain."""
    result = cluster_bootstrap_ci(
        adaptive_after=[0.8, 0.8, 0.8, 0.8, 0.8],
        adaptive_before=[0.7, 0.7, 0.7, 0.7, 0.7],
        bm25_after=[0.5, 0.5, 0.5, 0.5, 0.5],
        bm25_before=[0.5, 0.5, 0.5, 0.5, 0.5],
        n_boot=200,
    )
    assert result["point_estimate"] == pytest.approx(0.25)
    assert result["ci_low"] == pytest.approx(0.25)
    assert result["ci_high"] == pytest.approx(0.25)


def test_cluster_bootstrap_ci_deterministic_and_paired():
    kwargs = {
        "adaptive_after": [1.0, 0.0, 0.5],
        "adaptive_before": [0.5, 0.5, 1.0],
        "bm25_after": [0.0, 0.0, 0.5],
        "bm25_before": [0.5, 0.0, 0.5],
        "n_boot": 500,
    }
    first = cluster_bootstrap_ci(**kwargs)
    second = cluster_bootstrap_ci(**kwargs)
    assert first == second
    assert first["ci_low"] <= first["point_estimate"] <= first["ci_high"]
    with pytest.raises(ValueError, match="paired arms"):
        cluster_bootstrap_ci(
            adaptive_after=[1.0, 0.0],
            adaptive_before=[0.5],
            bm25_after=[0.0, 0.0],
            bm25_before=[0.5],
            n_boot=10,
        )


# --- dataset-expansion integrity ---------------------------------------------------


def test_duplicate_screen_flags_near_duplicates():
    references = [{"example_id": "old_1", "query": "What is RAG-Sequence?"}]
    candidates = [
        {"example_id": "new_1", "query": "What is RAG-Sequence"},  # ~identical
        {"example_id": "new_2", "query": "How does BM25 saturation work?"},
    ]
    flags = screen_duplicates(candidates, references)
    assert len(flags) == 1
    assert flags[0]["example_id"] == "new_1"
    assert flags[0]["jaccard"] >= DUPLICATE_JACCARD_THRESHOLD


def test_duplicate_screen_flags_within_set_repeats():
    candidates = [
        {"example_id": "a", "query": "Explain reciprocal rank fusion"},
        {"example_id": "b", "query": "Explain reciprocal rank fusion!"},
    ]
    flags = screen_duplicates(candidates, [])
    assert [f["example_id"] for f in flags] == ["b"]


def test_normalize_query_is_case_and_punctuation_insensitive():
    assert normalize_query("What is RAG?") == normalize_query("what is rag")
    assert token_jaccard("a b c", "a b c") == pytest.approx(1.0)
    assert token_jaccard("a b c", "x y z") == pytest.approx(0.0)


def test_quota_checker_accepts_exact_plan():
    queries = []
    for category, count in CATEGORY_QUOTAS.items():
        for i in range(count):
            queries.append(
                {"example_id": f"q_{category}_{i}", "category": category,
                 "relevant_documents": []}
            )
    assert len(queries) == POWERED_N_QUERIES
    # Three rotating papers per query overshoots every minimum (each paper
    # lands ~23 assignments against a maximum minimum of 18); the checker
    # enforces minimums, not exact counts.
    papers = list(PAPER_MINIMUMS)
    for i, query in enumerate(queries):
        query["relevant_documents"] = [
            papers[(i + k) % len(papers)] for k in range(3)
        ]
    report = check_quotas(queries)
    assert report["category_gaps"] == {}
    assert report["paper_gaps"] == {}
    assert report["satisfied"] is True


def test_quota_checker_rejects_wrong_size_and_category():
    queries = [
        {"example_id": "q1", "category": "factual", "relevant_documents": []}
    ]
    report = check_quotas(queries)
    assert report["satisfied"] is False
    assert report["n_queries"] == 1
    assert "factual" in report["category_gaps"]


# --- metric aggregation --------------------------------------------------------------


def test_adaptive_blend_and_confusion_taxonomy():
    decisions = [True, True, False, False]
    labels = [True, False, True, False]
    counts = confusion_counts(decisions, labels)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert classify_decision(True, True) == "true_escalation"
    assert classify_decision(False, True) == "false_escalation"
    assert classify_decision(True, False) == "missed_escalation"
    assert classify_decision(False, False) == "correct_stop"


# --- structural separation -----------------------------------------------------------------


def test_powered_code_has_no_retrieval_or_provider_imports():
    """Pre-routing integrity at the import level (extends the allowlist)."""
    text = (REPO_ROOT / "src" / "adaptive_rag" / "evaluation" / "powered.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "adaptive_rag.retrieval",
        "adaptive_rag.routing",
        "adaptive_rag.embeddings",
        "adaptive_rag.indexing",
        "qdrant",
        "AICredits",
        "groq",
    ):
        assert token.lower() not in text.lower(), token


def test_phase15_scripts_have_no_retrieval_or_provider_imports():
    """Curation/evaluation scripts must not reach into retrieval machinery."""
    for name in (
        "phase15_build_oracle.py",
        "phase15_evaluate.py",
        "phase15_figures.py",
    ):
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        lowered = text.lower()
        for token in (
            "adaptive_rag.retrieval",
            "adaptive_rag.routing",
            "adaptive_rag.embeddings",
            "adaptive_rag.indexing",
            "qdrant",
            "aicredits",
            "groq",
        ):
            assert token not in lowered, f"{name}: {token}"


# --- dataset-gate test-only accommodation ----------------------------------------------


def _write_mini_corpus(root: Path) -> tuple[Path, Path]:
    chunks_dir = root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "The first stage builds an index over the passage collection and "
        "the second stage reranks the candidates with a cross encoder."
    )
    record = {
        "chunk_id": "doc_x::structure_aware_v1::c00001",
        "document_id": "doc_x",
        "text": text,
        "metadata": {"section_path": ["1 Intro"]},
    }
    (chunks_dir / "doc_x.chunks.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    manifest = root / "papers.json"
    manifest.write_text(json.dumps([{"document_id": "doc_x"}]), encoding="utf-8")
    return chunks_dir, manifest


def _write_mini_dataset(root: Path) -> tuple[Path, Path]:
    dataset = root / "ds.jsonl"
    ledger = root / "ds.ledger.jsonl"
    quote = "The first stage builds an index over the passage collection"
    with open(dataset, "w", encoding="utf-8") as handle:
        for i in (1, 2):
            handle.write(
                json.dumps(
                    {
                        "example_id": f"t{i}",
                        "query": f"How is the index built in stage {i}?",
                        "reference_answer": (
                            "The first stage builds an index over the "
                            "passage collection."
                        ),
                        "relevant_documents": ["doc_x"],
                        "relevant_sections": [
                            {
                                "document_id": "doc_x",
                                "section_path_prefix": ["1 Intro"],
                            }
                        ],
                        "relevant_chunks": [],
                        "category": "factual",
                        "requires_multi_hop": False,
                        "notes": "",
                        "dataset_version": "mini",
                        "split": "test",
                    }
                )
                + "\n"
            )
    with open(ledger, "w", encoding="utf-8") as handle:
        for i in (1, 2):
            handle.write(
                json.dumps(
                    {
                        "example_id": f"t{i}",
                        "split": "test",
                        "relevant_documents": ["doc_x"],
                        "evidence": [
                            {
                                "document_id": "doc_x",
                                "chunk_id": "doc_x::structure_aware_v1::c00001",
                                "quote": quote,
                            }
                        ],
                        "answer_terms": ["index", "passage"],
                        "curator_notes": "grounded in the chunk text",
                    }
                )
                + "\n"
            )
    return dataset, ledger


def test_gate_rejects_test_only_by_default(tmp_path: Path):
    """S4 behavior is unchanged without the flag (pins the default)."""
    from scripts.validate_phase7_dataset import run_gate

    chunks_dir, manifest = _write_mini_corpus(tmp_path)
    dataset, ledger = _write_mini_dataset(tmp_path)
    report = run_gate(
        dataset,
        ledger,
        chunks_dir=chunks_dir,
        manifest_path=manifest,
        min_per_category=1,
    )
    assert report["summary"]["gate_open"] is False
    assert any(f["check"] == "S4" for f in report["failures"])


def test_gate_allows_test_only_with_flag(tmp_path: Path):
    """The pre-registered Phase 15 accommodation passes a test-only set."""
    from scripts.validate_phase7_dataset import run_gate

    chunks_dir, manifest = _write_mini_corpus(tmp_path)
    dataset, ledger = _write_mini_dataset(tmp_path)
    report = run_gate(
        dataset,
        ledger,
        chunks_dir=chunks_dir,
        manifest_path=manifest,
        min_per_category=1,
        allow_test_only=True,
    )
    assert report["summary"]["gate_open"] is True
    assert report["parameters"]["single_split_test_only"] is True
