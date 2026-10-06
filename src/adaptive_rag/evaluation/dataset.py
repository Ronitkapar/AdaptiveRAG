"""
evaluation.dataset
------------------
Benchmark dataset loading, validation, and retriever-independent relevance rules.
"""

import json
from pathlib import Path

from adaptive_rag.config.paths import EVALUATION_DIR, PAPERS_MANIFEST_PATH
from adaptive_rag.errors import EvaluationError
from adaptive_rag.schemas import EvaluationExample, ExperimentTrace

# The Phase 2-6 benchmark set: 20 records, one per labelled document, written
# before the calibration/test split existed. Phases 2-6 reproducibility depends on
# it being loadable by this exact path, so it is kept, named, and untouched.
LEGACY_DENSE_DATASET_PATH = EVALUATION_DIR / "dense_eval_v1.jsonl"

# The frozen Phase 7 benchmark set (107 records: 47 calibration, 60 test). This is
# the set every Phase 7 measurement is defined over.
PHASE7_DATASET_PATH = EVALUATION_DIR / "phase7_eval_v1.jsonl"

# Still the legacy 20-record set, deliberately. It is the default *argument* of
# `load_evaluation_dataset`, which the Phase 2-6 callers and their tests call with
# no argument and expect 20 records from; silently repointing it would change what
# those phases measure. Safety for Phase 7 is enforced where benchmarking actually
# happens instead -- `scripts/run_phase7_suite.py` requires `--dataset` and refuses
# the legacy file unless it is asked for by name -- so omitting it can no longer
# select the wrong set by accident.
DEFAULT_DATASET_PATH = LEGACY_DENSE_DATASET_PATH

VALID_CATEGORIES = {
    "factual",
    "terminology",
    "conceptual",
    "comparative",
    "multi_document",
    "fine_grained",
}

# Split names, mirroring schemas.experiment.EVAL_SPLITS. Kept as named constants
# so call sites read as intent rather than as bare strings.
SPLIT_CALIBRATION = "calibration"
SPLIT_TEST = "test"

VALID_SPLITS = (SPLIT_CALIBRATION, SPLIT_TEST)


def is_legacy_dataset(path: Path) -> bool:
    """Whether `path` names the Phase 2-6 20-record set.

    Compared by resolved path so a relative path, a symlink or a differently
    spelled copy of the same file is still recognised as the legacy set.
    """
    try:
        return path.resolve() == LEGACY_DENSE_DATASET_PATH.resolve()
    except OSError:  # pragma: no cover -- an unresolvable path is not the legacy file
        return False


def load_evaluation_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[EvaluationExample]:
    """Load and validate an evaluation dataset from JSONL."""
    if not path.is_file():
        raise EvaluationError(f"Evaluation dataset not found: {path}")

    examples: list[EvaluationExample] = []
    seen_ids: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                examples.append(EvaluationExample.model_validate_json(line))
            except Exception as exc:
                raise EvaluationError(f"Invalid dataset record at line {line_no}: {exc}") from exc

    if not examples:
        raise EvaluationError(f"Evaluation dataset is empty: {path}")

    for ex in examples:
        if ex.example_id in seen_ids:
            raise EvaluationError(f"Duplicate example_id in dataset: {ex.example_id}")
        seen_ids.add(ex.example_id)
        if not ex.relevant_documents:
            raise EvaluationError(f"{ex.example_id}: relevant_documents must not be empty")
        if ex.category not in VALID_CATEGORIES:
            raise EvaluationError(f"{ex.example_id}: invalid category '{ex.category}'")

    validate_against_corpus(examples)
    return examples


def validate_against_corpus(
    examples: list[EvaluationExample],
    manifest_path: Path = PAPERS_MANIFEST_PATH,
) -> None:
    """Ensure every labelled document_id exists in the Phase 1 corpus manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known_ids = {p["document_id"] for p in manifest}

    for ex in examples:
        unknown = [d for d in ex.relevant_documents if d not in known_ids]
        if unknown:
            raise EvaluationError(
                f"{ex.example_id}: unknown relevant document(s) {unknown}; "
                f"not present in {manifest_path.name}"
            )
        unknown_secs = [
            ref.document_id for ref in ex.relevant_sections if ref.document_id not in known_ids
        ]
        if unknown_secs:
            raise EvaluationError(
                f"{ex.example_id}: unknown section reference document(s) {unknown_secs}"
            )


def partition_by_split(
    examples: list[EvaluationExample],
) -> tuple[list[EvaluationExample], list[EvaluationExample]]:
    """Split examples into (calibration, test), preserving the input order.

    No label semantics are applied here: membership is exactly the `split` field,
    which defaults to "calibration" for records written before the split existed.
    """
    calibration: list[EvaluationExample] = []
    test: list[EvaluationExample] = []
    for ex in examples:
        if ex.split == SPLIT_TEST:
            test.append(ex)
        else:
            calibration.append(ex)
    return calibration, test


def validate_split_separation(examples: list[EvaluationExample]) -> None:
    """Ensure an example_id is unique overall and confined to a single split.

    Duplicate ids inside one split double-count that query; the same id in both
    splits is the calibration/test leakage Phase 7 must not measure.
    """
    seen: dict[str, str] = {}
    for ex in examples:
        previous = seen.get(ex.example_id)
        if previous is None:
            seen[ex.example_id] = ex.split
            continue
        if previous == ex.split:
            raise EvaluationError(
                f"Duplicate example_id in '{ex.split}' split: {ex.example_id}"
            )
        raise EvaluationError(
            f"example_id {ex.example_id} appears in both '{previous}' and "
            f"'{ex.split}' splits; calibration and test must be disjoint"
        )


def relevant_chunk_ids(
    example: EvaluationExample,
    trace: ExperimentTrace,
) -> set[str]:
    """Derive chunk-level relevance for a trace using a retriever-independent rule.

    A retrieved chunk counts as relevant when its document is labelled relevant
    and, when section labels exist for that document, its section path matches
    one of the labelled prefixes. Labels never depend on retrieval output.
    """
    if not trace.retrieval:
        return set()

    section_prefixes = [
        ref.section_path_prefix
        for ref in example.relevant_sections
        if ref.document_id in example.relevant_documents
    ]

    relevant: set[str] = set()
    for result in trace.retrieval.results:
        if result.metadata.document_id not in example.relevant_documents:
            continue
        if section_prefixes:
            path = result.metadata.section_path
            if not any(path[: len(prefix)] == prefix for prefix in section_prefixes):
                continue
        relevant.add(result.chunk_id)
    return relevant
