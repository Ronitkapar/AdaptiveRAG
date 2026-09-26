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

DEFAULT_DATASET_PATH = EVALUATION_DIR / "dense_eval_v1.jsonl"

VALID_CATEGORIES = {
    "factual",
    "terminology",
    "conceptual",
    "comparative",
    "multi_document",
    "fine_grained",
}


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
