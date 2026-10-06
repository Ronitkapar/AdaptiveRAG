"""
tests.test_dataset_split
------------------------
Calibration/test split contract for the evaluation benchmark.

Purely offline and deterministic: records are constructed in-process, plus one
read of the committed dense_eval_v1.jsonl.
"""

import json

import pytest
from pydantic import ValidationError

from adaptive_rag.errors import EvaluationError
from adaptive_rag.evaluation import (
    DEFAULT_DATASET_PATH,
    SPLIT_CALIBRATION,
    SPLIT_TEST,
    load_evaluation_dataset,
    partition_by_split,
    validate_split_separation,
)
from adaptive_rag.schemas import EVAL_SPLITS, EvaluationExample

LEGACY_RECORD = {
    "example_id": "dense_eval_001",
    "query": "What problem does dense retrieval solve in RAG?",
    "reference_answer": "It addresses lexical mismatch.",
    "relevant_documents": ["doc_a"],
    "category": "factual",
}


def _record(example_id: str, **overrides) -> dict:
    payload = dict(LEGACY_RECORD)
    payload["example_id"] = example_id
    payload.update(overrides)
    return payload


def test_legacy_record_without_split_key_still_validates():
    """A record written before the split existed must load unchanged."""
    assert "split" not in LEGACY_RECORD

    example = EvaluationExample.model_validate(LEGACY_RECORD)

    assert example.split == SPLIT_CALIBRATION
    assert example.split != SPLIT_TEST


def test_explicit_splits_round_trip():
    assert EvaluationExample.model_validate(_record("e1")).split == SPLIT_CALIBRATION
    assert (
        EvaluationExample.model_validate(_record("e2", split=SPLIT_CALIBRATION)).split
        == SPLIT_CALIBRATION
    )
    assert (
        EvaluationExample.model_validate(_record("e3", split=SPLIT_TEST)).split == SPLIT_TEST
    )


def test_invalid_split_value_is_rejected():
    with pytest.raises(ValidationError):
        EvaluationExample.model_validate(_record("e1", split="dev"))


def test_eval_splits_constant_matches_schema_literal():
    assert set(EVAL_SPLITS) == {SPLIT_CALIBRATION, SPLIT_TEST}


def test_partition_by_split_is_correct_and_order_preserving():
    examples = [
        EvaluationExample.model_validate(_record("c1")),
        EvaluationExample.model_validate(_record("t1", split=SPLIT_TEST)),
        EvaluationExample.model_validate(_record("c2", split=SPLIT_CALIBRATION)),
        EvaluationExample.model_validate(_record("t2", split=SPLIT_TEST)),
    ]

    calibration, test = partition_by_split(examples)

    assert [ex.example_id for ex in calibration] == ["c1", "c2"]
    assert [ex.example_id for ex in test] == ["t1", "t2"]
    assert calibration[0] is examples[0]
    assert calibration[1] is examples[2]
    assert test[0] is examples[1]
    assert test[1] is examples[3]
    assert len(calibration) + len(test) == len(examples)


def test_partition_of_empty_and_single_split_lists():
    examples = [EvaluationExample.model_validate(_record("c1"))]
    assert partition_by_split([]) == ([], [])
    assert partition_by_split(examples) == (examples, [])


def test_validate_split_separation_accepts_disjoint_splits():
    examples = [
        EvaluationExample.model_validate(_record("c1")),
        EvaluationExample.model_validate(_record("t1", split=SPLIT_TEST)),
    ]
    assert validate_split_separation(examples) is None


def test_validate_split_separation_rejects_cross_split_overlap():
    examples = [
        EvaluationExample.model_validate(_record("dup")),
        EvaluationExample.model_validate(_record("dup", split=SPLIT_TEST)),
    ]

    with pytest.raises(EvaluationError) as excinfo:
        validate_split_separation(examples)

    message = str(excinfo.value)
    assert "dup" in message
    assert SPLIT_CALIBRATION in message
    assert SPLIT_TEST in message


def test_validate_split_separation_rejects_within_split_duplicates():
    for split in (SPLIT_CALIBRATION, SPLIT_TEST):
        examples = [
            EvaluationExample.model_validate(_record("dup", split=split)),
            EvaluationExample.model_validate(_record("dup", split=split)),
        ]
        with pytest.raises(EvaluationError) as excinfo:
            validate_split_separation(examples)
        assert "dup" in str(excinfo.value)
        assert split in str(excinfo.value)


def test_real_dataset_loads_and_all_records_are_calibration_by_default():
    dataset = load_evaluation_dataset()

    assert len(dataset) == 20
    assert all(ex.split == SPLIT_CALIBRATION for ex in dataset)
    assert validate_split_separation(dataset) is None
    assert partition_by_split(dataset) == (dataset, [])


def test_real_dataset_file_carries_no_explicit_split_key():
    """The committed records are unchanged; the default is what supplies the split."""
    records = [
        json.loads(line)
        for line in DEFAULT_DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 20
    assert all("split" not in r for r in records)
