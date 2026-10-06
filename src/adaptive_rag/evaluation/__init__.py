"""
evaluation package initialization.
"""

from adaptive_rag.evaluation.base import (
    Evaluator,
    build_error_breakdown,
    count_errors,
    failed_traces,
    mean,
    metric,
    percentile,
)
from adaptive_rag.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    SPLIT_CALIBRATION,
    SPLIT_TEST,
    load_evaluation_dataset,
    partition_by_split,
    relevant_chunk_ids,
    validate_against_corpus,
    validate_split_separation,
)
from adaptive_rag.evaluation.efficiency import EfficiencyEvaluator
from adaptive_rag.evaluation.generation import (
    EVALUATOR_VERSION as GENERATION_EVALUATOR_VERSION,
    GenerationEvaluator,
    rouge_l,
    token_f1,
)
from adaptive_rag.evaluation.judge import GroqLLMJudge, JudgeCache, judge_cache_key
from adaptive_rag.evaluation.rows import (
    COLUMNS as ROW_COLUMNS,
    ROWS_VERSION,
    EvaluationRow,
    build_rows,
    read_rows_jsonl,
    row_to_record,
    rows_to_csv,
    rows_to_jsonl,
    write_rows,
)
from adaptive_rag.evaluation.retrieval import RetrievalEvaluator
from adaptive_rag.evaluation.routing import RoutingEvaluator

__all__ = [
    "DEFAULT_DATASET_PATH",
    "EvaluationRow",
    "EfficiencyEvaluator",
    "Evaluator",
    "GENERATION_EVALUATOR_VERSION",
    "GenerationEvaluator",
    "GroqLLMJudge",
    "JudgeCache",
    "ROW_COLUMNS",
    "ROWS_VERSION",
    "RetrievalEvaluator",
    "RoutingEvaluator",
    "SPLIT_CALIBRATION",
    "SPLIT_TEST",
    "build_error_breakdown",
    "build_rows",
    "count_errors",
    "failed_traces",
    "judge_cache_key",
    "load_evaluation_dataset",
    "mean",
    "metric",
    "partition_by_split",
    "percentile",
    "read_rows_jsonl",
    "relevant_chunk_ids",
    "row_to_record",
    "rows_to_csv",
    "rows_to_jsonl",
    "rouge_l",
    "token_f1",
    "validate_against_corpus",
    "validate_split_separation",
    "write_rows",
]
