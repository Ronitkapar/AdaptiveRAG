"""
errors.py
---------
Typed exception hierarchy for AdaptiveRAG.
Distinguishes retrieval failures, missing credentials, index errors,
and API errors from normal 'no results' or 'empty answer' states.
"""

from typing import Any


class AdaptiveRAGError(Exception):
    """Base exception for all AdaptiveRAG errors."""


class ConfigurationError(AdaptiveRAGError):
    """Raised when required configuration or settings are invalid."""


class MissingCredentialError(ConfigurationError):
    """Raised when a required API key or secret is missing."""


# --- Ingestion Errors ---


class IngestionError(AdaptiveRAGError):
    """Base exception for document ingestion failures."""


class CorpusIntegrityError(IngestionError):
    """Raised when a source PDF fails checksum or magic byte verification."""


class PdfExtractionError(IngestionError):
    """Raised when a PDF cannot be opened or parsed."""


# --- Chunking Errors ---


class ChunkingError(AdaptiveRAGError):
    """Raised when chunking fails or violates structural constraints."""


# --- Embedding Errors ---


class EmbeddingError(AdaptiveRAGError):
    """Base exception for embedding failures."""


class EmbeddingAPIError(EmbeddingError):
    """Raised when the embedding provider returns an API error (e.g. rate limit, bad request)."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding request times out."""


# --- Indexing & Vector Store Errors ---


class VectorStoreError(AdaptiveRAGError):
    """Base exception for vector store failures."""


class IndexUnavailableError(VectorStoreError):
    """Raised when the vector index cannot be connected to or loaded."""


class IndexConfigMismatchError(VectorStoreError):
    """Raised when existing collection dimensions or metadata conflict with current config."""


# --- Retrieval Errors ---


class RetrievalError(AdaptiveRAGError):
    """Raised when a retrieval query fails due to underlying component errors."""


class InvalidQueryError(RetrievalError):
    """Raised when a query is empty or violates retrieval preconditions."""


# --- Generation Errors ---


class GenerationError(AdaptiveRAGError):
    """Base exception for answer generation failures."""


class GenerationAPIError(GenerationError):
    """Raised when the generation provider returns an API error."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GenerationTimeoutError(GenerationError):
    """Raised when generation request times out."""


# --- Evaluation & Experiment Errors ---


class EvaluationError(AdaptiveRAGError):
    """Base exception for evaluation metric calculation failures."""


class ExperimentRunError(AdaptiveRAGError):
    """Raised when an experiment execution encounters an unrecoverable failure."""
