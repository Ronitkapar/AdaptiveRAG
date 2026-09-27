"""
retrieval.base
--------------
Retriever protocol definition.
Keeps dense and future retrieval strategies cleanly interchangeable.
"""

from typing import Any, Protocol, runtime_checkable

from adaptive_rag.schemas import RetrievalResponse


@runtime_checkable
class Retriever(Protocol):
    """Protocol for document retrievers."""

    method: str

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Retrieve ranked chunks for a query."""
        ...
