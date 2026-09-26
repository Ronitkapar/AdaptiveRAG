"""
rag.py
------
DenseRAGPipeline: Runtime question-answering composition root.
Connects Retriever -> ContextBuilder -> Generator.
Deliberately isolated from evaluation and experiment reporting.
"""

from typing import Any

from adaptive_rag.generation.base import Generator
from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.schemas import (
    GenerationRequest,
    GenerationResult,
    RetrievalResponse,
)


class DenseRAGPipeline:
    """Fixed Dense-RAG runtime pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder | None = None,
        generator: Generator | None = None,
    ):
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Execute end-to-end question answering."""
        # 1. Retrieve
        retrieval_response: RetrievalResponse = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # 2. Build Context
        generation_request: GenerationRequest = self.context_builder.build(
            retrieval_response
        )

        # 3. Generate Answer
        if self.generator is None:
            return {
                "question": question,
                "retrieval": retrieval_response.model_dump(mode="json"),
                "context": [c.model_dump(mode="json") for c in generation_request.context],
                "answer": None,
            }

        generation_result: GenerationResult = self.generator.generate(
            generation_request
        )

        return {
            "question": question,
            "answer": generation_result.answer,
            "cited_source_ids": generation_result.source_chunk_ids,
            "model": generation_result.model,
            "latency_ms": {
                "retrieval": retrieval_response.retrieval_metadata.latency_ms,
                "generation": generation_result.latency_ms,
                "total": retrieval_response.retrieval_metadata.latency_ms
                + generation_result.latency_ms,
            },
            "tokens": generation_result.usage.model_dump(mode="json"),
            "retrieval": retrieval_response.model_dump(mode="json"),
            "context_chunks_used": len(generation_request.context),
        }
