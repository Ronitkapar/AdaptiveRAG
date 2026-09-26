"""
generation.base
---------------
Protocol definition for text generators.
"""

from typing import Protocol

from adaptive_rag.schemas import GenerationRequest, GenerationResult


class Generator(Protocol):
    """Protocol for answer generators."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an answer grounded in the provided retrieved context."""
        ...
