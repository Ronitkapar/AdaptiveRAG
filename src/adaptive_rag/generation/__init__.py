"""
generation package initialization.
"""

from adaptive_rag.generation.base import Generator
from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.generation.groq import GroqGenerator
from adaptive_rag.generation.prompts import (
    CONTEXT_TEMPLATE_V1,
    JUDGE_PROMPT_V1,
    SYSTEM_PROMPT_V1,
)

__all__ = [
    "CONTEXT_TEMPLATE_V1",
    "ContextBuilder",
    "Generator",
    "GroqGenerator",
    "JUDGE_PROMPT_V1",
    "SYSTEM_PROMPT_V1",
]
