"""
embeddings package initialization.
"""

from adaptive_rag.embeddings.aicredits import AICreditsEmbeddingModel
from adaptive_rag.embeddings.base import ActiveEmbeddingInfo, EmbeddingModel
from adaptive_rag.embeddings.cache import EmbeddingCache, derive_cache_key
from adaptive_rag.embeddings.pipeline import EmbeddingPipeline

__all__ = [
    "AICreditsEmbeddingModel",
    "ActiveEmbeddingInfo",
    "EmbeddingCache",
    "EmbeddingModel",
    "EmbeddingPipeline",
    "derive_cache_key",
]
