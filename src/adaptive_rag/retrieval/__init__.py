"""
retrieval package initialization.
"""

from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.retrieval.dense import DenseRetriever

__all__ = [
    "DenseRetriever",
    "Retriever",
]
