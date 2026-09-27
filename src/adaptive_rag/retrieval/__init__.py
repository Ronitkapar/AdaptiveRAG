"""
retrieval package initialization.
"""

from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.retrieval.bm25 import BM25Retriever
from adaptive_rag.retrieval.dense import DenseRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "Retriever",
]
