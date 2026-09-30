"""
retrieval package initialization.
"""

from adaptive_rag.retrieval.base import Retriever
from adaptive_rag.retrieval.bm25 import BM25Retriever
from adaptive_rag.retrieval.dense import DenseRetriever
from adaptive_rag.retrieval.fusion import reciprocal_rank_fusion
from adaptive_rag.retrieval.hybrid import HybridRetriever
from adaptive_rag.retrieval.reranked import RerankedRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankedRetriever",
    "Retriever",
    "reciprocal_rank_fusion",
]
