"""
reranking package initialization.

Second-stage passage scoring, isolated from every first-stage strategy.
"""

from adaptive_rag.reranking.base import Reranker
from adaptive_rag.reranking.onnx_backend import OnnxCrossEncoderReranker

__all__ = [
    "OnnxCrossEncoderReranker",
    "Reranker",
]