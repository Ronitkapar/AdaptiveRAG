"""
ingestion package initialization.
"""

from adaptive_rag.ingestion.extractor import PdfExtractor, PdfplumberExtractor, RawChar, RawImage, RawPage, RawTable
from adaptive_rag.ingestion.normalizer import normalize_pages_to_document
from adaptive_rag.ingestion.pipeline import IngestionPipeline

__all__ = [
    "IngestionPipeline",
    "PdfExtractor",
    "PdfplumberExtractor",
    "RawChar",
    "RawImage",
    "RawPage",
    "RawTable",
    "normalize_pages_to_document",
]
