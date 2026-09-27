"""
schemas.config
--------------
Pydantic configuration models for all pipeline stages.
Leaf models with no downstream dependencies to avoid circular imports.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class IngestionConfig(BaseModel):
    """Configuration for PDF extraction and document ingestion."""

    model_config = ConfigDict(extra="forbid")

    extractor_backend: Literal["pdfplumber"] = "pdfplumber"
    ingestion_version: str = "ingestion_v2"
    min_page_chars_warning: int = 100
    extract_tables: bool = True
    extract_figures: bool = True
    extract_equations: bool = True


class ChunkingConfig(BaseModel):
    """Configuration for structure-aware document chunking."""

    model_config = ConfigDict(extra="forbid")

    chunking_version: str = "structure_aware_v1"
    target_tokens: int = 500
    max_tokens: int = 800
    min_tokens: int = 150
    overlap_tokens: int = 80
    include_section_header: bool = True
    include_reference_sections: bool = True
    merge_undersized_sections: bool = True


class EmbeddingConfig(BaseModel):
    """Configuration for text embeddings."""

    model_config = ConfigDict(extra="forbid")

    embedding_pipeline_version: str = "embedding_v1"
    provider: Literal["aicredits"] = "aicredits"
    model_id: str = "text-embedding-3-large"
    dimensions: int | None = None
    normalize: bool = True
    batch_size: int = 50
    max_retries: int = 4


class IndexConfig(BaseModel):
    """Configuration for vector store indexing."""

    model_config = ConfigDict(extra="forbid")

    index_version: str = "index_v1"
    collection_name: str = "adaptiverag_dense_v1"
    distance: Literal["cosine", "dot", "euclidean"] = "cosine"
    batch_size: int = 128


class DenseRetrievalConfig(BaseModel):
    """Configuration specifically for dense vector retrieval."""

    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "dense_v1"
    retrieval_method: Literal["dense"] = "dense"
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None


class BM25RetrievalConfig(BaseModel):
    """Configuration specifically for BM25 lexical retrieval."""

    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "bm25_v1"
    retrieval_method: Literal["bm25"] = "bm25"
    k1: float = 1.2
    b: float = 0.75
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None


class RetrievalConfig(BaseModel):
    """Configuration for retrieval (dense or bm25)."""

    model_config = ConfigDict(extra="forbid")

    retriever_version: str = "dense_v1"
    retrieval_method: Literal["dense", "bm25"] = "dense"
    top_k: int = 10
    score_threshold: float | None = None
    filters: dict[str, Any] | None = None
    # BM25-specific parameters
    k1: float = 1.2
    b: float = 0.75

    @model_validator(mode="after")
    def validate_method_and_version(self) -> "RetrievalConfig":
        if self.retrieval_method == "bm25" and self.retriever_version == "dense_v1":
            # Auto-align default dense version to bm25 version
            object.__setattr__(self, "retriever_version", "bm25_v1")
        elif self.retrieval_method == "dense" and self.retriever_version == "bm25_v1":
            object.__setattr__(self, "retriever_version", "dense_v1")
        return self


class ContextConfig(BaseModel):
    """Configuration for retrieved context assembly."""

    model_config = ConfigDict(extra="forbid")

    context_template_version: str = "context_v1"
    max_context_tokens: int = 6000
    max_chunks: int = 5


class GenerationConfig(BaseModel):
    """Configuration for Groq answer generation."""

    model_config = ConfigDict(extra="forbid")

    generator_version: str = "groq_v1"
    model: str = "openai/gpt-oss-120b"
    prompt_version: str = "generation_prompt_v1"
    temperature: float = 0.0
    max_completion_tokens: int = 1024
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"


class EvaluationConfig(BaseModel):
    """Configuration for retrieval, generation, and efficiency evaluation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_version: str = "eval_v1"
    k_grid: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    judge_model: str = "openai/gpt-oss-20b"
    judge_prompt_version: str = "judge_prompt_v1"
    enable_llm_judge: bool = True
    use_judge_cache: bool = True
