"""
fakes.py
--------
Deterministic offline fake components for unit testing without live API keys.
"""

import hashlib
import numpy as np
from typing import Any, Sequence

from adaptive_rag.schemas import (
    Chunk,
    ContextChunk,
    GenerationRequest,
    GenerationResult,
    RetrievalMetadata,
    RetrievalResponse,
    RetrievalResult,
    TokenUsage,
)


class FakeEmbeddingModel:
    """Generates deterministic pseudo-random unit vectors based on text hash."""

    def __init__(self, dimension: int = 64, normalize: bool = True):
        self.dimension = dimension
        self.normalize = normalize
        self.model_id = "fake-embedding-model"
        self.config_hash = "fake_cfg_hash"
        self.call_count = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        # Hash query to seed a deterministic vector
        seed = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec.tolist()


class FakeGenerator:
    """Generates deterministic answers citing retrieved context sources."""

    def __init__(self, model_name: str = "fake-groq-model"):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1
        citations = []
        source_ids = []
        for ctx in request.context:
            citations.append(f"[Source {ctx.source_index}]")
            source_ids.append(ctx.chunk_id)

        cite_str = " ".join(citations) if citations else "[No sources cited]"
        answer = f"Synthesized answer for '{request.query}'. Verified against {cite_str}."

        return GenerationResult(
            answer=answer,
            source_chunk_ids=source_ids,
            model=self.model_name,
            usage=TokenUsage(input_tokens=150, output_tokens=40, total_tokens=190),
            latency_ms=45.2,
            status="ok",
            generator_version="fake_v1",
            prompt_version="fake_prompt_v1",
        )
