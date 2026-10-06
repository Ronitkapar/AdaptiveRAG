"""
embeddings.pipeline
-------------------
Embeds chunks using EmbeddingModel and EmbeddingCache.
Saves canonical embeddings JSONL and manifest.
"""

import base64
import json
from pathlib import Path
import struct
from typing import Sequence

from adaptive_rag.config.hashing import canonical_json, compute_config_hash, compute_sha256
from adaptive_rag.config.paths import CHUNKS_DIR, EMBEDDINGS_DIR
from adaptive_rag.embeddings.base import EmbeddingModel
from adaptive_rag.embeddings.cache import EmbeddingCache, derive_cache_key
from adaptive_rag.schemas import Chunk, EmbeddingConfig


def _encode_vector_b64(vec: Sequence[float]) -> str:
    packed = struct.pack(f"{len(vec)}f", *vec)
    return base64.b64encode(packed).decode("ascii")


def _decode_vector_b64(b64_str: str) -> list[float]:
    raw = base64.b64decode(b64_str.encode("ascii"))
    num_floats = len(raw) // 4
    return list(struct.unpack(f"{num_floats}f", raw))


class EmbeddingPipeline:
    """Manages chunk embedding with caching and artifact persistence."""

    def __init__(
        self,
        model: EmbeddingModel,
        cache: EmbeddingCache | None = None,
        config: EmbeddingConfig | None = None,
    ):
        self.model = model
        self.cache = cache or EmbeddingCache()
        self.config = config or EmbeddingConfig()

    def embed_chunks(
        self,
        chunks: list[Chunk],
        base_url_host: str = "api.aicredits.in",
    ) -> list[list[float]]:
        """Embed a list of chunks, leveraging cache hits and updating misses."""
        if not chunks:
            return []

        results: list[list[float] | None] = [None] * len(chunks)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        miss_keys: list[str] = []

        for idx, c in enumerate(chunks):
            key = derive_cache_key(
                text=c.text,
                model_id=self.model.model_id,
                base_url_host=base_url_host,
                dimensions=self.config.dimensions,
                normalize=self.model.normalize,
                pipeline_version=self.config.embedding_pipeline_version,
            )
            cached_vec = self.cache.get(key)
            if cached_vec is not None:
                results[idx] = cached_vec
            else:
                miss_indices.append(idx)
                miss_texts.append(c.text)
                miss_keys.append(key)

        # Batch embed misses
        if miss_texts:
            fresh_vectors = self.model.embed(miss_texts)
            for m_idx, key, vec, c_idx in zip(miss_indices, miss_keys, fresh_vectors, miss_indices):
                results[c_idx] = vec
                self.cache.put(
                    cache_key=key,
                    vector=vec,
                    chunk_id=chunks[c_idx].chunk_id,
                    model_id=self.model.model_id,
                    text_sha256=compute_sha256(chunks[c_idx].text),
                )

        return [r for r in results if r is not None]

    def process_corpus_embeddings(
        self,
        chunks_dir: Path = CHUNKS_DIR,
        out_dir: Path = EMBEDDINGS_DIR,
        only_id: str | None = None,
    ) -> dict[str, int]:
        """Embed all chunk JSONL artifacts in chunks_dir and save embeddings JSONL."""
        chunk_files = sorted(list(chunks_dir.glob("*.chunks.jsonl")))
        if only_id:
            chunk_files = [f for f in chunk_files if f.name.startswith(only_id)]

        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {}

        for cf in chunk_files:
            doc_id = cf.name.replace(".chunks.jsonl", "")
            chunks: list[Chunk] = []
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(Chunk.model_validate_json(line))

            vectors = self.embed_chunks(chunks)
            dest_file = out_dir / f"{doc_id}.embeddings.jsonl"
            with open(dest_file, "w", encoding="utf-8") as f_out:
                for c, v in zip(chunks, vectors):
                    record = {
                        "chunk_id": c.chunk_id,
                        "model_id": self.model.model_id,
                        "dimension": len(v),
                        "vector_b64": _encode_vector_b64(v),
                    }
                    f_out.write(canonical_json(record) + "\n")

            summary[doc_id] = len(vectors)

        # Beside the embeddings it describes, not at a fixed location: Phase 8
        # embeds into its own directory while Phase 7's artifacts stay in place as
        # the study's "before" arm, and one corpus's manifest must not overwrite
        # the other's.
        manifest_path = out_dir.parent / "embeddings_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            canonical_json({
                "model_id": self.model.model_id,
                "dimension": self.model.dimension,
                "normalized": self.model.normalize,
                "pipeline_version": self.config.embedding_pipeline_version,
                "total_embedded_chunks": sum(summary.values()),
                "documents": summary,
            }),
            encoding="utf-8",
        )

        return summary
