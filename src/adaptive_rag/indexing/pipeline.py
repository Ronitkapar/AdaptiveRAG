"""
indexing.pipeline
-----------------
Pipeline for building Qdrant index from chunk and embedding artifacts.
Enforces vector dimension guards and records index metadata.
"""

import base64
import json
from pathlib import Path
import struct

from adaptive_rag.config.hashing import canonical_json
from adaptive_rag.config.paths import CHUNKS_DIR, EMBEDDINGS_DIR, QDRANT_DIR
from adaptive_rag.indexing.base import VectorStore
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.schemas import Chunk, IndexConfig


def _decode_vector_b64(b64_str: str) -> list[float]:
    raw = base64.b64decode(b64_str.encode("ascii"))
    num_floats = len(raw) // 4
    return list(struct.unpack(f"{num_floats}f", raw))


class IndexingPipeline:
    """Manages loading chunk/embedding artifacts and upserting into the VectorStore."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        config: IndexConfig | None = None,
    ):
        self.config = config or IndexConfig()
        self.vector_store = vector_store or QdrantVectorStore(config=self.config)

    def build_index_from_artifacts(
        self,
        chunks_dir: Path = CHUNKS_DIR,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        recreate: bool = False,
    ) -> int:
        """Read chunks and embeddings from disk and upsert into collection."""
        emb_files = sorted(list(embeddings_dir.glob("*.embeddings.jsonl")))
        if not emb_files:
            return 0

        # Read first vector to infer dimension
        sample_line = emb_files[0].read_text(encoding="utf-8").splitlines()[0]
        sample_rec = json.loads(sample_line)
        dim = sample_rec["dimension"]
        model_id = sample_rec["model_id"]

        self.vector_store.ensure_collection(
            dim=dim,
            distance=self.config.distance,
            recreate=recreate,
        )

        total_upserted = 0
        batch_size = self.config.batch_size

        for ef in emb_files:
            doc_id = ef.name.replace(".embeddings.jsonl", "")
            cf = chunks_dir / f"{doc_id}.chunks.jsonl"
            if not cf.is_file():
                continue

            chunks: list[Chunk] = []
            with open(cf, "r", encoding="utf-8") as f_c:
                for line in f_c:
                    if line.strip():
                        chunks.append(Chunk.model_validate_json(line))

            vectors: list[list[float]] = []
            with open(ef, "r", encoding="utf-8") as f_e:
                for line in f_e:
                    if line.strip():
                        rec = json.loads(line)
                        vectors.append(_decode_vector_b64(rec["vector_b64"]))

            # Batch upsert
            for i in range(0, len(chunks), batch_size):
                c_batch = chunks[i : i + batch_size]
                v_batch = vectors[i : i + batch_size]
                total_upserted += self.vector_store.upsert(c_batch, v_batch)

        # Save index metadata
        index_meta_path = Path(QDRANT_DIR) / "index_meta.json"
        index_meta_path.parent.mkdir(parents=True, exist_ok=True)
        index_meta_path.write_text(
            canonical_json({
                "collection_name": self.config.collection_name,
                "vector_dimension": dim,
                "embedding_model_id": model_id,
                "distance": self.config.distance,
                "total_points": total_upserted,
                "index_version": self.config.index_version,
            }),
            encoding="utf-8",
        )

        return total_upserted
