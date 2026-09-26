"""
indexing.qdrant
---------------
Local Qdrant vector store adapter.
Supports embedded path mode (default) and optional server URL.
Stores structured chunk payload and enforces index configuration guards.
"""

from pathlib import Path
from typing import Any, Sequence
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from adaptive_rag.config.settings import Settings, settings as default_settings
from adaptive_rag.errors import IndexConfigMismatchError, IndexUnavailableError
from adaptive_rag.indexing.base import IndexMetadata, ScoredPoint
from adaptive_rag.schemas import Chunk, IndexConfig

NAMESPACE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID from chunk_id for Qdrant point IDs."""
    return str(uuid.uuid5(NAMESPACE_UUID, chunk_id))


class QdrantVectorStore:
    """Qdrant vector store implementation with embedded local file storage."""

    def __init__(
        self,
        config: IndexConfig | None = None,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
    ):
        self.config = config or IndexConfig()
        self.settings = settings or default_settings
        self.collection_name = self.config.collection_name

        if client is not None:
            self.client = client
        else:
            try:
                if self.settings.QDRANT_MODE == "server" and self.settings.QDRANT_URL:
                    self.client = QdrantClient(
                        url=self.settings.QDRANT_URL,
                        api_key=self.settings.QDRANT_API_KEY,
                    )
                else:
                    path = Path(self.settings.QDRANT_PATH)
                    path.mkdir(parents=True, exist_ok=True)
                    self.client = QdrantClient(path=str(path))
            except Exception as exc:
                raise IndexUnavailableError(f"Failed to initialize Qdrant client: {exc}") from exc

    def ensure_collection(
        self,
        dim: int,
        distance: str = "cosine",
        recreate: bool = False,
    ) -> None:
        """Create or verify the Qdrant collection with expected dimension."""
        dist_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclidean": models.Distance.EUCLID,
        }
        q_dist = dist_map.get(distance.lower(), models.Distance.COSINE)

        collections = [c.name for c in self.client.get_collections().collections]
        exists = self.collection_name in collections

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dim, distance=q_dist),
            )
        else:
            # Dimension check guard
            info = self.client.get_collection(self.collection_name)
            current_dim = info.config.params.vectors.size
            if current_dim != dim:
                raise IndexConfigMismatchError(
                    f"Collection '{self.collection_name}' has dimension {current_dim}, "
                    f"but expected {dim}. Pass recreate=True to rebuild."
                )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Upsert a batch of chunks and their vectors into Qdrant."""
        if not chunks:
            return 0

        points: list[models.PointStruct] = []
        for c, v in zip(chunks, vectors):
            point_id = _chunk_id_to_uuid(c.chunk_id)
            payload = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "section_path": c.metadata.section_path,
                "headings": c.metadata.headings,
                "element_ids": c.metadata.element_ids,
                "element_types": c.metadata.element_types,
                "page_start": c.metadata.page_start,
                "page_end": c.metadata.page_end,
                "token_count": c.metadata.token_count,
                "doc_title": c.metadata.doc_title,
                "source_sha256": c.provenance.source_sha256,
                "chunking_version": c.chunking_metadata.chunking_version,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=list(v),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        """Search nearest points using cosine vector search."""
        q_filter = None
        if filters:
            conditions = []
            for k, val in filters.items():
                if isinstance(val, list):
                    conditions.append(models.FieldCondition(key=k, match=models.MatchAny(any=val)))
                else:
                    conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=val)))
            if conditions:
                q_filter = models.Filter(must=conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=q_filter,
            with_payload=True,
        )

        results: list[ScoredPoint] = []
        for hit in response.points:
            chunk_id = hit.payload.get("chunk_id", str(hit.id)) if hit.payload else str(hit.id)
            results.append(
                ScoredPoint(
                    chunk_id=chunk_id,
                    score=float(hit.score),
                    payload=dict(hit.payload or {}),
                )
            )
        return results

    def count(self) -> int:
        """Count total vectors in collection."""
        try:
            return self.client.count(self.collection_name).count
        except Exception:
            return 0

    def close(self) -> None:
        """Close Qdrant client."""
        try:
            self.client.close()
        except Exception:
            pass
