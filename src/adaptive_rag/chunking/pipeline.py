"""
chunking.pipeline
-----------------
Chunking pipeline orchestrating chunk generation across documents.
Saves deterministic chunk JSONL files and manifest statistics.
"""

import json
from pathlib import Path

from adaptive_rag.chunking.structure_aware import StructureAwareChunker
from adaptive_rag.config.hashing import canonical_json
from adaptive_rag.config.paths import CHUNKS_DIR, DOCUMENTS_DIR, PROCESSED_DATA_DIR, STATS_DIR
from adaptive_rag.schemas import Chunk, ChunkingConfig, Document


class ChunkingPipeline:
    """Manages chunking of processed Documents and persistence of derived chunks."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self.chunker = StructureAwareChunker(config=self.config)

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Chunk a single document."""
        return self.chunker.chunk_document(doc)

    def save_chunks(self, document_id: str, chunks: list[Chunk], out_dir: Path = CHUNKS_DIR) -> Path:
        """Persist chunks for a document as a JSONL file."""
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / f"{document_id}.chunks.jsonl"
        with open(dest_path, "w", encoding="utf-8") as f:
            for c in chunks:
                line = canonical_json(c.model_dump(mode="json"))
                f.write(line + "\n")
        return dest_path

    def chunk_corpus(
        self,
        docs_dir: Path = DOCUMENTS_DIR,
        out_dir: Path = CHUNKS_DIR,
        only_id: str | None = None,
    ) -> dict[str, list[Chunk]]:
        """Chunk all persisted Document JSON artifacts."""
        doc_files = sorted(list(docs_dir.glob("*.json")))
        if only_id:
            doc_files = [p for p in doc_files if p.stem == only_id]

        results: dict[str, list[Chunk]] = {}
        all_token_counts: list[int] = []
        doc_summaries = []

        for df in doc_files:
            doc_data = json.loads(df.read_text(encoding="utf-8"))
            doc = Document.model_validate(doc_data)
            chunks = self.chunk_document(doc)
            self.save_chunks(doc.document_id, chunks, out_dir=out_dir)
            results[doc.document_id] = chunks

            tokens = [c.metadata.token_count for c in chunks]
            all_token_counts.extend(tokens)
            doc_summaries.append({
                "document_id": doc.document_id,
                "chunk_count": len(chunks),
                "total_tokens": sum(tokens),
                "avg_tokens": round(sum(tokens) / len(chunks), 1) if chunks else 0,
            })

        # Save manifest
        manifest_path = PROCESSED_DATA_DIR / "chunks_manifest.json"
        manifest_path.write_text(
            canonical_json({
                "chunking_version": self.config.chunking_version,
                "total_documents": len(results),
                "total_chunks": sum(len(c) for c in results.values()),
                "config": self.config.model_dump(mode="json"),
                "documents": doc_summaries,
            }),
            encoding="utf-8",
        )

        # Save stats
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        stats_path = STATS_DIR / "chunk_stats.json"
        if all_token_counts:
            sorted_t = sorted(all_token_counts)
            stats_path.write_text(
                canonical_json({
                    "count": len(all_token_counts),
                    "min": sorted_t[0],
                    "max": sorted_t[-1],
                    "median": sorted_t[len(sorted_t) // 2],
                    "p95": sorted_t[int(len(sorted_t) * 0.95)],
                    "total_tokens": sum(all_token_counts),
                }),
                encoding="utf-8",
            )

        return results
