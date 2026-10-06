"""
indexing.bm25
-------------
Deterministic Okapi BM25 index implementation.
Operates directly over canonical chunk corpus without modifications.
Stores inverted index, document statistics, metadata, and handles persistence.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence
import unicodedata

from adaptive_rag.config.hashing import canonical_json
from adaptive_rag.config.paths import BM25_INDEX_PATH
from adaptive_rag.errors import IndexConfigMismatchError, IndexUnavailableError
from adaptive_rag.schemas import Chunk, ChunkMetadata, ChunkProvenance

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize_text(text: str) -> list[str]:
    """Symmetric deterministic tokenization: Unicode NFKD (strip diacritics) -> lower -> alphanumeric."""
    if not text:
        return []
    # Strip diacritical marks/accents
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _TOKEN_PATTERN.findall(stripped.lower())


@dataclass
class ScoredBM25Point:
    """A scored search match from the BM25 index."""

    chunk_id: str
    score: float
    metadata: ChunkMetadata
    provenance: ChunkProvenance
    text: str


class BM25Index:
    """In-memory Okapi BM25 index serialized to / loaded from JSON."""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        corpus_version: str = "",
        index_version: str = "bm25_v1",
        corpus_arm: str = "",
    ):
        self.k1 = k1
        self.b = b
        self.corpus_version = corpus_version
        self.index_version = index_version
        # Which corpus arm this index holds. `corpus_version` cannot identify it:
        # the Phase 8 arms differ only in reading order, so both fingerprint
        # identically, and a guard keyed on the version would wave through an index
        # built from the other arm -- which is a run that compares a corpus with
        # itself and reports it as a result.
        self.corpus_arm = corpus_arm

        # Document storage
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.doc_texts: list[str] = []
        self.doc_metadata: list[dict[str, Any]] = []
        self.doc_provenance: list[dict[str, Any]] = []

        # Inverted index: term -> list of (doc_index, term_frequency)
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        # Precomputed IDF: term -> Robertson IDF (bounded >= 0)
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    @property
    def total_docs(self) -> int:
        return len(self.doc_ids)

    def build_from_chunks(
        self,
        chunks: Sequence[Chunk],
        corpus_version: str = "",
        corpus_arm: str | None = None,
    ) -> int:
        """Build the inverted index from a sequence of Chunk objects."""
        self.corpus_version = corpus_version
        if corpus_arm is not None:
            self.corpus_arm = corpus_arm
        self.doc_ids = []
        self.doc_lengths = []
        self.doc_texts = []
        self.doc_metadata = []
        self.doc_provenance = []
        self.inverted_index = {}
        self.idf = {}

        if not chunks:
            self.avgdl = 0.0
            return 0

        total_length = 0
        df: dict[str, int] = {}

        for doc_idx, chunk in enumerate(chunks):
            self.doc_ids.append(chunk.chunk_id)
            self.doc_texts.append(chunk.text)
            self.doc_metadata.append(chunk.metadata.model_dump(mode="json"))
            self.doc_provenance.append(chunk.provenance.model_dump(mode="json"))

            tokens = tokenize_text(chunk.text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Count term frequencies in this document
            tf_counts: dict[str, int] = {}
            for t in tokens:
                tf_counts[t] = tf_counts.get(t, 0) + 1

            for term, freq in tf_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))
                df[term] = df.get(term, 0) + 1

        n_docs = len(self.doc_ids)
        self.avgdl = total_length / n_docs if n_docs > 0 else 0.0

        for term, doc_freq in df.items():
            raw_idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            self.idf[term] = max(0.0, raw_idf)

        return n_docs

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredBM25Point]:
        """Score documents against tokenized query using Okapi BM25 formula."""
        if self.total_docs == 0:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            term_idf = self.idf.get(term, 0.0)
            if term_idf <= 0.0:
                continue

            postings = self.inverted_index[term]
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                norm = 1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0)
                term_score = term_idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        if not scores:
            return []

        ranked_doc_indices = sorted(scores.keys(), key=lambda idx: (-scores[idx], idx))

        results: list[ScoredBM25Point] = []
        for doc_idx in ranked_doc_indices:
            score = scores[doc_idx]
            if score_threshold is not None and score < score_threshold:
                continue

            meta_dict = self.doc_metadata[doc_idx]
            if filters:
                matches_filters = True
                for k, v in filters.items():
                    if meta_dict.get(k) != v:
                        matches_filters = False
                        break
                if not matches_filters:
                    continue

            results.append(
                ScoredBM25Point(
                    chunk_id=self.doc_ids[doc_idx],
                    score=score,
                    metadata=ChunkMetadata.model_validate(meta_dict),
                    provenance=ChunkProvenance.model_validate(self.doc_provenance[doc_idx]),
                    text=self.doc_texts[doc_idx],
                )
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, file_path: Path = BM25_INDEX_PATH) -> None:
        """Serialize index data to JSON on disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "index_version": self.index_version,
            "corpus_version": self.corpus_version,
            "corpus_arm": self.corpus_arm,
            "k1": self.k1,
            "b": self.b,
            "total_docs": self.total_docs,
            "avgdl": self.avgdl,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "doc_texts": self.doc_texts,
            "doc_metadata": self.doc_metadata,
            "doc_provenance": self.doc_provenance,
            "inverted_index": self.inverted_index,
            "idf": self.idf,
        }
        file_path.write_text(canonical_json(data), encoding="utf-8")

    @classmethod
    def load(
        cls,
        file_path: Path = BM25_INDEX_PATH,
        expected_corpus_version: str | None = None,
        expected_corpus_arm: str | None = None,
    ) -> "BM25Index":
        """Load index from JSON disk file with integrity and staleness checks.

        `expected_corpus_arm` is checked separately from the version because the two
        are not the same question. The version asks "were these PDFs and this
        pipeline the ones this run expects?"; the arm asks "is this the reading
        order this run is measuring?". Only the second one can catch a Phase 8 run
        pointed at the wrong arm, since both arms share a version by construction.
        """
        if not file_path.is_file():
            raise IndexUnavailableError(f"BM25 index file not found at {file_path}")

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise IndexUnavailableError(f"Corrupted BM25 index at {file_path}: {exc}") from exc

        if expected_corpus_version and data.get("corpus_version") != expected_corpus_version:
            raise IndexConfigMismatchError(
                f"BM25 index corpus version '{data.get('corpus_version')}' does not match "
                f"expected '{expected_corpus_version}'"
            )

        index_arm = data.get("corpus_arm", "")
        if expected_corpus_arm and index_arm != expected_corpus_arm:
            raise IndexConfigMismatchError(
                f"BM25 index at {file_path} holds corpus arm '{index_arm or 'unset'}' but "
                f"'{expected_corpus_arm}' was expected. The Phase 8 arms share a corpus "
                "version, so this is the only check that catches the wrong arm."
            )

        index = cls(
            k1=data.get("k1", 1.2),
            b=data.get("b", 0.75),
            corpus_version=data.get("corpus_version", ""),
            index_version=data.get("index_version", "bm25_v1"),
            corpus_arm=index_arm,
        )
        index.doc_ids = data.get("doc_ids", [])
        index.doc_lengths = data.get("doc_lengths", [])
        index.doc_texts = data.get("doc_texts", [])
        index.doc_metadata = data.get("doc_metadata", [])
        index.doc_provenance = data.get("doc_provenance", [])
        raw_inv = data.get("inverted_index", {})
        index.inverted_index = {
            term: [tuple(entry) for entry in postings] for term, postings in raw_inv.items()
        }
        index.idf = data.get("idf", {})
        index.avgdl = float(data.get("avgdl", 0.0))
        return index
