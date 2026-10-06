"""
embeddings.aicredits
--------------------
AICredits embedding provider adapter for text-embedding-3-large.
Uses the OpenAI SDK transport against the OpenAI-compatible gateway.
Handles batch auto-splitting on 413, retries with backoff, and dimension probing.
"""

import time
from typing import Sequence
import numpy as np

from adaptive_rag.config.hashing import compute_config_hash
from adaptive_rag.config.settings import Settings, settings as default_settings
from adaptive_rag.errors import EmbeddingAPIError, EmbeddingTimeoutError
from adaptive_rag.schemas import EmbeddingConfig


def _l2_normalize(vector: list[float]) -> list[float]:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


class AICreditsEmbeddingModel:
    """Embedding model targeting text-embedding-3-large via AICredits."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        settings: Settings | None = None,
    ):
        self.config = config or EmbeddingConfig()
        self.settings = settings or default_settings
        self.model_id = self.config.model_id
        self.normalize = self.config.normalize
        self.config_hash = compute_config_hash(self.config)

        self._client = None
        self._probed_dim: int | None = self.config.dimensions

    def _get_client(self):
        if self._client is None:
            api_key = self.settings.require_aicredits_key()
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.settings.AICREDITS_BASE_URL,
                api_key=api_key,
                timeout=45.0,
                # `max_retries=0` disables the SDK's own retry layer
                # (`openai._constants.DEFAULT_MAX_RETRIES = 2`), which silently
                # re-issues a timed-out request *inside* a single call. With it
                # on, one stalled request cost 3 x 45s + backoff ~= 136s of
                # complete silence, on top of the explicit `max_retries` loop in
                # `_embed_batch`, which already owns backoff and 429/5xx
                # handling. Do not re-add it.
                max_retries=0,
            )
        return self._client

    @property
    def dimension(self) -> int:
        """Return the vector dimension, probing live if necessary."""
        if self._probed_dim is None:
            probe_vec = self.embed_query("probe")
            self._probed_dim = len(probe_vec)
        return self._probed_dim

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        client = self._get_client()
        params = {
            "model": self.model_id,
            "input": batch,
        }
        if self.config.dimensions is not None:
            params["dimensions"] = self.config.dimensions

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = client.embeddings.create(**params)
                vectors = [item.embedding for item in response.data]
                if self.normalize:
                    vectors = [_l2_normalize(v) for v in vectors]
                return vectors
            except Exception as exc:
                exc_str = str(exc).lower()
                status = getattr(exc, "status_code", None)
                # Classify by type, never by message text. The SDK renders a
                # timeout as "Request timed out." and a dropped connection as
                # "Connection error.", so the old `"timeout" in str(exc)` test
                # matched neither and let every timeout escape unretried on
                # attempt 1 as a plain EmbeddingAPIError. Imported locally to
                # keep this module importable without the SDK, as in
                # `_get_client`.
                from openai import APITimeoutError

                is_timeout = isinstance(exc, APITimeoutError)

                # Batch size cap check (413 or 400 payload too large)
                if (status == 413 or (status == 400 and any(h in exc_str for h in ("batch", "too many", "maximum", "array", "input")))) and len(batch) > 1:
                    mid = len(batch) // 2
                    return self._embed_batch(batch[:mid]) + self._embed_batch(batch[mid:])

                # Retryable status
                if (status in (429, 500, 502, 503, 504) or "rate limit" in exc_str or is_timeout) and attempt < self.config.max_retries:
                    wait_sec = 2 ** attempt
                    time.sleep(wait_sec)
                    continue

                if is_timeout:
                    raise EmbeddingTimeoutError(f"Embedding request timed out: {exc}") from exc

                raise EmbeddingAPIError(
                    f"AICredits embedding request failed: {exc}",
                    status_code=status,
                    response_body=str(exc),
                ) from exc

        raise EmbeddingAPIError(f"AICredits embedding failed after {self.config.max_retries} attempts.")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of texts with automatic batching."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        batch_size = max(1, self.config.batch_size)

        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        res = self._embed_batch([query])
        return res[0]
