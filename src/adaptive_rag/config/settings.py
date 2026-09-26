"""
config.settings
---------------
Pydantic Settings for environment variables, paths, and secrets.
Secrets are never defaulted and are validated lazily at access time.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from adaptive_rag.config.paths import QDRANT_DIR, REPO_ROOT
from adaptive_rag.errors import MissingCredentialError


class Settings(BaseSettings):
    """AdaptiveRAG environment and secrets settings."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- AICredits (Embeddings) ---
    AICREDITS_API_KEY: str | None = Field(default=None)
    AICREDITS_BASE_URL: str = "https://api.aicredits.in/v1"
    AICREDITS_EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_BATCH_SIZE: int = 50
    EMBEDDING_DIMENSIONS: int | None = None

    # --- Groq (Generation & LLM Judge) ---
    GROQ_API_KEY: str | None = Field(default=None)
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_JUDGE_MODEL: str = "openai/gpt-oss-20b"

    # --- Qdrant Vector Store ---
    QDRANT_MODE: Literal["local", "server"] = "local"
    QDRANT_PATH: str = str(QDRANT_DIR)
    QDRANT_COLLECTION: str = "adaptiverag_dense_v1"
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    # --- Logging & Tokenizer Cache ---
    LOG_LEVEL: str = "INFO"
    TIKTOKEN_CACHE_DIR: str | None = None

    def require_aicredits_key(self) -> str:
        """Return the AICredits API key or raise MissingCredentialError."""
        if not self.AICREDITS_API_KEY or not self.AICREDITS_API_KEY.strip():
            raise MissingCredentialError(
                "AICREDITS_API_KEY is not set. Add it to .env or pass it as an environment variable."
            )
        return self.AICREDITS_API_KEY.strip()

    def require_groq_key(self) -> str:
        """Return the Groq API key or raise MissingCredentialError."""
        if not self.GROQ_API_KEY or not self.GROQ_API_KEY.strip():
            raise MissingCredentialError(
                "GROQ_API_KEY is not set. Add it to .env or pass it as an environment variable."
            )
        return self.GROQ_API_KEY.strip()


settings = Settings()
