"""
generation.groq
---------------
Groq answer generator implementation.
Decoupled completely from embedding providers.
Parses [Source N] markers into source chunk IDs and maps token usage.
"""

import re
import time
from typing import Any

from adaptive_rag.config.settings import Settings, settings as default_settings
from adaptive_rag.errors import GenerationAPIError, GenerationTimeoutError
from adaptive_rag.schemas import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    TokenUsage,
)

SOURCE_CITATION_REGEX = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


class GroqGenerator:
    """Answer generator invoking Groq chat completions."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        settings: Settings | None = None,
    ):
        self.config = config or GenerationConfig()
        self.settings = settings or default_settings
        self.generator_version = self.config.generator_version
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = self.settings.require_groq_key()
            from groq import Groq
            self._client = Groq(api_key=api_key, timeout=60.0)
        return self._client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute chat completion over retrieved context passages."""
        client = self._get_client()

        # Build context passage blocks
        context_blocks = "\n\n".join(ctx.text for ctx in request.context)
        user_message = (
            f"Question: {request.query}\n\n"
            f"Sources:\n{context_blocks}\n\n"
            f"Answer the question concisely with citations [Source N]:"
        )

        messages = [
            {"role": "system", "content": request.system_instruction},
            {"role": "user", "content": user_message},
        ]

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_completion_tokens,
        }

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "timeout" in exc_str:
                raise GenerationTimeoutError(f"Groq generation timed out: {exc}") from exc
            status = getattr(exc, "status_code", None)
            raise GenerationAPIError(
                f"Groq generation failed: {exc}",
                status_code=status,
                response_body=str(exc),
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = response.choices[0]
        answer_text = choice.message.content or ""
        finish_reason = choice.finish_reason

        # Parse [Source N] mentions to link canonical chunk IDs
        source_indices = {
            int(m) for m in SOURCE_CITATION_REGEX.findall(answer_text)
        }
        chunk_by_idx = {ctx.source_index: ctx.chunk_id for ctx in request.context}
        cited_chunk_ids = [
            chunk_by_idx[idx] for idx in sorted(source_indices) if idx in chunk_by_idx
        ]

        usage = TokenUsage()
        if response.usage:
            usage.input_tokens = getattr(response.usage, "prompt_tokens", 0)
            usage.output_tokens = getattr(response.usage, "completion_tokens", 0)
            usage.total_tokens = getattr(response.usage, "total_tokens", 0)

        status = "empty_answer" if not answer_text.strip() else "ok"

        return GenerationResult(
            answer=answer_text,
            source_chunk_ids=cited_chunk_ids,
            model=response.model or self.config.model,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            status=status,
            generator_version=self.generator_version,
            prompt_version=self.config.prompt_version,
            metadata={
                "groq_id": getattr(response, "id", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
        )
