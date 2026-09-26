"""
tests.test_generation
---------------------
Unit tests for the Groq generator: citation parsing, usage mapping,
empty-answer semantics, and typed API failure behavior — all offline.
"""

from typing import Any

import pytest

from adaptive_rag.errors import GenerationAPIError
from adaptive_rag.generation.groq import SOURCE_CITATION_REGEX, GroqGenerator
from adaptive_rag.schemas import (
    ContextChunk,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    TokenUsage,
)


class _Usage:
    def __init__(self):
        self.prompt_tokens = 120
        self.completion_tokens = 30
        self.total_tokens = 150


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]
        self.usage = _Usage()
        self.model = "openai/gpt-oss-120b"
        self.id = "chatcmpl-fake"
        self.system_fingerprint = "fp-fake"

    def create(self, **kwargs: Any):
        return self


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs: Any):
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeGroqClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


def _request() -> GenerationRequest:
    context = [
        ContextChunk(
            source_index=1,
            chunk_id="doc1::c00001",
            document_id="doc1",
            doc_title="Paper One",
            section_path=["1. Intro"],
            page_start=1,
            page_end=1,
            score=0.9,
            rank=1,
            token_count=10,
            text="[Source 1] first passage",
        ),
        ContextChunk(
            source_index=2,
            chunk_id="doc1::c00002",
            document_id="doc1",
            doc_title="Paper One",
            section_path=["2. Method"],
            page_start=2,
            page_end=2,
            score=0.8,
            rank=2,
            token_count=12,
            text="[Source 2] second passage",
        ),
    ]
    return GenerationRequest(
        query="What is the method?",
        context=context,
        system_instruction="Answer with citations.",
        generation_config=GenerationConfig(),
    )


def test_citation_regex_extracts_source_indices():
    assert SOURCE_CITATION_REGEX.findall("See [Source 1] and [source 2].") == ["1", "2"]


def test_generate_parses_sources_and_usage():
    generator = GroqGenerator()
    generator._client = _FakeGroqClient("The method is X [Source 2] and Y [Source 1].")

    result = generator.generate(_request())

    assert isinstance(result, GenerationResult)
    assert result.status == "ok"
    assert result.source_chunk_ids == ["doc1::c00001", "doc1::c00002"]
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 30
    assert result.usage.total_tokens == 150
    assert result.finish_reason == "stop"


def test_generate_marks_empty_answer_not_failed():
    generator = GroqGenerator()
    generator._client = _FakeGroqClient("   ")

    result = generator.generate(_request())
    assert result.status == "empty_answer"
    assert result.answer == "   "


def test_generate_wraps_provider_errors_as_typed_errors():
    class _ExplodingClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: Any):
                    raise RuntimeError("boom: 500 internal error")

    generator = GroqGenerator()
    generator._client = _ExplodingClient()

    with pytest.raises(GenerationAPIError):
        generator.generate(_request())
