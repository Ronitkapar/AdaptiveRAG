#!/usr/bin/env python3
"""
check_providers.py
------------------
Live smoke checks for AICredits embeddings and Groq generation providers.
Never used inside the automated test suite; requires real API keys in .env.
"""

import argparse
import logging
import sys
import time

from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.settings import settings
from adaptive_rag.errors import MissingCredentialError

logger = logging.getLogger("check_providers")


def check_aicredits() -> bool:
    """Probe text-embedding-3-large through the AICredits gateway."""
    from adaptive_rag.embeddings.aicredits import AICreditsEmbeddingModel

    try:
        model = AICreditsEmbeddingModel()
        started = time.perf_counter()
        vector = model.embed_query("AdaptiveRAG provider smoke probe")
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info(
            "AICredits OK: model=%s dim=%d latency=%.1fms",
            model.model_id,
            len(vector),
            elapsed,
        )
        return len(vector) > 0
    except MissingCredentialError as exc:
        logger.error("AICredits skipped: %s", exc)
        return False
    except Exception as exc:
        logger.error("AICredits FAILED: %s: %s", type(exc).__name__, exc)
        return False


def check_groq() -> bool:
    """Probe the configured Groq generation model."""
    from adaptive_rag.generation.context import ContextBuilder
    from adaptive_rag.generation.groq import GroqGenerator
    from adaptive_rag.schemas import ContextChunk, GenerationRequest

    try:
        generator = GroqGenerator()
        request = GenerationRequest(
            query="Reply with exactly: provider smoke test passed.",
            context=[
                ContextChunk(
                    source_index=1,
                    chunk_id="smoke::c00001",
                    document_id="smoke",
                    doc_title="Smoke Test",
                    section_path=["1. Smoke"],
                    score=1.0,
                    rank=1,
                    token_count=5,
                    text="The smoke test says hello.",
                )
            ],
            system_instruction="Follow the user instruction exactly.",
            generation_config=generator.config,
        )
        started = time.perf_counter()
        result = generator.generate(request)
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Groq OK: model=%s status=%s latency=%.1fms tokens=%s",
            result.model,
            result.status,
            elapsed,
            result.usage.model_dump(mode="json"),
        )
        return result.status in ("ok", "empty_answer")
    except MissingCredentialError as exc:
        logger.error("Groq skipped: %s", exc)
        return False
    except Exception as exc:
        logger.error("Groq FAILED: %s: %s", type(exc).__name__, exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Live provider smoke checks (needs .env keys).")
    parser.add_argument("--embeddings", action="store_true", help="Check AICredits embeddings")
    parser.add_argument("--groq", action="store_true", help="Check Groq generation")
    args = parser.parse_args()

    setup_logging()
    run_all = not (args.embeddings or args.groq)

    ok = True
    if args.embeddings or run_all:
        ok = check_aicredits() and ok
    if args.groq or run_all:
        ok = check_groq() and ok

    logger.info("AICREDITS_API_KEY set: %s", bool(settings.AICREDITS_API_KEY))
    logger.info("GROQ_API_KEY set: %s", bool(settings.GROQ_API_KEY))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
