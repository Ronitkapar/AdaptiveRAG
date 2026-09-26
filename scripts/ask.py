#!/usr/bin/env python3
"""
ask.py
------
Runtime query demo script executing end-to-end Dense-RAG answering.
Retrieves context from local Qdrant, formats context, and generates grounded answer via Groq.
"""

import argparse
import json
import logging
import sys

from adaptive_rag.config.logging import setup_logging
from adaptive_rag.config.settings import settings
from adaptive_rag.embeddings.aicredits import AICreditsEmbeddingModel
from adaptive_rag.generation.context import ContextBuilder
from adaptive_rag.generation.groq import GroqGenerator
from adaptive_rag.indexing.qdrant import QdrantVectorStore
from adaptive_rag.rag import DenseRAGPipeline
from adaptive_rag.retrieval.dense import DenseRetriever

logger = logging.getLogger("ask")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the AdaptiveRAG Dense baseline.")
    parser.add_argument("question", type=str, help="Research question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--retrieve-only", action="store_true", help="Only show retrieved chunks")
    args = parser.parse_args()

    setup_logging()
    logger.info("Initializing Dense-RAG components...")

    embedding_model = AICreditsEmbeddingModel()
    vector_store = QdrantVectorStore()

    if vector_store.count() == 0:
        logger.error("Vector store collection is empty. Run build_index.py first.")
        return 1

    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    context_builder = ContextBuilder()
    generator = None if args.retrieve_only else GroqGenerator()

    pipeline = DenseRAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
    )

    result = pipeline.query(args.question, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("Question:", result["question"])
    print("=" * 60)

    if args.retrieve_only or result.get("answer") is None:
        print("\n--- Retrieved Context Chunks ---")
        for idx, item in enumerate(result["context"], start=1):
            print(f"\n[Result {idx}] (ID: {item['chunk_id']})")
            print(item["text"][:300] + "..." if len(item["text"]) > 300 else item["text"])
    else:
        print("\nAnswer:")
        print(result["answer"])
        print("\nCited Sources:", result["cited_source_ids"])
        print("Model:", result["model"])
        print(f"Latency: {result['latency_ms']['total']:.1f}ms (Retrieval: {result['latency_ms']['retrieval']:.1f}ms, Generation: {result['latency_ms']['generation']:.1f}ms)")
        print("Tokens:", result["tokens"])

    print("=" * 60 + "\n")
    vector_store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
