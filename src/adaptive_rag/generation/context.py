"""
generation.context
------------------
ContextBuilder constructing GenerationRequests from RetrievalResponses.
Enforces token and chunk budgets while preserving retrieval order.
"""

from typing import Any

from adaptive_rag.chunking.tokenizer import count_tokens
from adaptive_rag.generation.prompts import CONTEXT_TEMPLATE_V1, SYSTEM_PROMPT_V1
from adaptive_rag.schemas import (
    ContextChunk,
    ContextConfig,
    GenerationConfig,
    GenerationRequest,
    RetrievalResponse,
)


class ContextBuilder:
    """Formats retrieved chunks into clean generation context respecting token budgets."""

    def __init__(
        self,
        context_config: ContextConfig | None = None,
        generation_config: GenerationConfig | None = None,
        system_instruction: str = SYSTEM_PROMPT_V1,
    ):
        self.context_config = context_config or ContextConfig()
        self.generation_config = generation_config or GenerationConfig()
        self.system_instruction = system_instruction

    def build(self, response: RetrievalResponse) -> GenerationRequest:
        """Construct a structured GenerationRequest from a RetrievalResponse."""
        context_chunks: list[ContextChunk] = []
        dropped_ids: list[str] = []
        total_tokens = 0

        # Sort results by rank (preserving retriever order)
        sorted_results = sorted(response.results, key=lambda r: r.rank)

        for idx, res in enumerate(sorted_results, start=1):
            if len(context_chunks) >= self.context_config.max_chunks:
                dropped_ids.append(res.chunk_id)
                continue

            # Format passage text
            sec_path = " > ".join(res.metadata.section_path) if res.metadata.section_path else "General"
            pages_str = (
                f"{res.metadata.page_start}-{res.metadata.page_end}"
                if res.metadata.page_start
                else "N/A"
            )
            formatted = CONTEXT_TEMPLATE_V1.format(
                source_index=len(context_chunks) + 1,
                doc_title=res.metadata.doc_title or res.metadata.document_id,
                section_path=sec_path,
                pages=pages_str,
                score=res.score,
                text=res.text.strip(),
            )
            toks = count_tokens(formatted)

            # Check token budget
            if total_tokens + toks > self.context_config.max_context_tokens:
                # Include at least 1 passage even if it alone exceeds budget
                if not context_chunks:
                    context_chunks.append(
                        ContextChunk(
                            source_index=1,
                            chunk_id=res.chunk_id,
                            document_id=res.metadata.document_id,
                            doc_title=res.metadata.doc_title,
                            section_path=res.metadata.section_path,
                            page_start=res.metadata.page_start,
                            page_end=res.metadata.page_end,
                            score=res.score,
                            rank=res.rank,
                            token_count=toks,
                            text=formatted,
                        )
                    )
                    total_tokens += toks
                else:
                    dropped_ids.append(res.chunk_id)
                continue

            context_chunks.append(
                ContextChunk(
                    source_index=len(context_chunks) + 1,
                    chunk_id=res.chunk_id,
                    document_id=res.metadata.document_id,
                    doc_title=res.metadata.doc_title,
                    section_path=res.metadata.section_path,
                    page_start=res.metadata.page_start,
                    page_end=res.metadata.page_end,
                    score=res.score,
                    rank=res.rank,
                    token_count=toks,
                    text=formatted,
                )
            )
            total_tokens += toks

        metadata = {
            "retrieval_query": response.query,
            "total_retrieved": len(response.results),
            "included_chunks": len(context_chunks),
            "dropped_chunks": len(dropped_ids),
            "dropped_chunk_ids": dropped_ids,
            "context_tokens": total_tokens,
            "template_version": self.context_config.context_template_version,
        }

        return GenerationRequest(
            query=response.query,
            context=context_chunks,
            system_instruction=self.system_instruction,
            generation_config=self.generation_config,
            metadata=metadata,
        )
