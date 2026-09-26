"""
chunking.structure_aware
------------------------
Structure-aware chunker operating over Document hierarchy.
Groups coherent elements within sections, preserves atomic elements,
and supports conservative boundary overlap.
"""

from typing import Any
import re

from adaptive_rag.chunking.tokenizer import count_tokens
from adaptive_rag.config.hashing import compute_config_hash
from adaptive_rag.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkProvenance,
    ChunkingConfig,
    ChunkingMetadata,
    Document,
    Element,
    Section,
)

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def _render_element_text(element: Element) -> str:
    """Render element content to structured text representation."""
    if element.content.text:
        if element.type == "heading":
            level = element.metadata.get("heading_level", 2)
            prefix = "#" * min(level, 4)
            return f"{prefix} {element.content.text}"
        elif element.type == "table":
            return f"\n{element.content.text}\n"
        elif element.type == "equation":
            return f"\n$$\n{element.content.text}\n$$\n"
        elif element.type == "figure":
            return f"\n{element.content.text}\n"
        return element.content.text
    return ""


def _strip_section_header(text: str) -> str:
    """Remove a leading '[Section > Path]' header line from chunk text."""
    lines = text.split("\n", 1)
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        return lines[1] if len(lines) > 1 else ""
    return text


def _split_text_to_parts(
    text: str,
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    """Split text into parts bounded by max_tokens, aiming for target_tokens.

    Splits on sentence boundaries first; any sentence larger than max_tokens is
    further split on word boundaries so no part ever exceeds the hard cap.
    """
    units: list[str] = []
    for sentence in SENTENCE_SPLIT_REGEX.split(text):
        if not sentence.strip():
            continue
        if count_tokens(sentence) <= max_tokens:
            units.append(sentence)
            continue

        # Sentence exceeds the hard cap: split on word boundaries
        words = sentence.split()
        buf: list[str] = []
        buf_toks = 0
        for word in words:
            w_toks = count_tokens(word + " ")
            if buf_toks + w_toks > max_tokens and buf:
                units.append(" ".join(buf))
                buf, buf_toks = [word], w_toks
            else:
                buf.append(word)
                buf_toks += w_toks
        if buf:
            units.append(" ".join(buf))

    parts: list[str] = []
    cur: list[str] = []
    cur_toks = 0
    for unit in units:
        u_toks = count_tokens(unit)
        if cur_toks + u_toks > target_tokens and cur:
            parts.append(" ".join(cur))
            cur, cur_toks = [unit], u_toks
        else:
            cur.append(unit)
            cur_toks += u_toks
    if cur:
        parts.append(" ".join(cur))
    return parts


class StructureAwareChunker:
    """Chunks documents while respecting logical section hierarchy and atomic elements."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self.version = self.config.chunking_version

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Convert a hierarchical Document into structured Chunks."""
        chunks: list[Chunk] = []
        elem_by_id: dict[str, Element] = {el.element_id: el for el in document.elements}
        config_hash = compute_config_hash(self.config)
        chunk_ordinal = 0

        # Walk sections in document order
        for section in document.sections:
            if not section.element_ids:
                continue

            sec_elements = [elem_by_id[eid] for eid in section.element_ids if eid in elem_by_id]
            if not sec_elements:
                continue

            # Check if this is a references section
            is_ref = "References" in section.title or "Bibliography" in section.title
            if is_ref and not self.config.include_reference_sections:
                continue

            current_elements: list[Element] = []
            current_tokens = 0
            overlap_tail_ids: list[str] = []

            for elem in sec_elements:
                # Never split headings into own chunk without following content if possible
                elem_text = _render_element_text(elem)
                elem_tokens = count_tokens(elem_text)

                is_atomic = elem.type in ("table", "figure", "equation")

                # If single textual element exceeds max_tokens, split it
                if not is_atomic and elem_tokens > self.config.max_tokens:
                    # Flush pending buffer first
                    if current_elements:
                        chunk_ordinal += 1
                        c = self._create_chunk(
                            document=document,
                            section=section,
                            elements=current_elements,
                            ordinal=chunk_ordinal,
                            config_hash=config_hash,
                            overlap_ids=overlap_tail_ids,
                        )
                        chunks.append(c)
                        current_elements = []
                        current_tokens = 0

                    # Split oversized textual element at sentence/word boundaries
                    parts = _split_text_to_parts(
                        elem.content.text or "",
                        target_tokens=self.config.target_tokens,
                        max_tokens=self.config.max_tokens,
                    )
                    for part_idx, part_text in enumerate(parts, start=1):
                        chunk_ordinal += 1
                        c = self._create_split_chunk(
                            document=document,
                            section=section,
                            element=elem,
                            split_text=part_text,
                            ordinal=chunk_ordinal,
                            config_hash=config_hash,
                            part_index=part_idx,
                            part_count=len(parts),
                        )
                        chunks.append(c)

                    overlap_tail_ids = [elem.element_id]
                    continue

                # Buffer accumulation (account for header/separator rendering overhead)
                overhead = self._assembly_overhead(section, len(current_elements) + 1)
                if (
                    current_elements
                    and current_tokens + elem_tokens + overhead > self.config.max_tokens
                ):
                    chunk_ordinal += 1
                    c = self._create_chunk(
                        document=document,
                        section=section,
                        elements=current_elements,
                        ordinal=chunk_ordinal,
                        config_hash=config_hash,
                        overlap_ids=overlap_tail_ids,
                    )
                    chunks.append(c)

                    # Determine overlap tail from previous elements
                    overlap_tail_ids = []
                    accum_overlap = 0
                    for prev_el in reversed(current_elements):
                        if prev_el.type not in ("table", "figure", "equation"):
                            t = count_tokens(_render_element_text(prev_el))
                            if accum_overlap + t <= self.config.overlap_tokens:
                                overlap_tail_ids.insert(0, prev_el.element_id)
                                accum_overlap += t
                            else:
                                break

                    current_elements = [elem]
                    current_tokens = elem_tokens
                else:
                    current_elements.append(elem)
                    current_tokens += elem_tokens

            # Flush remaining elements in section
            if current_elements:
                chunk_ordinal += 1
                c = self._create_chunk(
                    document=document,
                    section=section,
                    elements=current_elements,
                    ordinal=chunk_ordinal,
                    config_hash=config_hash,
                    overlap_ids=overlap_tail_ids,
                )
                chunks.append(c)

        if self.config.merge_undersized_sections:
            chunks = self._merge_undersized(chunks, document, config_hash)

        return chunks

    def _assembly_overhead(self, section: Section, element_count: int) -> int:
        """Token overhead added when elements are joined into chunk text."""
        overhead = 2 * element_count  # blank-line separators between elements
        if self.config.include_section_header and section.path:
            overhead += count_tokens(f"[{' > '.join(section.path)}]")
        return overhead

    @staticmethod
    def _compatible_for_merge(a: Chunk, b: Chunk) -> tuple[bool, list[str]]:
        """Return (compatible, shared_section_path) for two adjacent chunks."""
        if a.document_id != b.document_id:
            return False, []
        pa, pb = a.metadata.section_path, b.metadata.section_path
        if pa and pa == pb:
            return True, pa
        if len(pa) >= 2 and len(pb) >= 2 and pa[:-1] == pb[:-1]:
            return True, pa[:-1]
        return False, []

    def _merge_pair(self, a: Chunk, b: Chunk, shared_path: list[str]) -> Chunk:
        """Merge two undersized sibling chunks into one coherent chunk."""
        body_a = _strip_section_header(a.text)
        body_b = _strip_section_header(b.text)
        body = "\n\n".join(t for t in (body_a, body_b) if t.strip())

        if self.config.include_section_header and shared_path:
            text = f"[{' > '.join(shared_path)}]\n{body}"
        else:
            text = body

        element_ids = a.metadata.element_ids + b.metadata.element_ids
        element_types = a.metadata.element_types + b.metadata.element_types
        headings = list(dict.fromkeys(a.metadata.headings + b.metadata.headings))
        pages = sorted(set(a.provenance.pages + b.provenance.pages))

        meta = a.metadata.model_copy(
            update={
                "section_path": shared_path,
                "headings": headings,
                "element_ids": element_ids,
                "element_types": element_types,
                "page_start": pages[0] if pages else None,
                "page_end": pages[-1] if pages else None,
                "token_count": count_tokens(text),
                "char_count": len(text),
            }
        )
        prov = a.provenance.model_copy(update={"pages": pages})
        c_meta = a.chunking_metadata.model_copy(
            update={
                "warnings": a.chunking_metadata.warnings
                + [f"merged_undersized_chunks:{a.chunk_id},{b.chunk_id}"],
                "oversized_atomic": a.chunking_metadata.oversized_atomic
                or b.chunking_metadata.oversized_atomic,
                "overlap_tokens": a.chunking_metadata.overlap_tokens,
            }
        )

        return Chunk(
            chunk_id=a.chunk_id,
            document_id=a.document_id,
            text=text,
            metadata=meta,
            provenance=prov,
            chunking_metadata=c_meta,
        )

    def _merge_undersized(
        self,
        chunks: list[Chunk],
        document: Document,
        config_hash: str,
    ) -> list[Chunk]:
        """Iteratively merge sub-min_tokens chunks with adjacent compatible siblings."""
        current = list(chunks)
        while True:
            merged_any = False
            result: list[Chunk] = []
            i = 0
            while i < len(current):
                a = current[i]
                if i + 1 < len(current) and a.metadata.token_count < self.config.min_tokens:
                    b = current[i + 1]
                    compatible, shared_path = self._compatible_for_merge(a, b)
                    combined = a.metadata.token_count + b.metadata.token_count
                    if compatible and combined <= self.config.max_tokens:
                        result.append(self._merge_pair(a, b, shared_path))
                        i += 2
                        merged_any = True
                        continue
                result.append(a)
                i += 1
            current = result
            if not merged_any:
                break

        # Renumber ordinals so chunk IDs stay contiguous and deterministic
        renumbered: list[Chunk] = []
        for idx, c in enumerate(current, start=1):
            new_id = f"{document.document_id}::{self.version}::c{idx:05d}"
            renumbered.append(
                c.model_copy(
                    update={
                        "chunk_id": new_id,
                        "chunking_metadata": c.chunking_metadata.model_copy(
                            update={"ordinal": idx, "config_hash": config_hash}
                        ),
                    }
                )
            )
        return renumbered

    def _create_chunk(
        self,
        document: Document,
        section: Section,
        elements: list[Element],
        ordinal: int,
        config_hash: str,
        overlap_ids: list[str],
    ) -> Chunk:
        rendered_texts = [_render_element_text(el) for el in elements]
        body_text = "\n\n".join(t for t in rendered_texts if t.strip())

        if self.config.include_section_header and section.path:
            header_prefix = " > ".join(section.path)
            full_text = f"[{header_prefix}]\n{body_text}"
        else:
            full_text = body_text

        elem_ids = [el.element_id for el in elements]
        elem_types = [el.type for el in elements]
        pages = sorted(list({el.provenance.page for el in elements if el.provenance.page is not None}))

        t_count = count_tokens(full_text)
        chunk_id = f"{document.document_id}::{self.version}::c{ordinal:05d}"

        meta = ChunkMetadata(
            document_id=document.document_id,
            doc_title=document.metadata.title,
            section_id=section.section_id,
            section_path=section.path,
            headings=[section.title],
            element_ids=elem_ids,
            element_types=elem_types,
            page_start=pages[0] if pages else None,
            page_end=pages[-1] if pages else None,
            token_count=t_count,
            char_count=len(full_text),
        )

        prov = ChunkProvenance(
            document_id=document.document_id,
            pages=pages,
            source_sha256=document.metadata.source_sha256,
        )

        c_meta = ChunkingMetadata(
            chunking_version=self.version,
            config_hash=config_hash,
            ordinal=ordinal,
            overlap_source_element_ids=overlap_ids,
            overlap_tokens=self.config.overlap_tokens if overlap_ids else 0,
            oversized_atomic=any(el.type in ("table", "figure") and t_count > self.config.max_tokens for el in elements),
        )

        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            text=full_text,
            metadata=meta,
            provenance=prov,
            chunking_metadata=c_meta,
        )

    def _create_split_chunk(
        self,
        document: Document,
        section: Section,
        element: Element,
        split_text: str,
        ordinal: int,
        config_hash: str,
        part_index: int,
        part_count: int = 1,
    ) -> Chunk:
        if self.config.include_section_header and section.path:
            header_prefix = " > ".join(section.path)
            full_text = f"[{header_prefix}]\n{split_text}"
        else:
            full_text = split_text

        page = element.provenance.page
        pages = [page] if page is not None else []
        t_count = count_tokens(full_text)
        chunk_id = f"{document.document_id}::{self.version}::c{ordinal:05d}"

        meta = ChunkMetadata(
            document_id=document.document_id,
            doc_title=document.metadata.title,
            section_id=section.section_id,
            section_path=section.path,
            headings=[section.title],
            element_ids=[element.element_id],
            element_types=[element.type],
            page_start=page,
            page_end=page,
            token_count=t_count,
            char_count=len(full_text),
        )

        prov = ChunkProvenance(
            document_id=document.document_id,
            pages=pages,
            source_sha256=document.metadata.source_sha256,
        )

        c_meta = ChunkingMetadata(
            chunking_version=self.version,
            config_hash=config_hash,
            ordinal=ordinal,
            split_of_element_id=element.element_id,
            part_index=part_index,
            part_count=part_count,
            oversized_atomic=t_count > self.config.max_tokens,
            warnings=(
                [f"split_part_exceeds_max_tokens:{t_count}"]
                if t_count > self.config.max_tokens
                else []
            ),
        )

        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            text=full_text,
            metadata=meta,
            provenance=prov,
            chunking_metadata=c_meta,
        )
