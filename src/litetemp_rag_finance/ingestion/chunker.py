from __future__ import annotations

import re
from typing import List, Optional

from litetemp_rag_finance.schema import Chunk


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunk_by: str = "sentence",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_by = chunk_by

    def chunk_text(self, text: str) -> list[str]:
        if self.chunk_by == "sentence":
            return self._chunk_by_sentences(text)
        elif self.chunk_by == "paragraph":
            return self._chunk_by_paragraphs(text)
        elif self.chunk_by == "section":
            return self._chunk_by_sections(text)
        else:
            return self._chunk_by_sentences(text)

    def _chunk_by_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                overlap_tokens = max(
                    0, current_len - self.chunk_overlap
                )
                overlap_idx = 0
                acc = 0
                for i, s in enumerate(current):
                    acc += len(s.split())
                    if acc > overlap_tokens:
                        overlap_idx = i
                        break
                current = current[overlap_idx:]
                current_len = sum(len(s.split()) for s in current)
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _chunk_by_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = []
        current_len = 0
        for para in paragraphs:
            para_len = len(para.split())
            if current_len + para_len > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += para_len
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _chunk_by_sections(self, text: str) -> list[str]:
        sections = re.split(
            r"(?=\n(?:[A-Z][A-Za-z\s]+)\n-+\n)", text
        )
        return [s.strip() for s in sections if s.strip()]

    def create_chunks(
        self,
        text: str,
        source_id: str,
        source_type: str,
        version: str,
        valid_from: str,
        valid_to: str | None,
        jurisdiction: str = "",
        topic_tags: list[str] | None = None,
    ) -> list[Chunk]:
        segments = self.chunk_text(text)
        chunks = []
        for i, segment in enumerate(segments):
            chunk = Chunk(
                chunk_id=f"{source_id}_{version}_{i:06d}",
                text=segment,
                source_id=source_id,
                source_type=source_type,
                version=version,
                valid_from=valid_from,
                valid_to=valid_to,
                jurisdiction=jurisdiction,
                topic_tags=topic_tags or [],
            )
            chunk.content_hash = chunk.compute_hash()
            chunks.append(chunk)
        return chunks
