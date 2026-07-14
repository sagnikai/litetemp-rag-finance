from __future__ import annotations

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from litetemp_rag_finance.hot_tier.faiss_index import FaissIndex
from litetemp_rag_finance.hot_tier.metadata_db import MetadataDB
from litetemp_rag_finance.schema import SearchResult


class LatestQuery:
    def __init__(
        self,
        embedder: SentenceTransformer,
        faiss_index: FaissIndex,
        metadata_db: MetadataDB,
    ):
        self.embedder = embedder
        self.faiss = faiss_index
        self.metadata = metadata_db

    def search(
        self,
        query: str,
        k: int = 10,
        jurisdiction: str | None = None,
        topic_filters: list[str] | None = None,
    ) -> list[SearchResult]:
        query_vec = self.embedder.encode(query, show_progress_bar=False)
        indices, distances = self.faiss.search(query_vec, k=k)

        if len(indices) == 0:
            return []

        results = []
        for idx, dist in zip(indices, distances):
            chunk = self._lookup_chunk(int(idx))
            if chunk is None:
                continue
            if jurisdiction and chunk.jurisdiction != jurisdiction:
                continue
            if topic_filters and not any(
                t in chunk.topic_tags for t in topic_filters
            ):
                continue
            score = float(1.0 / (1.0 + dist))
            result = SearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=score,
                valid_from=chunk.valid_from,
                valid_to=chunk.valid_to,
                source_id=chunk.source_id,
                version=chunk.version,
                jurisdiction=chunk.jurisdiction,
            )
            result.build_citation()
            results.append(result)

        return results

    def _lookup_chunk(self, idx: int) -> Optional[object]:
        cursor = self.metadata.conn.execute(
            "SELECT * FROM current_version WHERE rowid = ?", (idx,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        from datetime import date
        from litetemp_rag_finance.schema import Chunk
        return Chunk(
            chunk_id=row["chunk_id"],
            text="",
            version=row["version"],
            valid_from=date.fromisoformat(row["valid_from"]),
            valid_to=date.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
            content_hash=row["content_hash"],
            source_id=row["source_id"],
            source_type=row["source_type"],
            jurisdiction=row["jurisdiction"],
        )
