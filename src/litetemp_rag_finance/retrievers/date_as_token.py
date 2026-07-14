from __future__ import annotations

from datetime import date
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from litetemp_rag_finance.hot_tier.faiss_index import FaissIndex
from litetemp_rag_finance.schema import Chunk, SearchResult


class DateAsToken:
    def __init__(
        self,
        embedder: SentenceTransformer,
        faiss_index: FaissIndex,
    ):
        self.embedder = embedder
        self.faiss = faiss_index

    def _prepend_date(self, text: str, as_of: date) -> str:
        return f"[Date: {as_of.isoformat()}] {text}"

    def encode_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [
            self._prepend_date(c.text, c.valid_from)
            for c in chunks
        ]
        return self.embedder.encode(texts, show_progress_bar=False)

    def search(
        self,
        query: str,
        as_of: date,
        k: int = 10,
    ) -> list[SearchResult]:
        dated_query = self._prepend_date(query, as_of)
        query_vec = self.embedder.encode(dated_query, show_progress_bar=False)
        indices, distances = self.faiss.search(query_vec, k=k)

        results = []
        for idx, dist in zip(indices, distances):
            score = float(1.0 / (1.0 + dist))
            results.append(SearchResult(
                chunk_id=str(idx),
                text="",
                score=score,
                valid_from=as_of,
                valid_to=None,
                source_id="",
                version="",
                jurisdiction="",
            ))
        return results
