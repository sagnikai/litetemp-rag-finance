from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from litetemp_rag_finance.hot_tier.faiss_index import FaissIndex
from litetemp_rag_finance.schema import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class BaselineDPR:
    def __init__(
        self,
        embedder,
        faiss_index: FaissIndex,
    ):
        self.embedder = embedder
        self.faiss = faiss_index

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        query_vec = self.embedder.encode(query, show_progress_bar=False)
        indices, distances = self.faiss.search(query_vec, k=k)

        results = []
        for idx, dist in zip(indices, distances):
            score = float(1.0 / (1.0 + dist))
            results.append(SearchResult(
                chunk_id=str(idx),
                text="",
                score=score,
                valid_from=None,
                valid_to=None,
                source_id="",
                version="",
                jurisdiction="",
            ))
        return results
