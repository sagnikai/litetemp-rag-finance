from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from litetemp_rag_finance.hot_tier.faiss_index import FaissIndex
from litetemp_rag_finance.schema import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class TimeFusionRetriever:
    def __init__(
        self,
        embedder,
        faiss_index: FaissIndex,
        alpha: float = 0.7,
        decay_rate: float = 0.01,
    ):
        self.embedder = embedder
        self.faiss = faiss_index
        self.alpha = alpha
        self.decay_rate = decay_rate

    def _temporal_score(self, valid_from: date, query_date: date) -> float:
        days_diff = abs((query_date - valid_from).days)
        return float(np.exp(-self.decay_rate * days_diff))

    def search(
        self,
        query: str,
        query_date: date,
        k: int = 10,
        chunk_dates: list[date] | None = None,
    ) -> list[SearchResult]:
        query_vec = self.embedder.encode(query, show_progress_bar=False)
        indices, distances = self.faiss.search(query_vec, k=k * 2)

        results = []
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            semantic_score = float(1.0 / (1.0 + dist))
            if chunk_dates and i < len(chunk_dates):
                temporal = self._temporal_score(chunk_dates[i], query_date)
            else:
                temporal = 1.0
            fused = self.alpha * semantic_score + (1 - self.alpha) * temporal
            results.append(SearchResult(
                chunk_id=str(idx),
                text="",
                score=fused,
                valid_from=chunk_dates[i] if chunk_dates and i < len(chunk_dates) else query_date,
                valid_to=None,
                source_id="",
                version="",
                jurisdiction="",
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
