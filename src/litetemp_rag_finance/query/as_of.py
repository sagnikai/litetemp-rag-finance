from __future__ import annotations

from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from litetemp_rag_finance.cold_tier.parquet_store import ParquetStore
from litetemp_rag_finance.schema import SearchResult


class AsOfQuery:
    def __init__(
        self,
        embedder: SentenceTransformer,
        cold_store: ParquetStore,
    ):
        self.embedder = embedder
        self.cold = cold_store

    def search(
        self,
        query: str,
        as_of: date,
        k: int = 10,
        jurisdiction: str | None = None,
    ) -> list[SearchResult]:
        df = self.cold.read_as_of(as_of)
        if df.empty:
            return []

        if jurisdiction:
            df = df[df["jurisdiction"] == jurisdiction]

        texts = df["text"].tolist()
        if not texts:
            return []

        query_vec = self.embedder.encode(query, show_progress_bar=False)
        chunk_vecs = np.array(
            [np.frombuffer(v, dtype=np.float32) if isinstance(v, bytes) else np.array(v, dtype=np.float32)
             for v in df["embedding"].tolist()]
        )
        if chunk_vecs.ndim == 1:
            chunk_vecs = chunk_vecs.reshape(1, -1)
        if chunk_vecs.shape[0] == 0:
            return []

        scores = chunk_vecs @ query_vec
        top_k = min(k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            result = SearchResult(
                chunk_id=row["chunk_id"],
                text=row["text"],
                score=float(scores[idx]),
                valid_from=row["valid_from"],
                valid_to=row.get("valid_to"),
                source_id=row["source_id"],
                version=row["version"],
                jurisdiction=row["jurisdiction"],
            )
            result.build_citation()
            results.append(result)
        return results
