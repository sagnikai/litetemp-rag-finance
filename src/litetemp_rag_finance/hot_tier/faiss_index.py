from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class FaissIndex:
    """FAISS vector index for the hot tier (current chunks only)."""
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "HNSW",
        index_path: str | Path = "data/processed/hot_index.faiss",
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = Path(index_path)
        self.index: Optional[faiss.Index] = None
        self._build_index()

    def _build_index(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            return

        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        self.index = faiss.IndexIDMap(self.index)

    def search(self, query_vector: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.index.ntotal == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        return indices[0], distances[0]

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self.index.add_with_ids(vectors.astype(np.float32), ids.astype(np.int64))

    def remove(self, ids: np.ndarray) -> None:
        self.index.remove_ids(ids.astype(np.int64))

    def update(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self.remove(ids)
        self.add(ids, vectors)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
