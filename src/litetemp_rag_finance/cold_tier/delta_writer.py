from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from litetemp_rag_finance.schema import Chunk


class DeltaWriter:
    def __init__(self, base_path: str | Path = "data/processed/delta"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._delta_available = False
        try:
            import deltalake
            self._delta_available = True
            self._delta = deltalake
        except ImportError:
            pass

    def append(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        if not self._delta_available:
            self._append_fallback(chunks)
            return
        records = [self._chunk_to_record(c) for c in chunks]
        df = pd.DataFrame(records)
        df["ingested_at"] = datetime.utcnow()
        table_path = str(self.base_path)
        try:
            self._delta.write_table(
                df, table_path, mode="append", partition_by=["source_type"],
            )
        except Exception:
            self._append_fallback(chunks)

    def _append_fallback(self, chunks: list[Chunk]) -> None:
        records = [self._chunk_to_record(c) for c in chunks]
        df = pd.DataFrame(records)
        fallback_path = self.base_path / "fallback"
        fallback_path.mkdir(parents=True, exist_ok=True)
        file_path = fallback_path / f"delta_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(file_path, index=False)

    def _chunk_to_record(self, chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source_id": chunk.source_id,
            "source_type": chunk.source_type,
            "valid_from": chunk.valid_from.isoformat(),
            "valid_to": chunk.valid_to.isoformat() if chunk.valid_to else None,
            "version": chunk.version,
            "jurisdiction": chunk.jurisdiction,
            "topic_tags": ",".join(chunk.topic_tags),
            "content_hash": chunk.content_hash,
            "embedding": chunk.embedding,
        }
