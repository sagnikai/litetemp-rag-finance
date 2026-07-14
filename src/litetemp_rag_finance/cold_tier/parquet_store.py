from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from litetemp_rag_finance.schema import Chunk


class ParquetStore:
    def __init__(self, base_path: str | Path = "data/processed/cold"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _chunk_to_dict(self, chunk: Chunk) -> dict:
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
            "ingested_at": datetime.utcnow().isoformat(),
        }

    def append(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        df = pd.DataFrame([self._chunk_to_dict(c) for c in chunks])
        for group_keys, group_df in df.groupby(["source_type", "year"]):
            source_type, year = group_keys
            self._write_partition(group_df, source_type, year)

    def _write_partition(
        self, df: pd.DataFrame, source_type: str, year: str
    ) -> None:
        part_path = self.base_path / source_type / str(year)
        part_path.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        file_path = part_path / f"chunks_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        pq.write_table(table, file_path)

    def read_as_of(self, as_of: date) -> pd.DataFrame:
        fragments = []
        for part_path in self.base_path.rglob("*.parquet"):
            table = pq.read_table(part_path)
            df = table.to_pandas()
            df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
            df["valid_to"] = pd.to_datetime(df["valid_to"]).dt.date if df["valid_to"].notna().any() else None
            mask = (df["valid_from"] <= as_of) & (
                df["valid_to"].isna() | (df["valid_to"] >= as_of)
            )
            fragments.append(df[mask])
        return pd.concat(fragments, ignore_index=True) if fragments else pd.DataFrame()

    def read_between(self, t1: date, t2: date) -> pd.DataFrame:
        fragments = []
        for part_path in self.base_path.rglob("*.parquet"):
            table = pq.read_table(part_path)
            df = table.to_pandas()
            df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
            df["valid_to"] = pd.to_datetime(df["valid_to"]).dt.date if df["valid_to"].notna().any() else None
            mask = (df["valid_from"] <= t2) & (
                df["valid_to"].isna() | (df["valid_to"] >= t1)
            )
            fragments.append(df[mask])
        return pd.concat(fragments, ignore_index=True) if fragments else pd.DataFrame()
