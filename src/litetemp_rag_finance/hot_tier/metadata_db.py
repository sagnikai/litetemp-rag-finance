from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from litetemp_rag_finance.schema import Chunk, VersionRecord


class MetadataDB:
    def __init__(self, db_path: str | Path = "data/processed/hot_metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS current_version (
                chunk_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                content_hash TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                jurisdiction TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS version_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                version TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_current_source
                ON current_version(source_id, source_type);

            CREATE INDEX IF NOT EXISTS idx_version_history_chunk
                ON version_history(chunk_id);
        """)
        self.conn.commit()

    def upsert_current(self, chunk: Chunk) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO current_version
                (chunk_id, version, valid_from, valid_to, content_hash, source_id, source_type, jurisdiction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.chunk_id, chunk.version, chunk.valid_from.isoformat(),
            chunk.valid_to.isoformat() if chunk.valid_to else None,
            chunk.content_hash, chunk.source_id, chunk.source_type,
            chunk.jurisdiction,
        ))
        self.conn.commit()

    def upsert_current_batch(self, chunks: list[Chunk]) -> None:
        rows = [(
            c.chunk_id, c.version, c.valid_from.isoformat(),
            c.valid_to.isoformat() if c.valid_to else None,
            c.content_hash, c.source_id, c.source_type, c.jurisdiction,
        ) for c in chunks]
        self.conn.executemany("""
            INSERT OR REPLACE INTO current_version
                (chunk_id, version, valid_from, valid_to, content_hash, source_id, source_type, jurisdiction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()

    def remove_current(self, chunk_ids: list[str]) -> None:
        placeholders = ",".join("?" for _ in chunk_ids)
        self.conn.execute(
            f"DELETE FROM current_version WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self.conn.commit()

    def get_current_hash_map(self) -> Dict[str, str]:
        cursor = self.conn.execute(
            "SELECT chunk_id, content_hash FROM current_version"
        )
        return {row["chunk_id"]: row["content_hash"] for row in cursor.fetchall()}

    def get_current_chunks(self) -> list[Chunk]:
        cursor = self.conn.execute("SELECT * FROM current_version")
        chunks = []
        for row in cursor.fetchall():
            chunks.append(Chunk(
                chunk_id=row["chunk_id"],
                text="",
                version=row["version"],
                valid_from=date.fromisoformat(row["valid_from"]),
                valid_to=date.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
                content_hash=row["content_hash"],
                source_id=row["source_id"],
                source_type=row["source_type"],
                jurisdiction=row["jurisdiction"],
            ))
        return chunks

    def close(self) -> None:
        self.conn.close()
