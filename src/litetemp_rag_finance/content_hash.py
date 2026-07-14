from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from litetemp_rag_finance.schema import Chunk


def compute_chunk_hash(text: str, source_id: str, version: str, valid_from: str) -> str:
    raw = f"{text}|{source_id}|{version}|{valid_from}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_chunk_hashes(chunks: list[Chunk]) -> list[Chunk]:
    for chunk in chunks:
        chunk.content_hash = chunk.compute_hash()
    return chunks


def find_changed_chunks(
    new_chunks: list[Chunk],
    existing_hash_map: Dict[str, str],
) -> tuple[list[Chunk], list[Chunk]]:
    new_only: list[Chunk] = []
    changed: list[Chunk] = []

    for chunk in new_chunks:
        existing_hash = existing_hash_map.get(chunk.chunk_id)
        if existing_hash is None:
            new_only.append(chunk)
        elif chunk.content_hash != existing_hash:
            changed.append(chunk)

    return new_only, changed


def format_hash_map(chunks: list[Chunk]) -> Dict[str, str]:
    return {c.chunk_id: c.content_hash for c in chunks}


def load_hash_map(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_parquet(p / "current_version.parquet")
    return dict(zip(df["chunk_id"], df["content_hash"]))


def save_hash_map(hash_map: Dict[str, str], path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"chunk_id": k, "content_hash": v}
        for k, v in hash_map.items()
    ])
    df.to_parquet(p / "current_version.parquet", index=False)
