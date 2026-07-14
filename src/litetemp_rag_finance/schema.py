from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_id: str
    source_type: str
    valid_from: date
    valid_to: date | None = None
    version: str
    jurisdiction: str
    topic_tags: list[str] = Field(default_factory=list)
    content_hash: str = ""
    embedding: list[float] | None = None

    def compute_hash(self) -> str:
        raw = f"{self.text}|{self.source_id}|{self.version}|{self.valid_from.isoformat()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def is_current(self) -> bool:
        return self.valid_to is None or self.valid_to >= date.today()


class VersionRecord(BaseModel):
    chunk_id: str
    version: str
    valid_from: date
    valid_to: date | None
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TemporalQuery(BaseModel):
    query: str
    as_of: date | None = None
    t1: date | None = None
    t2: date | None = None
    k: int = 10
    jurisdiction: str | None = None
    topic_filters: list[str] = Field(default_factory=list)

    def validate_query(self) -> None:
        if self.as_of and (self.t1 or self.t2):
            raise ValueError("as_of cannot be combined with t1/t2")
        if (self.t1 is None) != (self.t2 is None):
            raise ValueError("t1 and t2 must both be set or both be None")


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    valid_from: date
    valid_to: date | None
    source_id: str
    version: str
    jurisdiction: str
    citation: str = ""

    def build_citation(self) -> str:
        valid_end = self.valid_to.isoformat() if self.valid_to else "present"
        self.citation = (
            f"[{self.source_id}] {self.chunk_id} "
            f"(v{self.version}, {self.valid_from.isoformat()}–{valid_end})"
        )
        return self.citation
