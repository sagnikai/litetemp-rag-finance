from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from litetemp_rag_finance.schema import Chunk


class TemporalQABuilder:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def build_absolute_time(
        self,
        chunks: list[Chunk],
        target_date: date,
    ) -> dict:
        matching = [
            c for c in chunks
            if c.valid_from <= target_date
            and (c.valid_to is None or c.valid_to >= target_date)
        ]
        if not matching:
            return {}
        chunk = self.rng.choice(matching)
        return {
            "question": f"What did {chunk.source_id} say about {self._random_topic(chunk)} on {target_date.isoformat()}?",
            "answer": chunk.text[:200],
            "type": "absolute_time",
            "as_of": target_date.isoformat(),
            "relevant_chunk_ids": [chunk.chunk_id],
            "source": chunk.source_id,
        }

    def build_relative_time(
        self,
        chunks: list[Chunk],
        target_date: date,
    ) -> dict:
        before_chunks = [
            c for c in chunks
            if c.valid_to is not None and c.valid_to < target_date
        ]
        if not before_chunks:
            return {}
        chunk = self.rng.choice(before_chunks)
        return {
            "question": f"What was the {chunk.source_id} policy before {target_date.isoformat()}?",
            "answer": chunk.text[:200],
            "type": "relative_time",
            "as_of": target_date.isoformat(),
            "relevant_chunk_ids": [chunk.chunk_id],
            "source": chunk.source_id,
        }

    def build_time_range(
        self,
        chunks: list[Chunk],
        t1: date,
        t2: date,
    ) -> dict:
        changes = [
            c for c in chunks
            if t1 <= c.valid_from <= t2
        ]
        if len(changes) < 2:
            return {}
        sorted_chunks = sorted(changes, key=lambda c: c.valid_from)
        return {
            "question": f"How did {sorted_chunks[0].source_id} guidance change between {t1.isoformat()} and {t2.isoformat()}?",
            "answer": f"Changed from version {sorted_chunks[0].version} to {sorted_chunks[-1].version}: {sorted_chunks[0].text[:100]} → {sorted_chunks[-1].text[:100]}",
            "type": "time_range",
            "t1": t1.isoformat(),
            "t2": t2.isoformat(),
            "relevant_chunk_ids": [c.chunk_id for c in sorted_chunks],
            "source": sorted_chunks[0].source_id,
        }

    def build_dataset(
        self,
        chunks: list[Chunk],
        n_absolute: int = 300,
        n_relative: int = 100,
        n_range: int = 100,
        date_start: date = date(2019, 1, 1),
        date_end: date = date(2026, 12, 31),
    ) -> list[dict]:
        pairs = []
        for _ in range(n_absolute):
            d = self._random_date(date_start, date_end)
            pair = self.build_absolute_time(chunks, d)
            if pair:
                pairs.append(pair)
        for _ in range(n_relative):
            d = self._random_date(date_start, date_end)
            pair = self.build_relative_time(chunks, d)
            if pair:
                pairs.append(pair)
        for _ in range(n_range):
            t1 = self._random_date(date_start, date_end)
            t2 = t1 + timedelta(days=self.rng.randint(180, 365))
            if t2 > date_end:
                continue
            pair = self.build_time_range(chunks, t1, t2)
            if pair:
                pairs.append(pair)
        return pairs

    def _random_date(self, start: date, end: date) -> date:
        delta = (end - start).days
        return start + timedelta(days=self.rng.randint(0, delta))

    def _random_topic(self, chunk: Chunk) -> str:
        topics = ["inflation", "interest rates", "capital requirements", "AML policies", "liquidity coverage", "stress testing", "disclosure rules"]
        if chunk.topic_tags:
            return self.rng.choice(chunk.topic_tags)
        return self.rng.choice(topics)

    def save(self, pairs: list[dict], path: str = "data/processed/qa_pairs.jsonl") -> None:
        import jsonlines
        with jsonlines.open(path, mode="w") as writer:
            for pair in pairs:
                writer.write(pair)

    def load(self, path: str = "data/processed/qa_pairs.jsonl") -> list[dict]:
        import jsonlines
        with jsonlines.open(path) as reader:
            return list(reader)
