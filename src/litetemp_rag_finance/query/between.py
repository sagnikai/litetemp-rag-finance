from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd

from litetemp_rag_finance.cold_tier.parquet_store import ParquetStore


class BetweenQuery:
    def __init__(self, cold_store: ParquetStore):
        self.cold = cold_store

    def search(
        self,
        query: str,
        t1: date,
        t2: date,
        k: int = 10,
    ) -> list[dict]:
        df = self.cold.read_between(t1, t2)
        if df.empty:
            return []

        changes = self._detect_changes(df)
        summary = self._summarize_changes(changes, query, k)
        return summary

    def _detect_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        key = ["chunk_id", "source_id"]
        diff_records = []
        for _, group in df.groupby(key):
            group = group.sort_values("valid_from")
            if len(group) < 2:
                continue
            for i in range(1, len(group)):
                prev = group.iloc[i - 1]
                curr = group.iloc[i]
                if prev["content_hash"] != curr["content_hash"]:
                    change_type = "modified" if prev["valid_to"] == curr["valid_from"] else "new_version"
                    diff_records.append({
                        "chunk_id": curr["chunk_id"],
                        "source_id": curr["source_id"],
                        "change_type": change_type,
                        "from_version": prev["version"],
                        "to_version": curr["version"],
                        "valid_from": curr["valid_from"],
                        "valid_to": curr.get("valid_to"),
                        "text_before": prev["text"][:200] + "..." if len(prev["text"]) > 200 else prev["text"],
                        "text_after": curr["text"][:200] + "..." if len(curr["text"]) > 200 else curr["text"],
                    })
        return pd.DataFrame(diff_records)

    def _summarize_changes(
        self,
        changes: pd.DataFrame,
        query: str,
        k: int,
    ) -> list[dict]:
        if changes.empty:
            return []

        keyword_filter = changes[
            changes["text_before"].str.contains(query[:30], case=False, na=False)
            | changes["text_after"].str.contains(query[:30], case=False, na=False)
        ]

        top = keyword_filter.head(k) if not keyword_filter.empty else changes.head(k)
        return top.to_dict(orient="records")

    def format_summary(self, changes: list[dict]) -> str:
        if not changes:
            return "No changes detected in the specified window."
        lines = [f"Found {len(changes)} changes:"]
        for c in changes:
            lines.append(
                f"- [{c['change_type']}] {c['chunk_id']} ({c['from_version']} → {c['to_version']}, "
                f"{c['valid_from']}): {c['text_before'][:60]} → {c['text_after'][:60]}"
            )
        return "\n".join(lines)
