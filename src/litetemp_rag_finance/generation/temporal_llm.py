from __future__ import annotations

from datetime import date
from typing import List, Optional

from litetemp_rag_finance.schema import SearchResult


class TemporalLLM:
    SYSTEM_PROMPT = (
        "You are a financial policy research assistant. "
        "You must answer based ONLY on the evidence provided below, "
        "and you must cite the source chunk_id and validity window "
        "for each claim you make."
    )

    def build_prompt(
        self,
        query: str,
        as_of: date | None,
        results: list[SearchResult],
    ) -> str:
        parts = [self.SYSTEM_PROMPT]

        if as_of:
            parts.append(f"\n## Query context\nPoint-in-time: {as_of.isoformat()}")
        else:
            parts.append("\n## Query context\nLatest available information requested.")

        parts.append(f"\n## Question\n{query}")

        parts.append("\n## Evidence\n")
        for i, r in enumerate(results, 1):
            parts.append(
                f"### Evidence {i} (score: {r.score:.3f})\n"
                f"Source: {r.citation}\n"
                f"Content: {r.text}\n"
            )

        parts.append(
            "\n## Instructions\n"
            "- Answer concisely based only on the evidence above.\n"
            "- For each factual claim, cite the chunk_id and its valid_from–valid_to window.\n"
            "- If the evidence is insufficient for the query_date, say so explicitly.\n"
            "- Format: [source_id:chunk_id] claim (valid valid_from – valid_to).\n"
        )

        return "\n".join(parts)

    def parse_response(self, response: str) -> dict:
        citations = []
        import re
        for match in re.finditer(
            r"\[([^:]+):([^\]]+)\]\s*(.+)",
            response,
        ):
            citations.append({
                "source_id": match.group(1),
                "chunk_id": match.group(2),
                "claim": match.group(3).strip(),
            })

        return {
            "answer": response,
            "citations": citations,
            "citation_count": len(citations),
        }
