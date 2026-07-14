from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np
from sklearn.metrics import ndcg_score

from litetemp_rag_finance.schema import SearchResult


class TemporalMetrics:
    @staticmethod
    def temporal_precision(
        results: list[SearchResult],
        query_date: date,
    ) -> float:
        if not results:
            return 0.0
        valid = sum(
            1 for r in results
            if r.valid_from is not None and r.valid_from <= query_date
            and (r.valid_to is None or r.valid_to >= query_date)
        )
        return valid / len(results)

    @staticmethod
    def temporal_recall(
        results: list[SearchResult],
        relevant_valid_ids: set[str],
    ) -> float:
        if not relevant_valid_ids:
            return 0.0
        retrieved_valid = {r.chunk_id for r in results}
        if not retrieved_valid:
            return 0.0
        return len(relevant_valid_ids & retrieved_valid) / len(relevant_valid_ids)

    @staticmethod
    def top_k_accuracy(
        results: list[SearchResult],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        top_k = results[:k]
        if not top_k:
            return 0.0
        return float(any(r.chunk_id in relevant_ids for r in top_k))

    @staticmethod
    def ndcg(
        results: list[SearchResult],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        y_true = np.array([
            [1.0 if r.chunk_id in relevant_ids else 0.0 for r in results[:k]]
        ])
        y_score = np.array([[r.score for r in results[:k]]])
        if y_true.shape[1] < 2:
            return float(y_true[0, 0])
        return float(ndcg_score(y_true, y_score, k=min(k, y_true.shape[1])))

    @staticmethod
    def temporal_faithfulness(
        generated_answer: str,
        evidence_results: list[SearchResult],
    ) -> float:
        cited_chunks = set()
        import re
        for match in re.finditer(r"\[([^\]]+)\]\s*\((\d{4}-\d{2}-\d{2})\s*–\s*(\d{4}-\d{2}-\d{2}|present)\)", generated_answer):
            cited_chunks.add(match.group(1))

        if not cited_chunks:
            citation_count = generated_answer.count("chunk_id")
            if citation_count == 0:
                return 0.0
            cited_chunks = {f"Evidence {i}" for i in range(1, citation_count + 1)}

        evidence_ids = {r.chunk_id for r in evidence_results}
        if not evidence_ids:
            return 0.0
        return len(cited_chunks & evidence_ids) / len(cited_chunks)

    @staticmethod
    def citation_completeness(
        generated_answer: str,
    ) -> dict:
        import re
        citations = re.findall(
            r"\[([^:]+):([^\]]+)\]\s*\(valid\s+(\d{4}-\d{2}-\d{2})\s*–\s*(\d{4}-\d{2}-\d{2}|present)\)",
            generated_answer,
        )
        return {
            "citation_count": len(citations),
            "has_chunk_ids": bool(citations),
            "has_validity_windows": all(len(c) == 4 for c in citations),
        }

    @staticmethod
    def all_metrics(
        results: list[SearchResult],
        query_date: date,
        relevant_ids: set[str],
        relevant_valid_ids: set[str],
        generated_answer: str,
    ) -> dict:
        return {
            "temporal_precision": TemporalMetrics.temporal_precision(results, query_date),
            "temporal_recall": TemporalMetrics.temporal_recall(results, relevant_valid_ids),
            "top_1_accuracy": TemporalMetrics.top_k_accuracy(results, relevant_ids, 1),
            "top_5_accuracy": TemporalMetrics.top_k_accuracy(results, relevant_ids, 5),
            "top_10_accuracy": TemporalMetrics.top_k_accuracy(results, relevant_ids, 10),
            "ndcg@10": TemporalMetrics.ndcg(results, relevant_ids, 10),
            "temporal_faithfulness": TemporalMetrics.temporal_faithfulness(
                generated_answer, results
            ),
            **TemporalMetrics.citation_completeness(generated_answer),
        }
