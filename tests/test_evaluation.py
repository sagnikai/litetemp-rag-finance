from datetime import date

from litetemp_rag_finance.evaluation.metrics import TemporalMetrics
from litetemp_rag_finance.schema import SearchResult


class TestTemporalMetrics:
    def test_temporal_precision_all_valid(self):
        results = [
            SearchResult(chunk_id="a", text="", score=0.9,
                         valid_from=date(2024, 1, 1), valid_to=None,
                         source_id="s", version="1", jurisdiction="US"),
            SearchResult(chunk_id="b", text="", score=0.8,
                         valid_from=date(2024, 6, 1), valid_to=date(2024, 12, 31),
                         source_id="s", version="1", jurisdiction="US"),
        ]
        prec = TemporalMetrics.temporal_precision(results, date(2024, 6, 15))
        assert prec == 1.0

    def test_temporal_precision_partial(self):
        results = [
            SearchResult(chunk_id="a", text="", score=0.9,
                         valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1),
                         source_id="s", version="1", jurisdiction="US"),
            SearchResult(chunk_id="b", text="", score=0.8,
                         valid_from=date(2025, 1, 1), valid_to=None,
                         source_id="s", version="1", jurisdiction="US"),
        ]
        prec = TemporalMetrics.temporal_precision(results, date(2024, 3, 15))
        assert prec == 0.5

    def test_temporal_recall(self):
        results = [
            SearchResult(chunk_id="a", text="", score=0.9,
                         valid_from=date(2024, 1, 1), valid_to=None,
                         source_id="s", version="1", jurisdiction="US"),
        ]
        recall = TemporalMetrics.temporal_recall(results, {"a", "b"})
        assert recall == 0.5

    def test_top_k_accuracy(self):
        results = [
            SearchResult(chunk_id="a", text="", score=0.9,
                         valid_from=date(2024, 1, 1), valid_to=None,
                         source_id="s", version="1", jurisdiction="US"),
            SearchResult(chunk_id="b", text="", score=0.8,
                         valid_from=date(2024, 1, 1), valid_to=None,
                         source_id="s", version="1", jurisdiction="US"),
        ]
        acc = TemporalMetrics.top_k_accuracy(results, {"a"}, 1)
        assert acc == 1.0

    def test_citation_completeness(self):
        answer = "[fed:fed_1.0_000000] Inflation rose 2% (valid 2024-01-01 – 2024-12-31)"
        result = TemporalMetrics.citation_completeness(answer)
        assert result["citation_count"] == 1
        assert result["has_chunk_ids"] is True
