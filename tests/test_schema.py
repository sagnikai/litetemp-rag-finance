from datetime import date, datetime

import pytest

from litetemp_rag_finance.schema import Chunk, TemporalQuery, SearchResult


class TestChunk:
    def test_compute_hash(self):
        chunk = Chunk(
            chunk_id="fed_1.0_000000",
            text="The Federal Reserve maintains the federal funds rate.",
            source_id="fed",
            source_type="regulator",
            valid_from=date(2024, 1, 1),
            valid_to=None,
            version="1.0",
            jurisdiction="US",
        )
        h = chunk.compute_hash()
        assert len(h) == 64
        assert h == chunk.compute_hash()

    def test_is_current_no_valid_to(self):
        chunk = Chunk(
            chunk_id="test",
            text="test",
            source_id="s",
            source_type="t",
            valid_from=date(2024, 1, 1),
            version="1.0",
            jurisdiction="US",
        )
        assert chunk.is_current is True


class TestTemporalQuery:
    def test_as_of_and_range_mutual_exclusion(self):
        with pytest.raises(ValueError):
            TemporalQuery(
                query="test", as_of=date(2024, 1, 1),
                t1=date(2023, 1, 1), t2=date(2024, 6, 1),
            ).validate_query()

    def test_t1_t2_must_be_paired(self):
        with pytest.raises(ValueError):
            TemporalQuery(query="test", t1=date(2024, 1, 1)).validate_query()

    def test_valid_query(self):
        q = TemporalQuery(query="test", as_of=date(2024, 6, 1))
        q.validate_query()


class TestSearchResult:
    def test_build_citation(self):
        r = SearchResult(
            chunk_id="fed_1.0_000000",
            text="test",
            score=0.95,
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            source_id="fed",
            version="1.0",
            jurisdiction="US",
        )
        cit = r.build_citation()
        assert "fed" in cit
        assert "fed_1.0_000000" in cit
        assert "2024-01-01" in cit
        assert "2024-12-31" in cit
