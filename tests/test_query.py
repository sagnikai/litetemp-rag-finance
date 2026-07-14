from datetime import date

from litetemp_rag_finance.query.between import BetweenQuery


class MockParquetStore:
    def read_between(self, t1, t2):
        import pandas as pd
        return pd.DataFrame({
            "chunk_id": ["c1", "c1"],
            "source_id": ["fed", "fed"],
            "text": ["old version", "new version"],
            "version": ["1.0", "2.0"],
            "valid_from": [date(2024, 1, 1), date(2025, 1, 1)],
            "valid_to": [date(2024, 12, 31), None],
            "content_hash": ["aaa", "bbb"],
            "jurisdiction": ["US", "US"],
        })


class TestBetweenQuery:
    def test_detect_changes(self):
        store = MockParquetStore()
        bq = BetweenQuery(store)
        changes = bq.search("test", t1=date(2024, 1, 1), t2=date(2025, 6, 1))
        assert len(changes) >= 1
        assert changes[0]["change_type"] in ("modified", "new_version")
