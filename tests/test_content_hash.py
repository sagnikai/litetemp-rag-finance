from datetime import date

from litetemp_rag_finance.content_hash import (
    compute_chunk_hash,
    compute_chunk_hashes,
    find_changed_chunks,
    format_hash_map,
)
from litetemp_rag_finance.schema import Chunk


class TestContentHash:
    def test_compute_chunk_hash(self):
        h = compute_chunk_hash("test text", "src1", "1.0", "2024-01-01")
        assert len(h) == 64

    def test_consistent_hash(self):
        h1 = compute_chunk_hash("same text", "s", "1", "2024-01-01")
        h2 = compute_chunk_hash("same text", "s", "1", "2024-01-01")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = compute_chunk_hash("text a", "s", "1", "2024-01-01")
        h2 = compute_chunk_hash("text b", "s", "1", "2024-01-01")
        assert h1 != h2

    def test_compute_chunk_hashes(self):
        chunks = [
            Chunk(chunk_id="a", text="hello", source_id="s", source_type="t",
                  valid_from=date(2024, 1, 1), version="1", jurisdiction="US"),
            Chunk(chunk_id="b", text="world", source_id="s", source_type="t",
                  valid_from=date(2024, 1, 1), version="1", jurisdiction="US"),
        ]
        result = compute_chunk_hashes(chunks)
        assert all(c.content_hash for c in result)

    def test_find_changed(self):
        existing = {
            "a": "old_hash",
            "b": "same_hash",
        }
        new_chunks = [
            Chunk(chunk_id="a", text="changed", source_id="s", source_type="t",
                  valid_from=date(2024, 1, 1), version="1", jurisdiction="US",
                  content_hash="new_hash"),
            Chunk(chunk_id="b", text="same", source_id="s", source_type="t",
                  valid_from=date(2024, 1, 1), version="1", jurisdiction="US",
                  content_hash="same_hash"),
            Chunk(chunk_id="c", text="new", source_id="s", source_type="t",
                  valid_from=date(2024, 1, 1), version="1", jurisdiction="US",
                  content_hash="new_hash_c"),
        ]
        new_only, changed = find_changed_chunks(new_chunks, existing)
        assert len(new_only) == 1
        assert new_only[0].chunk_id == "c"
        assert len(changed) == 1
        assert changed[0].chunk_id == "a"

    def test_format_hash_map(self):
        chunks = [
            Chunk(chunk_id="a", text="x", source_id="s", source_type="t",
                  valid_from=date(2024, 1, 1), version="1", jurisdiction="US",
                  content_hash="h1"),
        ]
        hm = format_hash_map(chunks)
        assert hm == {"a": "h1"}
