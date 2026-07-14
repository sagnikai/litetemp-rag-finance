from litetemp_rag_finance.ingestion.chunker import TextChunker
from litetemp_rag_finance.ingestion.document_loader import DocumentLoader


class TestTextChunker:
    def test_chunk_size_respected(self):
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = ". ".join(["sentence one", "sentence two", "sentence three", "sentence four"])
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c.split()) <= 10 or len(c.split()) >= 8

    def test_create_chunks(self):
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.create_chunks(
            text="This is test content for chunking. It has multiple sentences.",
            source_id="fed",
            source_type="regulator",
            version="1.0",
            valid_from="2024-01-01",
            valid_to=None,
            jurisdiction="US",
        )
        assert len(chunks) >= 1
        assert chunks[0].source_id == "fed"
        assert chunks[0].chunk_id.startswith("fed_1.0_")


class TestDocumentLoader:
    def test_supported_extensions(self):
        loader = DocumentLoader()
        assert ".txt" in loader.SUPPORTED_EXTENSIONS
        assert ".md" in loader.SUPPORTED_EXTENSIONS

    def test_extract_text_from_html(self):
        loader = DocumentLoader()
        html = "<html><body><p>Hello World</p></body></html>"
        text = loader._extract_text_from_html(html)
        assert "Hello World" in text
