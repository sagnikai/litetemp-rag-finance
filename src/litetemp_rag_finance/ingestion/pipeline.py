from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from litetemp_rag_finance.content_hash import (
    find_changed_chunks,
    format_hash_map,
)
from litetemp_rag_finance.hot_tier.faiss_index import FaissIndex
from litetemp_rag_finance.hot_tier.metadata_db import MetadataDB
from litetemp_rag_finance.cold_tier.parquet_store import ParquetStore
from litetemp_rag_finance.ingestion.chunker import TextChunker
from litetemp_rag_finance.ingestion.document_loader import DocumentLoader
from litetemp_rag_finance.schema import Chunk

logger = logging.getLogger(__name__)


class UpdatePipeline:
    def __init__(
        self,
        embedder_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        dimension: int = 384,
        device: str = "cpu",
    ):
        logger.info(f"Loading embedder: {embedder_model} on {device}")
        self.embedder = SentenceTransformer(embedder_model, device=device)
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.loader = DocumentLoader()
        self.faiss = FaissIndex(dimension=dimension)
        self.metadata = MetadataDB()
        self.cold = ParquetStore()
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, show_progress_bar=False)

    def run_full(self) -> dict:
        logger.info("Starting full index build")
        all_chunks: list[Chunk] = []
        for source_id, filepath, text in self.loader.load_all():
            chunks = self.chunker.create_chunks(
                text=text,
                source_id=source_id,
                source_type="regulator",
                version="1.0",
                valid_from="2024-01-01",
                valid_to=None,
                jurisdiction="",
            )
            all_chunks.extend(chunks)
            logger.info(f"  {source_id}: {len(chunks)} chunks")

        if not all_chunks:
            logger.warning("No chunks found to index")
            return {"total_chunks": 0, "new": 0, "changed": 0}

        texts = [c.text for c in all_chunks]
        embeddings = self.embed_texts(texts)
        for i, chunk in enumerate(all_chunks):
            chunk.embedding = embeddings[i].tolist()

        ids = np.array([hash(c.chunk_id) % (2**63) for c in all_chunks], dtype=np.int64)
        self.faiss.add(ids, embeddings)
        self.faiss.save()

        self.metadata.upsert_current_batch(all_chunks)
        self.cold.append(all_chunks)

        logger.info(f"Full build complete: {len(all_chunks)} chunks indexed")
        return {
            "total_chunks": len(all_chunks),
            "new": len(all_chunks),
            "changed": 0,
        }

    def run_incremental(self) -> dict:
        logger.info("Starting incremental update")
        new_chunks: list[Chunk] = []
        for source_id, filepath, text in self.loader.load_all():
            chunks = self.chunker.create_chunks(
                text=text,
                source_id=source_id,
                source_type="regulator",
                version="1.0",
                valid_from="2024-01-01",
                valid_to=None,
                jurisdiction="",
            )
            new_chunks.extend(chunks)

        if not new_chunks:
            return {"total_chunks": 0, "new": 0, "changed": 0, "skipped": 0}

        existing_hash_map = self.metadata.get_current_hash_map()
        for c in new_chunks:
            c.content_hash = c.compute_hash()

        new_only, changed = find_changed_chunks(new_chunks, existing_hash_map)
        unchanged = [
            c for c in new_chunks
            if c.chunk_id in existing_hash_map
            and c.content_hash == existing_hash_map[c.chunk_id]
        ]

        logger.info(
            f"  new={len(new_only)}, changed={len(changed)}, "
            f"skipped={len(unchanged)}"
        )

        chunks_to_embed = new_only + changed
        if chunks_to_embed:
            texts = [c.text for c in chunks_to_embed]
            embeddings = self.embed_texts(texts)
            for i, chunk in enumerate(chunks_to_embed):
                chunk.embedding = embeddings[i].tolist()
            ids = np.array(
                [hash(c.chunk_id) % (2**63) for c in chunks_to_embed],
                dtype=np.int64,
            )
            to_remove = np.array(
                [hash(c.chunk_id) % (2**63) for c in changed],
                dtype=np.int64,
            )
            self.faiss.remove(to_remove)
            self.faiss.add(ids, embeddings)
            self.faiss.save()
            self.metadata.upsert_current_batch(chunks_to_embed)
            self.cold.append(chunks_to_embed)

        return {
            "total_chunks": len(new_chunks),
            "new": len(new_only),
            "changed": len(changed),
            "skipped": len(unchanged),
        }
