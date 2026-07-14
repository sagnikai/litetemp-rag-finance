# LiteTemp-RAG-Finance Design Document

## Overview

LiteTemp-RAG-Finance is a lightweight, dual-tier temporal RAG system for retrieving point-in-time policy and regulatory content. It uses a hot/cold architecture to serve both low-latency "latest" queries and auditable "as-of" queries from a 100k–500k chunk corpus.

## Architecture

### Hot Tier (Current Knowledge)

- **Index**: FAISS HNSW (or Flat) storing only current chunks (`valid_to = ∞`)
- **Metadata**: SQLite with `current_version` table mapping `chunk_id → version_id, content_hash`
- **Target Latency**: Sub-100–200 ms P50 for `latest` queries
- **Update**: Content-addressable via SHA-256; only changed chunks trigger re-embedding

### Cold Tier (History and Audit)

- **Storage**: Parquet (or Delta Lake) with all historical versions
- **Partitioning**: By `source_type` and `year` for efficient slice filtering
- **Query**: Full scan of applicable partitions filtered by `valid_from ≤ t < valid_to`
- **Target Latency**: Sub-2 s for `as_of` queries on a 100k–500k corpus

## SCD2 Validity Model

### Validity Windows

Every chunk carries `valid_from` and `valid_to` (nullable for current):

```
valid_from ≤ t < valid_to  →  chunk is valid at time t
valid_to = NULL            →  chunk is currently valid
```

### Overlap Resolution

When two versions have overlapping validity windows:

1. Exact match on `valid_from` → later version wins
2. Overlapping windows → split into non-overlapping sub-windows
3. Gaps → treated as "no policy in effect" (informatative null)

### Corrections

Policy corrections are handled via explicit repair versions:

- **Retroactive correction**: `valid_from` set to original effective date, `valid_to` set to next version date; the corrected chunk replaces the original for the entire window
- **Prospective correction**: `valid_from` set to correction date, `valid_to = NULL`; supersedes prior version from correction date onward

## Content-Addressable Chunk Sync

```
new chunks → compute SHA-256 → compare with existing_hash_map
  ├── hash matches → skip (no re-embed)
  ├── hash differs → re-embed, update hot tier, append to cold tier
  └── new chunk_id → embed, add to hot tier, append to cold tier
```

## Query Operators

### `latest(query, k)`

1. Encode query with sentence-transformer
2. Search FAISS HNSW index
3. Filter by `jurisdiction` / `topic_tags` if provided
4. Return top-k `SearchResult` with citations

### `as_of(query, t, k)`

1. Scan cold tier partitions for `valid_from ≤ t < valid_to`
2. Load eligible chunk texts + embeddings
3. Compute cosine similarity with query embedding
4. Return top-k results

### `between(query, t1, t2, k)`

1. Scan cold tier for chunks valid overlapping `[t1, t2]`
2. Group by `(chunk_id, source_id)`
3. Detect hash changes within each group
4. Return change descriptions with before/after text

## Evaluation Protocol

### Metrics

| Metric | Definition |
|--------|-----------|
| Temporal Precision | Fraction of retrieved chunks valid at query time |
| Temporal Recall | Fraction of relevant valid chunks retrieved |
| Top-k Accuracy | 1 if any relevant chunk in top-k |
| NDCG@10 | Normalized discounted cumulative gain |
| Temporal Faithfulness | Claim-level entailment vs time-valid evidence |
| Citation Completeness | Fraction of claims with `chunk_id + validity` citation |
| Re-embed Fraction | Chunks re-embedded / total chunks per update |
| Latency P50/P95 | Median and 95th percentile query latency |

### Baseline Comparisons

- **Vanilla RAG**: No time awareness (standard DPR)
- **DateAsToken**: Prepend date to chunks and query
- **Time-Fusion**: Semantic score × temporal proximity decay
- **Dual-Tier (this work)**: Hot/cold with explicit temporal operators
