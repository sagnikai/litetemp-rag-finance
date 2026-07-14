# LiteTemp-RAG-Finance

**A Lightweight Dual-Tier Architecture for Temporal RAG over Policy Statements and Regulatory Guidance**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

## Motivation

In financial services, *"what we knew then"* matters more than *"what we know now."* Standard RAG answers past questions with present-day truth, creating hindsight bias that breaks complaint handling, remediation, AML investigations, and conduct-risk reviews. Regulators expect point-in-time (PIT) evidence, not a continuously overwriting index.

LiteTemp-RAG-Finance conditions every query on an explicit `as_of` timestamp and retrieves only from PIT slices using a dual-tier architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Query Interface                       │
│  latest(q, k)    as_of(q, t, k)    between(q, t1, t2)  │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│   Hot Tier       │          │   Cold Tier      │
│   (FAISS +       │          │   (Parquet/      │
│    SQLite)       │          │    Delta Lake)   │
│   Current only   │          │   All versions   │
│   valid_to = ∞   │          │   partitioned    │
│   sub-200ms      │          │   sub-2s         │
└──────────────────┘          └──────────────────┘
```

## Research Questions

- **RQ1**: Can a dual-tier index deliver sub-200 ms latency for "current policy" queries and sub-2 s for `as_of(t)` queries on a 100k–500k chunk policy/regulatory corpus?
- **RQ2**: How much re-embedding cost is saved by content-addressable chunk sync when policies change incrementally vs. full reindex?
- **RQ3**: Do temporal operators (`as_of`, `between`, `latest`) reduce time-inconsistent answers and improve temporal precision/recall on finance QA?

## Chunk Schema (SCD2)

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Unique identifier |
| `text` | string | Chunk content |
| `source_id` | string | Source document ID |
| `source_type` | string | `regulator` / `internal` |
| `valid_from` | date | Start of validity window |
| `valid_to` | date or null | End of validity window (null = current) |
| `version` | string | Document version |
| `jurisdiction` | string | US, EU, UK, CA |
| `topic_tags` | list[str] | Categorical labels |
| `content_hash` | string | SHA-256 for change detection |
| `embedding` | list[float] | Vector embedding |

## Quick Start

```bash
# Install
pip install -e .

# Build index from raw documents
python scripts/build_index.py --config config/default.yaml

# Incremental update (content-addressable)
python scripts/update_index.py --config config/default.yaml

# Run benchmark
python scripts/run_benchmark.py --config config/default.yaml
```

## Query Operators

```python
from datetime import date
from src.query.latest import LatestQuery
from src.query.as_of import AsOfQuery
from src.query.between import BetweenQuery

# Latest policy
results = latest_q.search("What did the Fed say about inflation?", k=10)

# Point-in-time
results = as_of_q.search(
    "What was the AML policy?",
    as_of=date(2024, 3, 15),
    k=10,
)

# Change detection
changes = between_q.search(
    "capital requirements",
    t1=date(2022, 1, 1),
    t2=date(2024, 12, 31),
)
```

## Evaluation

- **Finance-Policy-TimeCorpus**: 100k–500k chunks from 5–10 sources with version snapshots (2019–2026)
- **500–1,000 temporal QA pairs**: absolute time, relative time, time-range
- **Metrics**: Top-1/5/10 accuracy, NDCG@10, temporal precision/recall, temporal faithfulness, citation completeness, re-embed fraction, P50/P95 latency

## Reproducing the Paper

1. Place source documents in `data/raw/{source_id}/`
2. Run `python scripts/build_index.py` for full index
3. Run `python scripts/run_benchmark.py` to evaluate

## License

MIT
