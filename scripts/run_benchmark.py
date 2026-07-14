#!/usr/bin/env python
"""
Benchmark runner: evaluate temporal retrievers against Finance-Policy-TimeCorpus.
Usage:
    python scripts/run_benchmark.py [--config config/default.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from litetemp_rag_finance.cold_tier.parquet_store import ParquetStore
from litetemp_rag_finance.evaluation.metrics import TemporalMetrics
from litetemp_rag_finance.hot_tier.faiss_index import FaissIndex
from litetemp_rag_finance.hot_tier.metadata_db import MetadataDB
from litetemp_rag_finance.query.as_of import AsOfQuery
from litetemp_rag_finance.query.latest import LatestQuery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


def load_qa_pairs(path: str = "data/processed/qa_pairs.jsonl") -> list[dict]:
    import jsonlines
    with jsonlines.open(path) as reader:
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description="Run temporal RAG benchmark")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--qa-pairs", default="data/processed/qa_pairs.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    embedder = SentenceTransformer(config["embedder"]["model"], device=config["embedder"]["device"])
    faiss = FaissIndex(
        dimension=config["embedder"]["dimension"],
        index_type=config["hot_tier"]["index_type"],
        index_path=config["hot_tier"]["index_path"],
    )
    metadata = MetadataDB(config["hot_tier"]["metadata_db"])
    cold = ParquetStore(config["cold_tier"]["base_path"])

    latest_q = LatestQuery(embedder, faiss, metadata)
    as_of_q = AsOfQuery(embedder, cold)

    qa_pairs = load_qa_pairs(args.qa_pairs)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")

    latencies_latest = []
    latencies_asof = []
    metrics_results = []

    for pair in qa_pairs[:50]:
        query = pair["question"]
        relevant_ids = set(pair.get("relevant_chunk_ids", []))

        if pair["type"] == "absolute_time":
            as_of = date.fromisoformat(pair["as_of"])
            t0 = time.perf_counter()
            as_of_results = as_of_q.search(query, as_of, k=10)
            latencies_asof.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            latest_results = latest_q.search(query, k=10)
            latencies_latest.append((time.perf_counter() - t0) * 1000)

            m = TemporalMetrics.all_metrics(
                as_of_results, as_of,
                relevant_ids, relevant_ids,
                generated_answer="",
            )
            m["method"] = "as_of"
            m["qa_id"] = pair.get("question", "")[:40]
            metrics_results.append(m)

    report = {
        "num_queries": len(metrics_results),
        "latency_latest_ms_p50": float(np.percentile(latencies_latest, 50)) if latencies_latest else 0,
        "latency_latest_ms_p95": float(np.percentile(latencies_latest, 95)) if latencies_latest else 0,
        "latency_asof_ms_p50": float(np.percentile(latencies_asof, 50)) if latencies_asof else 0,
        "latency_asof_ms_p95": float(np.percentile(latencies_asof, 95)) if latencies_asof else 0,
    }

    if metrics_results:
        avg_metrics = {k: np.mean([m[k] for m in metrics_results])
                       for k in metrics_results[0]
                       if isinstance(metrics_results[0][k], (int, float))}
        report["average_metrics"] = avg_metrics

    output_path = "data/processed/benchmark_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    logger.info(f"Benchmark report written to {output_path}")


if __name__ == "__main__":
    main()
