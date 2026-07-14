#!/usr/bin/env python
"""
Full index build: load raw documents, chunk, embed, and write hot + cold tiers.
Usage:
    python scripts/build_index.py [--config config/default.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from litetemp_rag_finance.ingestion.pipeline import UpdatePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build full FAISS + Parquet index")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--embedder", default=None, help="Override embedder model name")
    args = parser.parse_args()

    config = load_config(args.config)
    embedder = args.embedder or config["embedder"]["model"]

    logger.info(f"Using embedder: {embedder}")
    pipeline = UpdatePipeline(
        embedder_model=embedder,
        chunk_size=config["ingestion"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunk_overlap"],
        dimension=config["embedder"]["dimension"],
        device=config["embedder"]["device"],
    )

    result = pipeline.run_full()
    logger.info(f"Build complete: {result}")
    print(result)


if __name__ == "__main__":
    main()
