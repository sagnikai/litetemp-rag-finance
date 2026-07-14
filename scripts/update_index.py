#!/usr/bin/env python
"""
Incremental index update: detect changed chunks, re-embed only what changed.
Usage:
    python scripts/update_index.py [--config config/default.yaml]
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
logger = logging.getLogger("update_index")


def main():
    parser = argparse.ArgumentParser(description="Incremental index update (content-addressable)")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = UpdatePipeline(
        embedder_model=config["embedder"]["model"],
        chunk_size=config["ingestion"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunk_overlap"],
        dimension=config["embedder"]["dimension"],
        device=config["embedder"]["device"],
    )

    result = pipeline.run_incremental()
    logger.info(f"Update complete: {result}")
    print(result)


if __name__ == "__main__":
    main()
