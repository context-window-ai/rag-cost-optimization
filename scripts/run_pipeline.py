#!/usr/bin/env python3
"""Run RAG pipeline with cost instrumentation."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_retrieval.data import download_and_load_dataset
from rag_retrieval.pipeline import RAGPipeline, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline with cost tracking")
    parser.add_argument("--config", required=True, help="Config name (e.g., naive)")
    parser.add_argument("--configs_dir", default="configs", help="Directory containing config files")
    parser.add_argument("--data_dir", default=None, help="Directory to store BEIR datasets (overrides config)")
    parser.add_argument("--demo", type=int, default=None, help="Run only first N queries (overrides config)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args.configs_dir)
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Dataset: {config.get('dataset', 'scifact')}")
    logger.info(f"LLM model: {config.get('llm_model', 'gpt-4o')}")
    logger.info(f"Top-k: {config.get('top_k', 100)}")
    logger.info(f"Rerank: {config.get('rerank', False)}")

    # Override data_dir if provided
    data_dir = args.data_dir or config.get("data_dir", "datasets")

    # Download and load dataset
    dataset = config.get("dataset", "scifact")
    logger.info(f"Loading dataset: {dataset}")
    corpus, queries, qrels = download_and_load_dataset(dataset, data_dir)
    logger.info(f"Corpus size: {len(corpus)}, Queries: {len(queries)}")

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = RAGPipeline(config)

    # Determine demo subset (CLI overrides config)
    demo_subset = args.demo if args.demo is not None else config.get("demo_subset")
    if demo_subset:
        logger.info(f"Running on demo subset: {demo_subset} queries")
    else:
        logger.info(f"Running on full dataset: {len(queries)} queries")

    # Run pipeline
    pipeline.run(corpus, queries, demo_subset=demo_subset)

    # Save config copy
    config_copy_path = pipeline.output_dir / "config.json"
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    total_cost = sum(r.estimated_cost_usd for r in pipeline.cost_records)
    avg_latency = (
        sum(r.latency_ms for r in pipeline.cost_records) / len(pipeline.cost_records)
        if pipeline.cost_records
        else 0
    )

    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Queries processed: {len(pipeline.cost_records)}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Average latency: {avg_latency:.0f}ms")
    logger.info(f"Outputs saved to: {pipeline.output_dir}")
    logger.info(f"  - answers.jsonl")
    logger.info(f"  - retrieval_metadata.json")
    logger.info(f"  - costs.csv")


if __name__ == "__main__":
    main()
