#!/usr/bin/env python3
"""Run RAG pipeline with cost instrumentation."""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_retrieval.data import download_and_load_dataset
from rag_retrieval.retriever import FAISSRetriever
from rag_retrieval.pipeline import NaiveRAGPipeline, OptimizedRAGPipeline, write_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_name: str, configs_dir: str = "configs") -> dict:
    """Load a YAML config file by name."""
    config_path = Path(configs_dir) / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


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

    # Override data_dir if provided
    data_dir = args.data_dir or config.get("data_dir", "datasets")

    # Download and load dataset
    dataset = config["dataset"]
    logger.info(f"Loading dataset: {dataset}")
    corpus, queries, qrels = download_and_load_dataset(dataset, data_dir)
    logger.info(f"Corpus size: {len(corpus)}, Queries: {len(queries)}")

    # Build retriever
    logger.info("Building retriever index...")
    retriever = FAISSRetriever(model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    retriever.build_index(corpus)
    logger.info("Index built successfully")

    # Create pipeline based on config type
    # Use OptimizedRAGPipeline if any optimization levers are defined
    is_optimized = any(key in config for key in [
        "conditional_rerank", "model_routing", "cheap_model"
    ])
    
    if is_optimized:
        logger.info("Using OptimizedRAGPipeline with optimization levers:")
        logger.info(f"  - lower_top_k: {config.get('top_k', 10)} (vs naive 100)")
        logger.info(f"  - conditional_rerank: {config.get('conditional_rerank', True)}")
        logger.info(f"  - model_routing: {config.get('model_routing', True)}")
        pipeline = OptimizedRAGPipeline(
            retriever=retriever,
            model=config.get("llm_model", "openai/gpt-4o"),
            cheap_model=config.get("cheap_model", "openai/gpt-4o-mini"),
            top_k=config.get("top_k", 10),
            rerank_top_k=config.get("rerank_k", 10),
            corpus=corpus,
            rerank_model=config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            temperature=config.get("llm_temperature", 0.0),
            lower_top_k=True,  # Enabled by setting lower top_k
            conditional_rerank=config.get("conditional_rerank", True),
            model_routing=config.get("model_routing", True),
            rerank_skip_threshold=config.get("rerank_skip_threshold", 0.85),
            cheap_model_threshold=config.get("cheap_model_threshold", 0.80),
        )
    else:
        logger.info("Using NaiveRAGPipeline (baseline)")
        pipeline = NaiveRAGPipeline(
            retriever=retriever,
            model=config.get("llm_model", "openai/gpt-4o"),
            top_k=config.get("top_k", 100),
            rerank_top_k=config.get("rerank_k", 10),
            corpus=corpus,
            rerank_model=config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            temperature=config.get("llm_temperature", 0.0),
        )

    # Determine demo subset (CLI overrides config)
    demo_subset = args.demo if args.demo is not None else config.get("demo_subset")

    # Get query IDs to process
    query_ids = list(queries.keys())
    if demo_subset:
        query_ids = query_ids[:demo_subset]
        logger.info(f"Running on demo subset: {demo_subset} queries")
    else:
        logger.info(f"Running on full dataset: {len(queries)} queries")

    # Run pipeline
    results = []
    total_cost = 0.0
    start_time = time.time()

    for i, query_id in enumerate(query_ids):
        query = queries[query_id]
        logger.info(f"Processing query {i+1}/{len(query_ids)}: {query_id}")
        
        result = pipeline.run_query(query_id, query, rerank=config.get("rerank", True))
        results.append(result)
        total_cost += result.cost_record.estimated_cost_usd

    total_time = time.time() - start_time

    # Write outputs
    output_dir = Path(config.get("output_dir", "outputs/naive"))
    write_outputs(results, output_dir)

    # Write summary
    avg_latency = sum(r.cost_record.latency_ms for r in results) / len(results) if results else 0
    summary = {
        "config": config,
        "total_queries": len(results),
        "total_cost_usd": round(total_cost, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "total_time_seconds": round(total_time, 2),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Queries processed: {len(results)}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Average latency: {avg_latency:.0f}ms")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info(f"  - answers.jsonl")
    logger.info(f"  - retrieval_metadata.jsonl")
    logger.info(f"  - costs.csv")


if __name__ == "__main__":
    main()
