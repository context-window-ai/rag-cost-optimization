#!/usr/bin/env python3
"""Evaluate retrieval baseline using BEIR dataset."""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_retrieval import DenseRetriever
from rag_retrieval.data import download_and_load_dataset
from rag_retrieval.evaluation import evaluate_retrieval, save_metrics, save_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval baseline")
    parser.add_argument("--config", required=True, help="Config name (e.g., baseline)")
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Load config
    config_path = Path(f"configs/{args.config}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir

    print(f"Config: {config}")
    print("=" * 60)

    # Download and load dataset
    dataset = config["dataset"]
    print(f"\nLoading dataset: {dataset}")
    corpus, queries, qrels = download_and_load_dataset(
        dataset, 
        config.get("data_dir", "datasets")
    )

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)} queries with judgments")

    # Initialize retriever
    model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"\nInitializing retriever: {model_name}")
    retriever = DenseRetriever(model_name=model_name)

    # Index documents
    print("\nBuilding index...")
    start_time = time.time()
    retriever.build_index(corpus)
    index_time = time.time() - start_time
    print(f"  Index built in {index_time:.2f}s")

    # Retrieve
    top_k = config.get("top_k", 100)
    print(f"\nRetrieving top-{top_k} documents per query...")
    start_time = time.time()
    results = retriever.retrieve(queries, top_k=top_k)
    retrieval_time = time.time() - start_time
    print(f"  Retrieval completed in {retrieval_time:.2f}s")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_retrieval(results, qrels, k_values=[1, 3, 5, 10, 100])

    # Print key metrics
    print("\n=== Results ===")
    print(f"  nDCG@10:   {metrics.get('ndcg@10', 0):.4f}")
    print(f"  Recall@10: {metrics.get('recall@10', 0):.4f}")
    print(f"  MAP:       {metrics.get('map', 0):.4f}")

    # Add metadata
    output_metrics = {
        "dataset": dataset,
        "model": model_name,
        "top_k": top_k,
        "index_time_s": round(index_time, 2),
        "retrieval_time_s": round(retrieval_time, 2),
        **metrics
    }

    # Save outputs
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(output_metrics, output_dir)
    print(f"\nSaved metrics to {output_dir / 'metrics.json'} and {output_dir / 'metrics.csv'}")

    save_results(results, output_dir)
    print(f"Saved results to {output_dir / 'results.json'}")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {output_dir / 'config.json'}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
