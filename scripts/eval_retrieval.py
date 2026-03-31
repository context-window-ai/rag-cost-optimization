#!/usr/bin/env python3
"""Evaluate retrieval baseline using BEIR dataset."""
import argparse
import json
from pathlib import Path
import yaml

from rag_retrieval import DenseRetriever
from rag_retrieval.data import download_and_load_dataset
from rag_retrieval.evaluation import evaluate_retrieval, save_metrics, save_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval baseline")
    parser.add_argument("--config", required=True, help="Config name (e.g., baseline)")
    parser.add_argument("--data_dir", default="datasets", help="Directory to store BEIR datasets")
    args = parser.parse_args()

    # Load config
    config_path = Path(f"configs/{args.config}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config}")

    # Download and load dataset
    dataset = config["dataset"]
    print(f"Loading dataset: {dataset}")
    corpus, queries, qrels = download_and_load_dataset(dataset, args.data_dir)

    print(f"Corpus size: {len(corpus)}, Queries: {len(queries)}, Qrels: {len(qrels)}")

    # Initialize retriever
    print(f"Initializing retriever with model: {config['model_name']}")
    retriever = DenseRetriever(model_name=config["model_name"])

    # Index documents
    print("Indexing documents...")
    retriever.build_index(corpus)

    # Search
    print("Retrieving documents...")
    results = retriever.retrieve(queries, top_k=config["top_k"])

    # Evaluate
    print("Evaluating results...")
    metrics = evaluate_retrieval(results, qrels, k_values=[1, 3, 5, 10, 100])

    # Add metadata to metrics
    output_metrics = {
        "dataset": dataset,
        "model": config["model_name"],
        **metrics
    }

    # Print metrics
    print("\n=== Evaluation Results ===")
    for metric, value in sorted(output_metrics.items()):
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

    # Save outputs
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(output_metrics, str(output_dir))
    print(f"\nSaved metrics to {output_dir}/metrics.json and {output_dir}/metrics.csv")

    save_results(results, str(output_dir))
    print(f"Saved raw results to {output_dir}/results.json")

    # Save config copy
    config_copy_path = output_dir / "config.json"
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_copy_path}")


if __name__ == "__main__":
    main()
