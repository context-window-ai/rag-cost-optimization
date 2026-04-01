#!/usr/bin/env python3
"""Build retriever comparison results from individual variant outputs."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


RETRIEVER_VARIANTS = [
    {"retriever": "dense", "dir": "outputs/retriever_comparison/dense"},
    {"retriever": "bm25", "dir": "outputs/retriever_comparison/bm25"},
    {"retriever": "hybrid", "dir": "outputs/retriever_comparison/hybrid"},
]


def load_csv_data(filepath: Path) -> List[Dict]:
    """Load CSV file and return list of dictionaries."""
    if not filepath.exists():
        return []
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_json_data(filepath: Path) -> Optional[Dict]:
    """Load JSON file and return dict."""
    if not filepath.exists():
        return None
    
    with open(filepath, "r") as f:
        return json.load(f)


def compute_averages(data: List[Dict], fields: List[str]) -> Dict[str, float]:
    """Compute averages for specified fields."""
    if not data:
        return {field: 0.0 for field in fields}
    
    averages = {}
    for field in fields:
        values = [float(row[field]) for row in data if field in row and row[field]]
        averages[field] = sum(values) / len(values) if values else 0.0
    
    return averages


def load_retrieval_metrics(variant_dir: Path) -> Dict[str, float]:
    """Load retrieval metrics if available."""
    metrics_path = variant_dir / "metrics.json"
    metrics = load_json_data(metrics_path)
    
    if metrics:
        return {
            "ndcg_at_10": metrics.get("ndcg@10"),
            "recall_at_10": metrics.get("recall@10"),
        }
    return {"ndcg_at_10": None, "recall_at_10": None}


def process_variant(variant: Dict) -> Optional[Dict]:
    """Process a single variant and return aggregated metrics."""
    variant_dir = Path(variant["dir"])
    
    # Load summary.json
    summary = load_json_data(variant_dir / "summary.json")
    if not summary:
        print(f"Warning: No summary.json found in {variant_dir}")
        return None
    
    # Load costs.csv
    costs_data = load_csv_data(variant_dir / "costs.csv")
    if not costs_data:
        print(f"Warning: No costs.csv found in {variant_dir}")
        return None
    
    # Load judge_scores.csv
    judge_data = load_csv_data(variant_dir / "judge_scores.csv")
    if not judge_data:
        print(f"Warning: No judge_scores.csv found in {variant_dir}")
        return None
    
    # Compute averages from costs.csv
    cost_avgs = compute_averages(costs_data, ["estimated_cost_usd", "latency_ms"])
    
    # Compute averages from judge_scores.csv
    judge_avgs = compute_averages(judge_data, ["faithfulness_score", "answer_relevance_score"])
    
    # Try to load retrieval metrics
    retrieval_metrics = load_retrieval_metrics(variant_dir)
    
    return {
        "retriever": variant["retriever"],
        "avg_cost": cost_avgs["estimated_cost_usd"],
        "avg_latency_ms": cost_avgs["latency_ms"],
        "avg_faithfulness": judge_avgs["faithfulness_score"],
        "avg_answer_relevance": judge_avgs["answer_relevance_score"],
        "ndcg_at_10": retrieval_metrics["ndcg_at_10"],
        "recall_at_10": retrieval_metrics["recall_at_10"],
    }


def write_csv(results: List[Dict], output_path: Path) -> None:
    """Write results to CSV file."""
    fieldnames = [
        "retriever",
        "avg_cost",
        "avg_latency_ms",
        "avg_faithfulness",
        "avg_answer_relevance",
        "ndcg_at_10",
        "recall_at_10",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            # Format N/A for missing retrieval metrics
            row = r.copy()
            if row["ndcg_at_10"] is None:
                row["ndcg_at_10"] = "N/A"
            if row["recall_at_10"] is None:
                row["recall_at_10"] = "N/A"
            writer.writerow(row)
    
    print(f"CSV results written to {output_path}")


def write_summary(results: List[Dict], output_path: Path) -> None:
    """Write markdown summary with analysis."""
    with open(output_path, "w") as f:
        f.write("# Retriever Comparison Results\n\n")
        f.write("Comparison of BM25 (sparse), Dense (FAISS), and Hybrid (RRF fusion) retrievers.\n\n")
        
        # Table
        f.write("| Retriever | Avg Cost (USD) | Avg Latency (ms) | Avg Faithfulness | Avg Answer Relevance | nDCG@10 | Recall@10 |\n")
        f.write("|-----------|----------------|------------------|------------------|---------------------|---------|----------|\n")
        
        for r in results:
            ndcg = f"{r['ndcg_at_10']:.4f}" if r['ndcg_at_10'] is not None else "N/A"
            recall = f"{r['recall_at_10']:.4f}" if r['recall_at_10'] is not None else "N/A"
            f.write(
                f"| {r['retriever']} | "
                f"${r['avg_cost']:.6f} | "
                f"{r['avg_latency_ms']:.1f} | "
                f"{r['avg_faithfulness']:.2f}/5 | "
                f"{r['avg_answer_relevance']:.2f}/5 | "
                f"{ndcg} | "
                f"{recall} |\n"
            )
        
        f.write("\n## Analysis\n\n")
        
        # Find best on different dimensions
        best_cost = min(results, key=lambda x: x["avg_cost"])
        best_faithfulness = max(results, key=lambda x: x["avg_faithfulness"])
        best_relevance = max(results, key=lambda x: x["avg_answer_relevance"])
        
        # Find best faithfulness/cost ratio
        best_ratio = 0
        best_ratio_retriever = None
        for r in results:
            if r["avg_cost"] > 0:
                ratio = r["avg_faithfulness"] / r["avg_cost"]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_ratio_retriever = r
        
        f.write("### Key Findings\n\n")
        
        f.write(f"- **Best cost efficiency**: `{best_cost['retriever']}` at ${best_cost['avg_cost']:.6f}/query\n")
        f.write(f"- **Highest faithfulness**: `{best_faithfulness['retriever']}` at {best_faithfulness['avg_faithfulness']:.2f}/5\n")
        f.write(f"- **Highest answer relevance**: `{best_relevance['retriever']}` at {best_relevance['avg_answer_relevance']:.2f}/5\n")
        
        if best_ratio_retriever:
            f.write(
                f"- **Best faithfulness/cost ratio**: `{best_ratio_retriever['retriever']}` "
                f"({best_ratio_retriever['avg_faithfulness']:.2f}/5 at ${best_ratio_retriever['avg_cost']:.6f})\n"
            )
        
        f.write("\n### Retriever Characteristics\n\n")
        
        # Analyze patterns
        dense_result = next((r for r in results if r["retriever"] == "dense"), None)
        bm25_result = next((r for r in results if r["retriever"] == "bm25"), None)
        hybrid_result = next((r for r in results if r["retriever"] == "hybrid"), None)
        
        if bm25_result and dense_result:
            if bm25_result["avg_latency_ms"] < dense_result["avg_latency_ms"]:
                f.write("- **BM25 is faster** than dense retrieval (no embedding computation)\n")
            else:
                f.write("- **Dense retrieval is comparable in speed** (FAISS is highly optimized)\n")
        
        if hybrid_result and dense_result and bm25_result:
            if hybrid_result["avg_faithfulness"] >= max(dense_result["avg_faithfulness"], bm25_result["avg_faithfulness"]):
                f.write("- **Hybrid retrieval matches or exceeds** both individual retrievers on faithfulness\n")
            else:
                f.write("- **Hybrid retrieval provides balanced** results between BM25 and dense\n")
        
        # Note about retrieval metrics
        has_retrieval_metrics = any(r["ndcg_at_10"] is not None for r in results)
        if not has_retrieval_metrics:
            f.write("\n### Note\n\n")
            f.write("Retrieval metrics (nDCG@10, Recall@10) were not computed for this comparison. ")
            f.write("To include them, run `scripts/eval_retrieval.py` on each variant's output.\n")
    
    print(f"Summary written to {output_path}")


def main():
    """Main entry point."""
    results = []
    
    for variant in RETRIEVER_VARIANTS:
        result = process_variant(variant)
        if result:
            results.append(result)
    
    if not results:
        print("Error: No variant data found")
        return 1
    
    # Sort by retriever name for consistent ordering
    retriever_order = {"bm25": 0, "dense": 1, "hybrid": 2}
    results.sort(key=lambda x: retriever_order.get(x["retriever"], 99))
    
    # Write outputs
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    write_csv(results, output_dir / "retriever_comparison.csv")
    write_summary(results, output_dir / "retriever_comparison_summary.md")
    
    print(f"\nProcessed {len(results)} retriever variants successfully")
    return 0


if __name__ == "__main__":
    exit(main())
