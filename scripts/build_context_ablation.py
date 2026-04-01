#!/usr/bin/env python3
"""Build context ablation results from individual variant outputs."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


ABLATION_VARIANTS = [
    {"context_count": 3, "dir": "outputs/context_ablation/ctx3"},
    {"context_count": 5, "dir": "outputs/context_ablation/ctx5"},
    {"context_count": 10, "dir": "outputs/context_ablation/ctx10"},
    {"context_count": 20, "dir": "outputs/context_ablation/ctx20"},
]


def load_csv_data(filepath: Path) -> List[Dict]:
    """Load CSV file and return list of dictionaries."""
    if not filepath.exists():
        return []
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_averages(data: List[Dict], fields: List[str]) -> Dict[str, float]:
    """Compute averages for specified fields."""
    if not data:
        return {field: 0.0 for field in fields}
    
    averages = {}
    for field in fields:
        values = [float(row[field]) for row in data if field in row and row[field]]
        averages[field] = sum(values) / len(values) if values else 0.0
    
    return averages


def process_variant(variant: Dict) -> Optional[Dict]:
    """Process a single variant and return aggregated metrics."""
    variant_dir = Path(variant["dir"])
    
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
    cost_avgs = compute_averages(costs_data, ["estimated_cost_usd", "latency_ms", "prompt_tokens"])
    
    # Compute averages from judge_scores.csv
    judge_avgs = compute_averages(judge_data, ["faithfulness_score", "answer_relevance_score"])
    
    return {
        "context_count": variant["context_count"],
        "avg_cost_usd": cost_avgs["estimated_cost_usd"],
        "avg_latency_ms": cost_avgs["latency_ms"],
        "avg_prompt_tokens": cost_avgs["prompt_tokens"],
        "avg_faithfulness": judge_avgs["faithfulness_score"],
        "avg_answer_relevance": judge_avgs["answer_relevance_score"],
    }


def write_csv(results: List[Dict], output_path: Path) -> None:
    """Write results to CSV file."""
    fieldnames = [
        "context_count",
        "avg_cost_usd",
        "avg_latency_ms",
        "avg_prompt_tokens",
        "avg_faithfulness",
        "avg_answer_relevance",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"CSV results written to {output_path}")


def write_summary(results: List[Dict], output_path: Path) -> None:
    """Write markdown summary with analysis."""
    with open(output_path, "w") as f:
        f.write("# Context Count Ablation Results\n\n")
        
        # Table
        f.write("| Context Count | Avg Cost (USD) | Avg Latency (ms) | Avg Prompt Tokens | Avg Faithfulness | Avg Answer Relevance |\n")
        f.write("|---------------|----------------|------------------|-------------------|------------------|---------------------|\n")
        
        for r in results:
            f.write(
                f"| {r['context_count']} | "
                f"${r['avg_cost_usd']:.6f} | "
                f"{r['avg_latency_ms']:.1f} | "
                f"{r['avg_prompt_tokens']:.1f} | "
                f"{r['avg_faithfulness']:.2f}/5 | "
                f"{r['avg_answer_relevance']:.2f}/5 |\n"
            )
        
        f.write("\n## Analysis\n\n")
        
        # Find best faithfulness/cost ratio
        best_ratio = 0
        best_variant = None
        for r in results:
            if r["avg_cost_usd"] > 0:
                ratio = r["avg_faithfulness"] / r["avg_cost_usd"]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_variant = r
        
        if best_variant:
            f.write(f"### Sweet Spot\n\n")
            f.write(
                f"**{best_variant['context_count']} documents** provides the best faithfulness/cost ratio "
                f"({best_variant['avg_faithfulness']:.2f}/5 faithfulness at ${best_variant['avg_cost_usd']:.6f} per query).\n\n"
            )
        
        # Quality vs context count
        f.write("### Quality vs. Context Count\n\n")
        
        faithfulness_trend = [r["avg_faithfulness"] for r in results]
        relevance_trend = [r["avg_answer_relevance"] for r in results]
        
        if all(faithfulness_trend[i] >= faithfulness_trend[i+1] for i in range(len(faithfulness_trend)-1)):
            f.write("- **Faithfulness decreases** as context count increases (more docs = more noise).\n")
        elif all(faithfulness_trend[i] <= faithfulness_trend[i+1] for i in range(len(faithfulness_trend)-1)):
            f.write("- **Faithfulness improves** as context count increases (more docs = better coverage).\n")
        else:
            f.write("- **Faithfulness is mixed** - neither consistently improves nor degrades with more context.\n")
        
        if all(relevance_trend[i] >= relevance_trend[i+1] for i in range(len(relevance_trend)-1)):
            f.write("- **Answer relevance decreases** as context count increases.\n")
        elif all(relevance_trend[i] <= relevance_trend[i+1] for i in range(len(relevance_trend)-1)):
            f.write("- **Answer relevance improves** as context count increases.\n")
        else:
            f.write("- **Answer relevance is relatively stable** across different context counts.\n")
        
        f.write("\n### Cost Scaling\n\n")
        
        if len(results) >= 2:
            first_cost = results[0]["avg_cost_usd"]
            last_cost = results[-1]["avg_cost_usd"]
            scaling_factor = last_cost / first_cost if first_cost > 0 else 0
            
            f.write(
                f"- Cost scales from ${first_cost:.6f} (3 docs) to ${last_cost:.6f} (20 docs).\n"
            )
            f.write(
                f"- Adding ~17 more documents increases cost by {scaling_factor:.1f}x.\n"
            )
    
    print(f"Summary written to {output_path}")


def main():
    """Main entry point."""
    results = []
    
    for variant in ABLATION_VARIANTS:
        result = process_variant(variant)
        if result:
            results.append(result)
    
    if not results:
        print("Error: No variant data found")
        return 1
    
    # Sort by context_count ascending
    results.sort(key=lambda x: x["context_count"])
    
    # Write outputs
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    write_csv(results, output_dir / "context_ablation.csv")
    write_summary(results, output_dir / "context_ablation_summary.md")
    
    print(f"\nProcessed {len(results)} variants successfully")
    return 0


if __name__ == "__main__":
    exit(main())
