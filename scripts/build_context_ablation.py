#!/usr/bin/env python3
"""Build context count ablation summary from outputs."""

import csv
from pathlib import Path
from typing import Dict, List, Optional


# Define the ablation variants in order
ABLATION_VARIANTS = [
    {"context_count": 3, "dir": "outputs/context_ablation/ctx3"},
    {"context_count": 5, "dir": "outputs/context_ablation/ctx5"},
    {"context_count": 10, "dir": "outputs/context_ablation/ctx10"},
    {"context_count": 20, "dir": "outputs/context_ablation/ctx20"},
]


def load_costs(dir_path: Path) -> Optional[Dict]:
    """Load costs.csv and compute averages."""
    costs_file = dir_path / "costs.csv"
    if not costs_file.exists():
        return None
    
    costs = []
    latencies = []
    prompt_tokens = []
    
    with open(costs_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            costs.append(float(row["estimated_cost_usd"]))
            latencies.append(float(row["latency_ms"]))
            prompt_tokens.append(int(row["prompt_tokens"]))
    
    if not costs:
        return None
    
    return {
        "avg_cost_usd": sum(costs) / len(costs),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
        "count": len(costs)
    }


def load_judge_scores(dir_path: Path) -> Optional[Dict]:
    """Load judge_scores.csv and compute averages."""
    scores_file = dir_path / "judge_scores.csv"
    if not scores_file.exists():
        return None
    
    faithfulness_scores = []
    relevance_scores = []
    
    with open(scores_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            faithfulness_scores.append(float(row["faithfulness_score"]))
            relevance_scores.append(float(row["answer_relevance_score"]))
    
    if not faithfulness_scores:
        return None
    
    return {
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "avg_answer_relevance": sum(relevance_scores) / len(relevance_scores),
        "count": len(faithfulness_scores)
    }


def write_context_ablation_csv(variants_data: List[Dict], output_path: Path):
    """Write context_ablation.csv with all metrics."""
    fieldnames = [
        "context_count",
        "avg_cost_usd",
        "avg_latency_ms",
        "avg_prompt_tokens",
        "avg_faithfulness",
        "avg_answer_relevance"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for variant in variants_data:
            writer.writerow({
                "context_count": variant["context_count"],
                "avg_cost_usd": f"{variant['avg_cost_usd']:.6f}",
                "avg_latency_ms": f"{variant['avg_latency_ms']:.1f}",
                "avg_prompt_tokens": f"{variant['avg_prompt_tokens']:.0f}",
                "avg_faithfulness": f"{variant['avg_faithfulness']:.2f}" if variant.get("avg_faithfulness") is not None else "N/A",
                "avg_answer_relevance": f"{variant['avg_answer_relevance']:.2f}" if variant.get("avg_answer_relevance") is not None else "N/A"
            })


def write_context_ablation_summary(variants_data: List[Dict], output_path: Path):
    """Write context_ablation_summary.md with table and key takeaways."""
    with open(output_path, "w") as f:
        f.write("# Context Count Ablation Summary\n\n")
        f.write("Comparing quality and cost across different context window sizes (3, 5, 10, 20 docs).\n")
        f.write("All variants use the same optimized retrieval settings (conditional reranking, gpt-5.4-mini, top_k=20).\n\n")
        
        # Write table
        f.write("## Results\n\n")
        f.write("| Context Count | Avg Cost ($) | Avg Latency (ms) | Avg Prompt Tokens | Faithfulness | Answer Relevance |\n")
        f.write("|---------------|--------------|------------------|-------------------|--------------|------------------|\n")
        
        for variant in variants_data:
            faithfulness = f"{variant['avg_faithfulness']:.2f}" if variant.get("avg_faithfulness") is not None else "N/A"
            relevance = f"{variant['avg_answer_relevance']:.2f}" if variant.get("avg_answer_relevance") is not None else "N/A"
            
            f.write(f"| {variant['context_count']} | ${variant['avg_cost_usd']:.4f} | {variant['avg_latency_ms']:.0f} | {variant['avg_prompt_tokens']:.0f} | {faithfulness} | {relevance} |\n")
        
        # Find sweet spot (best faithfulness/cost ratio)
        f.write("\n## Sweet Spot Analysis\n\n")
        
        valid_variants = [v for v in variants_data if v.get("avg_faithfulness") is not None]
        
        if valid_variants:
            # Calculate faithfulness/cost ratio for each variant
            best_ratio = 0
            best_variant = None
            
            for variant in valid_variants:
                if variant["avg_cost_usd"] > 0:
                    ratio = variant["avg_faithfulness"] / variant["avg_cost_usd"]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_variant = variant
            
            if best_variant:
                f.write(f"**Best faithfulness/cost ratio**: {best_variant['context_count']} docs\n")
                f.write(f"- Faithfulness: {best_variant['avg_faithfulness']:.2f}/5\n")
                f.write(f"- Cost: ${best_variant['avg_cost_usd']:.4f} per query\n")
                f.write(f"- Ratio: {best_ratio:.1f} (faithfulness per dollar)\n\n")
            
            # Analyze whether more docs helps or hurts
            f.write("## Quality vs Context Size\n\n")
            
            # Check faithfulness trend
            faith_values = [v["avg_faithfulness"] for v in valid_variants]
            max_faith_variant = max(valid_variants, key=lambda v: v["avg_faithfulness"])
            min_faith_variant = min(valid_variants, key=lambda v: v["avg_faithfulness"])
            
            f.write(f"- **Highest faithfulness**: {max_faith_variant['context_count']} docs ({max_faith_variant['avg_faithfulness']:.2f}/5)\n")
            f.write(f"- **Lowest faithfulness**: {min_faith_variant['context_count']} docs ({min_faith_variant['avg_faithfulness']:.2f}/5)\n")
            
            # Check if more docs hurts or helps
            if len(valid_variants) >= 2:
                first = valid_variants[0]
                last = valid_variants[-1]
                
                if last["avg_faithfulness"] > first["avg_faithfulness"]:
                    f.write(f"- **Trend**: More documents in context tends to **improve** faithfulness ({first['avg_faithfulness']:.2f} → {last['avg_faithfulness']:.2f})\n")
                elif last["avg_faithfulness"] < first["avg_faithfulness"]:
                    f.write(f"- **Trend**: More documents in context tends to **hurt** faithfulness ({first['avg_faithfulness']:.2f} → {last['avg_faithfulness']:.2f})\n")
                else:
                    f.write(f"- **Trend**: Faithfulness remains **stable** across context sizes ({first['avg_faithfulness']:.2f} → {last['avg_faithfulness']:.2f})\n")
            
            # Cost scaling
            f.write("\n## Cost Scaling\n\n")
            first_cost = valid_variants[0]["avg_cost_usd"]
            last_cost = valid_variants[-1]["avg_cost_usd"]
            cost_increase = ((last_cost - first_cost) / first_cost * 100) if first_cost > 0 else 0
            
            f.write(f"- **Cost increase from 3→20 docs**: {cost_increase:.1f}% (${first_cost:.4f} → ${last_cost:.4f})\n")
            f.write(f"- Prompt tokens scale roughly linearly with context count\n")
            
            # Recommendation
            f.write("\n## Recommendation\n\n")
            if best_variant:
                if best_variant["context_count"] <= 5:
                    f.write(f"**Recommendation**: Use **{best_variant['context_count']} docs** for optimal quality/cost trade-off.\n")
                    f.write("Smaller context windows are sufficient for this dataset and provide significant cost savings.\n")
                elif best_variant["context_count"] >= 10:
                    f.write(f"**Recommendation**: Consider using **{best_variant['context_count']} docs** for best quality.\n")
                    f.write("Larger context windows provide meaningful quality improvements on this dataset.\n")
                else:
                    f.write(f"**Recommendation**: **{best_variant['context_count']} docs** offers the best balance.\n")


def main():
    """Main entry point."""
    # Collect data for each variant
    variants_data = []
    
    for variant in ABLATION_VARIANTS:
        dir_path = Path(variant["dir"])
        
        # Load costs
        costs = load_costs(dir_path)
        if costs is None:
            print(f"Warning: Could not load costs from {dir_path}")
            continue
        
        # Load judge scores
        judge_scores = load_judge_scores(dir_path)
        if judge_scores is None:
            print(f"Warning: Could not load judge scores from {dir_path}")
        
        # Combine data
        variant_data = {
            "context_count": variant["context_count"],
            "dir": variant["dir"],
            "avg_cost_usd": costs["avg_cost_usd"],
            "avg_latency_ms": costs["avg_latency_ms"],
            "avg_prompt_tokens": costs["avg_prompt_tokens"],
            "avg_faithfulness": judge_scores["avg_faithfulness"] if judge_scores else None,
            "avg_answer_relevance": judge_scores["avg_answer_relevance"] if judge_scores else None,
        }
        
        variants_data.append(variant_data)
    
    # Write outputs
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "context_ablation.csv"
    summary_path = output_dir / "context_ablation_summary.md"
    
    write_context_ablation_csv(variants_data, csv_path)
    print(f"✓ Wrote {csv_path}")
    
    write_context_ablation_summary(variants_data, summary_path)
    print(f"✓ Wrote {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Context Count Ablation Summary")
    print("=" * 60)
    print(f"Variants processed: {len(variants_data)}")
    
    if variants_data:
        print("\nContext | Cost     | Faithfulness | Relevance")
        print("-" * 50)
        for v in variants_data:
            f = f"{v['avg_faithfulness']:.2f}" if v.get("avg_faithfulness") else "N/A"
            r = f"{v['avg_answer_relevance']:.2f}" if v.get("avg_answer_relevance") else "N/A"
            print(f"{v['context_count']:7} | ${v['avg_cost_usd']:.4f} | {f:12} | {r}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
