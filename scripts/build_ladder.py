#!/usr/bin/env python3
"""Build optimization ladder summary from step outputs."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


# Define the ladder steps in order
LADDER_STEPS = [
    {
        "name": "step1_gpt4o",
        "label": "Baseline (gpt-4o)",
        "dir": "outputs/ladder/step1_gpt4o"
    },
    {
        "name": "step2_model_swap",
        "label": "Model swap (gpt-5.4-mini)",
        "dir": "outputs/naive"
    },
    {
        "name": "step3_lower_topk",
        "label": "Lower top-k (100→10)",
        "dir": "outputs/ladder/step3_lower_topk"
    },
    {
        "name": "step4_cond_rerank",
        "label": "Conditional reranking",
        "dir": "outputs/ladder/step4_cond_rerank"
    },
    {
        "name": "step5_model_routing",
        "label": "Model routing (final)",
        "dir": "outputs/optimized"
    },
]


def load_costs(dir_path: Path) -> Dict:
    """Load costs.csv and compute averages."""
    costs_file = dir_path / "costs.csv"
    if not costs_file.exists():
        return {"avg_cost_usd": 0.0, "avg_latency_ms": 0.0, "count": 0}
    
    costs = []
    latencies = []
    
    with open(costs_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            costs.append(float(row["estimated_cost_usd"]))
            latencies.append(float(row["latency_ms"]))
    
    if not costs:
        return {"avg_cost_usd": 0.0, "avg_latency_ms": 0.0, "count": 0}
    
    return {
        "avg_cost_usd": sum(costs) / len(costs),
        "avg_latency_ms": sum(latencies) / len(latencies),
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


def compute_deltas(steps_data: List[Dict]) -> List[Dict]:
    """Compute delta vs previous step and cumulative delta vs baseline."""
    baseline_cost = steps_data[0]["avg_cost_usd"] if steps_data else 0.0
    
    for i, step in enumerate(steps_data):
        # Delta vs previous step
        if i == 0:
            step["cost_delta_vs_prev"] = 0.0
            step["cost_pct_change_vs_prev"] = 0.0
        else:
            prev_cost = steps_data[i - 1]["avg_cost_usd"]
            delta = step["avg_cost_usd"] - prev_cost
            step["cost_delta_vs_prev"] = delta
            if prev_cost > 0:
                step["cost_pct_change_vs_prev"] = (delta / prev_cost) * 100
            else:
                step["cost_pct_change_vs_prev"] = 0.0
        
        # Delta vs baseline
        delta_vs_baseline = step["avg_cost_usd"] - baseline_cost
        step["cost_delta_vs_baseline"] = delta_vs_baseline
        if baseline_cost > 0:
            step["cost_pct_change_vs_baseline"] = (delta_vs_baseline / baseline_cost) * 100
        else:
            step["cost_pct_change_vs_baseline"] = 0.0
    
    return steps_data


def write_ladder_csv(steps_data: List[Dict], output_path: Path):
    """Write ladder.csv with all metrics."""
    fieldnames = [
        "step",
        "label",
        "avg_cost_usd",
        "avg_latency_ms",
        "avg_faithfulness",
        "avg_answer_relevance",
        "cost_delta_vs_prev",
        "cost_pct_change_vs_prev",
        "cost_delta_vs_baseline",
        "cost_pct_change_vs_baseline"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for step in steps_data:
            writer.writerow({
                "step": step["name"],
                "label": step["label"],
                "avg_cost_usd": f"{step['avg_cost_usd']:.6f}",
                "avg_latency_ms": f"{step['avg_latency_ms']:.1f}",
                "avg_faithfulness": f"{step['avg_faithfulness']:.2f}" if step.get("avg_faithfulness") is not None else "N/A",
                "avg_answer_relevance": f"{step['avg_answer_relevance']:.2f}" if step.get("avg_answer_relevance") is not None else "N/A",
                "cost_delta_vs_prev": f"{step['cost_delta_vs_prev']:.6f}",
                "cost_pct_change_vs_prev": f"{step['cost_pct_change_vs_prev']:.1f}",
                "cost_delta_vs_baseline": f"{step['cost_delta_vs_baseline']:.6f}",
                "cost_pct_change_vs_baseline": f"{step['cost_pct_change_vs_baseline']:.1f}"
            })


def write_ladder_summary(steps_data: List[Dict], output_path: Path):
    """Write ladder_summary.md with table and key takeaways."""
    with open(output_path, "w") as f:
        f.write("# Optimization Ladder Summary\n\n")
        
        # Write table
        f.write("## Step-by-Step Results\n\n")
        f.write("| Step | Label | Avg Cost ($) | Avg Latency (ms) | Faithfulness | Answer Relevance | Cost Δ vs Prev | Cost % Δ vs Baseline |\n")
        f.write("|------|-------|--------------|------------------|--------------|------------------|----------------|---------------------|\n")
        
        for step in steps_data:
            faithfulness = f"{step['avg_faithfulness']:.2f}" if step.get("avg_faithfulness") is not None else "N/A"
            relevance = f"{step['avg_answer_relevance']:.2f}" if step.get("avg_answer_relevance") is not None else "N/A"
            delta_prev = f"{step['cost_delta_vs_prev']:.4f}" if step['cost_delta_vs_prev'] < 0 else f"+{step['cost_delta_vs_prev']:.4f}"
            pct_baseline = f"{step['cost_pct_change_vs_baseline']:.1f}%"
            
            f.write(f"| {step['name']} | {step['label']} | ${step['avg_cost_usd']:.4f} | {step['avg_latency_ms']:.0f} | {faithfulness} | {relevance} | {delta_prev} | {pct_baseline} |\n")
        
        # Write key takeaways
        f.write("\n## Key Takeaways\n\n")
        
        # Find biggest cost lever
        max_savings_idx = 0
        max_savings = 0.0
        for i in range(1, len(steps_data)):
            savings = abs(steps_data[i]["cost_delta_vs_prev"])
            if savings > max_savings:
                max_savings = savings
                max_savings_idx = i
        
        if max_savings_idx > 0:
            step = steps_data[max_savings_idx]
            f.write(f"- **Biggest single cost reduction**: {step['label']} saved ${abs(step['cost_delta_vs_prev']):.4f} per query ({abs(step['cost_pct_change_vs_prev']):.1f}%)\n")
        
        # Final cumulative savings
        if len(steps_data) > 1:
            final_step = steps_data[-1]
            baseline_step = steps_data[0]
            f.write(f"- **Final cumulative savings**: ${abs(final_step['cost_delta_vs_baseline']):.4f} per query ({abs(final_step['cost_pct_change_vs_baseline']):.1f}%) from baseline\n")
        
        # Quality impact
        if steps_data[0].get("avg_faithfulness") is not None and steps_data[-1].get("avg_faithfulness") is not None:
            baseline_f = steps_data[0]["avg_faithfulness"]
            final_f = steps_data[-1]["avg_faithfulness"]
            baseline_r = steps_data[0]["avg_answer_relevance"]
            final_r = steps_data[-1]["avg_answer_relevance"]
            
            f.write(f"- **Quality impact**: Faithfulness {baseline_f:.2f} → {final_f:.2f}, Answer relevance {baseline_r:.2f} → {final_r:.2f}\n")
            f.write(f"  - Quality maintained while reducing cost by {abs(final_step['cost_pct_change_vs_baseline']):.1f}%\n")


def main():
    """Main entry point."""
    # Collect data for each step
    steps_data = []
    
    for step in LADDER_STEPS:
        dir_path = Path(step["dir"])
        
        # Load costs
        costs = load_costs(dir_path)
        
        # Load judge scores (may not exist for all steps)
        judge_scores = load_judge_scores(dir_path)
        
        # Combine data
        step_data = {
            "name": step["name"],
            "label": step["label"],
            "dir": step["dir"],
            "avg_cost_usd": costs["avg_cost_usd"],
            "avg_latency_ms": costs["avg_latency_ms"],
            "avg_faithfulness": judge_scores["avg_faithfulness"] if judge_scores else None,
            "avg_answer_relevance": judge_scores["avg_answer_relevance"] if judge_scores else None,
        }
        
        steps_data.append(step_data)
    
    # Compute deltas
    steps_data = compute_deltas(steps_data)
    
    # Write outputs
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "ladder.csv"
    summary_path = output_dir / "ladder_summary.md"
    
    write_ladder_csv(steps_data, csv_path)
    print(f"✓ Wrote {csv_path}")
    
    write_ladder_summary(steps_data, summary_path)
    print(f"✓ Wrote {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Optimization Ladder Summary")
    print("=" * 60)
    print(f"Steps processed: {len(steps_data)}")
    
    if len(steps_data) > 1:
        baseline = steps_data[0]["avg_cost_usd"]
        final = steps_data[-1]["avg_cost_usd"]
        savings = baseline - final
        pct = (savings / baseline * 100) if baseline > 0 else 0
        print(f"Baseline cost: ${baseline:.4f} per query")
        print(f"Final cost: ${final:.4f} per query")
        print(f"Total savings: ${savings:.4f} per query ({pct:.1f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
