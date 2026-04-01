#!/usr/bin/env python3
"""Build optimal config comparison summary from naive vs optimal outputs."""

import csv
import json
from pathlib import Path
from typing import Dict, Optional


def load_json_data(filepath: Path) -> Optional[Dict]:
    """Load JSON file and return dict."""
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        return json.load(f)


def load_csv_data(filepath: Path) -> list:
    """Load CSV file and return list of dictionaries."""
    if not filepath.exists():
        return []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_averages(data: list, fields: list) -> Dict[str, float]:
    """Compute averages for specified fields."""
    if not data:
        return {field: 0.0 for field in fields}

    averages = {}
    for field in fields:
        values = [float(row[field]) for row in data if field in row and row[field]]
        averages[field] = sum(values) / len(values) if values else 0.0

    return averages


def format_currency(amount: float) -> str:
    """Format currency with commas and dollar sign."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage with sign."""
    return f"{value:+.1f}%"


def process_config(config_name: str, output_dir: str) -> Optional[Dict]:
    """Process a single config and return aggregated metrics."""
    dir_path = Path(output_dir)

    # Load summary.json
    summary = load_json_data(dir_path / "summary.json")
    if not summary:
        print(f"Warning: No summary.json found in {dir_path}")
        return None

    # Load costs.csv
    costs_data = load_csv_data(dir_path / "costs.csv")
    if not costs_data:
        print(f"Warning: No costs.csv found in {dir_path}")
        return None

    # Load judge_scores.csv
    judge_data = load_csv_data(dir_path / "judge_scores.csv")
    if not judge_data:
        print(f"Warning: No judge_scores.csv found in {dir_path}")
        return None

    # Compute averages from costs.csv
    cost_avgs = compute_averages(costs_data, ["estimated_cost_usd", "latency_ms"])

    # Compute averages from judge_scores.csv
    judge_avgs = compute_averages(
        judge_data, ["faithfulness_score", "answer_relevance_score"]
    )

    return {
        "config": config_name,
        "avg_cost_usd": cost_avgs["estimated_cost_usd"],
        "avg_latency_ms": cost_avgs["latency_ms"],
        "avg_faithfulness": judge_avgs["faithfulness_score"],
        "avg_answer_relevance": judge_avgs["answer_relevance_score"],
    }


def write_csv(results: list, output_path: Path) -> None:
    """Write results to CSV file."""
    fieldnames = [
        "config",
        "avg_cost_usd",
        "avg_latency_ms",
        "avg_faithfulness",
        "avg_answer_relevance",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"CSV results written to {output_path}")


def write_summary(results: list, output_path: Path) -> None:
    """Write markdown summary with analysis."""
    naive = next((r for r in results if r["config"] == "naive"), None)
    optimal = next((r for r in results if r["config"] == "optimal"), None)

    if not naive or not optimal:
        print("Error: Both naive and optimal results are required")
        return

    # Calculate deltas
    cost_delta_pct = (
        (optimal["avg_cost_usd"] - naive["avg_cost_usd"]) / naive["avg_cost_usd"] * 100
        if naive["avg_cost_usd"] > 0
        else 0
    )
    faithfulness_delta_pct = (
        (optimal["avg_faithfulness"] - naive["avg_faithfulness"])
        / naive["avg_faithfulness"]
        * 100
        if naive["avg_faithfulness"] > 0
        else 0
    )

    # Calculate cost at scale
    cost_per_1m_naive = naive["avg_cost_usd"] * 1_000_000
    cost_per_1m_optimal = optimal["avg_cost_usd"] * 1_000_000
    daily_savings_at_1m_qpd = (naive["avg_cost_usd"] - optimal["avg_cost_usd"]) * 1_000_000
    annual_savings = daily_savings_at_1m_qpd * 365

    with open(output_path, "w") as f:
        f.write("# Optimal Config Benchmark Results\n\n")
        f.write(
            "Comparison of naive baseline vs optimized configuration combining all winning levers.\n\n"
        )

        # Results table
        f.write("## Results\n\n")
        f.write(
            "| Config | Avg Cost (USD) | Avg Latency (ms) | Avg Faithfulness | Avg Answer Relevance |\n"
        )
        f.write(
            "|--------|----------------|------------------|------------------|---------------------|\n"
        )

        for r in results:
            f.write(
                f"| {r['config']} | "
                f"${r['avg_cost_usd']:.6f} | "
                f"{r['avg_latency_ms']:.1f} | "
                f"{r['avg_faithfulness']:.2f}/5 | "
                f"{r['avg_answer_relevance']:.2f}/5 |\n"
            )

        # Cost and quality deltas
        f.write("\n## Deltas\n\n")
        f.write(f"- **Cost change**: {format_percentage(cost_delta_pct)}\n")
        f.write(f"- **Faithfulness change**: {format_percentage(faithfulness_delta_pct)}\n")

        # Cost at scale section
        f.write("\n## Cost at Scale\n\n")
        f.write("### Cost per 1M queries\n\n")
        f.write(f"- **Naive**: {format_currency(cost_per_1m_naive)}\n")
        f.write(f"- **Optimal**: {format_currency(cost_per_1m_optimal)}\n")
        f.write(f"- **Savings per 1M queries**: {format_currency(cost_per_1m_naive - cost_per_1m_optimal)}\n\n")

        f.write("### Annual savings at 1M queries/day\n\n")
        f.write(f"- **Daily savings**: {format_currency(daily_savings_at_1m_qpd)}\n")
        f.write(f"- **Annual savings**: {format_currency(annual_savings)}\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        findings = []

        # Cost reduction
        if cost_delta_pct < 0:
            findings.append(
                f"- **{abs(cost_delta_pct):.1f}% cost reduction** achieved through optimization levers"
            )

        # Quality impact
        if abs(faithfulness_delta_pct) < 5:
            findings.append(
                f"- **Quality maintained** with faithfulness change of {format_percentage(faithfulness_delta_pct)}"
            )
        elif faithfulness_delta_pct > 0:
            findings.append(
                f"- **Quality improved** with faithfulness increase of {format_percentage(faithfulness_delta_pct)}"
            )
        else:
            findings.append(
                f"- **Minor quality trade-off** with faithfulness decrease of {format_percentage(faithfulness_delta_pct)}"
            )

        # Latency impact
        latency_delta = optimal["avg_latency_ms"] - naive["avg_latency_ms"]
        latency_delta_pct = (
            latency_delta / naive["avg_latency_ms"] * 100
            if naive["avg_latency_ms"] > 0
            else 0
        )
        if abs(latency_delta_pct) < 10:
            findings.append(
                f"- **Latency comparable** at {optimal['avg_latency_ms']:.0f}ms vs {naive['avg_latency_ms']:.0f}ms baseline"
            )
        elif latency_delta_pct < 0:
            findings.append(
                f"- **Latency improved** by {abs(latency_delta_pct):.1f}% ({naive['avg_latency_ms']:.0f}ms → {optimal['avg_latency_ms']:.0f}ms)"
            )

        # Scale impact
        if annual_savings > 1_000_000:
            findings.append(
                f"- **At scale**: {format_currency(annual_savings)} annual savings at 1M queries/day"
            )

        for finding in findings:
            f.write(f"{finding}\n")

    print(f"Summary written to {output_path}")


def main():
    """Main entry point."""
    results = []

    # Process naive config
    naive_result = process_config("naive", "outputs/naive")
    if naive_result:
        results.append(naive_result)

    # Process optimal config
    optimal_result = process_config("optimal", "outputs/optimal")
    if optimal_result:
        results.append(optimal_result)

    if len(results) < 2:
        print("Error: Both naive and optimal results are required")
        return 1

    # Write outputs
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    write_csv(results, output_dir / "optimal_comparison.csv")
    write_summary(results, output_dir / "optimal_comparison_summary.md")

    print(f"\nProcessed {len(results)} configs successfully")
    return 0


if __name__ == "__main__":
    exit(main())
