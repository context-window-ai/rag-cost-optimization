#!/usr/bin/env python3
"""
Build comparison report from naive and optimized pipeline outputs.

Joins cost, latency, retrieval, and judge score metrics into:
- outputs/comparison.csv: Per-query comparison
- outputs/comparison_summary.md: Aggregate statistics
- outputs/review_sheet.csv (with --review-sheet flag): Manual review sheet
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_costs_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load costs.csv into a dict keyed by query_id."""
    if not path.exists():
        return {}
    
    costs = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row['query_id']
            costs[query_id] = {
                'model': row['model'],
                'prompt_tokens': int(row['prompt_tokens']),
                'completion_tokens': int(row['completion_tokens']),
                'rerank_count': int(row['rerank_count']),
                'latency_ms': float(row['latency_ms']),
                'estimated_cost_usd': float(row['estimated_cost_usd'])
            }
    return costs


def load_judge_scores_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load judge_scores.csv into a dict keyed by query_id."""
    if not path.exists():
        return {}
    
    scores = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row['query_id']
            scores[query_id] = {
                'faithfulness_score': int(row['faithfulness_score']),
                'answer_relevance_score': int(row['answer_relevance_score']),
                'judge_cost_usd': float(row['estimated_cost_usd'])
            }
    return scores


def load_answers_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load answers.jsonl into a dict keyed by query_id."""
    if not path.exists():
        return {}
    
    answers = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            query_id = data['query_id']
            answers[query_id] = {
                'query': data['query'],
                'answer': data['answer'],
                'doc_ids': data['doc_ids']
            }
    return answers


def load_retrieval_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load retrieval_metadata.json into a dict."""
    if not path.exists():
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_comparison_csv(
    naive_dir: Path,
    optimized_dir: Path,
    output_path: Path
) -> List[Dict[str, Any]]:
    """Build comparison.csv joining all metrics."""
    
    # Load naive data
    naive_costs = load_costs_csv(naive_dir / 'costs.csv')
    naive_judge = load_judge_scores_csv(naive_dir / 'judge_scores.csv')
    naive_answers = load_answers_jsonl(naive_dir / 'answers.jsonl')
    
    # Load optimized data
    optimized_costs = load_costs_csv(optimized_dir / 'costs.csv')
    optimized_judge = load_judge_scores_csv(optimized_dir / 'judge_scores.csv')
    optimized_answers = load_answers_jsonl(optimized_dir / 'answers.jsonl')
    
    # Get all query IDs from both configs
    all_query_ids = set(naive_costs.keys()) | set(optimized_costs.keys())
    
    # Build comparison rows
    rows = []
    for query_id in sorted(all_query_ids, key=int):
        # Get query text (prefer naive, fallback to optimized)
        query_text = ''
        if query_id in naive_answers:
            query_text = naive_answers[query_id]['query']
        elif query_id in optimized_answers:
            query_text = optimized_answers[query_id]['query']
        
        row = {'query_id': query_id, 'query': query_text}
        
        # Naive metrics
        if query_id in naive_costs:
            cost_data = naive_costs[query_id]
            row['naive_cost_usd'] = cost_data['estimated_cost_usd']
            row['naive_latency_ms'] = cost_data['latency_ms']
            row['naive_prompt_tokens'] = cost_data['prompt_tokens']
            row['naive_completion_tokens'] = cost_data['completion_tokens']
            row['naive_rerank_count'] = cost_data['rerank_count']
        else:
            row['naive_cost_usd'] = None
            row['naive_latency_ms'] = None
            row['naive_prompt_tokens'] = None
            row['naive_completion_tokens'] = None
            row['naive_rerank_count'] = None
        
        # Naive judge scores
        if query_id in naive_judge:
            row['naive_faithfulness'] = naive_judge[query_id]['faithfulness_score']
            row['naive_answer_relevance'] = naive_judge[query_id]['answer_relevance_score']
            row['naive_judge_cost_usd'] = naive_judge[query_id]['judge_cost_usd']
        else:
            row['naive_faithfulness'] = None
            row['naive_answer_relevance'] = None
            row['naive_judge_cost_usd'] = None
        
        # Optimized metrics
        if query_id in optimized_costs:
            cost_data = optimized_costs[query_id]
            row['optimized_cost_usd'] = cost_data['estimated_cost_usd']
            row['optimized_latency_ms'] = cost_data['latency_ms']
            row['optimized_prompt_tokens'] = cost_data['prompt_tokens']
            row['optimized_completion_tokens'] = cost_data['completion_tokens']
            row['optimized_rerank_count'] = cost_data['rerank_count']
        else:
            row['optimized_cost_usd'] = None
            row['optimized_latency_ms'] = None
            row['optimized_prompt_tokens'] = None
            row['optimized_completion_tokens'] = None
            row['optimized_rerank_count'] = None
        
        # Optimized judge scores
        if query_id in optimized_judge:
            row['optimized_faithfulness'] = optimized_judge[query_id]['faithfulness_score']
            row['optimized_answer_relevance'] = optimized_judge[query_id]['answer_relevance_score']
            row['optimized_judge_cost_usd'] = optimized_judge[query_id]['judge_cost_usd']
        else:
            row['optimized_faithfulness'] = None
            row['optimized_answer_relevance'] = None
            row['optimized_judge_cost_usd'] = None
        
        rows.append(row)
    
    # Write CSV
    fieldnames = [
        'query_id', 'query',
        'naive_cost_usd', 'naive_latency_ms', 'naive_prompt_tokens', 
        'naive_completion_tokens', 'naive_rerank_count',
        'naive_faithfulness', 'naive_answer_relevance', 'naive_judge_cost_usd',
        'optimized_cost_usd', 'optimized_latency_ms', 'optimized_prompt_tokens',
        'optimized_completion_tokens', 'optimized_rerank_count',
        'optimized_faithfulness', 'optimized_answer_relevance', 'optimized_judge_cost_usd'
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return rows


def calculate_mean(values: List[float]) -> Optional[float]:
    """Calculate mean of a list, ignoring None values."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def build_summary_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Build comparison_summary.md with aggregate statistics."""
    
    # Extract values for each metric
    naive_costs = [r['naive_cost_usd'] for r in rows]
    naive_latencies = [r['naive_latency_ms'] for r in rows]
    naive_faithfulness = [r['naive_faithfulness'] for r in rows]
    naive_relevance = [r['naive_answer_relevance'] for r in rows]
    
    optimized_costs = [r['optimized_cost_usd'] for r in rows]
    optimized_latencies = [r['optimized_latency_ms'] for r in rows]
    optimized_faithfulness = [r['optimized_faithfulness'] for r in rows]
    optimized_relevance = [r['optimized_answer_relevance'] for r in rows]
    
    # Calculate means
    avg_naive_cost = calculate_mean(naive_costs)
    avg_naive_latency = calculate_mean(naive_latencies)
    avg_naive_faithfulness = calculate_mean(naive_faithfulness)
    avg_naive_relevance = calculate_mean(naive_relevance)
    
    avg_optimized_cost = calculate_mean(optimized_costs)
    avg_optimized_latency = calculate_mean(optimized_latencies)
    avg_optimized_faithfulness = calculate_mean(optimized_faithfulness)
    avg_optimized_relevance = calculate_mean(optimized_relevance)
    
    # Calculate deltas and percent changes
    def calc_delta(naive: Optional[float], optimized: Optional[float]) -> tuple:
        if naive is None or optimized is None:
            return None, None
        delta = optimized - naive
        if naive != 0:
            pct_change = (delta / naive) * 100
        else:
            pct_change = None
        return delta, pct_change
    
    cost_delta, cost_pct = calc_delta(avg_naive_cost, avg_optimized_cost)
    latency_delta, latency_pct = calc_delta(avg_naive_latency, avg_optimized_latency)
    faith_delta, faith_pct = calc_delta(avg_naive_faithfulness, avg_optimized_faithfulness)
    rel_delta, rel_pct = calc_delta(avg_naive_relevance, avg_optimized_relevance)
    
    # Build markdown
    md_lines = [
        "# Comparison Summary\n",
        "## Aggregate Metrics\n",
        "| Metric | Naive | Optimized | Delta | % Change |",
        "|--------|-------|-----------|-------|----------|"
    ]
    
    # Add rows
    def format_row(metric: str, naive: Optional[float], optimized: Optional[float], 
                   delta: Optional[float], pct: Optional[float], decimals: int = 6) -> str:
        def fmt(val: Optional[float]) -> str:
            if val is None:
                return "N/A"
            return f"{val:.{decimals}f}"
        
        def fmt_pct(val: Optional[float]) -> str:
            if val is None:
                return "N/A"
            return f"{val:.1f}%"
        
        return f"| {metric} | {fmt(naive)} | {fmt(optimized)} | {fmt(delta)} | {fmt_pct(pct)} |"
    
    md_lines.append(format_row("avg_cost_usd", avg_naive_cost, avg_optimized_cost, cost_delta, cost_pct, 6))
    md_lines.append(format_row("avg_latency_ms", avg_naive_latency, avg_optimized_latency, latency_delta, latency_pct, 2))
    md_lines.append(format_row("avg_faithfulness", avg_naive_faithfulness, avg_optimized_faithfulness, faith_delta, faith_pct, 2))
    md_lines.append(format_row("avg_answer_relevance", avg_naive_relevance, avg_optimized_relevance, rel_delta, rel_pct, 2))
    md_lines.append(f"| total_queries | {len(rows)} | {len(rows)} | 0 | 0.0% |")
    
    # Add interpretation note
    md_lines.append("\n## Interpretation\n")
    
    if cost_delta is not None and cost_delta < 0:
        savings_pct = abs(cost_pct) if cost_pct else 0
        md_lines.append(f"**Cost Savings:** The optimized pipeline achieves **{savings_pct:.1f}% cost reduction** on average.")
        md_lines.append(f"- Naive average cost: ${avg_naive_cost:.6f} per query")
        md_lines.append(f"- Optimized average cost: ${avg_optimized_cost:.6f} per query")
        md_lines.append(f"- Savings: ${abs(cost_delta):.6f} per query")
    elif cost_delta is not None and cost_delta > 0:
        md_lines.append(f"**Cost Increase:** The optimized pipeline costs **{cost_pct:.1f}% more** on average.")
    else:
        md_lines.append("**Cost:** No significant cost difference between pipelines.")
    
    # Quality note
    md_lines.append("\n**Quality Impact:**")
    if faith_delta is not None and rel_delta is not None:
        if faith_delta >= 0 and rel_delta >= 0:
            md_lines.append("The optimized pipeline maintains or improves quality metrics.")
        elif faith_delta < 0 or rel_delta < 0:
            md_lines.append("The optimized pipeline shows some quality degradation.")
            if faith_delta < 0:
                md_lines.append(f"- Faithfulness decreased by {abs(faith_delta):.2f} points")
            if rel_delta < 0:
                md_lines.append(f"- Answer relevance decreased by {abs(rel_delta):.2f} points")
    else:
        md_lines.append("Quality metrics not available for comparison (some judge scores missing).")
    
    # Write markdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')


def build_review_sheet(
    naive_dir: Path,
    optimized_dir: Path,
    output_path: Path
) -> None:
    """Build review_sheet.csv for manual spot-checking."""
    
    # Load data
    naive_answers = load_answers_jsonl(naive_dir / 'answers.jsonl')
    optimized_answers = load_answers_jsonl(optimized_dir / 'answers.jsonl')
    naive_judge = load_judge_scores_csv(naive_dir / 'judge_scores.csv')
    optimized_judge = load_judge_scores_csv(optimized_dir / 'judge_scores.csv')
    
    # Get all query IDs
    all_query_ids = set(naive_answers.keys()) | set(optimized_answers.keys())
    
    # Build rows
    rows = []
    for query_id in sorted(all_query_ids, key=int):
        # Get query text
        query_text = ''
        if query_id in naive_answers:
            query_text = naive_answers[query_id]['query']
        elif query_id in optimized_answers:
            query_text = optimized_answers[query_id]['query']
        
        # Get answers
        naive_answer = naive_answers.get(query_id, {}).get('answer', '')
        optimized_answer = optimized_answers.get(query_id, {}).get('answer', '')
        
        # Get judge scores
        naive_faith = naive_judge.get(query_id, {}).get('faithfulness_score', '')
        naive_rel = naive_judge.get(query_id, {}).get('answer_relevance_score', '')
        opt_faith = optimized_judge.get(query_id, {}).get('faithfulness_score', '')
        opt_rel = optimized_judge.get(query_id, {}).get('answer_relevance_score', '')
        
        rows.append({
            'query_id': query_id,
            'query': query_text,
            'naive_answer': naive_answer,
            'optimized_answer': optimized_answer,
            'naive_faithfulness': naive_faith if naive_faith != '' else None,
            'naive_answer_relevance': naive_rel if naive_rel != '' else None,
            'optimized_faithfulness': opt_faith if opt_faith != '' else None,
            'optimized_answer_relevance': opt_rel if opt_rel != '' else None,
            'reviewer_notes': '',
            'reviewer_correct': ''
        })
    
    # Write CSV
    fieldnames = [
        'query_id', 'query', 'naive_answer', 'optimized_answer',
        'naive_faithfulness', 'naive_answer_relevance',
        'optimized_faithfulness', 'optimized_answer_relevance',
        'reviewer_notes', 'reviewer_correct'
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Build comparison report from pipeline outputs')
    parser.add_argument(
        '--naive-dir',
        type=Path,
        default=Path('outputs/naive'),
        help='Directory containing naive pipeline outputs (default: outputs/naive)'
    )
    parser.add_argument(
        '--optimized-dir',
        type=Path,
        default=Path('outputs/optimized'),
        help='Directory containing optimized pipeline outputs (default: outputs/optimized)'
    )
    parser.add_argument(
        '--review-sheet',
        action='store_true',
        help='Also generate review_sheet.csv for manual spot-checking'
    )
    
    args = parser.parse_args()
    
    # Build comparison CSV
    print(f"Building comparison.csv from {args.naive_dir} and {args.optimized_dir}...")
    rows = build_comparison_csv(
        args.naive_dir,
        args.optimized_dir,
        Path('outputs/comparison.csv')
    )
    print(f"  Created outputs/comparison.csv with {len(rows)} queries")
    
    # Build summary markdown
    print("Building comparison_summary.md...")
    build_summary_markdown(rows, Path('outputs/comparison_summary.md'))
    print("  Created outputs/comparison_summary.md")
    
    # Build review sheet if requested
    if args.review_sheet:
        print("Building review_sheet.csv...")
        build_review_sheet(args.naive_dir, args.optimized_dir, Path('outputs/review_sheet.csv'))
        print("  Created outputs/review_sheet.csv")
    
    print("Done!")


if __name__ == '__main__':
    main()
