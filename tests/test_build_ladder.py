"""Tests for build_ladder.py script."""

import csv
import tempfile
from pathlib import Path

import pytest

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_ladder


def test_ladder_csv_has_correct_columns():
    """Test that ladder CSV has all required columns."""
    csv_path = Path("outputs/ladder.csv")
    
    if not csv_path.exists():
        pytest.skip("ladder.csv not found - run build_ladder.py first")
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        
        required_columns = [
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
        
        for col in required_columns:
            assert col in columns, f"Missing required column: {col}"


def test_ladder_csv_steps_in_correct_order():
    """Test that steps appear in correct order."""
    csv_path = Path("outputs/ladder.csv")
    
    if not csv_path.exists():
        pytest.skip("ladder.csv not found - run build_ladder.py first")
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        steps = [row["step"] for row in reader]
        
        expected_order = [
            "step1_gpt4o",
            "step2_model_swap",
            "step3_lower_topk",
            "step4_cond_rerank",
            "step5_model_routing"
        ]
        
        assert steps == expected_order, f"Steps not in correct order: {steps}"


def test_graceful_handling_of_missing_judge_scores():
    """Test that missing judge_scores.csv is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create costs.csv
        costs_file = tmpdir_path / "costs.csv"
        with open(costs_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "model", "prompt_tokens", "completion_tokens",
                "rerank_count", "latency_ms", "estimated_cost_usd"
            ])
            writer.writerow(["q1", "gpt-4o", 100, 50, 1, 1000, 0.01])
        
        # Test load_judge_scores with missing file
        result = build_ladder.load_judge_scores(tmpdir_path)
        
        assert result is None, "Should return None when judge_scores.csv is missing"


def test_load_costs_with_valid_file():
    """Test loading costs from a valid costs.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create costs.csv
        costs_file = tmpdir_path / "costs.csv"
        with open(costs_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "model", "prompt_tokens", "completion_tokens",
                "rerank_count", "latency_ms", "estimated_cost_usd"
            ])
            writer.writerow(["q1", "gpt-4o", 100, 50, 1, 1000, 0.01])
            writer.writerow(["q2", "gpt-4o", 150, 75, 1, 2000, 0.02])
        
        result = build_ladder.load_costs(tmpdir_path)
        
        assert result["avg_cost_usd"] == 0.015
        assert result["avg_latency_ms"] == 1500.0
        assert result["count"] == 2


def test_load_costs_with_missing_file():
    """Test loading costs when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        result = build_ladder.load_costs(tmpdir_path)
        
        assert result["avg_cost_usd"] == 0.0
        assert result["avg_latency_ms"] == 0.0
        assert result["count"] == 0


def test_load_judge_scores_with_valid_file():
    """Test loading judge scores from a valid file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create judge_scores.csv
        scores_file = tmpdir_path / "judge_scores.csv"
        with open(scores_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "faithfulness_score", "faithfulness_reasoning",
                "answer_relevance_score", "answer_relevance_reasoning",
                "prompt_tokens", "completion_tokens", "estimated_cost_usd"
            ])
            writer.writerow(["q1", "4", "Good", "5", "Excellent", 100, 50, 0.01])
            writer.writerow(["q2", "5", "Perfect", "4", "Good", 100, 50, 0.01])
        
        result = build_ladder.load_judge_scores(tmpdir_path)
        
        assert result is not None
        assert result["avg_faithfulness"] == 4.5
        assert result["avg_answer_relevance"] == 4.5
        assert result["count"] == 2


def test_compute_deltas():
    """Test computation of cost deltas."""
    steps_data = [
        {
            "name": "step1",
            "avg_cost_usd": 0.01,
        },
        {
            "name": "step2",
            "avg_cost_usd": 0.005,
        },
        {
            "name": "step3",
            "avg_cost_usd": 0.003,
        },
    ]
    
    result = build_ladder.compute_deltas(steps_data)
    
    # Step 1 (baseline)
    assert result[0]["cost_delta_vs_prev"] == 0.0
    assert result[0]["cost_delta_vs_baseline"] == 0.0
    assert result[0]["cost_pct_change_vs_baseline"] == 0.0
    
    # Step 2
    assert result[1]["cost_delta_vs_prev"] == -0.005
    assert result[1]["cost_delta_vs_baseline"] == -0.005
    assert result[1]["cost_pct_change_vs_baseline"] == -50.0
    
    # Step 3
    assert result[2]["cost_delta_vs_prev"] == -0.002
    assert result[2]["cost_delta_vs_baseline"] == -0.007
    assert result[2]["cost_pct_change_vs_baseline"] == -70.0


def test_ladder_summary_md_exists():
    """Test that ladder_summary.md was created."""
    summary_path = Path("outputs/ladder_summary.md")
    
    if not summary_path.exists():
        pytest.skip("ladder_summary.md not found - run build_ladder.py first")
    
    content = summary_path.read_text()
    
    # Check for key sections
    assert "# Optimization Ladder Summary" in content
    assert "## Step-by-Step Results" in content
    assert "## Key Takeaways" in content
    assert "Biggest single cost reduction" in content
    assert "Final cumulative savings" in content
    assert "Quality impact" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
