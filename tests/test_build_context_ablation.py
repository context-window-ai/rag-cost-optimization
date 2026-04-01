#!/usr/bin/env python3
"""Tests for build_context_ablation.py script."""

import csv
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_context_ablation as bca


def write_csv(filepath: Path, data: List[Dict], fieldnames: List[str]) -> None:
    """Helper to write CSV data."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def test_load_csv_data_existing_file():
    """Test loading CSV data from an existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.csv"
        
        data = [
            {"col1": "1", "col2": "a"},
            {"col1": "2", "col2": "b"},
        ]
        write_csv(test_file, data, ["col1", "col2"])
        
        result = bca.load_csv_data(test_file)
        assert len(result) == 2
        assert result[0]["col1"] == "1"
        assert result[1]["col2"] == "b"


def test_load_csv_data_missing_file():
    """Test loading CSV data from a non-existent file."""
    result = bca.load_csv_data(Path("/nonexistent/file.csv"))
    assert result == []


def test_compute_averages_basic():
    """Test computing averages from data."""
    data = [
        {"val": "10", "other": "5"},
        {"val": "20", "other": "15"},
        {"val": "30", "other": "25"},
    ]
    
    result = bca.compute_averages(data, ["val", "other"])
    
    assert result["val"] == 20.0
    assert result["other"] == 15.0


def test_compute_averages_empty_data():
    """Test computing averages with empty data."""
    result = bca.compute_averages([], ["val1", "val2"])
    
    assert result["val1"] == 0.0
    assert result["val2"] == 0.0


def test_compute_averages_missing_field():
    """Test computing averages with missing fields."""
    data = [
        {"val": "10"},
        {"val": "20"},
    ]
    
    result = bca.compute_averages(data, ["val", "missing"])
    
    assert result["val"] == 15.0
    assert result["missing"] == 0.0


def test_process_variant_success():
    """Test processing a variant with valid data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "ctx3"
        variant_dir.mkdir()
        
        # Write costs.csv
        costs_data = [
            {"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100", "prompt_tokens": "500"},
            {"query_id": "2", "estimated_cost_usd": "0.002", "latency_ms": "200", "prompt_tokens": "600"},
        ]
        write_csv(variant_dir / "costs.csv", costs_data, 
                  ["query_id", "estimated_cost_usd", "latency_ms", "prompt_tokens"])
        
        # Write judge_scores.csv
        judge_data = [
            {"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"},
            {"query_id": "2", "faithfulness_score": "4", "answer_relevance_score": "5"},
        ]
        write_csv(variant_dir / "judge_scores.csv", judge_data,
                  ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        variant = {"context_count": 3, "dir": str(variant_dir)}
        result = bca.process_variant(variant)
        
        assert result is not None
        assert result["context_count"] == 3
        assert result["avg_cost_usd"] == 0.0015
        assert result["avg_latency_ms"] == 150.0
        assert result["avg_prompt_tokens"] == 550.0
        assert result["avg_faithfulness"] == 4.5
        assert result["avg_answer_relevance"] == 4.5


def test_process_variant_missing_costs():
    """Test processing a variant with missing costs.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "ctx3"
        variant_dir.mkdir()
        
        # Only write judge_scores.csv
        judge_data = [{"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"}]
        write_csv(variant_dir / "judge_scores.csv", judge_data,
                  ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        variant = {"context_count": 3, "dir": str(variant_dir)}
        result = bca.process_variant(variant)
        
        assert result is None


def test_process_variant_missing_judge_scores():
    """Test processing a variant with missing judge_scores.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "ctx3"
        variant_dir.mkdir()
        
        # Only write costs.csv
        costs_data = [{"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100", "prompt_tokens": "500"}]
        write_csv(variant_dir / "costs.csv", costs_data,
                  ["query_id", "estimated_cost_usd", "latency_ms", "prompt_tokens"])
        
        variant = {"context_count": 3, "dir": str(variant_dir)}
        result = bca.process_variant(variant)
        
        assert result is None


def test_csv_output_columns():
    """Test that CSV output has correct columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_path = tmpdir / "test.csv"
        
        results = [
            {
                "context_count": 3,
                "avg_cost_usd": 0.001,
                "avg_latency_ms": 100.0,
                "avg_prompt_tokens": 500.0,
                "avg_faithfulness": 4.5,
                "avg_answer_relevance": 4.0,
            }
        ]
        
        bca.write_csv(results, output_path)
        
        # Read back and check columns
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert "context_count" in rows[0]
        assert "avg_cost_usd" in rows[0]
        assert "avg_latency_ms" in rows[0]
        assert "avg_prompt_tokens" in rows[0]
        assert "avg_faithfulness" in rows[0]
        assert "avg_answer_relevance" in rows[0]


def test_variants_in_ascending_order():
    """Test that variants appear in ascending context_count order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create variant directories
        for count in [10, 3, 20, 5]:
            variant_dir = tmpdir / f"ctx{count}"
            variant_dir.mkdir()
            
            costs_data = [{"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100", "prompt_tokens": "500"}]
            write_csv(variant_dir / "costs.csv", costs_data,
                      ["query_id", "estimated_cost_usd", "latency_ms", "prompt_tokens"])
            
            judge_data = [{"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"}]
            write_csv(variant_dir / "judge_scores.csv", judge_data,
                      ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        # Create results in random order
        results = []
        for count in [10, 3, 20, 5]:
            variant = {"context_count": count, "dir": str(tmpdir / f"ctx{count}")}
            result = bca.process_variant(variant)
            if result:
                results.append(result)
        
        # Sort as the script does
        results.sort(key=lambda x: x["context_count"])
        
        context_counts = [r["context_count"] for r in results]
        assert context_counts == [3, 5, 10, 20]


def test_graceful_missing_files():
    """Test that the script handles missing files gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create variant with no files
        variant_dir = tmpdir / "empty"
        variant_dir.mkdir()
        
        variant = {"context_count": 3, "dir": str(variant_dir)}
        result = bca.process_variant(variant)
        
        # Should return None and not crash
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
