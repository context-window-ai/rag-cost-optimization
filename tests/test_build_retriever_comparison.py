#!/usr/bin/env python3
"""Tests for build_retriever_comparison.py script."""

import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_retriever_comparison as brc


def write_csv(filepath: Path, data: List[Dict], fieldnames: List[str]) -> None:
    """Helper to write CSV data."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def write_json(filepath: Path, data: Dict) -> None:
    """Helper to write JSON data."""
    with open(filepath, "w") as f:
        json.dump(data, f)


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
        
        result = brc.load_csv_data(test_file)
        assert len(result) == 2
        assert result[0]["col1"] == "1"
        assert result[1]["col2"] == "b"


def test_load_csv_data_missing_file():
    """Test loading CSV data from a non-existent file."""
    result = brc.load_csv_data(Path("/nonexistent/file.csv"))
    assert result == []


def test_load_json_data_existing_file():
    """Test loading JSON data from an existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.json"
        
        data = {"key": "value", "number": 42}
        write_json(test_file, data)
        
        result = brc.load_json_data(test_file)
        assert result["key"] == "value"
        assert result["number"] == 42


def test_load_json_data_missing_file():
    """Test loading JSON data from a non-existent file."""
    result = brc.load_json_data(Path("/nonexistent/file.json"))
    assert result is None


def test_compute_averages_basic():
    """Test computing averages from data."""
    data = [
        {"val": "10", "other": "5"},
        {"val": "20", "other": "15"},
        {"val": "30", "other": "25"},
    ]
    
    result = brc.compute_averages(data, ["val", "other"])
    
    assert result["val"] == 20.0
    assert result["other"] == 15.0


def test_compute_averages_empty_data():
    """Test computing averages with empty data."""
    result = brc.compute_averages([], ["val1", "val2"])
    
    assert result["val1"] == 0.0
    assert result["val2"] == 0.0


def test_compute_averages_missing_field():
    """Test computing averages with missing fields."""
    data = [
        {"val": "10"},
        {"val": "20"},
    ]
    
    result = brc.compute_averages(data, ["val", "missing"])
    
    assert result["val"] == 15.0
    assert result["missing"] == 0.0


def test_process_variant_success():
    """Test processing a variant with valid data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "dense"
        variant_dir.mkdir()
        
        # Write summary.json
        summary_data = {"retriever_type": "dense", "total_queries": 50}
        write_json(variant_dir / "summary.json", summary_data)
        
        # Write costs.csv
        costs_data = [
            {"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100"},
            {"query_id": "2", "estimated_cost_usd": "0.002", "latency_ms": "200"},
        ]
        write_csv(variant_dir / "costs.csv", costs_data, 
                  ["query_id", "estimated_cost_usd", "latency_ms"])
        
        # Write judge_scores.csv
        judge_data = [
            {"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"},
            {"query_id": "2", "faithfulness_score": "4", "answer_relevance_score": "5"},
        ]
        write_csv(variant_dir / "judge_scores.csv", judge_data,
                  ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        variant = {"retriever": "dense", "dir": str(variant_dir)}
        result = brc.process_variant(variant)
        
        assert result is not None
        assert result["retriever"] == "dense"
        assert result["avg_cost"] == 0.0015
        assert result["avg_latency_ms"] == 150.0
        assert result["avg_faithfulness"] == 4.5
        assert result["avg_answer_relevance"] == 4.5


def test_process_variant_with_retrieval_metrics():
    """Test processing a variant with retrieval metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "dense"
        variant_dir.mkdir()
        
        # Write summary.json
        write_json(variant_dir / "summary.json", {"total_queries": 50})
        
        # Write costs.csv
        write_csv(variant_dir / "costs.csv", 
                  [{"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100"}],
                  ["query_id", "estimated_cost_usd", "latency_ms"])
        
        # Write judge_scores.csv
        write_csv(variant_dir / "judge_scores.csv",
                  [{"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"}],
                  ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        # Write metrics.json
        write_json(variant_dir / "metrics.json", {"ndcg@10": 0.65, "recall@10": 0.75})
        
        variant = {"retriever": "dense", "dir": str(variant_dir)}
        result = brc.process_variant(variant)
        
        assert result is not None
        assert result["ndcg_at_10"] == 0.65
        assert result["recall_at_10"] == 0.75


def test_process_variant_missing_summary():
    """Test processing a variant with missing summary.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "dense"
        variant_dir.mkdir()
        
        # Only write costs.csv and judge_scores.csv
        write_csv(variant_dir / "costs.csv", 
                  [{"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100"}],
                  ["query_id", "estimated_cost_usd", "latency_ms"])
        write_csv(variant_dir / "judge_scores.csv",
                  [{"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"}],
                  ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        variant = {"retriever": "dense", "dir": str(variant_dir)}
        result = brc.process_variant(variant)
        
        assert result is None


def test_process_variant_missing_costs():
    """Test processing a variant with missing costs.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "dense"
        variant_dir.mkdir()
        
        write_json(variant_dir / "summary.json", {"total_queries": 50})
        write_csv(variant_dir / "judge_scores.csv",
                  [{"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"}],
                  ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        variant = {"retriever": "dense", "dir": str(variant_dir)}
        result = brc.process_variant(variant)
        
        assert result is None


def test_process_variant_missing_judge_scores():
    """Test processing a variant with missing judge_scores.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        variant_dir = tmpdir / "dense"
        variant_dir.mkdir()
        
        write_json(variant_dir / "summary.json", {"total_queries": 50})
        write_csv(variant_dir / "costs.csv", 
                  [{"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100"}],
                  ["query_id", "estimated_cost_usd", "latency_ms"])
        
        variant = {"retriever": "dense", "dir": str(variant_dir)}
        result = brc.process_variant(variant)
        
        assert result is None


def test_csv_output_format():
    """Test that CSV output has correct columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_path = tmpdir / "test.csv"
        
        results = [
            {
                "retriever": "dense",
                "avg_cost": 0.001,
                "avg_latency_ms": 100.0,
                "avg_faithfulness": 4.5,
                "avg_answer_relevance": 4.0,
                "ndcg_at_10": 0.65,
                "recall_at_10": 0.75,
            }
        ]
        
        brc.write_csv(results, output_path)
        
        # Read back and check columns
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert "retriever" in rows[0]
        assert "avg_cost" in rows[0]
        assert "avg_latency_ms" in rows[0]
        assert "avg_faithfulness" in rows[0]
        assert "avg_answer_relevance" in rows[0]
        assert "ndcg_at_10" in rows[0]
        assert "recall_at_10" in rows[0]


def test_csv_output_with_na_values():
    """Test that CSV output handles N/A values correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_path = tmpdir / "test.csv"
        
        results = [
            {
                "retriever": "bm25",
                "avg_cost": 0.001,
                "avg_latency_ms": 100.0,
                "avg_faithfulness": 4.5,
                "avg_answer_relevance": 4.0,
                "ndcg_at_10": None,
                "recall_at_10": None,
            }
        ]
        
        brc.write_csv(results, output_path)
        
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert rows[0]["ndcg_at_10"] == "N/A"
        assert rows[0]["recall_at_10"] == "N/A"


def test_retriever_ordering():
    """Test that retrievers appear in consistent order (bm25, dense, hybrid)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create variant directories
        for name in ["hybrid", "bm25", "dense"]:
            variant_dir = tmpdir / name
            variant_dir.mkdir()
            
            write_json(variant_dir / "summary.json", {"total_queries": 50})
            write_csv(variant_dir / "costs.csv", 
                      [{"query_id": "1", "estimated_cost_usd": "0.001", "latency_ms": "100"}],
                      ["query_id", "estimated_cost_usd", "latency_ms"])
            write_csv(variant_dir / "judge_scores.csv",
                      [{"query_id": "1", "faithfulness_score": "5", "answer_relevance_score": "4"}],
                      ["query_id", "faithfulness_score", "answer_relevance_score"])
        
        # Process variants
        results = []
        for name in ["hybrid", "bm25", "dense"]:
            variant = {"retriever": name, "dir": str(tmpdir / name)}
            result = brc.process_variant(variant)
            if result:
                results.append(result)
        
        # Sort as the script does
        retriever_order = {"bm25": 0, "dense": 1, "hybrid": 2}
        results.sort(key=lambda x: retriever_order.get(x["retriever"], 99))
        
        retriever_names = [r["retriever"] for r in results]
        assert retriever_names == ["bm25", "dense", "hybrid"]


def test_graceful_missing_files():
    """Test that the script handles missing files gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create variant with no files
        variant_dir = tmpdir / "empty"
        variant_dir.mkdir()
        
        variant = {"retriever": "dense", "dir": str(variant_dir)}
        result = brc.process_variant(variant)
        
        # Should return None and not crash
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
