"""Tests for build_optimal_comparison.py script."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from scripts.build_optimal_comparison import (
    compute_averages,
    format_currency,
    format_percentage,
    load_csv_data,
    load_json_data,
    process_config,
    write_csv,
    write_summary,
)


class TestLoadJsonData:
    """Tests for load_json_data function."""

    def test_load_json_valid_file(self, tmp_path: Path):
        """Test loading a valid JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        result = load_json_data(json_file)

        assert result == {"key": "value", "number": 42}

    def test_load_json_missing_file(self, tmp_path: Path):
        """Test handling of missing JSON file."""
        result = load_json_data(tmp_path / "missing.json")

        assert result is None

    def test_load_json_invalid_json(self, tmp_path: Path):
        """Test handling of invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{not valid json}")

        with pytest.raises(json.JSONDecodeError):
            load_json_data(json_file)


class TestLoadCsvData:
    """Tests for load_csv_data function."""

    def test_load_csv_valid_file(self, tmp_path: Path):
        """Test loading a valid CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\nval1,val2\nval3,val4\n")

        result = load_csv_data(csv_file)

        assert len(result) == 2
        assert result[0] == {"col1": "val1", "col2": "val2"}
        assert result[1] == {"col1": "val3", "col2": "val4"}

    def test_load_csv_missing_file(self, tmp_path: Path):
        """Test handling of missing CSV file."""
        result = load_csv_data(tmp_path / "missing.csv")

        assert result == []

    def test_load_csv_empty_file(self, tmp_path: Path):
        """Test handling of empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("col1,col2\n")

        result = load_csv_data(csv_file)

        assert result == []


class TestComputeAverages:
    """Tests for compute_averages function."""

    def test_compute_averages_valid_data(self):
        """Test computing averages from valid data."""
        data = [
            {"cost": "0.001", "latency": "100"},
            {"cost": "0.002", "latency": "200"},
            {"cost": "0.003", "latency": "300"},
        ]

        result = compute_averages(data, ["cost", "latency"])

        assert result["cost"] == pytest.approx(0.002)
        assert result["latency"] == pytest.approx(200.0)

    def test_compute_averages_empty_data(self):
        """Test computing averages from empty data."""
        result = compute_averages([], ["cost", "latency"])

        assert result["cost"] == 0.0
        assert result["latency"] == 0.0

    def test_compute_averages_missing_field(self):
        """Test computing averages when some fields are missing."""
        data = [
            {"cost": "0.001"},
            {"cost": "0.002", "latency": "200"},
        ]

        result = compute_averages(data, ["cost", "latency"])

        assert result["cost"] == pytest.approx(0.0015)
        assert result["latency"] == pytest.approx(200.0)


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_format_currency_small_amount(self):
        """Test formatting small currency amounts."""
        result = format_currency(123.45)
        assert result == "$123.45"

    def test_format_currency_large_amount(self):
        """Test formatting large currency amounts with commas."""
        result = format_currency(1234567.89)
        assert result == "$1,234,567.89"

    def test_format_currency_zero(self):
        """Test formatting zero currency."""
        result = format_currency(0.0)
        assert result == "$0.00"


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_format_percentage_positive(self):
        """Test formatting positive percentage."""
        result = format_percentage(25.5)
        assert result == "+25.5%"

    def test_format_percentage_negative(self):
        """Test formatting negative percentage."""
        result = format_percentage(-30.2)
        assert result == "-30.2%"

    def test_format_percentage_zero(self):
        """Test formatting zero percentage."""
        result = format_percentage(0.0)
        assert result == "+0.0%"


class TestProcessConfig:
    """Tests for process_config function."""

    def test_process_config_valid_data(self, tmp_path: Path):
        """Test processing a config with valid data."""
        # Create summary.json
        summary = {
            "config": {"output_dir": str(tmp_path)},
            "total_queries": 50,
            "total_cost_usd": 0.1,
        }
        (tmp_path / "summary.json").write_text(json.dumps(summary))

        # Create costs.csv
        (tmp_path / "costs.csv").write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,model,100,50,10,1500,0.001\n"
            "2,model,200,60,10,1600,0.002\n"
        )

        # Create judge_scores.csv
        (tmp_path / "judge_scores.csv").write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,answer_relevance_reasoning\n"
            "1,5,Good,4,Relevant\n"
            "2,4,OK,5,Very relevant\n"
        )

        result = process_config("test", str(tmp_path))

        assert result is not None
        assert result["config"] == "test"
        assert result["avg_cost_usd"] == pytest.approx(0.0015)
        assert result["avg_latency_ms"] == pytest.approx(1550.0)
        assert result["avg_faithfulness"] == pytest.approx(4.5)
        assert result["avg_answer_relevance"] == pytest.approx(4.5)

    def test_process_config_missing_summary(self, tmp_path: Path):
        """Test processing a config with missing summary.json."""
        result = process_config("test", str(tmp_path))

        assert result is None

    def test_process_config_missing_costs(self, tmp_path: Path):
        """Test processing a config with missing costs.csv."""
        (tmp_path / "summary.json").write_text('{"config": {}}')

        result = process_config("test", str(tmp_path))

        assert result is None

    def test_process_config_missing_judge_scores(self, tmp_path: Path):
        """Test processing a config with missing judge_scores.csv."""
        (tmp_path / "summary.json").write_text('{"config": {}}')
        (tmp_path / "costs.csv").write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
        )

        result = process_config("test", str(tmp_path))

        assert result is None


class TestWriteCsv:
    """Tests for write_csv function."""

    def test_csv_has_correct_columns(self, tmp_path: Path):
        """Test that output CSV has correct columns."""
        output_file = tmp_path / "test.csv"

        results = [
            {
                "config": "naive",
                "avg_cost_usd": 0.005,
                "avg_latency_ms": 2000.0,
                "avg_faithfulness": 4.0,
                "avg_answer_relevance": 4.5,
            },
            {
                "config": "optimal",
                "avg_cost_usd": 0.002,
                "avg_latency_ms": 1500.0,
                "avg_faithfulness": 4.2,
                "avg_answer_relevance": 4.6,
            },
        ]

        write_csv(results, output_file)

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        expected_columns = [
            "config",
            "avg_cost_usd",
            "avg_latency_ms",
            "avg_faithfulness",
            "avg_answer_relevance",
        ]

        assert fieldnames == expected_columns

    def test_csv_contains_both_configs(self, tmp_path: Path):
        """Test that CSV contains both naive and optimal configs."""
        output_file = tmp_path / "test.csv"

        results = [
            {
                "config": "naive",
                "avg_cost_usd": 0.005,
                "avg_latency_ms": 2000.0,
                "avg_faithfulness": 4.0,
                "avg_answer_relevance": 4.5,
            },
            {
                "config": "optimal",
                "avg_cost_usd": 0.002,
                "avg_latency_ms": 1500.0,
                "avg_faithfulness": 4.2,
                "avg_answer_relevance": 4.6,
            },
        ]

        write_csv(results, output_file)

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["config"] == "naive"
        assert rows[1]["config"] == "optimal"


class TestWriteSummary:
    """Tests for write_summary function."""

    def test_summary_has_required_sections(self, tmp_path: Path):
        """Test that summary markdown has required sections."""
        output_file = tmp_path / "test_summary.md"

        results = [
            {
                "config": "naive",
                "avg_cost_usd": 0.005,
                "avg_latency_ms": 2000.0,
                "avg_faithfulness": 4.0,
                "avg_answer_relevance": 4.5,
            },
            {
                "config": "optimal",
                "avg_cost_usd": 0.002,
                "avg_latency_ms": 1500.0,
                "avg_faithfulness": 4.2,
                "avg_answer_relevance": 4.6,
            },
        ]

        write_summary(results, output_file)

        content = output_file.read_text()

        assert "# Optimal Config Benchmark Results" in content
        assert "## Results" in content
        assert "## Deltas" in content
        assert "## Cost at Scale" in content
        assert "## Key Findings" in content

    def test_summary_includes_cost_at_scale(self, tmp_path: Path):
        """Test that summary includes cost at scale calculations."""
        output_file = tmp_path / "test_summary.md"

        results = [
            {
                "config": "naive",
                "avg_cost_usd": 0.005,
                "avg_latency_ms": 2000.0,
                "avg_faithfulness": 4.0,
                "avg_answer_relevance": 4.5,
            },
            {
                "config": "optimal",
                "avg_cost_usd": 0.002,
                "avg_latency_ms": 1500.0,
                "avg_faithfulness": 4.2,
                "avg_answer_relevance": 4.6,
            },
        ]

        write_summary(results, output_file)

        content = output_file.read_text()

        # Check for cost per 1M queries
        assert "Cost per 1M queries" in content
        assert "$5,000.00" in content  # 0.005 * 1M
        assert "$2,000.00" in content  # 0.002 * 1M

        # Check for annual savings
        assert "Annual savings" in content

    def test_summary_shows_cost_reduction(self, tmp_path: Path):
        """Test that summary shows cost reduction percentage."""
        output_file = tmp_path / "test_summary.md"

        results = [
            {
                "config": "naive",
                "avg_cost_usd": 0.005,
                "avg_latency_ms": 2000.0,
                "avg_faithfulness": 4.0,
                "avg_answer_relevance": 4.5,
            },
            {
                "config": "optimal",
                "avg_cost_usd": 0.002,
                "avg_latency_ms": 1500.0,
                "avg_faithfulness": 4.2,
                "avg_answer_relevance": 4.6,
            },
        ]

        write_summary(results, output_file)

        content = output_file.read_text()

        # Cost reduction is (0.002 - 0.005) / 0.005 * 100 = -60%
        assert "-60.0%" in content

    def test_summary_handles_missing_optimal(self, tmp_path: Path):
        """Test that summary handles missing optimal config."""
        output_file = tmp_path / "test_summary.md"

        results = [
            {
                "config": "naive",
                "avg_cost_usd": 0.005,
                "avg_latency_ms": 2000.0,
                "avg_faithfulness": 4.0,
                "avg_answer_relevance": 4.5,
            }
        ]

        write_summary(results, output_file)

        # Should not create file or return early
        # The function prints an error and returns without writing
        assert not output_file.exists()


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, tmp_path: Path):
        """Test the full workflow from data loading to summary generation."""
        # Create naive outputs
        naive_dir = tmp_path / "naive"
        naive_dir.mkdir()

        (naive_dir / "summary.json").write_text(
            json.dumps({"config": {"output_dir": str(naive_dir)}, "total_queries": 50})
        )
        (naive_dir / "costs.csv").write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,model,100,50,10,2000,0.005\n"
        )
        (naive_dir / "judge_scores.csv").write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,answer_relevance_reasoning\n"
            "1,4,Good,4.5,Relevant\n"
        )

        # Create optimal outputs
        optimal_dir = tmp_path / "optimal"
        optimal_dir.mkdir()

        (optimal_dir / "summary.json").write_text(
            json.dumps({"config": {"output_dir": str(optimal_dir)}, "total_queries": 50})
        )
        (optimal_dir / "costs.csv").write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,model,100,50,10,1500,0.002\n"
        )
        (optimal_dir / "judge_scores.csv").write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,answer_relevance_reasoning\n"
            "1,4.2,Good,4.6,Relevant\n"
        )

        # Process both configs
        naive_result = process_config("naive", str(naive_dir))
        optimal_result = process_config("optimal", str(optimal_dir))

        assert naive_result is not None
        assert optimal_result is not None

        results = [naive_result, optimal_result]

        # Write outputs
        csv_file = tmp_path / "comparison.csv"
        summary_file = tmp_path / "comparison_summary.md"

        write_csv(results, csv_file)
        write_summary(results, summary_file)

        # Verify outputs exist
        assert csv_file.exists()
        assert summary_file.exists()

        # Verify CSV content
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

        # Verify summary content
        summary_content = summary_file.read_text()
        assert "Optimal Config Benchmark Results" in summary_content
        assert "Cost at Scale" in summary_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
