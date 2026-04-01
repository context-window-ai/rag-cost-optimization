"""Tests for build_context_ablation.py script."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_context_ablation import (
    ABLATION_VARIANTS,
    compute_faithfulness_cost_ratio,
    load_costs,
    load_judge_scores,
    write_context_ablation_csv,
    write_context_ablation_summary,
)


class TestLoadCosts:
    """Tests for load_costs function."""

    def test_load_costs_valid_file(self, tmp_path: Path):
        """Test loading a valid costs.csv file."""
        costs_file = tmp_path / "costs.csv"
        costs_file.write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,model,100,50,10,1500,0.001\n"
            "2,model,200,60,10,1600,0.002\n"
        )

        result = load_costs(tmp_path)

        assert result["avg_cost_usd"] == pytest.approx(0.0015)
        assert result["avg_latency_ms"] == pytest.approx(1550.0)
        assert result["avg_prompt_tokens"] == pytest.approx(150.0)
        assert result["count"] == 2

    def test_load_costs_missing_file(self, tmp_path: Path):
        """Test handling of missing costs.csv file."""
        result = load_costs(tmp_path)

        assert result["avg_cost_usd"] == 0.0
        assert result["avg_latency_ms"] == 0.0
        assert result["avg_prompt_tokens"] == 0.0
        assert result["count"] == 0

    def test_load_costs_empty_file(self, tmp_path: Path):
        """Test handling of empty costs.csv file."""
        costs_file = tmp_path / "costs.csv"
        costs_file.write_text("query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n")

        result = load_costs(tmp_path)

        assert result["avg_cost_usd"] == 0.0
        assert result["avg_latency_ms"] == 0.0
        assert result["avg_prompt_tokens"] == 0.0
        assert result["count"] == 0


class TestLoadJudgeScores:
    """Tests for load_judge_scores function."""

    def test_load_judge_scores_valid_file(self, tmp_path: Path):
        """Test loading a valid judge_scores.csv file."""
        scores_file = tmp_path / "judge_scores.csv"
        scores_file.write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,answer_relevance_reasoning\n"
            "1,5,Good,4,Relevant\n"
            "2,4,OK,5,Very relevant\n"
        )

        result = load_judge_scores(tmp_path)

        assert result is not None
        assert result["avg_faithfulness"] == pytest.approx(4.5)
        assert result["avg_answer_relevance"] == pytest.approx(4.5)
        assert result["count"] == 2

    def test_load_judge_scores_missing_file(self, tmp_path: Path):
        """Test handling of missing judge_scores.csv file."""
        result = load_judge_scores(tmp_path)

        assert result is None

    def test_load_judge_scores_empty_file(self, tmp_path: Path):
        """Test handling of empty judge_scores.csv file."""
        scores_file = tmp_path / "judge_scores.csv"
        scores_file.write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,answer_relevance_reasoning\n"
        )

        result = load_judge_scores(tmp_path)

        assert result is None


class TestComputeFaithfulnessCostRatio:
    """Tests for compute_faithfulness_cost_ratio function."""

    def test_ratio_positive_values(self):
        """Test ratio with positive values."""
        data = {
            "avg_cost_usd": 0.001,
            "avg_faithfulness": 4.5
        }

        ratio = compute_faithfulness_cost_ratio(data)

        assert ratio == pytest.approx(4500.0)

    def test_ratio_zero_cost(self):
        """Test ratio with zero cost."""
        data = {
            "avg_cost_usd": 0.0,
            "avg_faithfulness": 4.5
        }

        ratio = compute_faithfulness_cost_ratio(data)

        assert ratio == 0.0

    def test_ratio_missing_faithfulness(self):
        """Test ratio with missing faithfulness."""
        data = {
            "avg_cost_usd": 0.001,
        }

        ratio = compute_faithfulness_cost_ratio(data)

        assert ratio == 0.0


class TestWriteContextAblationCSV:
    """Tests for write_context_ablation_csv function."""

    def test_csv_has_correct_columns(self, tmp_path: Path):
        """Test that output CSV has correct columns."""
        output_file = tmp_path / "test.csv"

        variants_data = [
            {
                "context_count": 3,
                "avg_cost_usd": 0.001,
                "avg_latency_ms": 1500.0,
                "avg_prompt_tokens": 1000.0,
                "avg_faithfulness": 4.5,
                "avg_answer_relevance": 4.0,
            }
        ]

        write_context_ablation_csv(variants_data, output_file)

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        expected_columns = [
            "context_count",
            "avg_cost_usd",
            "avg_latency_ms",
            "avg_prompt_tokens",
            "avg_faithfulness",
            "avg_answer_relevance"
        ]

        assert fieldnames == expected_columns

    def test_csv_variants_in_ascending_order(self, tmp_path: Path):
        """Test that variants appear in ascending context_count order."""
        output_file = tmp_path / "test.csv"

        # Create variants in order (already sorted by ABLATION_VARIANTS)
        variants_data = [
            {"context_count": 3, "avg_cost_usd": 0.001, "avg_latency_ms": 1500.0,
             "avg_prompt_tokens": 1000.0, "avg_faithfulness": 4.5, "avg_answer_relevance": 4.0},
            {"context_count": 5, "avg_cost_usd": 0.002, "avg_latency_ms": 1600.0,
             "avg_prompt_tokens": 1500.0, "avg_faithfulness": 4.6, "avg_answer_relevance": 4.1},
            {"context_count": 10, "avg_cost_usd": 0.003, "avg_latency_ms": 1700.0,
             "avg_prompt_tokens": 2000.0, "avg_faithfulness": 4.7, "avg_answer_relevance": 4.2},
            {"context_count": 20, "avg_cost_usd": 0.004, "avg_latency_ms": 1800.0,
             "avg_prompt_tokens": 2500.0, "avg_faithfulness": 4.8, "avg_answer_relevance": 4.3},
        ]

        write_context_ablation_csv(variants_data, output_file)

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        context_counts = [int(row["context_count"]) for row in rows]
        assert context_counts == [3, 5, 10, 20]
        assert context_counts == sorted(context_counts)

    def test_csv_handles_missing_judge_scores(self, tmp_path: Path):
        """Test that CSV handles missing judge scores gracefully."""
        output_file = tmp_path / "test.csv"

        variants_data = [
            {
                "context_count": 3,
                "avg_cost_usd": 0.001,
                "avg_latency_ms": 1500.0,
                "avg_prompt_tokens": 1000.0,
                "avg_faithfulness": None,
                "avg_answer_relevance": None,
            }
        ]

        write_context_ablation_csv(variants_data, output_file)

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["avg_faithfulness"] == "N/A"
        assert row["avg_answer_relevance"] == "N/A"


class TestWriteContextAblationSummary:
    """Tests for write_context_ablation_summary function."""

    def test_summary_has_table_and_interpretation(self, tmp_path: Path):
        """Test that summary markdown has required sections."""
        output_file = tmp_path / "test_summary.md"

        variants_data = [
            {"context_count": 3, "avg_cost_usd": 0.001, "avg_latency_ms": 1500.0,
             "avg_prompt_tokens": 1000.0, "avg_faithfulness": 4.5, "avg_answer_relevance": 4.0,
             "faithfulness_cost_ratio": 4500.0},
            {"context_count": 5, "avg_cost_usd": 0.002, "avg_latency_ms": 1600.0,
             "avg_prompt_tokens": 1500.0, "avg_faithfulness": 4.6, "avg_answer_relevance": 4.1,
             "faithfulness_cost_ratio": 2300.0},
        ]

        write_context_ablation_summary(variants_data, output_file)

        content = output_file.read_text()

        assert "# Context Count Ablation Summary" in content
        assert "## Results" in content
        assert "## Sweet Spot Analysis" in content
        assert "## Interpretation" in content
        assert "## Recommendation" in content

    def test_summary_identifies_sweet_spot(self, tmp_path: Path):
        """Test that summary identifies the best faithfulness/cost ratio."""
        output_file = tmp_path / "test_summary.md"

        variants_data = [
            {"context_count": 3, "avg_cost_usd": 0.001, "avg_latency_ms": 1500.0,
             "avg_prompt_tokens": 1000.0, "avg_faithfulness": 4.5, "avg_answer_relevance": 4.0,
             "faithfulness_cost_ratio": 4500.0},
            {"context_count": 5, "avg_cost_usd": 0.002, "avg_latency_ms": 1600.0,
             "avg_prompt_tokens": 1500.0, "avg_faithfulness": 4.6, "avg_answer_relevance": 4.1,
             "faithfulness_cost_ratio": 2300.0},
        ]

        write_context_ablation_summary(variants_data, output_file)

        content = output_file.read_text()

        # The first variant has higher ratio
        assert "Best faithfulness/cost ratio**: 3 docs" in content


class TestAblationVariants:
    """Tests for ABLATION_VARIANTS configuration."""

    def test_variants_have_correct_context_counts(self):
        """Test that variants have expected context counts."""
        context_counts = [v["context_count"] for v in ABLATION_VARIANTS]

        assert context_counts == [3, 5, 10, 20]

    def test_variants_are_sorted_ascending(self):
        """Test that variants are in ascending order by context_count."""
        context_counts = [v["context_count"] for v in ABLATION_VARIANTS]

        assert context_counts == sorted(context_counts)


class TestGracefulErrorHandling:
    """Tests for graceful handling of errors and missing data."""

    def test_missing_costs_and_scores_directory(self, tmp_path: Path):
        """Test handling when output directory doesn't exist."""
        nonexistent_dir = tmp_path / "nonexistent"

        # Should not raise exceptions
        costs = load_costs(nonexistent_dir)
        scores = load_judge_scores(nonexistent_dir)

        assert costs["avg_cost_usd"] == 0.0
        assert scores is None

    def test_malformed_csv_handling(self, tmp_path: Path):
        """Test handling of malformed CSV files."""
        costs_file = tmp_path / "costs.csv"
        costs_file.write_text("this,is,not,a,valid,csv\n1,2,3\n")

        # Should handle gracefully (may raise or return empty)
        # The actual behavior depends on implementation
        try:
            result = load_costs(tmp_path)
            # If it doesn't raise, it should return valid structure
            assert "avg_cost_usd" in result
        except (KeyError, ValueError, IndexError):
            # If it raises, that's acceptable for malformed data
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
