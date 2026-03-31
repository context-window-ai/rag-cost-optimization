"""Tests for build_report.py script."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from scripts.build_report import (
    load_costs_csv,
    load_judge_scores_csv,
    load_answers_jsonl,
    load_retrieval_metadata,
    build_comparison_csv,
    build_summary_markdown,
    build_review_sheet,
    calculate_mean,
)


class TestLoadCostsCsv:
    """Tests for load_costs_csv function."""
    
    def test_load_costs_csv_basic(self, tmp_path):
        """Test loading a basic costs.csv file."""
        costs_file = tmp_path / "costs.csv"
        costs_file.write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,openai/gpt-4o,100,50,10,1000.5,0.00123\n"
            "2,openai/gpt-4o,200,75,10,1500.25,0.00234\n"
        )
        
        result = load_costs_csv(costs_file)
        
        assert len(result) == 2
        assert result['1']['model'] == 'openai/gpt-4o'
        assert result['1']['prompt_tokens'] == 100
        assert result['1']['completion_tokens'] == 50
        assert result['1']['rerank_count'] == 10
        assert result['1']['latency_ms'] == 1000.5
        assert result['1']['estimated_cost_usd'] == 0.00123
        assert result['2']['prompt_tokens'] == 200
    
    def test_load_costs_csv_missing_file(self, tmp_path):
        """Test handling of missing costs.csv file."""
        missing_file = tmp_path / "missing.csv"
        result = load_costs_csv(missing_file)
        assert result == {}


class TestLoadJudgeScoresCsv:
    """Tests for load_judge_scores_csv function."""
    
    def test_load_judge_scores_basic(self, tmp_path):
        """Test loading a basic judge_scores.csv file."""
        scores_file = tmp_path / "judge_scores.csv"
        scores_file.write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,"
            "answer_relevance_reasoning,prompt_tokens,completion_tokens,estimated_cost_usd\n"
            "1,5,Good,4,Relevant,100,50,0.001\n"
            "2,3,Okay,2,Somewhat,150,60,0.0015\n"
        )
        
        result = load_judge_scores_csv(scores_file)
        
        assert len(result) == 2
        assert result['1']['faithfulness_score'] == 5
        assert result['1']['answer_relevance_score'] == 4
        assert result['1']['judge_cost_usd'] == 0.001
        assert result['2']['faithfulness_score'] == 3
    
    def test_load_judge_scores_missing_file(self, tmp_path):
        """Test handling of missing judge_scores.csv file."""
        missing_file = tmp_path / "missing.csv"
        result = load_judge_scores_csv(missing_file)
        assert result == {}


class TestLoadAnswersJsonl:
    """Tests for load_answers_jsonl function."""
    
    def test_load_answers_basic(self, tmp_path):
        """Test loading a basic answers.jsonl file."""
        answers_file = tmp_path / "answers.jsonl"
        answers_file.write_text(
            '{"query_id": "1", "query": "Test query 1", "answer": "Answer 1", "doc_ids": ["a", "b"]}\n'
            '{"query_id": "2", "query": "Test query 2", "answer": "Answer 2", "doc_ids": ["c"]}\n'
        )
        
        result = load_answers_jsonl(answers_file)
        
        assert len(result) == 2
        assert result['1']['query'] == 'Test query 1'
        assert result['1']['answer'] == 'Answer 1'
        assert result['1']['doc_ids'] == ['a', 'b']
        assert result['2']['query'] == 'Test query 2'
    
    def test_load_answers_missing_file(self, tmp_path):
        """Test handling of missing answers.jsonl file."""
        missing_file = tmp_path / "missing.jsonl"
        result = load_answers_jsonl(missing_file)
        assert result == {}


class TestLoadRetrievalMetadata:
    """Tests for load_retrieval_metadata function."""
    
    def test_load_retrieval_metadata_basic(self, tmp_path):
        """Test loading a basic retrieval_metadata.json file."""
        metadata_file = tmp_path / "retrieval_metadata.json"
        data = {
            "1": {"retrieved_count": 100, "rerank_count": 10},
            "2": {"retrieved_count": 50, "rerank_count": 5}
        }
        metadata_file.write_text(json.dumps(data))
        
        result = load_retrieval_metadata(metadata_file)
        
        assert len(result) == 2
        assert result['1']['retrieved_count'] == 100
        assert result['2']['rerank_count'] == 5
    
    def test_load_retrieval_metadata_missing_file(self, tmp_path):
        """Test handling of missing retrieval_metadata.json file."""
        missing_file = tmp_path / "missing.json"
        result = load_retrieval_metadata(missing_file)
        assert result == {}


class TestCalculateMean:
    """Tests for calculate_mean function."""
    
    def test_calculate_mean_basic(self):
        """Test basic mean calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_mean(values)
        assert result == 3.0
    
    def test_calculate_mean_with_none(self):
        """Test mean calculation with None values."""
        values = [1.0, None, 3.0, None, 5.0]
        result = calculate_mean(values)
        assert result == 3.0
    
    def test_calculate_mean_all_none(self):
        """Test mean calculation with all None values."""
        values = [None, None, None]
        result = calculate_mean(values)
        assert result is None
    
    def test_calculate_mean_empty(self):
        """Test mean calculation with empty list."""
        values = []
        result = calculate_mean(values)
        assert result is None


class TestBuildComparisonCsv:
    """Tests for build_comparison_csv function."""
    
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Create test data files."""
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        naive_dir.mkdir()
        optimized_dir.mkdir()
        
        # Create naive costs.csv
        naive_costs = naive_dir / "costs.csv"
        naive_costs.write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,openai/gpt-4o,100,50,100,1000.0,0.001\n"
            "2,openai/gpt-4o,200,75,100,1500.0,0.002\n"
        )
        
        # Create naive judge_scores.csv
        naive_judge = naive_dir / "judge_scores.csv"
        naive_judge.write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,"
            "answer_relevance_reasoning,prompt_tokens,completion_tokens,estimated_cost_usd\n"
            "1,5,Good,4,Relevant,100,50,0.0001\n"
            "2,3,Okay,2,Somewhat,150,60,0.00015\n"
        )
        
        # Create naive answers.jsonl
        naive_answers = naive_dir / "answers.jsonl"
        naive_answers.write_text(
            '{"query_id": "1", "query": "Query 1", "answer": "Naive answer 1", "doc_ids": ["a"]}\n'
            '{"query_id": "2", "query": "Query 2", "answer": "Naive answer 2", "doc_ids": ["b"]}\n'
        )
        
        # Create optimized costs.csv
        opt_costs = optimized_dir / "costs.csv"
        opt_costs.write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,openai/gpt-5.4-mini,50,25,10,500.0,0.0005\n"
            "2,openai/gpt-5.4-mini,100,40,10,750.0,0.001\n"
        )
        
        # Create optimized answers.jsonl (no judge scores for optimized)
        opt_answers = optimized_dir / "answers.jsonl"
        opt_answers.write_text(
            '{"query_id": "1", "query": "Query 1", "answer": "Optimized answer 1", "doc_ids": ["c"]}\n'
            '{"query_id": "2", "query": "Query 2", "answer": "Optimized answer 2", "doc_ids": ["d"]}\n'
        )
        
        return tmp_path
    
    def test_build_comparison_csv_basic(self, setup_test_data):
        """Test basic comparison CSV generation."""
        tmp_path = setup_test_data
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        output_path = tmp_path / "comparison.csv"
        
        rows = build_comparison_csv(naive_dir, optimized_dir, output_path)
        
        # Check file was created
        assert output_path.exists()
        
        # Check rows
        assert len(rows) == 2
        
        # Check first row
        row1 = rows[0]
        assert row1['query_id'] == '1'
        assert row1['query'] == 'Query 1'
        assert row1['naive_cost_usd'] == 0.001
        assert row1['naive_latency_ms'] == 1000.0
        assert row1['naive_faithfulness'] == 5
        assert row1['optimized_cost_usd'] == 0.0005
        assert row1['optimized_latency_ms'] == 500.0
        assert row1['optimized_faithfulness'] is None  # No optimized judge scores
    
    def test_build_comparison_csv_handles_missing_optimized_judge(self, setup_test_data):
        """Test that missing optimized judge scores are handled gracefully."""
        tmp_path = setup_test_data
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        output_path = tmp_path / "comparison.csv"
        
        rows = build_comparison_csv(naive_dir, optimized_dir, output_path)
        
        # All optimized judge scores should be None
        for row in rows:
            assert row['optimized_faithfulness'] is None
            assert row['optimized_answer_relevance'] is None
            assert row['optimized_judge_cost_usd'] is None
    
    def test_build_comparison_csv_file_format(self, setup_test_data):
        """Test that the output CSV has the correct column order."""
        tmp_path = setup_test_data
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        output_path = tmp_path / "comparison.csv"
        
        build_comparison_csv(naive_dir, optimized_dir, output_path)
        
        # Read the CSV and check headers
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        expected_headers = [
            'query_id', 'query',
            'naive_cost_usd', 'naive_latency_ms', 'naive_prompt_tokens', 
            'naive_completion_tokens', 'naive_rerank_count',
            'naive_faithfulness', 'naive_answer_relevance', 'naive_judge_cost_usd',
            'optimized_cost_usd', 'optimized_latency_ms', 'optimized_prompt_tokens',
            'optimized_completion_tokens', 'optimized_rerank_count',
            'optimized_faithfulness', 'optimized_answer_relevance', 'optimized_judge_cost_usd'
        ]
        
        assert headers == expected_headers


class TestBuildSummaryMarkdown:
    """Tests for build_summary_markdown function."""
    
    def test_build_summary_markdown_basic(self, tmp_path):
        """Test basic summary markdown generation."""
        rows = [
            {
                'query_id': '1',
                'query': 'Query 1',
                'naive_cost_usd': 0.001,
                'naive_latency_ms': 1000.0,
                'naive_faithfulness': 5,
                'naive_answer_relevance': 4,
                'optimized_cost_usd': 0.0005,
                'optimized_latency_ms': 500.0,
                'optimized_faithfulness': 4,
                'optimized_answer_relevance': 4
            },
            {
                'query_id': '2',
                'query': 'Query 2',
                'naive_cost_usd': 0.002,
                'naive_latency_ms': 1500.0,
                'naive_faithfulness': 3,
                'naive_answer_relevance': 2,
                'optimized_cost_usd': 0.001,
                'optimized_latency_ms': 750.0,
                'optimized_faithfulness': 3,
                'optimized_answer_relevance': 3
            }
        ]
        
        output_path = tmp_path / "comparison_summary.md"
        build_summary_markdown(rows, output_path)
        
        # Check file was created
        assert output_path.exists()
        
        # Read and check content
        content = output_path.read_text()
        assert "# Comparison Summary" in content
        assert "## Aggregate Metrics" in content
        assert "avg_cost_usd" in content
        assert "avg_latency_ms" in content
        assert "avg_faithfulness" in content
        assert "total_queries" in content
        assert "Cost Savings" in content
        assert "50.0%" in content  # Expected cost reduction (0.0005 vs 0.001 is 50% reduction)
    
    def test_build_summary_markdown_with_none_values(self, tmp_path):
        """Test summary generation with missing values."""
        rows = [
            {
                'query_id': '1',
                'query': 'Query 1',
                'naive_cost_usd': 0.001,
                'naive_latency_ms': 1000.0,
                'naive_faithfulness': 5,
                'naive_answer_relevance': 4,
                'optimized_cost_usd': 0.0005,
                'optimized_latency_ms': 500.0,
                'optimized_faithfulness': None,
                'optimized_answer_relevance': None
            }
        ]
        
        output_path = tmp_path / "comparison_summary.md"
        build_summary_markdown(rows, output_path)
        
        # Check file was created
        assert output_path.exists()
        
        # Read and check content
        content = output_path.read_text()
        assert "Quality metrics not available" in content


class TestBuildReviewSheet:
    """Tests for build_review_sheet function."""
    
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Create test data files."""
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        naive_dir.mkdir()
        optimized_dir.mkdir()
        
        # Create naive answers.jsonl
        naive_answers = naive_dir / "answers.jsonl"
        naive_answers.write_text(
            '{"query_id": "1", "query": "Query 1", "answer": "Naive answer 1", "doc_ids": ["a"]}\n'
        )
        
        # Create naive judge_scores.csv
        naive_judge = naive_dir / "judge_scores.csv"
        naive_judge.write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,"
            "answer_relevance_reasoning,prompt_tokens,completion_tokens,estimated_cost_usd\n"
            "1,5,Good,4,Relevant,100,50,0.0001\n"
        )
        
        # Create optimized answers.jsonl
        opt_answers = optimized_dir / "answers.jsonl"
        opt_answers.write_text(
            '{"query_id": "1", "query": "Query 1", "answer": "Optimized answer 1", "doc_ids": ["c"]}\n'
        )
        
        return tmp_path
    
    def test_build_review_sheet_basic(self, setup_test_data):
        """Test basic review sheet generation."""
        tmp_path = setup_test_data
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        output_path = tmp_path / "review_sheet.csv"
        
        build_review_sheet(naive_dir, optimized_dir, output_path)
        
        # Check file was created
        assert output_path.exists()
        
        # Read and check content
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        assert row['query_id'] == '1'
        assert row['query'] == 'Query 1'
        assert row['naive_answer'] == 'Naive answer 1'
        assert row['optimized_answer'] == 'Optimized answer 1'
        assert row['naive_faithfulness'] == '5'
        assert row['naive_answer_relevance'] == '4'
        assert row['reviewer_notes'] == ''
        assert row['reviewer_correct'] == ''
    
    def test_build_review_sheet_columns(self, setup_test_data):
        """Test that review sheet has the correct columns."""
        tmp_path = setup_test_data
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        output_path = tmp_path / "review_sheet.csv"
        
        build_review_sheet(naive_dir, optimized_dir, output_path)
        
        # Read headers
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        expected_headers = [
            'query_id', 'query', 'naive_answer', 'optimized_answer',
            'naive_faithfulness', 'naive_answer_relevance',
            'optimized_faithfulness', 'optimized_answer_relevance',
            'reviewer_notes', 'reviewer_correct'
        ]
        
        assert headers == expected_headers
    
    def test_build_review_sheet_handles_missing_judge(self, setup_test_data):
        """Test that missing judge scores are handled gracefully."""
        tmp_path = setup_test_data
        
        # Remove optimized judge scores (they don't exist in fixture)
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        output_path = tmp_path / "review_sheet.csv"
        
        build_review_sheet(naive_dir, optimized_dir, output_path)
        
        # Read and check content
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        # Optimized scores should be empty (None becomes empty string in CSV)
        assert row['optimized_faithfulness'] == ''
        assert row['optimized_answer_relevance'] == ''


class TestIntegration:
    """Integration tests using actual test data."""
    
    @pytest.fixture
    def setup_integration_data(self, tmp_path):
        """Create comprehensive test data matching real outputs."""
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        naive_dir.mkdir(parents=True)
        optimized_dir.mkdir(parents=True)
        
        # Naive costs
        naive_costs = naive_dir / "costs.csv"
        naive_costs.write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,openai/gpt-4o,4110,33,100,3151.73,0.010605\n"
            "3,openai/gpt-4o,2417,93,100,3959.5,0.006973\n"
        )
        
        # Naive judge scores
        naive_judge = naive_dir / "judge_scores.csv"
        naive_judge.write_text(
            "query_id,faithfulness_score,faithfulness_reasoning,answer_relevance_score,"
            "answer_relevance_reasoning,prompt_tokens,completion_tokens,estimated_cost_usd\n"
            "1,5,Good,2,Relevant,4783,137,0.004204\n"
            "3,4,Okay,3,Somewhat,3227,156,0.003122\n"
        )
        
        # Naive answers
        naive_answers = naive_dir / "answers.jsonl"
        naive_answers.write_text(
            '{"query_id": "1", "query": "Query 1 text", "answer": "Naive answer 1", "doc_ids": ["a"]}\n'
            '{"query_id": "3", "query": "Query 3 text", "answer": "Naive answer 3", "doc_ids": ["b"]}\n'
        )
        
        # Naive retrieval metadata
        naive_meta = naive_dir / "retrieval_metadata.json"
        naive_meta.write_text(json.dumps({
            "1": {"retrieved_count": 100, "rerank_count": 100, "context_count": 10},
            "3": {"retrieved_count": 100, "rerank_count": 100, "context_count": 10}
        }))
        
        # Optimized costs
        opt_costs = optimized_dir / "costs.csv"
        opt_costs.write_text(
            "query_id,model,prompt_tokens,completion_tokens,rerank_count,latency_ms,estimated_cost_usd\n"
            "1,openai/gpt-5.4-mini,2073,114,10,1947.27,0.002068\n"
            "3,openai/gpt-5.4-mini,2173,169,10,1722.28,0.00239\n"
        )
        
        # Optimized answers (no judge scores for optimized)
        opt_answers = optimized_dir / "answers.jsonl"
        opt_answers.write_text(
            '{"query_id": "1", "query": "Query 1 text", "answer": "Optimized answer 1", "doc_ids": ["c"]}\n'
            '{"query_id": "3", "query": "Query 3 text", "answer": "Optimized answer 3", "doc_ids": ["d"]}\n'
        )
        
        return tmp_path
    
    def test_full_pipeline(self, setup_integration_data):
        """Test the full report generation pipeline."""
        tmp_path = setup_integration_data
        naive_dir = tmp_path / "naive"
        optimized_dir = tmp_path / "optimized"
        
        # Build comparison CSV
        comparison_path = tmp_path / "comparison.csv"
        rows = build_comparison_csv(naive_dir, optimized_dir, comparison_path)
        
        # Build summary markdown
        summary_path = tmp_path / "comparison_summary.md"
        build_summary_markdown(rows, summary_path)
        
        # Build review sheet
        review_path = tmp_path / "review_sheet.csv"
        build_review_sheet(naive_dir, optimized_dir, review_path)
        
        # Verify all files exist
        assert comparison_path.exists()
        assert summary_path.exists()
        assert review_path.exists()
        
        # Verify comparison CSV
        with open(comparison_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        
        assert len(csv_rows) == 2
        
        # Verify summary markdown
        summary_content = summary_path.read_text()
        assert "# Comparison Summary" in summary_content
        assert "Cost Savings" in summary_content
        
        # Verify review sheet
        with open(review_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            review_rows = list(reader)
        
        assert len(review_rows) == 2
