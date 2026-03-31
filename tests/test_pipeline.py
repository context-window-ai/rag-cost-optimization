"""Tests for RAG pipeline with cost tracking."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_retrieval.pipeline import (
    QueryCost,
    RAGPipeline,
    estimate_cost,
    load_config,
)


@pytest.fixture
def sample_config():
    """Sample pipeline configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "dataset": "scifact",
            "data_dir": tmpdir,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "rerank": True,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm_model": "gpt-4o",
            "llm_temperature": 0.0,
            "top_k": 10,
            "rerank_k": 5,
            "output_dir": str(Path(tmpdir) / "outputs"),
            "demo_subset": None,
        }


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return {
        "doc1": {"title": "Python Programming", "text": "Python is a popular programming language."},
        "doc2": {"title": "Machine Learning", "text": "Machine learning is a subset of artificial intelligence."},
        "doc3": {"title": "Data Science", "text": "Data science combines statistics and programming."},
    }


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "q1": "What is Python?",
        "q2": "What is machine learning?",
    }


class TestQueryCost:
    """Tests for QueryCost dataclass."""

    def test_query_cost_creation(self):
        """Test creating a QueryCost record."""
        record = QueryCost(
            query_id="q1",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            rerank_count=10,
            latency_ms=150.5,
            estimated_cost_usd=0.001234,
        )
        assert record.query_id == "q1"
        assert record.model == "gpt-4o"
        assert record.prompt_tokens == 100
        assert record.completion_tokens == 50
        assert record.rerank_count == 10
        assert record.latency_ms == 150.5
        assert record.estimated_cost_usd == 0.001234


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_estimate_cost_gpt4o(self):
        """Test cost estimation for GPT-4o."""
        # GPT-4o: $2.50/1M prompt, $10.00/1M completion
        cost = estimate_cost("gpt-4o", 1000, 500)
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert abs(cost - expected) < 1e-9

    def test_estimate_cost_gpt4o_mini(self):
        """Test cost estimation for GPT-4o-mini."""
        # GPT-4o-mini: $0.15/1M prompt, $0.60/1M completion
        cost = estimate_cost("gpt-4o-mini", 1000, 500)
        expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
        assert abs(cost - expected) < 1e-9

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model returns 0."""
        cost = estimate_cost("unknown-model", 1000, 500)
        assert cost == 0.0


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization."""
        pipeline = RAGPipeline(sample_config)
        assert pipeline.config == sample_config
        assert pipeline.llm_model == "gpt-4o"
        assert pipeline.output_dir.exists()
        assert pipeline.cost_records == []

    def test_pipeline_without_reranker(self, sample_config):
        """Test pipeline without reranker."""
        sample_config["rerank"] = False
        pipeline = RAGPipeline(sample_config)
        assert pipeline.reranker is None

    @patch("rag_retrieval.pipeline.OpenAI")
    def test_pipeline_run_mock(self, mock_openai, sample_config, sample_corpus, sample_queries):
        """Test pipeline run with mocked LLM."""
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        pipeline = RAGPipeline(sample_config)
        pipeline.run(sample_corpus, sample_queries, demo_subset=1)

        # Check outputs
        assert len(pipeline.cost_records) == 1
        assert pipeline.cost_records[0].query_id == "q1"
        assert pipeline.cost_records[0].model == "gpt-4o"
        assert pipeline.cost_records[0].prompt_tokens == 100
        assert pipeline.cost_records[0].completion_tokens == 50

        # Check files were created
        answers_path = pipeline.output_dir / "answers.jsonl"
        assert answers_path.exists()

        costs_path = pipeline.output_dir / "costs.csv"
        assert costs_path.exists()

        metadata_path = pipeline.output_dir / "retrieval_metadata.json"
        assert metadata_path.exists()


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_config_file_not_found(self):
        """Test loading non-existent config raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_config("nonexistent", tmpdir)

    def test_load_config_success(self):
        """Test loading a valid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            config_path.write_text("dataset: scifact\nllm_model: gpt-4o\n")
            config = load_config("test", tmpdir)
            assert config["dataset"] == "scifact"
            assert config["llm_model"] == "gpt-4o"


class TestCostCSV:
    """Tests for cost CSV output."""

    def test_cost_csv_format(self, sample_config):
        """Test that cost CSV has correct columns."""
        import csv

        pipeline = RAGPipeline(sample_config)
        pipeline.cost_records = [
            QueryCost(
                query_id="q1",
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                rerank_count=10,
                latency_ms=150.5,
                estimated_cost_usd=0.001234,
            ),
        ]
        pipeline._write_cost_csv()

        costs_path = pipeline.output_dir / "costs.csv"
        with open(costs_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["query_id"] == "q1"
        assert rows[0]["model"] == "gpt-4o"
        assert rows[0]["prompt_tokens"] == "100"
        assert rows[0]["completion_tokens"] == "50"
        assert rows[0]["rerank_count"] == "10"
        assert rows[0]["latency_ms"] == "150.5"
        assert rows[0]["estimated_cost_usd"] == "0.001234"
