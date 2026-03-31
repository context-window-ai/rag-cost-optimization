"""Integration tests for the retrieval pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag_retrieval import DenseRetriever, evaluate_retrieval, save_metrics, save_results


def test_full_pipeline():
    """Test the complete retrieval and evaluation pipeline."""
    # Create a small corpus
    corpus = {
        "doc1": {"title": "Machine Learning", "text": "Machine learning is a subset of AI."},
        "doc2": {"title": "Deep Learning", "text": "Deep learning uses neural networks."},
        "doc3": {"title": "NLP", "text": "Natural language processing deals with text."},
        "doc4": {"title": "Computer Vision", "text": "Computer vision processes images."},
    }

    queries = {
        "q1": "What is machine learning?",
        "q2": "Tell me about neural networks",
    }

    qrels = {
        "q1": {"doc1": 1, "doc2": 0, "doc3": 0, "doc4": 0},
        "q2": {"doc1": 0, "doc2": 1, "doc3": 0, "doc4": 0},
    }

    # Initialize retriever
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build index
    retriever.build_index(corpus)

    # Retrieve
    results = retriever.retrieve(queries, top_k=4)

    # Evaluate
    metrics = evaluate_retrieval(results, qrels, k_values=[1, 3])

    # Check metrics structure
    assert "ndcg@1" in metrics
    assert "recall@1" in metrics
    assert "map" in metrics

    # Check metrics are in valid range
    for metric_name, metric_value in metrics.items():
        assert 0.0 <= metric_value <= 1.0, f"{metric_name} out of range: {metric_value}"

    # Test save functions
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Save metrics
        save_metrics(metrics, output_dir)
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "metrics.csv").exists()

        # Load and verify JSON
        with open(output_dir / "metrics.json") as f:
            loaded_metrics = json.load(f)
        assert loaded_metrics == metrics

        # Save results
        save_results(results, output_dir)
        assert (output_dir / "results.json").exists()

        # Load and verify results
        with open(output_dir / "results.json") as f:
            loaded_results = json.load(f)
        assert len(loaded_results) == len(queries)
        for query_id in queries:
            assert query_id in loaded_results
            assert len(loaded_results[query_id]) == 4  # top_k=4


def test_retriever_save_load():
    """Test saving and loading FAISS index."""
    corpus = {
        "doc1": {"title": "Test", "text": "Test document"},
        "doc2": {"title": "Another", "text": "Another document"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create and save index
        retriever1 = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever1.build_index(corpus)
        retriever1.save_index(tmpdir_path)

        # Load index
        retriever2 = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever2.load_index(tmpdir_path)

        # Verify
        assert retriever2.index is not None
        assert len(retriever2.doc_ids) == 2
        assert retriever2.doc_ids == retriever1.doc_ids


def test_empty_corpus():
    """Test handling of empty corpus."""
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with pytest.raises((ValueError, IndexError)):
        # Should fail gracefully with empty corpus
        retriever.build_index({})


def test_metrics_with_no_relevant_docs():
    """Test evaluation when queries have no relevant documents."""
    results = {
        "q1": [("d1", 0.9), ("d2", 0.8)],
    }

    qrels = {
        "q1": {"d1": 0, "d2": 0},  # No relevant docs
    }

    metrics = evaluate_retrieval(results, qrels, k_values=[1])

    # Should handle gracefully (query with no relevant docs is skipped)
    assert len(metrics) > 0
