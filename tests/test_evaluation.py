"""Tests for evaluation metrics."""

import numpy as np
import pytest

from rag_retrieval.evaluation import (
    _dcg_at_k,
    _ndcg_at_k,
    _recall_at_k,
    _precision_at_k,
    _map_at_k,
    evaluate_retrieval,
)


def test_dcg_at_k():
    """Test DCG computation with binary relevance."""
    # Binary relevance: 1, 1, 0, 0, 0
    relevances = [1, 1, 0, 0, 0]
    dcg = _dcg_at_k(relevances, k=5)
    # DCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
    expected = 1.0 / np.log2(2) + 1.0 / np.log2(3)
    np.testing.assert_allclose(dcg, expected, rtol=1e-5)

    # Empty relevance
    assert _dcg_at_k([], k=5) == 0.0


def test_ndcg_at_k():
    """Test nDCG computation with binary relevance."""
    # Perfect ranking (all relevant docs first)
    relevances = [1, 1, 1, 0, 0]
    ndcg = _ndcg_at_k(relevances, k=5)
    assert ndcg == 1.0

    # Imperfect ranking
    relevances = [0, 1, 1, 0, 0]
    ndcg = _ndcg_at_k(relevances, k=5)
    assert 0.0 < ndcg < 1.0

    # All zeros
    relevances = [0, 0, 0]
    assert _ndcg_at_k(relevances, k=3) == 0.0


def test_recall_at_k():
    """Test Recall computation with binary relevance."""
    relevances = [1, 0, 1, 0, 1]
    total_relevant = 3

    # At k=3, we have [1, 0, 1] = 2 relevant out of 3 total
    recall = _recall_at_k(relevances, k=3, total_relevant=total_relevant)
    assert recall == 2 / 3

    recall = _recall_at_k(relevances, k=5, total_relevant=total_relevant)
    assert recall == 1.0

    # No relevant documents
    recall = _recall_at_k([0, 0, 0], k=3, total_relevant=0)
    assert recall == 0.0


def test_precision_at_k():
    """Test Precision computation with binary relevance."""
    relevances = [1, 1, 0, 0, 1]

    precision = _precision_at_k(relevances, k=2)
    assert precision == 1.0

    precision = _precision_at_k(relevances, k=4)
    assert precision == 0.5

    precision = _precision_at_k(relevances, k=0)
    assert precision == 0.0


def test_map_at_k():
    """Test MAP computation with binary relevance."""
    # Perfect ranking
    relevances = [1, 1, 1, 0, 0]
    map_score = _map_at_k(relevances, k=5)
    assert map_score == 1.0

    # Mixed ranking
    relevances = [1, 0, 1, 0, 0]
    map_score = _map_at_k(relevances, k=5)
    # Precision at rank 1: 1/1 = 1.0
    # Precision at rank 3: 2/3 = 0.667
    # MAP = (1.0 + 0.667) / 2 = 0.833
    expected = (1.0 + 2 / 3) / 2
    np.testing.assert_allclose(map_score, expected, rtol=1e-3)

    # No relevant documents
    map_score = _map_at_k([0, 0, 0], k=3)
    assert map_score == 0.0


def test_evaluate_retrieval():
    """Test full evaluation pipeline."""
    # Setup - results format: query_id -> {doc_id: score}
    results = {
        "q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7},
        "q2": {"d2": 0.9, "d1": 0.8, "d4": 0.7},
    }

    qrels = {
        "q1": {"d1": 2, "d2": 1, "d3": 0},
        "q2": {"d1": 1, "d2": 2, "d4": 0},
    }

    metrics = evaluate_retrieval(results, qrels, k_values=[1, 3])

    # Check that all metrics are present
    assert "ndcg@1" in metrics
    assert "ndcg@3" in metrics
    assert "recall@1" in metrics
    assert "recall@3" in metrics
    assert "precision@1" in metrics
    assert "precision@3" in metrics
    assert "map" in metrics  # Overall MAP

    # Check that metrics are between 0 and 1
    for metric_name, metric_value in metrics.items():
        assert 0.0 <= metric_value <= 1.0, f"{metric_name} out of range: {metric_value}"


def test_evaluate_retrieval_missing_query():
    """Test that queries without qrels are skipped."""
    results = {
        "q1": {"d1": 0.9},
        "q2": {"d2": 0.8},  # No qrels for q2
    }

    qrels = {
        "q1": {"d1": 1},
    }

    metrics = evaluate_retrieval(results, qrels, k_values=[1])

    # Should only evaluate q1
    assert len(metrics) > 0
    # All metrics should be valid (from q1 only)
    for metric_name, metric_value in metrics.items():
        assert 0.0 <= metric_value <= 1.0
