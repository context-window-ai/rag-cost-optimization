"""Tests for dense retriever."""

import numpy as np
import pytest

from rag_retrieval.retriever import DenseRetriever, BM25Retriever, HybridRetriever


def test_retriever_initialization():
    """Test retriever initialization."""
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert retriever.model is not None
    assert retriever.index is None
    assert retriever.doc_ids == []


def test_encode():
    """Test text encoding."""
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = ["hello world", "test query"]
    embeddings = retriever.encode(texts, batch_size=2)

    assert embeddings.shape[0] == 2
    assert embeddings.dtype == np.float32
    # Check normalization (L2 norm should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_index_documents():
    """Test document indexing."""
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

    doc_ids = ["doc1", "doc2", "doc3"]
    doc_texts = [
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "Machine learning is fascinating.",
    ]

    retriever.index_documents(doc_ids, doc_texts, batch_size=2)

    assert retriever.index is not None
    assert len(retriever.doc_ids) == 3
    assert retriever.index.ntotal == 3


def test_index_documents_mismatch():
    """Test that mismatched doc_ids and doc_texts raises error."""
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with pytest.raises(ValueError, match="must have same length"):
        retriever.index_documents(["doc1", "doc2"], ["text1"])


def test_search_without_index():
    """Test that searching without indexing raises error."""
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with pytest.raises(RuntimeError, match="Index not built"):
        retriever.search(["query"])


def test_search():
    """Test search functionality."""
    retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Index documents about different topics
    doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    doc_texts = [
        "The cat sat on the mat.",
        "A feline animal sleeping.",
        "Machine learning and AI.",
        "Deep neural networks.",
    ]

    retriever.index_documents(doc_ids, doc_texts, batch_size=2)

    # Search for cat-related query
    results = retriever.search(["cat animal"], top_k=2)

    assert len(results) == 1
    assert "cat animal" in results
    assert len(results["cat animal"]) == 2
    # Check that results are tuples of (doc_id, score)
    assert all(isinstance(doc_id, str) for doc_id, _ in results["cat animal"])
    assert all(isinstance(score, float) for _, score in results["cat animal"])
    # Results should be sorted by score descending
    scores = [score for _, score in results["cat animal"]]
    assert scores == sorted(scores, reverse=True)


# BM25Retriever Tests

def test_bm25_retriever_initialization():
    """Test BM25 retriever initialization."""
    retriever = BM25Retriever()
    assert retriever.index is None
    assert retriever.doc_ids == []


def test_bm25_index_documents():
    """Test BM25 document indexing."""
    retriever = BM25Retriever()

    doc_ids = ["doc1", "doc2", "doc3"]
    doc_texts = [
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "Machine learning is fascinating.",
    ]

    retriever.index_documents(doc_ids, doc_texts)

    assert retriever.index is not None
    assert len(retriever.doc_ids) == 3


def test_bm25_index_documents_mismatch():
    """Test that mismatched doc_ids and doc_texts raises error for BM25."""
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="must have same length"):
        retriever.index_documents(["doc1", "doc2"], ["text1"])


def test_bm25_search_without_index():
    """Test that searching without indexing raises error for BM25."""
    retriever = BM25Retriever()

    with pytest.raises(RuntimeError, match="Index not built"):
        retriever.search(["query"])


def test_bm25_search():
    """Test BM25 search functionality."""
    retriever = BM25Retriever()

    doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    doc_texts = [
        "The cat sat on the mat.",
        "A feline animal sleeping.",
        "Machine learning and AI.",
        "Deep neural networks.",
    ]

    retriever.index_documents(doc_ids, doc_texts)

    # Search for cat-related query
    results = retriever.search(["cat animal"], top_k=2)

    assert len(results) == 1
    assert "cat animal" in results
    assert len(results["cat animal"]) == 2
    # Check that results are tuples of (doc_id, score)
    assert all(isinstance(doc_id, str) for doc_id, _ in results["cat animal"])
    assert all(isinstance(score, float) for _, score in results["cat animal"])
    # Scores should be normalized to [0, 1]
    for _, score in results["cat animal"]:
        assert 0.0 <= score <= 1.0


def test_bm25_search_normalized_scores():
    """Test that BM25 scores are normalized to [0, 1]."""
    retriever = BM25Retriever()

    # Use enough documents to get non-zero IDF scores
    doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    doc_texts = ["cat cat cat", "dog dog", "bird bird", "fish fish", "snake snake"]

    retriever.index_documents(doc_ids, doc_texts)
    results = retriever.search(["cat"], top_k=5)

    # Top result should have score 1.0 (max score normalized)
    assert results["cat"][0][1] == 1.0
    # All scores should be in [0, 1]
    for _, score in results["cat"]:
        assert 0.0 <= score <= 1.0


# HybridRetriever Tests

def test_hybrid_retriever_initialization():
    """Test hybrid retriever initialization."""
    retriever = HybridRetriever(dense_weight=0.7)
    assert retriever.dense_weight == 0.7
    assert retriever.dense_retriever is not None
    assert retriever.bm25_retriever is not None


def test_hybrid_invalid_weight():
    """Test that invalid dense_weight raises error."""
    with pytest.raises(ValueError, match="dense_weight must be between 0 and 1"):
        HybridRetriever(dense_weight=1.5)

    with pytest.raises(ValueError, match="dense_weight must be between 0 and 1"):
        HybridRetriever(dense_weight=-0.1)


def test_hybrid_index_documents():
    """Test hybrid document indexing."""
    retriever = HybridRetriever()

    doc_ids = ["doc1", "doc2", "doc3"]
    doc_texts = [
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "Machine learning is fascinating.",
    ]

    retriever.index_documents(doc_ids, doc_texts, batch_size=2)

    assert retriever.dense_retriever.index is not None
    assert retriever.bm25_retriever.index is not None
    assert len(retriever.doc_ids) == 3


def test_hybrid_search_without_index():
    """Test that searching without indexing raises error for hybrid."""
    retriever = HybridRetriever()

    with pytest.raises(RuntimeError, match="Index not built"):
        retriever.search(["query"])


def test_hybrid_search():
    """Test hybrid search with RRF fusion."""
    retriever = HybridRetriever(dense_weight=0.5)

    doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    doc_texts = [
        "The cat sat on the mat.",
        "A feline animal sleeping.",
        "Machine learning and AI.",
        "Deep neural networks.",
    ]

    retriever.index_documents(doc_ids, doc_texts, batch_size=2)

    # Search for cat-related query
    results = retriever.search(["cat animal"], top_k=2)

    assert len(results) == 1
    assert "cat animal" in results
    assert len(results["cat animal"]) == 2
    # Check that results are tuples of (doc_id, score)
    assert all(isinstance(doc_id, str) for doc_id, _ in results["cat animal"])
    assert all(isinstance(score, float) for _, score in results["cat animal"])
    # Results should be sorted by RRF score descending
    scores = [score for _, score in results["cat animal"]]
    assert scores == sorted(scores, reverse=True)
