"""Tests for dense retriever."""

import numpy as np
import pytest

from rag_retrieval.retriever import DenseRetriever


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
