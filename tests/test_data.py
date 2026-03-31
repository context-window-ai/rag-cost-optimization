"""Tests for data module."""

import pytest
from unittest.mock import patch, MagicMock
from rag_retrieval.data import download_and_load_dataset


def test_download_and_load_dataset():
    """Test dataset loading (integration test with mock)."""
    mock_corpus = {"doc1": {"title": "Test", "text": "Content"}}
    mock_queries = {"q1": "test query"}
    mock_qrels = {"q1": {"doc1": 1}}
    
    with patch('rag_retrieval.data.util.download_and_unzip') as mock_download, \
         patch('rag_retrieval.data.GenericDataLoader') as mock_loader, \
         patch('os.path.exists', return_value=True):
        
        # Setup mock loader
        mock_instance = MagicMock()
        mock_instance.load.return_value = (mock_corpus, mock_queries, mock_qrels)
        mock_loader.return_value = mock_instance
        
        corpus, queries, qrels = download_and_load_dataset("scifact", "datasets")
        
        # Verify download was NOT called since path exists
        mock_download.assert_not_called()
        
        # Verify loader was called
        assert corpus == mock_corpus
        assert queries == mock_queries
        assert qrels == mock_qrels
