"""RAG Cost Optimization - Retrieval baseline with BEIR evaluation."""

from rag_retrieval.retriever import DenseRetriever, FAISSRetriever
from rag_retrieval.evaluation import evaluate_retrieval, save_metrics, save_results
from rag_retrieval.data import download_and_load_dataset

__version__ = "0.1.0"
__all__ = [
    "DenseRetriever",
    "FAISSRetriever",
    "evaluate_retrieval",
    "save_metrics",
    "save_results",
    "download_and_load_dataset",
]

from .retriever import DenseRetriever, FAISSRetriever
from .evaluation import evaluate_retrieval, save_metrics, save_results

__all__ = ["DenseRetriever", "FAISSRetriever", "evaluate_retrieval", "save_metrics", "save_results"]
