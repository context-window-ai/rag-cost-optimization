"""BEIR evaluation metrics."""
import json
from typing import Dict, List, Union
from pathlib import Path
import numpy as np


def _dcg_at_k(relevance: List[int], k: int) -> float:
    """Compute DCG@k using graded relevance.

    Args:
        relevance: List of graded relevance scores in ranked order
        k: Number of results to consider

    Returns:
        DCG@k score
    """
    if not relevance or k == 0:
        return 0.0

    relevance = relevance[:k]
    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)
    return dcg


def _ndcg_at_k(relevance: List[int], k: int) -> float:
    """Compute nDCG@k using graded relevance.

    Args:
        relevance: List of graded relevance scores in ranked order
        k: Number of results to consider

    Returns:
        nDCG@k score (0 to 1)
    """
    if not relevance or k == 0:
        return 0.0

    dcg = _dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = _dcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(relevance: List[int], k: int, total_relevant: int) -> float:
    """Compute Recall@k.

    Args:
        relevance: List of binary relevance indicators (0/1) in ranked order
        k: Number of results to consider
        total_relevant: Total number of relevant documents

    Returns:
        Recall@k score (0 to 1)
    """
    if total_relevant == 0 or k == 0:
        return 0.0

    relevance = relevance[:k]
    return sum(relevance) / total_relevant


def _precision_at_k(relevance: List[int], k: int) -> float:
    """Compute Precision@k.

    Args:
        relevance: List of binary relevance indicators (0/1) in ranked order
        k: Number of results to consider

    Returns:
        Precision@k score (0 to 1)
    """
    if k == 0:
        return 0.0

    relevance = relevance[:k]
    return sum(relevance) / k


def _map_at_k(relevance: List[int], k: int) -> float:
    """Compute Mean Average Precision@k.

    Args:
        relevance: List of binary relevance indicators (0/1) in ranked order
        k: Number of results to consider

    Returns:
        MAP@k score (0 to 1)
    """
    if not relevance or k == 0:
        return 0.0

    relevance = relevance[:k]
    num_relevant = sum(relevance)

    if num_relevant == 0:
        return 0.0

    precisions = []
    for i, rel in enumerate(relevance):
        if rel == 1:
            precisions.append(_precision_at_k(relevance, i + 1))

    return np.mean(precisions) if precisions else 0.0


def evaluate_retrieval(
    results: Dict[str, List[tuple]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = None
) -> Dict[str, float]:
    """Evaluate retrieval results using BEIR metrics.

    Args:
        results: Dict mapping query_id to list of (doc_id, score) tuples
        qrels: Dict mapping query_id to {doc_id: relevance}
        k_values: List of k values for evaluation

    Returns:
        Dict of all metrics (ndcg, recall, precision, map at each k, and global map)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 100]

    metrics = {}
    all_ap_scores = []

    # Initialize metric accumulators
    for k in k_values:
        metrics[f"ndcg@{k}"] = []
        metrics[f"recall@{k}"] = []
        metrics[f"precision@{k}"] = []
        metrics[f"map@{k}"] = []

    for query_id, doc_scores in results.items():
        if query_id not in qrels:
            continue

        query_qrels = qrels[query_id]
        total_relevant = sum(1 for rel in query_qrels.values() if rel > 0)

        if total_relevant == 0:
            continue

        # Sort results by score descending and extract doc_ids
        ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]

        for k in k_values:
            # Get graded relevance for nDCG
            graded_relevance = [query_qrels.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
            
            # Get binary relevance for recall/precision/MAP
            binary_relevance = [1 if query_qrels.get(doc_id, 0) > 0 else 0 for doc_id in ranked_doc_ids[:k]]

            # Compute metrics
            metrics[f"ndcg@{k}"].append(_ndcg_at_k(graded_relevance, k))
            metrics[f"recall@{k}"].append(_recall_at_k(binary_relevance, k, total_relevant))
            metrics[f"precision@{k}"].append(_precision_at_k(binary_relevance, k))
            metrics[f"map@{k}"].append(_map_at_k(binary_relevance, k))

        # Also compute MAP at max k for global MAP
        max_k = max(k_values)
        binary_relevance = [1 if query_qrels.get(doc_id, 0) > 0 else 0 for doc_id in ranked_doc_ids[:max_k]]
        all_ap_scores.append(_map_at_k(binary_relevance, max_k))

    # Average metrics
    final_metrics = {}
    for metric_name, values in metrics.items():
        final_metrics[metric_name] = float(np.mean(values)) if values else 0.0

    # Add global MAP (average of all query AP scores)
    final_metrics["map"] = float(np.mean(all_ap_scores)) if all_ap_scores else 0.0

    return final_metrics


def save_metrics(metrics: Dict[str, Union[str, float]], output_path: Union[str, Path]) -> None:
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_results(results: Dict[str, List[tuple]], output_path: Union[str, Path]) -> None:
    """Save raw retrieval results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tuples to lists for JSON serialization
    results_serializable = {
        query_id: [[doc_id, score] for doc_id, score in doc_scores]
        for query_id, doc_scores in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
