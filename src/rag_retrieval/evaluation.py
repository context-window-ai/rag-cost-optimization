"""BEIR evaluation metrics."""

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def _dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG@k using graded relevance.

    Args:
        relevances: List of graded relevance scores (0, 1, 2, ...) in ranked order
        k: Number of results to consider

    Returns:
        DCG@k score
    """
    if not relevances or k == 0:
        return 0.0

    relevances = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / np.log2(i + 2)
    return dcg


def _ndcg_at_k(relevances: List[int], k: int) -> float:
    """Compute nDCG@k using graded relevance.

    Args:
        relevances: List of graded relevance scores (0, 1, 2, ...) in ranked order
        k: Number of results to consider

    Returns:
        nDCG@k score (0 to 1)
    """
    if not relevances or k == 0:
        return 0.0

    dcg = _dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(relevances: List[int], k: int, total_relevant: int) -> float:
    """Compute Recall@k.

    Args:
        relevances: List of binary relevance indicators (0 or 1) in ranked order
        k: Number of results to consider
        total_relevant: Total number of relevant documents

    Returns:
        Recall@k score (0 to 1)
    """
    if total_relevant == 0 or k == 0:
        return 0.0

    relevances = relevances[:k]
    return sum(relevances) / total_relevant


def _precision_at_k(relevances: List[int], k: int) -> float:
    """Compute Precision@k.

    Args:
        relevances: List of binary relevance indicators (0 or 1) in ranked order
        k: Number of results to consider

    Returns:
        Precision@k score (0 to 1)
    """
    if k == 0:
        return 0.0

    relevances = relevances[:k]
    return sum(relevances) / k


def _map_at_k(relevances: List[int], k: int) -> float:
    """Compute Average Precision@k.

    Args:
        relevances: List of binary relevance indicators (0 or 1) in ranked order
        k: Number of results to consider

    Returns:
        AP@k score (0 to 1)
    """
    if not relevances or k == 0:
        return 0.0

    relevances = relevances[:k]
    num_relevant = sum(relevances)

    if num_relevant == 0:
        return 0.0

    precisions = []
    for i, rel in enumerate(relevances):
        if rel == 1:
            precisions.append(_precision_at_k(relevances, i + 1))

    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_retrieval(
    results: Dict[str, Union[Dict[str, float], List[tuple]]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = None
) -> Dict[str, float]:
    """Evaluate retrieval results using BEIR metrics.

    Args:
        results: Dict mapping query_id to {doc_id: score} or list of (doc_id, score) tuples
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

        # Handle both dict and list formats
        if isinstance(doc_scores, list):
            # List of (doc_id, score) tuples - already sorted
            ranked_doc_ids = [doc_id for doc_id, _ in doc_scores]
        else:
            # Dict format - sort by score descending
            ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]

        # Get graded relevance for nDCG
        graded_relevances = [query_qrels.get(doc_id, 0) for doc_id in ranked_doc_ids]
        
        # Get binary relevance for recall/precision/MAP
        binary_relevances = [1 if query_qrels.get(doc_id, 0) > 0 else 0 for doc_id in ranked_doc_ids]

        for k in k_values:
            metrics[f"ndcg@{k}"].append(_ndcg_at_k(graded_relevances, k))
            metrics[f"recall@{k}"].append(_recall_at_k(binary_relevances, k, total_relevant))
            metrics[f"precision@{k}"].append(_precision_at_k(binary_relevances, k))
            metrics[f"map@{k}"].append(_map_at_k(binary_relevances, k))

        # Compute AP for this query (at max k)
        max_k = max(k_values)
        all_ap_scores.append(_map_at_k(binary_relevances, max_k))

    # Average metrics
    final_metrics = {}
    for metric_name, values in metrics.items():
        final_metrics[metric_name] = float(np.mean(values)) if values else 0.0

    # Add global MAP (average of all query AP scores)
    final_metrics["map"] = float(np.mean(all_ap_scores)) if all_ap_scores else 0.0

    return final_metrics


def save_metrics(metrics: Dict[str, Union[str, float]], output_dir: Union[str, Path]) -> None:
    """Save metrics to JSON and CSV files.

    Args:
        metrics: Dict with metric names and values
        output_dir: Directory to save metrics files
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save CSV
    df = pd.DataFrame([metrics])
    df.to_csv(output_dir / "metrics.csv", index=False)


def save_results(
    results: Dict[str, Union[Dict[str, float], List[tuple]]],
    output_dir: Union[str, Path],
    query_ids: List[str] = None
) -> None:
    """Save raw retrieval results to JSON.

    Args:
        results: Dict mapping query_id to {doc_id: score} or list of (doc_id, score) tuples
        output_dir: Directory to save results file
        query_ids: Optional list of query IDs to include (if None, include all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by query_ids if provided
    if query_ids is not None:
        results = {qid: results[qid] for qid in query_ids if qid in results}

    # Convert to list format sorted by score
    results_serializable = {}
    for query_id, doc_scores in results.items():
        # Handle both dict and list formats
        if isinstance(doc_scores, list):
            sorted_docs = doc_scores  # Already a list of tuples
        else:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results_serializable[query_id] = [
            {"doc_id": doc_id, "score": round(score, 6)}
            for doc_id, score in sorted_docs
        ]

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
