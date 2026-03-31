"""Data loading utilities for BEIR datasets."""

from typing import Dict, Tuple
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os


def download_and_load_dataset(
    dataset_name: str = "scifact",
    data_dir: str = "datasets"
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict[str, int]]]:
    """Download and load a BEIR dataset.
    
    Args:
        dataset_name: Name of the BEIR dataset (e.g., 'scifact')
        data_dir: Directory to store dataset
        
    Returns:
        Tuple of (corpus, queries, qrels) where:
        - corpus: Dict[doc_id, Dict with 'title' and 'text']
        - queries: Dict[query_id, query_text]
        - qrels: Dict[query_id, Dict[doc_id, relevance_score]]
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    dataset_path = os.path.join(data_dir, dataset_name)
    
    if not os.path.exists(dataset_path):
        os.makedirs(data_dir, exist_ok=True)
        util.download_and_unzip(url, data_dir)
    
    loader = GenericDataLoader(data_folder=dataset_path)
    corpus, queries, qrels = loader.load(split="test")
    
    return corpus, queries, qrels
