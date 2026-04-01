"""Dense, BM25, and Hybrid retrieval implementations."""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """FAISS-based dense retriever using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode texts to dense vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def index_documents(self, doc_ids: list[str], doc_texts: list[str], batch_size: int = 32) -> None:
        """Build FAISS index from documents."""
        if len(doc_ids) != len(doc_texts):
            raise ValueError("doc_ids and doc_texts must have same length")

        self.doc_ids = doc_ids
        self.doc_texts = doc_texts

        # Encode documents
        doc_embeddings = self.encode(doc_texts, batch_size=batch_size, show_progress=True)

        # Build FAISS index (inner product for cosine similarity with normalized vectors)
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(doc_embeddings)

    def search(self, queries: list[str], top_k: int = 100, batch_size: int = 32) -> dict[str, list[tuple[str, float]]]:
        """Search for top-k documents per query.

        Returns:
            Dict mapping query_id -> list of (doc_id, score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index_documents first.")

        # Encode queries
        query_embeddings = self.encode(queries, batch_size=batch_size, show_progress=True)

        # Search
        scores, indices = self.index.search(query_embeddings, top_k)

        # Format results
        results = {}
        for i, query in enumerate(queries):
            results[query] = [
                (self.doc_ids[idx], float(scores[i, j]))
                for j, idx in enumerate(indices[i])
                if idx >= 0  # FAISS returns -1 for empty slots
            ]

        return results

    def save_index(self, path: Path) -> None:
        """Save FAISS index and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata
        metadata = {
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

    def load_index(self, path: Path) -> None:
        """Load FAISS index and metadata."""
        path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.doc_ids = metadata["doc_ids"]

    def build_index(self, corpus: dict) -> None:
        """Build FAISS index from BEIR corpus format.
        
        Args:
            corpus: Dict mapping doc_id -> {'title': str, 'text': str}
        """
        doc_ids = list(corpus.keys())
        doc_texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in doc_ids
        ]
        self.index_documents(doc_ids, doc_texts)

    def retrieve(self, queries: dict, top_k: int = 100) -> dict:
        """Retrieve top-k documents for each query.
        
        Args:
            queries: Dict mapping query_id -> query_text
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dict mapping query_id -> {doc_id: score}
        """
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        # Encode queries
        query_embeddings = self.encode(query_texts, batch_size=32, show_progress=True)
        
        # Search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Format results with query_id keys
        results = {}
        for i, query_id in enumerate(query_ids):
            results[query_id] = {
                self.doc_ids[idx]: float(scores[i, j])
                for j, idx in enumerate(indices[i])
                if idx >= 0
            }
        
        return results


# Alias for backwards compatibility
DenseRetriever = FAISSRetriever


class BM25Retriever:
    """BM25-based sparse retriever using rank_bm25."""

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []

    def index_documents(self, doc_ids: list[str], doc_texts: list[str]) -> None:
        """Build BM25Okapi index from documents."""
        if len(doc_ids) != len(doc_texts):
            raise ValueError("doc_ids and doc_texts must have same length")

        self.doc_ids = doc_ids
        self.doc_texts = doc_texts

        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [text.lower().split() for text in doc_texts]
        self.index = BM25Okapi(tokenized_docs)

    def search(self, queries: list[str], top_k: int = 100) -> dict[str, list[tuple[str, float]]]:
        """Search for top-k documents per query.

        Returns:
            Dict mapping query -> list of (doc_id, score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index_documents first.")

        results = {}
        for query in queries:
            tokenized_query = query.lower().split()
            scores = self.index.get_scores(tokenized_query)

            # Normalize scores to [0, 1] range (divide by max score)
            max_score = max(scores) if max(scores) > 0 else 1.0
            normalized_scores = [s / max_score for s in scores]

            # Get top-k indices
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]

            results[query] = [
                (self.doc_ids[idx], float(normalized_scores[idx]))
                for idx in top_indices
            ]

        return results

    def build_index(self, corpus: dict) -> None:
        """Build BM25 index from BEIR corpus format.
        
        Args:
            corpus: Dict mapping doc_id -> {'title': str, 'text': str}
        """
        doc_ids = list(corpus.keys())
        doc_texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in doc_ids
        ]
        self.index_documents(doc_ids, doc_texts)

    def retrieve(self, queries: dict, top_k: int = 100) -> dict:
        """Retrieve top-k documents for each query.
        
        Args:
            queries: Dict mapping query_id -> query_text
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dict mapping query_id -> {doc_id: score}
        """
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        raw_results = self.search(query_texts, top_k=top_k)
        
        # Remap to query_id keys
        results = {}
        for query_id, query_text in zip(query_ids, query_texts):
            results[query_id] = {
                doc_id: score for doc_id, score in raw_results[query_text]
            }
        
        return results


class HybridRetriever:
    """Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval."""

    def __init__(self, dense_weight: float = 0.5, dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize hybrid retriever.
        
        Args:
            dense_weight: Weight for dense scores (0-1). BM25 gets (1 - dense_weight).
            dense_model_name: Model name for dense retriever.
        """
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1")
        
        self.dense_weight = dense_weight
        self.dense_retriever = FAISSRetriever(model_name=dense_model_name)
        self.bm25_retriever = BM25Retriever()
        self.doc_ids: list[str] = []

    def index_documents(self, doc_ids: list[str], doc_texts: list[str], batch_size: int = 32) -> None:
        """Build both dense and sparse indices."""
        if len(doc_ids) != len(doc_texts):
            raise ValueError("doc_ids and doc_texts must have same length")
        
        self.doc_ids = doc_ids
        self.dense_retriever.index_documents(doc_ids, doc_texts, batch_size=batch_size)
        self.bm25_retriever.index_documents(doc_ids, doc_texts)

    def search(self, queries: list[str], top_k: int = 100, batch_size: int = 32) -> dict[str, list[tuple[str, float]]]:
        """Search using reciprocal rank fusion (RRF) with k=60.
        
        For each doc: score = dense_weight/(rank_dense+60) + (1-dense_weight)/(rank_bm25+60)
        Then take top_k by combined score.
        
        Returns:
            Dict mapping query -> list of (doc_id, score) tuples
        """
        if self.dense_retriever.index is None:
            raise RuntimeError("Index not built. Call index_documents first.")
        
        # Get results from both retrievers (with higher k for better fusion)
        retrieval_k = min(top_k * 5, len(self.doc_ids))  # Get more docs for better fusion
        dense_results = self.dense_retriever.search(queries, top_k=retrieval_k, batch_size=batch_size)
        bm25_results = self.bm25_retriever.search(queries, top_k=retrieval_k)
        
        results = {}
        for query in queries:
            # Build rank dictionaries (1-indexed)
            dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results[query])}
            bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_results[query])}
            
            # Get all doc_ids from both result sets
            all_doc_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
            
            # Compute RRF scores
            rrf_scores = {}
            for doc_id in all_doc_ids:
                dense_rank = dense_ranks.get(doc_id, retrieval_k + 1)
                bm25_rank = bm25_ranks.get(doc_id, retrieval_k + 1)
                
                # RRF formula: weighted combination
                rrf_score = (
                    self.dense_weight / (dense_rank + 60) +
                    (1 - self.dense_weight) / (bm25_rank + 60)
                )
                rrf_scores[doc_id] = rrf_score
            
            # Sort by RRF score and take top_k
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results[query] = sorted_docs
        
        return results

    def build_index(self, corpus: dict) -> None:
        """Build hybrid index from BEIR corpus format.
        
        Args:
            corpus: Dict mapping doc_id -> {'title': str, 'text': str}
        """
        doc_ids = list(corpus.keys())
        doc_texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in doc_ids
        ]
        self.index_documents(doc_ids, doc_texts)

    def retrieve(self, queries: dict, top_k: int = 100) -> dict:
        """Retrieve top-k documents for each query.
        
        Args:
            queries: Dict mapping query_id -> query_text
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dict mapping query_id -> {doc_id: score}
        """
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        raw_results = self.search(query_texts, top_k=top_k)
        
        # Remap to query_id keys
        results = {}
        for query_id, query_text in zip(query_ids, query_texts):
            results[query_id] = {
                doc_id: score for doc_id, score in raw_results[query_text]
            }
        
        return results
