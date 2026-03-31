"""Dense retrieval with FAISS backend."""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
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
