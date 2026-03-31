"""RAG pipeline with built-in cost instrumentation."""

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from openai import OpenAI
from sentence_transformers import CrossEncoder

from rag_retrieval.retriever import FAISSRetriever

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-1M-token pricing (USD)
MODEL_PRICING = {
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "openai/gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "openai/gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
}


@dataclass
class QueryCost:
    """Cost and latency record for a single query."""

    query_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    rerank_count: int
    latency_ms: float
    estimated_cost_usd: float


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a model given token counts."""
    pricing = MODEL_PRICING.get(model, {"prompt": 0.0, "completion": 0.0})
    return (
        prompt_tokens * pricing["prompt"]
        + completion_tokens * pricing["completion"]
    ) / 1_000_000


class RAGPipeline:
    """RAG pipeline that bakes in per-query cost tracking.
    
    This is the "before" in the before/after comparison:
    - High top_k retrieval (100 by default)
    - Always rerank with CrossEncoder
    - Call the most expensive model (gpt-4o)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retriever = FAISSRetriever(model_name=config["model_name"])
        self.reranker = (
            CrossEncoder(config["rerank_model"]) if config.get("rerank") else None
        )
        self.llm_model = config["llm_model"]
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cost_records: List[QueryCost] = []
        self._doc_text_map: Dict[str, str] = {}

    def _build_doc_map(self) -> None:
        """Build mapping from doc_id to doc_text."""
        self._doc_text_map = dict(
            zip(self.retriever.doc_ids, self.retriever.doc_texts)
        )

    def _call_llm(self, query: str, contexts: List[str]):
        """Call the LLM and return (answer_text, usage)."""
        client = OpenAI(base_url=OPENROUTER_BASE_URL)
        context_block = "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        )
        system_prompt = (
            "Answer the question based on the provided context passages. "
            "Cite passage numbers. If the context is insufficient, say so clearly."
        )
        response = client.chat.completions.create(
            model=self.llm_model,
            temperature=self.config.get("llm_temperature", 0.0),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content, response.usage

    def run(
        self,
        corpus: Dict,
        queries: Dict,
        demo_subset: Optional[int] = None,
    ) -> None:
        """Run the full pipeline: retrieve -> rerank -> generate -> record costs.
        
        Args:
            corpus: Dict mapping doc_id -> {'title': str, 'text': str}
            queries: Dict mapping query_id -> query_text
            demo_subset: If set, only process this many queries
        """
        logger.info("Building document index ...")
        self.retriever.build_index(corpus)
        self._build_doc_map()

        query_ids = list(queries.keys())
        if demo_subset is not None:
            query_ids = query_ids[:demo_subset]

        logger.info("Processing %d queries ...", len(query_ids))

        answers_path = self.output_dir / "answers.jsonl"
        answers_path.unlink(missing_ok=True)

        retrieval_meta: Dict[str, Any] = {}

        for qid in query_ids:
            query = queries[qid]
            t0 = time.time()

            # --- retrieve ---
            results = self.retriever.retrieve(
                {qid: query}, top_k=self.config["top_k"]
            )
            # retriever returns {qid: {doc_id: score}}, convert to list of tuples
            doc_scores = list(results.get(qid, {}).items())

            # --- rerank ---
            rerank_count = 0
            if self.reranker and doc_scores:
                rerank_count = len(doc_scores)
                pairs = [
                    (query, self._doc_text_map[did]) for did, _ in doc_scores
                ]
                ce_scores = self.reranker.predict(pairs)
                ranked = sorted(
                    zip(doc_scores, ce_scores), key=lambda x: x[1], reverse=True
                )
                doc_scores = [
                    (did, float(ce_sc))
                    for (did, _), ce_sc in ranked
                ][: self.config.get("rerank_k", len(doc_scores))]

            contexts = [self._doc_text_map[did] for did, _ in doc_scores]

            # --- generate ---
            answer, usage = self._call_llm(query, contexts)

            latency_ms = (time.time() - t0) * 1000
            cost_usd = estimate_cost(
                self.llm_model, usage.prompt_tokens, usage.completion_tokens
            )

            record = QueryCost(
                query_id=qid,
                model=self.llm_model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                rerank_count=rerank_count,
                latency_ms=round(latency_ms, 2),
                estimated_cost_usd=round(cost_usd, 6),
            )
            self.cost_records.append(record)

            logger.info(json.dumps(asdict(record)))

            retrieval_meta[qid] = {
                "retrieved_count": self.config["top_k"],
                "rerank_count": rerank_count,
                "context_count": len(contexts),
                "doc_ids": [did for did, _ in doc_scores],
                "scores": [round(sc, 4) for _, sc in doc_scores],
            }

            with open(answers_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "query": query,
                            "answer": answer,
                            "doc_ids": [did for did, _ in doc_scores],
                        }
                    )
                    + "\n"
                )

        # --- write sidecar outputs ---
        with open(self.output_dir / "retrieval_metadata.json", "w") as f:
            json.dump(retrieval_meta, f, indent=2)

        self._write_cost_csv()

        total = sum(r.estimated_cost_usd for r in self.cost_records)
        avg_lat = (
            sum(r.latency_ms for r in self.cost_records) / len(self.cost_records)
            if self.cost_records
            else 0
        )
        logger.info(
            "Done. %d queries | $%.4f total | %.0f ms avg latency",
            len(self.cost_records),
            total,
            avg_lat,
        )

    def _write_cost_csv(self) -> None:
        """Write per-query costs to CSV."""
        path = self.output_dir / "costs.csv"
        fields = [
            "query_id",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "rerank_count",
            "latency_ms",
            "estimated_cost_usd",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.cost_records:
                writer.writerow(asdict(r))


def load_config(config_name: str, configs_dir: str = "configs") -> Dict[str, Any]:
    """Load a YAML pipeline config by name."""
    config_path = Path(configs_dir) / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)
