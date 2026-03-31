#!/usr/bin/env python3
"""
LLM Judge Evaluation Script for RAG Answers.

Evaluates saved RAG answers on two dimensions:
1. Faithfulness: Is the answer supported by the retrieved context?
2. Answer Relevance: Does the answer address the query?

Usage:
    uv run python scripts/eval_judge.py --answers outputs/naive/answers.jsonl
    uv run python scripts/eval_judge.py --answers outputs/optimized/answers.jsonl
"""

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-1M-token pricing (USD) - from pipeline.py
MODEL_PRICING = {
    "openai/gpt-5.4-mini": {"prompt": 0.75, "completion": 4.50},
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "openai/gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "openai/gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
}


@dataclass
class JudgeScore:
    """Judge evaluation scores for a single query."""
    query_id: str
    faithfulness_score: int
    faithfulness_reasoning: str
    answer_relevance_score: int
    answer_relevance_reasoning: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a model given token counts."""
    pricing = MODEL_PRICING.get(model, {"prompt": 0.0, "completion": 0.0})
    return (
        prompt_tokens * pricing["prompt"]
        + completion_tokens * pricing["completion"]
    ) / 1_000_000


def load_judge_prompts(config_path: str = "configs/judge_prompts.yaml") -> Dict:
    """Load judge prompt templates from config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Judge prompts config not found: {config_path}")
    
    with open(path) as f:
        return yaml.safe_load(f)


def load_answers(answers_path: str) -> List[Dict]:
    """Load answers from JSONL file."""
    answers = []
    with open(answers_path) as f:
        for line in f:
            if line.strip():
                answers.append(json.loads(line))
    return answers


def load_retrieval_metadata(answers_path: str) -> Optional[Dict]:
    """Load retrieval metadata from the same directory as answers.
    
    Returns None if metadata file doesn't exist (doc_ids will be taken from answers).
    """
    answers_dir = Path(answers_path).parent
    metadata_path = answers_dir / "retrieval_metadata.json"
    
    if not metadata_path.exists():
        logger.warning(
            f"Retrieval metadata not found: {metadata_path}. "
            "Using doc_ids from answers file."
        )
        return None
    
    with open(metadata_path) as f:
        return json.load(f)


def load_corpus(dataset_name: str = "scifact", data_dir: str = "datasets") -> Dict:
    """Load corpus to get document texts."""
    corpus_path = Path(data_dir) / dataset_name / "corpus.jsonl"
    
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}. "
            "Make sure the dataset is downloaded."
        )
    
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_id = doc["_id"]
                corpus[doc_id] = {
                    "title": doc.get("title", ""),
                    "text": doc.get("text", "")
                }
    return corpus


def get_context_for_docs(doc_ids: List[str], corpus: Dict) -> str:
    """Build context string from document IDs."""
    contexts = []
    for i, doc_id in enumerate(doc_ids):
        if doc_id in corpus:
            doc = corpus[doc_id]
            text = f"{doc['title']}: {doc['text']}"
            contexts.append(f"[{i + 1}] {text}")
    return "\n\n".join(contexts)


def parse_score(response_text: str) -> Tuple[int, str]:
    """Parse score and reasoning from LLM response."""
    # Extract reasoning
    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=Score:|$)", response_text, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
    
    # Extract score - look for "Score: X" pattern
    score_match = re.search(r"Score:\s*(\d)", response_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        # Clamp to 1-5 range
        score = max(1, min(5, score))
    else:
        # Fallback: look for any single digit in the response
        digit_match = re.search(r"\b([1-5])\b", response_text)
        score = int(digit_match.group(1)) if digit_match else 3
        reasoning = f"[Fallback scoring] {reasoning}"
    
    return score, reasoning


def call_judge(
    client: OpenAI,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, int, int]:
    """Call the LLM judge and return (response_text, prompt_tokens, completion_tokens)."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def evaluate_answer(
    client: OpenAI,
    query_id: str,
    query: str,
    answer: str,
    context: str,
    judge_config: Dict,
) -> JudgeScore:
    """Evaluate a single answer on faithfulness and answer relevance."""
    model = judge_config["judge_model"]
    temperature = judge_config["judge_temperature"]
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Evaluate faithfulness
    faith_system = judge_config["faithfulness"]["system_prompt"]
    faith_user = judge_config["faithfulness"]["user_prompt"].format(
        context=context,
        query=query,
        answer=answer,
    )
    
    faith_response, p_tokens, c_tokens = call_judge(
        client, model, temperature, faith_system, faith_user
    )
    total_prompt_tokens += p_tokens
    total_completion_tokens += c_tokens
    
    faith_score, faith_reasoning = parse_score(faith_response)
    
    # Evaluate answer relevance
    rel_system = judge_config["answer_relevance"]["system_prompt"]
    rel_user = judge_config["answer_relevance"]["user_prompt"].format(
        query=query,
        answer=answer,
    )
    
    rel_response, p_tokens, c_tokens = call_judge(
        client, model, temperature, rel_system, rel_user
    )
    total_prompt_tokens += p_tokens
    total_completion_tokens += c_tokens
    
    rel_score, rel_reasoning = parse_score(rel_response)
    
    # Calculate cost
    cost_usd = estimate_cost(model, total_prompt_tokens, total_completion_tokens)
    
    return JudgeScore(
        query_id=query_id,
        faithfulness_score=faith_score,
        faithfulness_reasoning=faith_reasoning,
        answer_relevance_score=rel_score,
        answer_relevance_reasoning=rel_reasoning,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        estimated_cost_usd=round(cost_usd, 6),
    )


def write_scores_csv(scores: List[JudgeScore], output_path: Path) -> None:
    """Write judge scores to CSV file."""
    fields = [
        "query_id",
        "faithfulness_score",
        "faithfulness_reasoning",
        "answer_relevance_score",
        "answer_relevance_reasoning",
        "prompt_tokens",
        "completion_tokens",
        "estimated_cost_usd",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            writer.writerow(asdict(score))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers using LLM judge"
    )
    parser.add_argument(
        "--answers",
        required=True,
        help="Path to answers.jsonl file",
    )
    parser.add_argument(
        "--judge-config",
        default="configs/judge_prompts.yaml",
        help="Path to judge prompts config file",
    )
    parser.add_argument(
        "--dataset",
        default="scifact",
        help="Dataset name (default: scifact)",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets",
        help="Data directory (default: datasets)",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path (default: same directory as answers, judge_scores.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of answers to evaluate (for testing)",
    )
    
    args = parser.parse_args()
    
    # Load configurations
    logger.info(f"Loading judge prompts from {args.judge_config}")
    judge_config = load_judge_prompts(args.judge_config)
    logger.info(f"Judge prompt version: {judge_config['version']}")
    logger.info(f"Faithfulness prompt version: {judge_config['faithfulness']['version']}")
    logger.info(f"Answer relevance prompt version: {judge_config['answer_relevance']['version']}")
    
    # Load data
    logger.info(f"Loading answers from {args.answers}")
    answers = load_answers(args.answers)
    logger.info(f"Loaded {len(answers)} answers")
    
    logger.info("Loading retrieval metadata...")
    retrieval_metadata = load_retrieval_metadata(args.answers)
    # Note: retrieval_metadata may be None; doc_ids come from answers file
    
    logger.info(f"Loading corpus from {args.dataset}...")
    corpus = load_corpus(args.dataset, args.data_dir)
    logger.info(f"Loaded {len(corpus)} documents")
    
    if args.limit:
        answers = answers[:args.limit]
        logger.info(f"Limited to {len(answers)} answers")
    
    # Initialize OpenAI client
    client = OpenAI(base_url=OPENROUTER_BASE_URL)
    
    # Evaluate each answer
    scores: List[JudgeScore] = []
    total_cost = 0.0
    
    for i, answer_record in enumerate(answers):
        query_id = answer_record["query_id"]
        query = answer_record["query"]
        answer = answer_record["answer"]
        doc_ids = answer_record["doc_ids"]
        
        # Get context from retrieved documents
        context = get_context_for_docs(doc_ids, corpus)
        
        logger.info(f"Evaluating query {i + 1}/{len(answers)}: {query_id}")
        
        try:
            score = evaluate_answer(
                client=client,
                query_id=query_id,
                query=query,
                answer=answer,
                context=context,
                judge_config=judge_config,
            )
            scores.append(score)
            total_cost += score.estimated_cost_usd
            
            logger.info(
                f"  Faithfulness: {score.faithfulness_score}/5, "
                f"Relevance: {score.answer_relevance_score}/5, "
                f"Cost: ${score.estimated_cost_usd:.6f}"
            )
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error evaluating query {query_id}: {e}")
            continue
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.answers).parent / "judge_scores.csv"
    
    # Write results
    write_scores_csv(scores, output_path)
    
    # Print summary
    if scores:
        avg_faith = sum(s.faithfulness_score for s in scores) / len(scores)
        avg_rel = sum(s.answer_relevance_score for s in scores) / len(scores)
        
        logger.info("=" * 60)
        logger.info(f"Evaluation complete!")
        logger.info(f"  Total queries evaluated: {len(scores)}")
        logger.info(f"  Average faithfulness: {avg_faith:.2f}/5")
        logger.info(f"  Average answer relevance: {avg_rel:.2f}/5")
        logger.info(f"  Total cost: ${total_cost:.4f}")
        logger.info(f"  Results saved to: {output_path}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
