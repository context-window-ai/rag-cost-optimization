"""Tests for LLM judge evaluation script."""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.eval_judge import (
    JudgeScore,
    call_judge,
    estimate_cost,
    get_context_for_docs,
    load_answers,
    load_corpus,
    load_judge_prompts,
    load_retrieval_metadata,
    parse_score,
    write_scores_csv,
)


def test_estimate_cost():
    """Test cost estimation for different models."""
    # Test gpt-5.4-mini (the judge model)
    cost = estimate_cost("openai/gpt-5.4-mini", 1000, 500)
    expected = (1000 * 0.75 + 500 * 4.50) / 1_000_000
    assert cost == pytest.approx(expected)
    
    # Test unknown model returns 0
    cost = estimate_cost("unknown-model", 1000, 500)
    assert cost == 0.0


def test_load_judge_prompts():
    """Test loading judge prompts from config."""
    prompts = load_judge_prompts("configs/judge_prompts.yaml")
    
    assert "version" in prompts
    assert "faithfulness" in prompts
    assert "answer_relevance" in prompts
    assert "judge_model" in prompts
    assert prompts["judge_model"] == "openai/gpt-5.4-mini"
    
    # Check faithfulness prompt structure
    assert "system_prompt" in prompts["faithfulness"]
    assert "user_prompt" in prompts["faithfulness"]
    assert "version" in prompts["faithfulness"]
    
    # Check answer relevance prompt structure
    assert "system_prompt" in prompts["answer_relevance"]
    assert "user_prompt" in prompts["answer_relevance"]
    assert "version" in prompts["answer_relevance"]


def test_load_answers(tmp_path):
    """Test loading answers from JSONL file."""
    answers_file = tmp_path / "answers.jsonl"
    answers_file.write_text(
        '{"query_id": "1", "query": "What?", "answer": "Test", "doc_ids": ["d1"]}\n'
        '{"query_id": "2", "query": "Why?", "answer": "Because", "doc_ids": ["d2"]}\n'
    )
    
    answers = load_answers(str(answers_file))
    
    assert len(answers) == 2
    assert answers[0]["query_id"] == "1"
    assert answers[1]["query_id"] == "2"


def test_load_answers_empty_lines(tmp_path):
    """Test loading answers handles empty lines."""
    answers_file = tmp_path / "answers.jsonl"
    answers_file.write_text(
        '{"query_id": "1", "query": "What?", "answer": "Test", "doc_ids": ["d1"]}\n'
        '\n'
        '{"query_id": "2", "query": "Why?", "answer": "Because", "doc_ids": ["d2"]}\n'
    )
    
    answers = load_answers(str(answers_file))
    assert len(answers) == 2


def test_load_retrieval_metadata(tmp_path):
    """Test loading retrieval metadata."""
    # Create answers file
    answers_file = tmp_path / "answers.jsonl"
    answers_file.write_text('{"query_id": "1", "query": "What?", "answer": "Test", "doc_ids": ["d1"]}')
    
    # Create metadata file
    metadata = {
        "1": {
            "doc_ids": ["d1", "d2"],
            "scores": [0.9, 0.8]
        }
    }
    metadata_file = tmp_path / "retrieval_metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    
    loaded = load_retrieval_metadata(str(answers_file))
    
    assert "1" in loaded
    assert loaded["1"]["doc_ids"] == ["d1", "d2"]


def test_load_retrieval_metadata_missing(tmp_path):
    """Test when retrieval metadata is missing, returns None."""
    answers_file = tmp_path / "answers.jsonl"
    answers_file.write_text('{"query_id": "1"}')
    
    # Should return None when file is missing
    result = load_retrieval_metadata(str(answers_file))
    assert result is None


def test_get_context_for_docs():
    """Test building context from document IDs."""
    corpus = {
        "d1": {"title": "Cats", "text": "Cats are furry animals."},
        "d2": {"title": "Dogs", "text": "Dogs are loyal pets."},
    }
    
    context = get_context_for_docs(["d1", "d2"], corpus)
    
    assert "[1] Cats: Cats are furry animals." in context
    assert "[2] Dogs: Dogs are loyal pets." in context


def test_get_context_for_docs_missing():
    """Test context handles missing documents."""
    corpus = {
        "d1": {"title": "Cats", "text": "Cats are furry animals."},
    }
    
    context = get_context_for_docs(["d1", "d_missing"], corpus)
    
    assert "[1] Cats: Cats are furry animals." in context
    assert "[2]" not in context  # Missing doc not included


def test_parse_score():
    """Test parsing score and reasoning from LLM response."""
    response = "Reasoning: The answer is well-supported by the context.\nScore: 5"
    score, reasoning = parse_score(response)
    
    assert score == 5
    assert "well-supported" in reasoning


def test_parse_score_lowercase():
    """Test parsing handles case-insensitive matching."""
    response = "reasoning: Good answer.\nscore: 4"
    score, reasoning = parse_score(response)
    
    assert score == 4
    assert "Good" in reasoning


def test_parse_score_clamp():
    """Test score clamping to 1-5 range."""
    response = "Reasoning: Test\nScore: 7"
    score, _ = parse_score(response)
    
    assert score == 5  # Clamped to max


def test_parse_score_fallback():
    """Test fallback scoring when format not matched."""
    response = "I give this a 3 out of 5."
    score, reasoning = parse_score(response)
    
    assert score == 3
    assert "Fallback" in reasoning


def test_parse_score_no_digit():
    """Test fallback when no digit found."""
    response = "No score here."
    score, reasoning = parse_score(response)
    
    assert score == 3  # Default fallback
    assert "Fallback" in reasoning


def test_write_scores_csv(tmp_path):
    """Test writing scores to CSV."""
    scores = [
        JudgeScore(
            query_id="q1",
            faithfulness_score=5,
            faithfulness_reasoning="Good",
            answer_relevance_score=4,
            answer_relevance_reasoning="Relevant",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=0.001,
        ),
        JudgeScore(
            query_id="q2",
            faithfulness_score=3,
            faithfulness_reasoning="Partial",
            answer_relevance_score=5,
            answer_relevance_reasoning="Very relevant",
            prompt_tokens=120,
            completion_tokens=60,
            estimated_cost_usd=0.0012,
        ),
    ]
    
    output_path = tmp_path / "scores.csv"
    write_scores_csv(scores, output_path)
    
    with open(output_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 2
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["faithfulness_score"] == "5"
    assert rows[0]["answer_relevance_score"] == "4"
    assert rows[1]["query_id"] == "q2"
    assert rows[1]["faithfulness_score"] == "3"


@patch("scripts.eval_judge.OpenAI")
def test_call_judge(mock_openai):
    """Test calling LLM judge."""
    # Mock the client and response
    client = MagicMock()
    mock_openai.return_value = client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Score: 5"))]
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    client.chat.completions.create.return_value = mock_response
    
    response_text, p_tokens, c_tokens = call_judge(
        client=client,
        model="openai/gpt-5.4-mini",
        temperature=0.0,
        system_prompt="You are a judge.",
        user_prompt="Evaluate this.",
    )
    
    assert response_text == "Score: 5"
    assert p_tokens == 100
    assert c_tokens == 50
    
    # Verify the call was made correctly
    client.chat.completions.create.assert_called_once_with(
        model="openai/gpt-5.4-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": "Evaluate this."},
        ],
    )


def test_judge_score_dataclass():
    """Test JudgeScore dataclass."""
    score = JudgeScore(
        query_id="test",
        faithfulness_score=4,
        faithfulness_reasoning="Good",
        answer_relevance_score=5,
        answer_relevance_reasoning="Great",
        prompt_tokens=100,
        completion_tokens=50,
        estimated_cost_usd=0.001,
    )
    
    # Test asdict conversion
    from dataclasses import asdict
    d = asdict(score)
    
    assert d["query_id"] == "test"
    assert d["faithfulness_score"] == 4
    assert d["answer_relevance_score"] == 5
    assert d["estimated_cost_usd"] == 0.001


def test_load_corpus():
    """Test loading corpus from JSONL file."""
    # This test uses the actual scifact corpus if available
    # Skip if not available
    corpus_path = Path("datasets/scifact/corpus.jsonl")
    if not corpus_path.exists():
        pytest.skip("Scifact corpus not available")
    
    corpus = load_corpus("scifact", "datasets")
    
    assert len(corpus) > 0
    # Check structure of a document
    for doc_id, doc in corpus.items():
        assert "title" in doc
        assert "text" in doc
        break


def test_load_judge_prompts_missing():
    """Test error when judge prompts config is missing."""
    with pytest.raises(FileNotFoundError, match="Judge prompts config not found"):
        load_judge_prompts("nonexistent_config.yaml")


# Integration test (marked to skip by default)
@pytest.mark.integration
def test_eval_judge_end_to_end():
    """End-to-end test of judge evaluation (requires API key)."""
    import subprocess
    import os
    
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Check if answers file exists
    answers_path = Path("outputs/naive/answers.jsonl")
    if not answers_path.exists():
        pytest.skip("No answers file available for testing")
    
    # Run the script with limit=1
    result = subprocess.run(
        ["uv", "run", "python", "scripts/eval_judge.py", 
         "--answers", str(answers_path), "--limit", "1"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "judge_scores.csv" in result.stdout or "Evaluation complete" in result.stdout
