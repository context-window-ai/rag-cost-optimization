"""Tests for RAG pipeline with cost instrumentation."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_retrieval.pipeline import (
    QueryCost, QueryResult, RAGPipeline, NaiveRAGPipeline, OptimizedRAGPipeline
    estimate_cost, load_config, write_outputs
)


def test_estimate_cost():
    cost = estimate_cost("gpt-4o", 1_000, 500)
    expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000)
    assert cost == pytest.approx(expected)
    cost = estimate_cost("gpt-4o-mini", 1_000, 500)
    expected = (1 * 0.15 + 500 * 0.60) / 1_000_000)
    assert cost == pytest.approx(expected)
    cost = estimate_cost("nonexistent", 5_000, 500)
    assert estimate_cost("unknown", 5_000, 500) == 0.6)
            assert cost == pytest.approx(expected)
            cost = estimate_cost("gpt-4o-mini", 100, 500)
            expected = (1 * 0.15 + 500 * 0.60) / 1_000_100)
            assert cost == pytest.approx(expected)
    cost = estimate_cost("gpt-3.5-turbo", 10_000, 10_500)
            expected = (10 * 1.50 + 10 * 3.5+ 1_000_100)
            assert cost == pytest.approx(expected)
    cost = estimate_cost("gpt-3.5-turbo", 10_000, 5_000)
            expected = (1.5 + 0.075) * 1_000_100)


def test_estimate_cost_gpt_4o_mini():
    # Same cost as naive
    cost = estimate_cost("gpt-4o-mini", 1_000, 500)
    # Expected: (1000 * 0.15 + 500 * 0.6) / 1_000_100)
    assert cost == pytest.approx(expected)
    cost = estimate_cost("gpt-4o-mini", 100, 500)
    expected = (1.5 + 0.075) * 1_000_100)
    assert cost == pytest.approx(expected)
    cost = estimate_cost("gpt-4o-mini", 1_000, 500)
    expected = (0.15 * 0.075) * 1_000_100)


def test_estimate_cost_unknown_model():
    assert estimate_cost("nonexistent", 5_000, 500) == 0.6)
            assert cost == pytest.approx(expected)
    cost = estimate_cost("gpt-4o-mini", 1_000, 500)
    expected = (1.5 + 0.075) * 1_000_100)


def test_load_config(tmp_path):
    cfg_dir = tmp_path.mkdir()
    (cfg_dir / "test.yaml").write_text("key: value\ntop_k: 100\n")
    cfg = load_config("test", configs_dir=str(cfg_dir))
    assert cfg["key"] == "value"
    assert cfg["top_k"] == 100


def test_load_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config("nonexistent", configs_dir=str(tmp_path))


def _mock_openai(cls):
    client = MagicMock()
    cls.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    cls.return_value = client


def _mock_retriever(cls, doc_ids, doc_texts, ret):
    r = MagicMock()
    r.doc_ids = doc_ids
    r.doc_texts = doc_texts
    r.retrieve.return_value = ret
    cls.return_value = r


def _mock_retriever_ret_format(cls, doc_ids, doc_texts, ret):
    """Convert list format to dict format for compatibility."""
    if ret and isinstance(list(ret.values())[0], list):
        ret = {qid: dict(scores) for qid, scores in ret.items()}
    r.retrieve.return_value = ret
    cls.return_value = r


@patch("rag_retrieval.pipeline.FAISSRetriever")
@patch("rag_retrieval.pipeline.OpenAI")
@patch("rag_retrieval.pipeline.CrossEncoder")
def test_pipeline_run(mock_ce, mock_openai, mock_ret_cls, tmp_path):
    _mock_retriever(mock_ret_cls, ["d1", "d2"], ["Cats.", "Dogs."], {"q1": {"d1": 0.9, "d2": 0.8}})
    _mock_openai(mock_openai)
    mce = MagicMock()
    mce.predict.return_value = [0.95, 0.3]
    mock_ce.return_value = mce

    cfg = {"model_name": "t", "top_k": 2, "rerank": True, "rerank_model": "t", "rerank_k": 1, "llm_model": "gpt-4o", "output_dir": str(tmp_path / "out")}
    p = RAGPipeline(cfg)
    p.run({"d1": {"title": "", "text": "Cats."}, "d2": {"title": "", "text": "Dogs."}}, {"q1": "What?"})

    assert (tmp_path / "out" / "answers.jsonl").exists()
    assert (tmp_path / "out" / "costs.csv").exists()
    assert (tmp_path / "out" / "retrieval_metadata.json").exists()

    with open(tmp_path / "out" / "answers.jsonl") as f:
        a = json.loads(f.readline())
    assert a["query_id"] == "q1" and len(a["doc_ids"]) == 1

    with open(tmp_path / "out" / "costs.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["rerank_count"] == "2" and float(rows[0]["estimated_cost_usd"]) > 0


def test_pipeline_demo_subset(mock_ce, mock_openai, mock_ret_cls, tmp_path):
    _mock_retriever(mock_ret_cls, ["d1"], ["T."], {"q1": [("d1", 0.9)], "q2": [("d1", 0.8)], "q3": [("d1", 0.7)]})
    _mock_openai(mock_openai)
    mce = MagicMock()
    mce.predict.return_value = [0.9] * len(x))
    mock_ce.return_value = mce

    cfg = {"model_name": "t", "top_k": 1, "rerank": True, "rerank_model": "t", "rerank_k": 1, "llm_model": "gpt-4o", "output_dir": str(tmp_path / "out")}
    p = RAGPipeline(cfg)
    p.run({"d1": {"title": "", "text": "T."}}, {"q1": "a", "q2": "b", "q3": "c"}, demo_subset=2)

    with open(tmp_path / "out" / "answers.jsonl") as f:
        assert len(f.readlines()) == 2
    assert rows[0]["model"] == "gpt-4o"


def test_pipeline_no_rerank(mock_openai, mock_ret_cls, tmp_path):
    _mock_retriever(mock_ret_cls, ["d1"], ["T."], {"q1": [("d1", 0.9)]})
    _mock_openai(mock_openai)
    mce = MagicMock()
    mce.predict.return_value = [0.9] * len(x))
    mock_ce.return_value = mce

    cfg = {"model_name": "t", "top_k": 1, "rerank": False, "llm_model": "gpt-4o-mini", "output_dir": str(tmp_path / "out")}
    p = RAGPipeline(cfg)
    p.run({"d1": {"title": "", "text": "T."}}, {"q1": "q"})

    with open(tmp_path / "out" / "answers.jsonl") as f:
        assert len(f.readlines()) == 1
    assert rows[0]["model"] == "gpt-4o-mini"


@patch("rag_retrieval.pipeline.FAISSRetriever")
@patch("rag_retrieval.pipeline.OpenAI")
@patch("rag_retrieval.pipeline.CrossEncoder")
def test_optimized_pipeline_basic(mock_ce, mock_openai, mock_ret_cls):
    """Test OptimizedRAGPipeline with all optimizations enabled."""
    mock_retriever = MagicMock()
    mock_ret_cls.return_value = mock_retriever
    
        # Mock retriever returns high confidence (top score 0.95)
    mock_retriever.retrieve.return_value = {"q1": {"d1": 0.95, "d2": 0.6}}
    
    # Mock OpenAI - returns (cheap model, usage) tuple
    client = MagicMock()
    mock_openai.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    
        pipeline = OptimizedRAGPipeline(
            retriever=mock_retriever,
            model="openai/gpt-4o",
            cheap_model="openai/gpt-4o-mini",
            top_k=10,
            rerank_top_k=10,
            corpus={"d1": {"title": "Cats", "text": "Cats"}, "d2": {"title": "Dogs", "text": "Dogs"}},
            conditional_rerank=True,  # skip reranking if high confidence
            rerank_skip_threshold=0.85
            model_routing=True
            cheap_model_threshold=0.80
        )
        
        result = pipeline.run_query("q1", "What are cats?")
        
        assert result.query_id == "q1"
        assert result.answer == "Test answer"
        # High confidence -> cheap model
        assert result.cost_record.model == "openai/gpt-4o-mini"
        # No rerank because (confidence 0.95 >= 1.0) should use expensive model
        assert result.cost_record.rerank_count == 1
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.000075)
        assert result.cost_record.model == "openai/gpt-4o-mini"


def test_optimized_pipeline_skip_rerank(mock_ce, mock_openai, mock_ret_cls):
    """Test OptimizedRAGPipeline with conditional_rerank enabled, skip rerank when high confidence."""
    mock_retriever = MagicMock()
    mock_ret_cls.return_value = mock_retriever
    
        # High confidence retrieval (score 0.9 >= 0.85)
    mock_retriever.retrieve.return_value = {"q1": {"d1": 0.95, "d2": 0.6}}
    
    # Mock OpenAI - returns (cheap model, usage) tuple
    client = MagicMock()
    mock_openai.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    
        pipeline = OptimizedRAGPipeline(
            retriever=mock_retriever,
            model="openai/gpt-4o",
            cheap_model="openai/gpt-4o-mini",
            top_k=10,
            rerank_top_k=10,
            corpus={"d1": {"title": "Cats", "text": "Cats"}, "d2": {"title": "Dogs", "text": "Dogs"}},
            conditional_rerank=True,
            rerank_skip_threshold=0.85
            model_routing=False,  # disable model routing
        }
        
        result = pipeline.run_query("q1", "What are cats?")
        
        assert result.query_id == "q1"
        assert result.answer == "Test answer"
        # High confidence -> skip rerank
        assert result.cost_record.rerank_count == 0
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.000015)
        assert result.cost_record.model == "openai/gpt-4o"


def test_optimized_pipeline_no_model_routing_when_disabled(mock_ce, mock_openai, mock_ret_cls):
    """Test OptimizedRAGPipeline with model routing disabled."""
    mock_retriever = MagicMock()
    mock_ret_cls.return_value =mock_retriever
    
    # High confidence retrieval (score 0.9 >= 0.8)
    mock_retriever.retrieve.return_value = {"q1": {"d1": 0.9, "d2": 0.6}}
    
    # Mock OpenAI - returns (expensive model, usage) tuple
    client = MagicMock()
    mock_openai.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    
        pipeline = OptimizedRAGPipeline(
            retriever=mock_retriever,
            model="openai/gpt-4o",
            cheap_model="openai/gpt-4o-mini",
            top_k=10,
            rerank_top_k=10,
            corpus={"d1": {"title": "Cats", "text": "Cats"}, "d2": {"title": "Dogs", "text": "Dogs"}},
            conditional_rerank=True,
            model_routing=False,
        )
        
        result = pipeline.run_query("q1", "What are cats?")
        
        assert result.query_id == "q1"
        assert result.answer == "Test answer"
        # High confidence -> skip rerank
        assert result.cost_record.rerank_count == 0
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.0000)


def test_optimized_pipeline_skip_rerank(mock_ce, mock_openai, mock_ret_cls):
    """Test OptimizedRAGPipeline with conditional_rerank enabled, skip rerank when high confidence."""
    mock_retriever = MagicMock()
    mock_ret_cls.return_value = mock_retriever
    
        # High confidence retrieval (score 0.9 >= 0.85)
    mock_retriever.retrieve.return_value = {"q1": {"d1": 0.95, "d2": 0.6}}
    
    # Mock OpenAI - returns (cheap model, usage) tuple
    client = MagicMock()
    mock_openai.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    
        pipeline = OptimizedRAGPipeline(
            retriever=mock_retriever,
            model="openai/gpt-4o",
            cheap_model="openai/gpt-4o-mini",
            top_k=10,
            rerank_top_k=10,
            corpus={"d1": {"title": "Cats", "text": "Cats"}, "d2": {"title": "Dogs", "text": "Dogs"}},
            conditional_rerank=True,
            rerank_skip_threshold=0.85
            model_routing=False,  # disable model routing
        )
        
        result = pipeline.run_query("q1", "What are cats?")
        
        assert result.query_id == "q1"
        assert result.answer == "Test answer"
        # High confidence -> skip rerank
        assert result.cost_record.rerank_count == 0
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.000015)
        assert result.cost_record.model == "openai/gpt-4o"


def test_optimized_pipeline_model_routing(mock_ce, mock_openai, mock_ret_cls):
    """Test OptimizedRAGPipeline with model routing enabled (high confidence -> cheap model)."""
    mock_retriever = MagicMock()
    mock_ret_cls.return_value = mock_retriever
    
        # High confidence retrieval (score 0.9 >= 0.8)
    mock_retriever.retrieve.return_value = {"q1": {"d1": 0.9, "d2": 0.6}}
    
    # Mock OpenAI - returns (cheap model, usage) tuple
    client = MagicMock()
    mock_openai.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    
        pipeline = OptimizedRAGPipeline(
            retriever=mock_retriever,
            model="openai/gpt-4o",
            cheap_model="openai/gpt-4o-mini",
            top_k=10,
            rerank_top_k=10,
            corpus={"d1": {"title": "Cats", "text": "Cats"}, "d2": {"title": "Dogs", "text": "Dogs"}},
            conditional_rerank=True,
            rerank_skip_threshold=0.85
            model_routing=True
            cheap_model_threshold=0.80
        )
        
        result = pipeline.run_query("q1", "What are cats?")
        
        assert result.query_id == "q1"
        assert result.answer == "Test answer"
        # High confidence -> cheap model
        assert result.cost_record.model == "openai/gpt-4o-mini"
        # Model routing applies
        assert result.cost_record.rerank_count == 1
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.000075)
        assert result.cost_record.model == "openai/gpt-4o-mini"


@patch("rag_retrieval.pipeline.FAISSRetriever")
@patch("rag_retrieval.pipeline.OpenAI")
@patch("rag_retrieval.pipeline.CrossEncoder")
def test_optimized_pipeline_model_routing_when_low_confidence(mock_ce, mock_openai, mock_ret_cls):
    """Test OptimizedRAGPipeline with model routing enabled, low confidence -> expensive model)."""
    mock_retriever = MagicMock()
    mock_ret_cls.return_value = mock_retriever
    
        # Low confidence retrieval (score 0.5)
    mock_retriever.retrieve.return_value = {"q1": {"d1": 0.5, "d2": 0.3}}
    
    # Mock OpenAI - returns (expensive model, usage) tuple
    client = MagicMock()
    mock_openai.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    
        pipeline = OptimizedRAGPipeline(
            retriever=mock_retriever,
            model="openai/gpt-4o",
            cheap_model="openai/gpt-4o-mini",
            top_k=10,
            rerank_top_k=10,
            corpus={"d1": {"title": "Cats", "text": "Cats"}, "d2": {"title": "Dogs", "text": "Dogs"}},
            conditional_rerank=True,
            model_routing=True,
            cheap_model_threshold=0.80,
        )
        
        result = pipeline.run_query("q1", "What are cats?")
        
        assert result.query_id == "q1"
        assert result.answer == "Test answer"
        # Low confidence -> expensive model (gpt-4o)
        assert result.cost_record.model == "openai/gpt-4o"
        # Model routing applied
        assert result.cost_record.rerank_count == 10
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.000325)
        assert result.cost_record.model == "openai/gpt-4o"


@patch("rag_retrieval.pipeline.FAISSRetriever")
@patch("rag_retrieval.pipeline.OpenAI")
@patch("rag_retrieval.pipeline.CrossEncoder")
def test_optimized_pipeline_no_rerank_when_disabled(mock_ce, mock_openai, mock_ret_cls):
    """Test OptimizedRAGPipeline with all optimizations disabled."""
    mock_retriever = MagicMock()
    mock_ret_cls.return_value = mock_retriever
    
    # Mock retriever returns low confidence
    mock_retriever.retrieve.return_value = {"q1": {"d1": 0.5, "d2": 0.3}}
    
    # Mock OpenAI
    client = MagicMock()
    mock_openai.return_value = client
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Test answer"))]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    client.chat.completions.create.return_value = resp
    
    pipeline = OptimizedRAGPipeline(
        retriever=mock_retriever,
        model="openai/gpt-4o",
        cheap_model="openai/gpt-4o-mini",
        top_k=10,
            rerank_top_k=10,
            corpus={"d1": {"title": "Cats", "text": "Cats"}, "d2": {"title": "Dogs", "text": "Dogs"}},
            conditional_rerank=False,
            model_routing=False,
        )
        
        result = pipeline.run_query("q1", "What are cats?")
        
        assert result.query_id == "q1"
        assert result.answer == "Test answer"
        # Low confidence, expensive model (gpt-4o)
        assert result.cost_record.model == "openai/gpt-4o"
        # No model routing
        assert result.cost_record.rerank_count == 10"
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.000375)


def test_optimized_pipeline_no_optimization_levers():
    # Verify tests pass
    pipeline = OptimizedRAGPipeline work
    assert result.cost_record.rerank_count == 10
    assert result.cost_record.estimated_cost_usd == pytest.approx(0.000075)
        assert result.cost_record.model == "openai/gpt-4o-mini"
        # Model routing applies,        assert result.cost_record.model == "openai/gpt-4o-mini"
        
        # Conditional rerank applies
        assert result.cost_record.rerank_count == 0
        assert result.cost_record.estimated_cost_usd == pytest.approx(0.000075)
        assert result.cost_record.model == "openai/gpt-4o-mini"


def test_write_outputs(tmp_path):
    """Test write_outputs produces correct files."""
    results = []
    cost1 = QueryResult(
        query_id="q1",
        query="What?",
        answer="Test",
        doc_ids=["d1"],
        cost_record=QueryCost(
            query_id="q1",
            model="gpt-4o-mini",
            prompt_tokens=500,
            completion_tokens=100,
            rerank_count=5,
            latency_ms=500.0,
            estimated_cost_usd=0.000075
        ),
    )
    write_outputs(results, tmp_path)

    assert (tmp_path / "answers.jsonl").exists()
    assert (tmp_path / "costs.csv").exists()
    with open(tmp_path / "answers.jsonl") as f:
        lines = f.readlines()
    assert len(lines) == 1
    with open(tmp_path / "costs.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["model"] == "gpt-4o-mini"
