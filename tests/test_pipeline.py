"""Tests for RAG pipeline with cost instrumentation."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_retrieval.pipeline import QueryCost, RAGPipeline, estimate_cost, load_config


def test_estimate_cost():
    cost = estimate_cost("gpt-4o", 1_000, 500)
    expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
    assert cost == pytest.approx(expected)


def test_estimate_cost_unknown_model():
    assert estimate_cost("nonexistent", 1000, 500) == 0.0


def test_query_cost_dataclass():
    qc = QueryCost(
        query_id="q1", model="gpt-4o", prompt_tokens=100,
        completion_tokens=50, rerank_count=10, latency_ms=500.0,
        estimated_cost_usd=0.001,
    )
    assert qc.query_id == "q1"
    assert qc.prompt_tokens == 100


def test_load_config(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
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


def _mock_retriever(cls, doc_ids, doc_texts, ret):
    r = MagicMock()
    r.doc_ids = doc_ids
    r.doc_texts = doc_texts
    r.retrieve.return_value = ret
    cls.return_value = r


@patch("rag_retrieval.pipeline.FAISSRetriever")
@patch("rag_retrieval.pipeline.OpenAI")
@patch("rag_retrieval.pipeline.CrossEncoder")
def test_pipeline_run(mock_ce, mock_openai, mock_ret_cls, tmp_path):
    _mock_retriever(mock_ret_cls, ["d1", "d2"], ["Cats.", "Dogs."], {"q1": [("d1", 0.9), ("d2", 0.8)]})
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


@patch("rag_retrieval.pipeline.FAISSRetriever")
@patch("rag_retrieval.pipeline.OpenAI")
@patch("rag_retrieval.pipeline.CrossEncoder")
def test_pipeline_demo_subset(mock_ce, mock_openai, mock_ret_cls, tmp_path):
    _mock_retriever(mock_ret_cls, ["d1"], ["T."], {"q1": [("d1", 0.9)], "q2": [("d1", 0.8)], "q3": [("d1", 0.7)]})
    _mock_openai(mock_openai)
    mock_ce.return_value = MagicMock(predict=lambda x: [0.9] * len(x))

    cfg = {"model_name": "t", "top_k": 1, "rerank": True, "rerank_model": "t", "rerank_k": 1, "llm_model": "gpt-4o", "output_dir": str(tmp_path / "out")}
    p = RAGPipeline(cfg)
    p.run({"d1": {"title": "", "text": "T."}}, {"q1": "a", "q2": "b", "q3": "c"}, demo_subset=2)

    with open(tmp_path / "out" / "answers.jsonl") as f:
        assert len(f.readlines()) == 2


@patch("rag_retrieval.pipeline.FAISSRetriever")
@patch("rag_retrieval.pipeline.OpenAI")
def test_pipeline_no_rerank(mock_openai, mock_ret_cls, tmp_path):
    _mock_retriever(mock_ret_cls, ["d1"], ["T."], {"q1": [("d1", 0.9)]})
    _mock_openai(mock_openai)

    cfg = {"model_name": "t", "top_k": 1, "rerank": False, "llm_model": "gpt-4o-mini", "output_dir": str(tmp_path / "out")}
    p = RAGPipeline(cfg)
    p.run({"d1": {"title": "", "text": "T."}}, {"q1": "q"})

    with open(tmp_path / "out" / "costs.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["rerank_count"] == "0" and rows[0]["model"] == "gpt-4o-mini"
