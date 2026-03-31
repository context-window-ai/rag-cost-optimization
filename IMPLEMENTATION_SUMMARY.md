# Issue #4: Naive RAG Pipeline with Cost Instrumentation - Implementation Summary

## ✅ COMPLETED

### Implementation Overview
Built the intentionally expensive baseline RAG pipeline with built-in cost tracking for before/after cost optimization comparison.

### Files Created/Modified

#### Core Pipeline Module
- **src/rag_retrieval/pipeline.py** (290 lines)
  - `QueryCost` dataclass: Tracks per-query metrics
  - `estimate_cost()`: Calculates USD cost based on token counts
  - `RAGPipeline` class: Full RAG pipeline with cost instrumentation
    - Retrieves documents using FAISS
    - Reranks with CrossEncoder (always on for naive baseline)
    - Generates answers using OpenAI LLM (gpt-4o by default)
    - Tracks costs for every query

#### Configuration
- **configs/naive.yaml**: Expensive baseline configuration
  - Dataset: scifact
  - LLM: gpt-4o (most expensive)
  - Top K: 100 (high)
  - Rerank: true (always)
  - Rerank K: 20
  - Demo subset: 20 queries (for testing)

- **configs/test.yaml**: Quick test configuration
  - LLM: gpt-4o-mini (cheaper)
  - Top K: 10
  - Demo subset: 3 queries

#### Scripts
- **scripts/run_pipeline.py**: Command-line interface
  - `--config`: Config name (required)
  - `--demo`: Override demo_subset
  - `--configs_dir`: Custom config directory
  - `--data_dir`: Custom data directory

#### Tests
- **tests/test_pipeline.py**: Comprehensive test suite (8 tests)
  - Cost estimation
  - Config loading
  - Pipeline initialization
  - Pipeline execution with/without reranking
  - Demo subset handling
  - CSV output format verification

### Pipeline Outputs (per run)

For each query, the pipeline writes:

1. **answers.jsonl** (JSON Lines format)
   ```json
   {"query_id": "q1", "query": "...", "answer": "..."}
   ```

2. **retrieval_metadata.json** (JSON format)
   ```json
   {
     "q1": {
       "retrieved_count": 100,
       "rerank_count": 20,
       "context_count": 20,
       "doc_ids": ["doc1", "doc2", ...],
       "scores": [0.95, 0.89, ...]
     }
   }
   ```

3. **costs.csv** (CSV format with headers)
   - query_id
   - model
   - prompt_tokens
   - completion_tokens
   - rerank_count
   - latency_ms
   - estimated_cost_usd

### Cost Tracking Features

- **Per-query token counting**: Prompt and completion tokens
- **Latency measurement**: Milliseconds per query
- **Cost estimation**: USD based on model pricing
- **Rerank tracking**: Number of documents reranked

### Model Pricing (per 1M tokens)
```python
MODEL_PRICING = {
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
}
```

### Usage

```bash
# Run on full dataset
uv run python scripts/run_pipeline.py --config naive

# Run on demo subset (for testing)
uv run python scripts/run_pipeline.py --config naive --demo 5

# Run with custom config
uv run python scripts/run_pipeline.py --config test
```

### Test Results
```
27 tests passed ✓
- test_estimate_cost
- test_estimate_cost_unknown_model
- test_query_cost_dataclass
- test_load_config
- test_load_config_missing
- test_pipeline_run_with_rerank
- test_pipeline_demo_subset
- test_pipeline_no_rerank
- test_pipeline_cost_csv_columns
+ 18 other tests (retriever, evaluation, integration)
```

### Acceptance Criteria Verification

✅ **Pipeline runs with naive config**: `uv run python scripts/run_pipeline.py --config naive` works
✅ **Produces answers JSONL**: Each query writes answer to answers.jsonl
✅ **Produces retrieval metadata**: Writes retrieval_metadata.json
✅ **Produces cost CSV**: Writes costs.csv with per-query metrics
✅ **CSV includes required columns**: query_id, model, prompt_tokens, completion_tokens, rerank_count, latency_ms, estimated_cost_usd
✅ **Cost tracking built-in**: Not a separate module, integrated from the start
✅ **Intentionally expensive**: High top_k (100), always rerank, expensive model (gpt-4o)
✅ **No forbidden features**: No Jupyter, Streamlit, multiple backends, skip-LLM, or extra observability

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  RAGPipeline                            │
│                                                         │
│  1. Retrieve (FAISS) → top_k=100 docs                  │
│  2. Rerank (CrossEncoder) → top 20 docs                │
│  3. Generate (OpenAI GPT-4o) → answer                  │
│  4. Track costs → QueryCost record                     │
│                                                         │
│  Outputs:                                               │
│  - answers.jsonl                                        │
│  - retrieval_metadata.json                             │
│  - costs.csv                                           │
└─────────────────────────────────────────────────────────┘
```

### Next Steps (for future issues)
- Implement cost optimization strategies
- Add caching layers
- Implement adaptive retrieval
- Compare against this baseline

### Constraints Met (from AGENTS.md)
✅ No Jupyter notebooks
✅ No Streamlit or web apps
✅ No multiple retrieval backends
✅ No Skip-LLM optimization path
✅ No observability/tracing stack beyond built-in cost CSV
✅ Concise, clean code and tests

## Summary
The naive RAG pipeline is complete and ready for use as the expensive baseline in cost optimization experiments. All acceptance criteria are met, tests pass, and the pipeline produces structured outputs suitable for analysis.
