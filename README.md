# RAG Cost Optimization

Retrieval baseline with BEIR evaluation for comparing RAG outcomes against cost.

## Overview

This project implements a dense retrieval baseline using FAISS and evaluates it against the BEIR benchmark (specifically the SciFact dataset). The goal is to provide a foundation for comparing RAG system performance against operational costs.

## Features

- **Dense Retrieval**: FAISS-based retriever using sentence-transformers embeddings
- **BEIR Evaluation**: Standard BEIR metrics (nDCG@10, Recall@10, MAP, etc.)
- **Cost Tracking**: Built-in timing and cost instrumentation for downstream analysis
- **Modular Design**: Clean separation between retrieval, evaluation, and data loading

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-cost-optimization

# Install dependencies with uv
uv sync

# Or install in development mode
uv pip install -e .
```

## Quick Start

### Run Baseline Evaluation

```bash
uv run python scripts/eval_retrieval.py --config baseline
```

This will:
1. Download the SciFact dataset from BEIR
2. Build a FAISS index using all-MiniLM-L6-v2 embeddings
3. Retrieve top-100 documents for each query
4. Compute BEIR metrics
5. Save results to `outputs/baseline/`

### Output Files

- `metrics.json`: All evaluation metrics in JSON format
- `metrics.csv`: Metrics in CSV format for easy analysis
- `results.json`: Raw retrieval results (query -> ranked document list)
- `config.json`: Copy of the configuration used

### Example Metrics

```
nDCG@10:   0.6451
Recall@10: 0.7833
MAP:       0.6032
```

## Project Structure

```
rag-cost-optimization/
├── configs/
│   └── baseline.yaml          # Baseline configuration
├── scripts/
│   └── eval_retrieval.py      # Main evaluation script
├── src/
│   └── rag_retrieval/
│       ├── __init__.py
│       ├── retriever.py       # FAISS-based dense retriever
│       ├── evaluation.py      # BEIR metrics implementation
│       ├── data.py           # BEIR dataset loading
│       └── pipeline.py       # Full RAG pipeline (optional)
├── tests/
│   ├── test_retriever.py
│   ├── test_evaluation.py
│   ├── test_data.py
│   └── test_integration.py
└── outputs/
    └── baseline/              # Evaluation outputs
```

## Configuration

Edit `configs/baseline.yaml` to customize:

```yaml
dataset: scifact
model_name: sentence-transformers/all-MiniLM-L6-v2
top_k: 100
batch_size: 32
output_dir: outputs/baseline
data_dir: datasets
```

## Testing

Run all tests:

```bash
uv run pytest tests/ -v
```

Run with coverage:

```bash
uv run pytest tests/ --cov=rag_retrieval
```

## API Usage

```python
from rag_retrieval import DenseRetriever, evaluate_retrieval
from rag_retrieval.data import download_and_load_dataset

# Load dataset
corpus, queries, qrels = download_and_load_dataset("scifact", "datasets")

# Initialize retriever
retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build index
retriever.build_index(corpus)

# Retrieve
results = retriever.retrieve(queries, top_k=100)

# Evaluate
metrics = evaluate_retrieval(results, qrels, k_values=[1, 3, 5, 10, 100])
print(f"nDCG@10: {metrics['ndcg@10']:.4f}")
```

## Metrics Explained

- **nDCG@k**: Normalized Discounted Cumulative Gain at k - measures ranking quality
- **Recall@k**: Proportion of relevant documents retrieved in top k
- **Precision@k**: Proportion of retrieved documents that are relevant
- **MAP**: Mean Average Precision - average of precision scores at each relevant document

## Constraints

As per AGENTS.md, this project does NOT include:
- Jupyter notebooks
- Streamlit or other web apps
- Multiple retrieval backends
- Skip-LLM optimization path
- Observability/tracing stack beyond built-in cost CSV

## Development

### Adding New Datasets

1. Add dataset name to BEIR supported datasets
2. Create corresponding config file in `configs/`
3. Run evaluation with new config

### Adding New Metrics

1. Implement metric function in `src/rag_retrieval/evaluation.py`
2. Add to `evaluate_retrieval()` function
3. Add tests in `tests/test_evaluation.py`

## License

[Add license information]

## References

- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
