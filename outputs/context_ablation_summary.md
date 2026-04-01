# Context Count Ablation Summary

Comparing quality and cost across different context window sizes (3, 5, 10, 20 docs).
All variants use the same optimized retrieval settings (conditional reranking, gpt-5.4-mini, top_k=20).

## Results

| Context Count | Avg Cost ($) | Avg Latency (ms) | Avg Prompt Tokens | Faithfulness | Answer Relevance |
|---------------|--------------|------------------|-------------------|--------------|------------------|
| 3 | $0.0005 | 2630 | 1198 | 4.60 | 4.46 |
| 5 | $0.0007 | 2856 | 1856 | 4.64 | 4.54 |
| 10 | $0.0012 | 3443 | 3556 | 4.74 | 4.56 |
| 20 | $0.0020 | 3289 | 6529 | 4.56 | 4.62 |

## Sweet Spot Analysis

**Best faithfulness/cost ratio**: 3 docs
- Faithfulness: 4.60/5
- Cost: $0.0005 per query
- Ratio: 9350.4 (faithfulness per dollar)

## Quality vs Context Size

- **Highest faithfulness**: 10 docs (4.74/5)
- **Lowest faithfulness**: 20 docs (4.56/5)
- **Trend**: More documents in context tends to **hurt** faithfulness (4.60 → 4.56)

## Cost Scaling

- **Cost increase from 3→20 docs**: 304.9% ($0.0005 → $0.0020)
- Prompt tokens scale roughly linearly with context count

## Recommendation

**Recommendation**: Use **3 docs** for optimal quality/cost trade-off.
Smaller context windows are sufficient for this dataset and provide significant cost savings.
