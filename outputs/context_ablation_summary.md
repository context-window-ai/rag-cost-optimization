# Context Count Ablation Results

| Context Count | Avg Cost (USD) | Avg Latency (ms) | Avg Prompt Tokens | Avg Faithfulness | Avg Answer Relevance |
|---------------|----------------|------------------|-------------------|------------------|---------------------|
| 3 | $0.000492 | 2629.8 | 1197.7 | 4.74/5 | 4.50/5 |
| 5 | $0.000686 | 2856.1 | 1856.4 | 4.64/5 | 4.54/5 |
| 10 | $0.001174 | 3443.3 | 3555.6 | 4.74/5 | 4.56/5 |
| 20 | $0.001992 | 3289.5 | 6528.6 | 4.56/5 | 4.62/5 |

## Analysis

### Sweet Spot

**3 documents** provides the best faithfulness/cost ratio (4.74/5 faithfulness at $0.000492 per query).

### Quality vs. Context Count

- **Faithfulness is mixed** - neither consistently improves nor degrades with more context.
- **Answer relevance improves** as context count increases.

### Cost Scaling

- Cost scales from $0.000492 (3 docs) to $0.001992 (20 docs).
- Adding ~17 more documents increases cost by 4.0x.
