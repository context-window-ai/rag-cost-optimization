# Optimal Config Benchmark Results

Comparison of naive baseline vs optimized configuration combining all winning levers.

## Results

| Config | Avg Cost (USD) | Avg Latency (ms) | Avg Faithfulness | Avg Answer Relevance |
|--------|----------------|------------------|------------------|---------------------|
| naive | $0.003295 | 2061.6 | 4.74/5 | 4.58/5 |
| optimal | $0.000384 | 1698.8 | 4.78/5 | 4.60/5 |

## Deltas

- **Cost change**: -88.3%
- **Faithfulness change**: +0.8%

## Cost at Scale

### Cost per 1M queries

- **Naive**: $3,294.50
- **Optimal**: $384.10
- **Savings per 1M queries**: $2,910.40

### Annual savings at 1M queries/day

- **Daily savings**: $2,910.40
- **Annual savings**: $1,062,296.00

## Key Findings

- **88.3% cost reduction** achieved through optimization levers
- **Quality maintained** with faithfulness change of +0.8%
- **Latency improved** by 17.6% (2062ms → 1699ms)
- **At scale**: $1,062,296.00 annual savings at 1M queries/day
