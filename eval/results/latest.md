# Evaluation results

Run: 2026-11-03T19:06:46+00:00  
Index: `st:sentence-transformers/all-MiniLM-L6-v2`, 60,000 complaints, meta_in_text=True  
Gold set: {'retrieval': 55, 'aggregate': 24, 'unanswerable': 20}  


## Retrieval

| retrieval | n | hit@1 | recall@1 | hit@3 | recall@3 | hit@5 | recall@5 | hit@10 | recall@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 55 | 0.382 | 0.382 | 0.509 | 0.509 | 0.527 | 0.527 | 0.618 | 0.618 | 0.457 |
| bm25 | 55 | 0.527 | 0.527 | 0.636 | 0.636 | 0.655 | 0.655 | 0.709 | 0.709 | 0.584 |
| hybrid | 55 | 0.473 | 0.473 | 0.618 | 0.618 | 0.673 | 0.673 | 0.709 | 0.709 | 0.563 |
| hybrid+rerank | 55 | 0.564 | 0.564 | 0.600 | 0.600 | 0.673 | 0.673 | 0.800 | 0.800 | 0.610 |

## Routing

| router | accuracy | sql_recall | retrieve_recall | n |
|---|---|---|---|---|
| rules | 1.000 | 1.000 | 1.000 | 79 |

## SQL route (exact-match against SQL gold)

- n: 24
- routed_to_sql: 1.000
- top1_match: 1.000
- full_match: 1.000
