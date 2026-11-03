# CFPB Complaints Assistant: an evaluated RAG + SQL-routing system

A question-answering system over public [CFPB consumer complaint](https://www.consumerfinance.gov/data-research/consumer-complaints/) narratives. The focus is **measurement**: every design choice (chunking, embeddings, hybrid retrieval, reranking, abstention, routing) is evaluated against a hand-verified question set, not just demoed.

> **Complaints are unverified consumer statements.** The assistant reports what consumers said, with citations. It does not make findings about any company.

## What it does

```
question ──► router ──┬─► SQL aggregate tool ──► exact counts/rankings from DuckDB
                      │     (validated spec → parameterized SQL; the model never writes SQL)
                      └─► hybrid retrieval (dense + BM25, RRF, metadata filters, optional reranker)
                            └─► grounded answer with [complaint-ID] citations
                                  └─► guardrails: abstain when evidence is missing,
                                      strip fabricated citations, never show unsourced answers
```

- **Why a router?** Retrieval over narratives is bad at counting. "Which issue is most common?" goes to SQL, "what did consumers say about escrow shortages?" goes to retrieval, and the router's accuracy is measured separately.
- **Why a constrained SQL tool?** The model fills in a small structured spec (group-by, filters, limit) that is validated against allowlists and compiled to parameterized SQL, so it is safe and testable.

## Quickstart

Requires Python 3.10+.

```bash
make install                 # pip install -e ".[st,app,dev]"
make data                    # download real CFPB data and clean it into DuckDB (needs internet)
make index                   # build the embedding + BM25 index
export ANTHROPIC_API_KEY=...  # optional; without it a deterministic test stub is used (see below)
cfpb-rag ask "Which issue is most common for Mortgage complaints?"
make api                     # FastAPI on :8080  (POST /ask, GET /health)
make app                     # Streamlit UI
```

No internet or key? `make sample` builds a **synthetic** dataset so you can try the pipeline end to end. Never report numbers from synthetic data as results.

### LLM backends

| Backend | When | Notes |
|---|---|---|
| Anthropic (default model `claude-haiku-5-5`, override with `CFPB_RAG_MODEL`) | `ANTHROPIC_API_KEY` set | Responses cached in `data/cache/` to control cost |
| Extractive stub | no key | A deterministic **test double, not a language model**. Used by tests/CI. The UI shows a warning when it is active. |

## Evaluation (the point of the project)

Build the question set, then **review every retrieval question by hand**:

```bash
make goldset     # Claude drafts retrieval questions from sampled complaints; SQL computes aggregate gold
make review      # accept / edit / reject each retrieval question → eval/goldset.jsonl
make eval        # retrieval + routing + SQL-route metrics (no API key needed)
make eval-llm    # adds grounded-answer quality, citation validity, abstention (uses the API)
```

Three question types:

| Type | Gold | Metrics |
|---|---|---|
| Retrieval | the source complaint's ID | hit@k, recall@k, MRR for dense / BM25 / hybrid / hybrid+rerank |
| Aggregate | exact SQL result | routing accuracy, top-1 and full-match of the SQL route |
| Unanswerable | should abstain | abstention precision / recall, false-abstention rate |

With `--with-llm`: citation validity (share of citations pointing at retrieved complaints) and an LLM-judge score for groundedness and correctness. **Hand-check ~20 judge verdicts and report agreement.**

### Results

> Fill this in from `eval/results/latest.md` after running on the real data. The numbers below are intentionally empty: results from synthetic data or an unreviewed gold set should not be published.

| Retrieval | hit@5 | recall@10 | MRR |
|---|---|---|---|
| dense | | | |
| BM25 | | | |
| hybrid | | | |
| hybrid + rerank | | | |

Ablations worth running and writing up (`cfpb-rag index --embedder st:<model>` / `--no-meta-in-text`): embedding model, metadata in the embedded text, k, reranking, hybrid vs single retriever. A flat or negative result with a clear explanation is a fine result.

### Known limitations (state these in any write-up)

- **Retrieval gold lists only the source complaint.** Other complaints may also answer the question, so recall is a lower bound.
- **Questions drafted from a complaint overlap it lexically**, which flatters BM25. The draft prompt asks for paraphrase, but review for leakage.
- **Aggregate questions are templated**; they test the SQL route, not natural-language variety.
- **The unanswerable set is slice-specific.** Questions about auto loans or student loans are unanswerable only while the index covers just the configured product groups.
- **The LLM judge can be wrong.** Report its agreement with your manual check.

## Data quality

`cfpb-rag ingest` writes `data/quality_report.json` with how many rows were dropped and why: outside the product slice, old or unparseable date, missing narrative, too short, duplicate complaint ID, duplicate narrative text. Narratives contain CFPB redaction markers (`XXXX`); the report tracks the redaction ratio, and prompts tell the model not to infer redacted details.

## Project layout

```
src/cfpb_rag/
  ingest/      download, clean (quality report), synthetic sample generator
  index/       embedders (sentence-transformers or TF-IDF/SVD), BM25, index builder
  retrieve/    dense / BM25 / hybrid retrieval, metadata filters, optional cross-encoder rerank
  agent/       router, constrained SQL tool, pipeline with abstention + citation guardrails
  eval/        gold-set builder, review CLI, metrics, LLM judge, evaluation runner
  api/         FastAPI service
app/           Streamlit UI
tests/         41 tests: ingestion invariants, SQL safety, retrieval, guardrails, API
```

## Deployment

```bash
docker build -t cfpb-rag .
docker run -p 8080:8080 -v "$(pwd)/data:/data" -e ANTHROPIC_API_KEY cfpb-rag
```

The image ships code only. Build the database and index first and mount `data/`. The service works on Cloud Run with the data baked into a derived image or loaded from a bucket at startup.

Environment variables: `CFPB_RAG_MODEL`, `CFPB_RAG_LLM` (`auto|anthropic|stub`), `CFPB_RAG_ROUTER` (`rules|llm`), `CFPB_RAG_SPEC_WITH_LLM` (`1` to let the model propose the SQL spec), `CFPB_RAG_MIN_DENSE` (similarity floor for abstaining), `CFPB_DATA_DIR`.

## Testing

```bash
make test    # no network or API key required; uses synthetic data
```

CI runs the tests plus an end-to-end smoke run of the pipeline and eval harness, and gates on the exact-match SQL route and routing accuracy. Retrieval quality is deliberately **not** gated on synthetic data.
