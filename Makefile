.PHONY: install data index goldset review eval eval-llm test api app sample

install:
	pip install -e ".[st,app,dev]"

# Real data. CFPB may refuse scripted downloads (HTTP 403): if `cfpb-rag download` says so,
# download "All complaint data" in a browser and save it as data/raw/complaints.csv.zip first.
# The zip is read directly (no unzipping), and --max-rows keeps a 60k random sample.
data:
	cfpb-rag download
	cfpb-rag ingest --products "Mortgage,Credit reporting" --since 2024-01-01 --max-rows 60000

index:
	cfpb-rag index

# Draft candidate questions with Claude, then review every one by hand.
goldset:
	python -m cfpb_rag.eval.build_goldset --llm anthropic --n-retrieval 80 --n-aggregate 30

review:
	python -m cfpb_rag.eval.review

eval:
	python -m cfpb_rag.eval.run_eval --rerank

eval-llm:
	python -m cfpb_rag.eval.run_eval --rerank --with-llm

test:
	pytest -q

api:
	uvicorn cfpb_rag.api.main:app --reload --port 8080

app:
	streamlit run app/streamlit_app.py

# SYNTHETIC data, for trying the pipeline without downloading anything.
sample:
	cfpb-rag sample --n 3000
	cfpb-rag ingest
	cfpb-rag index --embedder tfidf-lsa
