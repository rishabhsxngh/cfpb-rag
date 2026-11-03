"""Paths and defaults. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CFPB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("CFPB_DB_PATH", DATA_DIR / "complaints.duckdb"))
INDEX_DIR = Path(os.environ.get("CFPB_INDEX_DIR", DATA_DIR / "index"))
CACHE_DIR = DATA_DIR / "cache"

CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
# Parquet mirror of the same public database, hosted on Hugging Face (not blocked like CFPB's server).
HF_REPO = "Mouwiya/cfpb-consumer-complaints"

# The slice of the CFPB database the assistant covers. Kept narrow on purpose so
# retrieval quality is measurable and the evaluation set stays tractable.
DEFAULT_PRODUCT_GROUPS = ["Mortgage", "Credit reporting"]
DEFAULT_SINCE = "2024-01-01"

# "auto" uses sentence-transformers if installed, otherwise a TF-IDF + SVD fallback.
DEFAULT_EMBEDDER = os.environ.get("CFPB_RAG_EMBEDDER", "auto")
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = os.environ.get("CFPB_RAG_MODEL", "claude-haiku-5-5")
