"""FastAPI service: POST /ask returns the answer, citations, route and sources."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..agent.pipeline import Pipeline, build_pipeline
from ..retrieve.retriever import Filters

app = FastAPI(title="CFPB complaints RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    product_group: str | None = None
    company: str | None = None
    state: str | None = Field(default=None, min_length=2, max_length=2)
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    return build_pipeline(
        llm=os.environ.get("CFPB_RAG_LLM", "auto"),
        router=os.environ.get("CFPB_RAG_ROUTER", "rules"),
        spec_with_llm=os.environ.get("CFPB_RAG_SPEC_WITH_LLM", "0") == "1",
        min_dense_score=float(os.environ.get("CFPB_RAG_MIN_DENSE", "0")),
    )


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {"status": "ok", "complaints": len(p.retriever), "index": p.retriever.info,
            "llm": getattr(p.llm, "name", "unknown")}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        pipe = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(503, f"Index or database not found: {e}") from e
    filters = Filters(req.product_group, req.company, req.state.upper() if req.state else None,
                      None, req.date_from, req.date_to)
    return pipe.ask(req.question, filters=None if filters.is_empty() else filters,
                    k=req.k, mode=req.mode)
