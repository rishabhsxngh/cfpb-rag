import pytest

from cfpb_rag.agent.pipeline import Pipeline
from cfpb_rag.agent.router import RuleRouter
from cfpb_rag.index.store import build_index
from cfpb_rag.ingest.clean import build_database
from cfpb_rag.ingest.sample import generate_sample
from cfpb_rag.llm import ExtractiveStubLLM
from cfpb_rag.retrieve.retriever import Retriever


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfpb")
    csv = d / "sample.csv"
    generate_sample(csv, n=1500, seed=3)
    db = d / "c.duckdb"
    report = build_database(csv, db, since="2024-01-01")
    idx = d / "index"
    build_index(db, idx, embedder_spec="tfidf-lsa")
    return {"csv": csv, "db": db, "index": idx, "report": report}


@pytest.fixture(scope="session")
def retriever(env):
    return Retriever(env["index"])


@pytest.fixture(scope="session")
def pipeline(env, retriever):
    return Pipeline(retriever, env["db"], ExtractiveStubLLM(), RuleRouter())
