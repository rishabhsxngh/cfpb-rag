import pytest
from fastapi.testclient import TestClient

from cfpb_rag.api import main as api


@pytest.fixture
def client(pipeline, monkeypatch):
    monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask_retrieval_with_filter(client):
    r = client.post("/ask", json={"question": "What did consumers report about the portal being down?",
                                  "product_group": "Mortgage", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "retrieve" and len(body["sources"]) <= 3
    assert all(s["meta"]["product_group"] == "Mortgage" for s in body["sources"])


def test_ask_sql(client):
    body = client.post("/ask", json={"question": "How many Mortgage complaints were filed in 2025?"}).json()
    assert body["route"] == "sql" and body["rows"][0]["n"] > 0


@pytest.mark.parametrize("payload", [
    {"question": "hi"},                                   # too short
    {"question": "x" * 501},                              # too long
    {"question": "valid question here", "k": 99},
    {"question": "valid question here", "mode": "magic"},
    {"question": "valid question here", "date_from": "01/02/2025"},
])
def test_validation_errors(client, payload):
    assert client.post("/ask", json=payload).status_code == 422
