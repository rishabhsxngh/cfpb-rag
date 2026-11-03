from cfpb_rag.agent.pipeline import Pipeline
from cfpb_rag.agent.router import RuleRouter
from cfpb_rag.eval import metrics as M


class FakeLLM:
    """Returns whatever it is told to, to test the guardrails around the model."""
    name = "fake"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, max_tokens=700):
        return self.reply


def test_router_cues():
    r = RuleRouter()
    assert r.route("How many Mortgage complaints were filed in 2025?") == "sql"
    assert r.route("Which company has the most complaints?") == "sql"
    assert r.route("What did consumers say about escrow shortages?") == "retrieve"


def test_sql_route_answers_with_rows_and_query(pipeline):
    out = pipeline.ask("Which issue is most common for Mortgage complaints?")
    assert out["route"] == "sql" and not out["abstained"]
    assert out["rows"] and out["sql"] and out["spec"]["group_by"] == "issue"


def test_unparseable_aggregate_falls_back_to_retrieval(pipeline):
    out = pipeline.ask("How many people were upset?")
    assert out["route"] == "retrieve"


def test_zero_row_aggregate_abstains(pipeline):
    out = pipeline.ask("How many complaints mention foreclosure?")
    assert out["abstained"] and out["abstain_reason"] == "no_matching_rows"


def test_out_of_scope_question_abstains(pipeline):
    out = pipeline.ask("What is the capital of France?")
    assert out["abstained"] and out["citations"] == []


def test_grounded_answer_has_valid_citations(pipeline):
    out = pipeline.ask("What did consumers report about the portal being down?")
    assert not out["abstained"]
    retrieved = {s["complaint_id"] for s in out["sources"]}
    assert out["citations"] and set(out["citations"]) <= retrieved


def test_unsourced_answer_is_never_shown(env, retriever):
    p = Pipeline(retriever, env["db"], FakeLLM("Consumers were unhappy with their servicer."), RuleRouter())
    out = p.ask("What did consumers report about late fees?")
    assert out["abstained"] and out["abstain_reason"] == "no_valid_citation"


def test_fabricated_citation_is_stripped(env, retriever):
    question = "What did consumers report about late fees?"
    real = retriever.search(question, k=5, mode="hybrid")[0].complaint_id  # same query the pipeline runs
    p = Pipeline(retriever, env["db"], FakeLLM(f"A late fee was charged [{real}] [1234567]."), RuleRouter())
    out = p.ask(question)
    assert out["citations"] == [real] and out["invalid_citations"] == ["1234567"]
    assert "1234567" not in out["answer"]


def test_model_abstention_is_respected(env, retriever):
    p = Pipeline(retriever, env["db"], FakeLLM("INSUFFICIENT_EVIDENCE"), RuleRouter())
    assert p.ask("What did consumers report about late fees?")["abstain_reason"] == "model_abstained"


def test_metrics():
    assert M.hit_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert M.hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert M.recall_at_k(["a", "b"], ["a", "z"], 5) == 0.5
    assert M.reciprocal_rank(["x", "y", "g"], ["g"]) == 1 / 3
    assert M.citation_validity(["a"], ["b"]) == 0.5 and M.citation_validity([], []) is None
    s = M.abstention_scores([(True, True), (True, False), (False, True), (False, False)])
    assert s["abstain_precision"] == 0.5 and s["abstain_recall"] == 0.5 and s["false_abstention_rate"] == 0.5
    exp = [{"group_value": "A", "n": 5}, {"group_value": "B", "n": 3}]
    assert M.aggregate_match(exp, exp)["full"]
    assert not M.aggregate_match(exp, [{"group_value": "A", "n": 4}])["top1"]


def test_goldset_export_and_drafted_roundtrip(tmp_path, env):
    import json
    from cfpb_rag.eval import build_goldset

    exp = tmp_path / "to_draft.jsonl"
    build_goldset.main(["--db", str(env["db"]), "--export-complaints", "10", "--export-path", str(exp)])
    rows = [json.loads(line) for line in exp.read_text().splitlines()]
    assert len(rows) == 10 and {"id", "complaint", "issue"} <= set(rows[0])

    drafted = tmp_path / "drafted.jsonl"
    drafted.write_text("\n".join(
        json.dumps({"id": r["id"], "question": f"Who complained about {r['issue'].lower()} at length?"})
        for r in rows))
    out = tmp_path / "cand.jsonl"
    build_goldset.main(["--db", str(env["db"]), "--drafted", str(drafted), "--out", str(out),
                        "--n-aggregate", "3"])
    items = [json.loads(line) for line in out.read_text().splitlines()]
    retr = [i for i in items if i["type"] == "retrieval"]
    assert len(retr) == 10 and all(not i["verified"] for i in retr)
    assert {i["gold_ids"][0] for i in retr} == {r["id"] for r in rows}


def test_review_decisions_applies_accept_reject_and_records_reviewer(tmp_path, env):
    import json
    from cfpb_rag.eval import build_goldset, review

    exp = tmp_path / "to_draft.jsonl"
    build_goldset.main(["--db", str(env["db"]), "--export-complaints", "6", "--export-path", str(exp)])
    rows = [json.loads(line) for line in exp.read_text().splitlines()]
    drafted = tmp_path / "d.jsonl"
    drafted.write_text("\n".join(json.dumps({"id": r["id"], "question": f"Question number {i} about {r['issue']}?"})
                                 for i, r in enumerate(rows)))
    cand = tmp_path / "c.jsonl"
    build_goldset.main(["--db", str(env["db"]), "--drafted", str(drafted), "--out", str(cand), "--n-aggregate", "2"])
    dec = tmp_path / "dec.json"
    dec.write_text(json.dumps({"reviewer": "tester", "reject": [rows[0]["id"]], "edit": {rows[1]["id"]: "An edited question text?"}}))
    out = tmp_path / "gold.jsonl"
    review.main(["--candidates", str(cand), "--out", str(out), "--db", str(env["db"]), "--decisions", str(dec)])
    items = [json.loads(line) for line in out.read_text().splitlines()]
    retr = [i for i in items if i["type"] == "retrieval"]
    assert len(retr) == 5 and rows[0]["id"] not in {i["gold_ids"][0] for i in retr}
    assert all(i["reviewed_by"] == "tester" for i in retr)
    assert any(i["question"] == "An edited question text?" for i in retr)
