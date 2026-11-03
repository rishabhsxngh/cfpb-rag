"""Build the evaluation set.

Three question types:
  retrieval     A question answerable by one specific complaint (gold = its ID). Drafted by an
                LLM, so they are written as candidates and MUST be reviewed (see review.py).
  aggregate     Counts / rankings. Gold answers are computed exactly in SQL; the question
                wording is templated, which is a limitation worth stating in the write-up.
  unanswerable  Questions the system should refuse (out of scope, legal conclusions, PII ...).

Retrieval gold lists only the source complaint, so recall is a lower bound: other complaints
may also answer the question. Questions are drafted to be specific to limit this.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import duckdb

from ..agent import sql_tool
from ..config import DB_PATH
from ..llm import extract_json, get_llm

DRAFT_SYSTEM = (
    "You write evaluation questions for a retrieval system over consumer complaints. Given ONE "
    "complaint, write a single question that this complaint answers.\n"
    "Requirements: be specific enough (company, issue, a distinctive event or detail) that few "
    "other complaints would match; PARAPHRASE instead of copying phrases from the complaint; "
    "do not include the complaint ID or any redaction markers; do not ask for personal details.\n"
    'Reply with JSON only: {"question": "<the question>"}'
)

UNANSWERABLE_FILE = Path(__file__).with_name("unanswerable.txt")


def _retrieval_candidates(con, n: int, seed: int, llm, allow_template: bool) -> list[dict]:
    rows = con.execute(
        "SELECT complaint_id, product_group, issue, company, narrative FROM complaints "
        "WHERE n_words BETWEEN 40 AND 250 AND xxxx_ratio < 0.08"
    ).fetchall()
    random.Random(seed).shuffle(rows)
    out = []
    for cid, pg, issue, company, narrative in rows:
        if len(out) >= n:
            break
        question, source = None, "llm_drafted"
        user = (f"TASK: draft_question\nProduct: {pg}\nIssue: {issue}\nCompany: {company}\n"
                f"Complaint: {narrative[:1500]}")
        try:
            obj = extract_json(llm.complete(DRAFT_SYSTEM, user, max_tokens=150))
        except Exception:
            obj = None
        if obj and isinstance(obj.get("question"), str) and len(obj["question"]) > 15:
            question = obj["question"].strip()
        elif allow_template:  # smoke tests only: low-quality templated question
            question, source = f"What did a {pg.lower()} consumer report about {issue.lower()} involving {company}?", "template"
        if question:
            out.append({"id": f"r{len(out) + 1:03d}", "type": "retrieval", "question": question,
                        "gold_ids": [str(cid)], "verified": False, "source": source})
    return out


def _aggregate_items(con, n: int, seed: int) -> list[dict]:
    known = sql_tool.known_values(con)
    cands: list[tuple[str, sql_tool.AggSpec]] = []
    for pg in known["product_group"]:
        cands.append((f"Which issue is most common for {pg} complaints?",
                      sql_tool.AggSpec("issue", {"product_group": pg}, "desc", 5)))
        cands.append((f"Which company has the most {pg} complaints?",
                      sql_tool.AggSpec("company", {"product_group": pg}, "desc", 5)))
        cands.append((f"What are the top 3 states by number of {pg} complaints?",
                      sql_tool.AggSpec("state", {"product_group": pg}, "desc", 3)))
        for y in known["year"]:
            cands.append((f"How many {pg} complaints were filed in {y}?",
                          sql_tool.AggSpec(None, {"product_group": pg, "year": y})))
            cands.append((f"Which issue is most common for {pg} complaints in {y}?",
                          sql_tool.AggSpec("issue", {"product_group": pg, "year": y}, "desc", 5)))
    for y in known["year"]:
        cands.append((f"Which company has the most complaints in {y}?",
                      sql_tool.AggSpec("company", {"year": y}, "desc", 5)))
        cands.append((f"How many complaints were filed in {y}?", sql_tool.AggSpec(None, {"year": y})))
    random.Random(seed).shuffle(cands)
    out = []
    for q, spec in cands:
        res = sql_tool.run(con, spec)
        if not res["rows"] or res["rows"][0]["n"] == 0:
            continue
        # skip ties at the top: the "most common" answer must be unambiguous
        if spec.group_by and len(res["rows"]) > 1 and res["rows"][0]["n"] == res["rows"][1]["n"]:
            continue
        out.append({"id": f"a{len(out) + 1:03d}", "type": "aggregate", "question": q,
                    "spec": spec.to_dict(), "expected": res["rows"], "verified": True,
                    "source": "sql_template"})
        if len(out) >= n:
            break
    return out


def _unanswerable_items() -> list[dict]:
    qs = [ln.strip() for ln in UNANSWERABLE_FILE.read_text().splitlines() if ln.strip()]
    return [{"id": f"u{i + 1:03d}", "type": "unanswerable", "question": q, "verified": True,
             "source": "handwritten"} for i, q in enumerate(qs)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build gold-set candidates")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--out", type=Path, default=Path("eval/candidates.jsonl"))
    ap.add_argument("--n-retrieval", type=int, default=80)
    ap.add_argument("--n-aggregate", type=int, default=30)
    ap.add_argument("--llm", default="auto", help="auto | anthropic | stub")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--allow-template", action="store_true",
                    help="Fallback templated questions when no LLM is available (smoke tests only)")
    ap.add_argument("--auto-verify", action="store_true",
                    help="Mark retrieval candidates verified without review (smoke tests / CI only)")
    args = ap.parse_args(argv)

    con = duckdb.connect(str(args.db), read_only=True)
    llm = get_llm(args.llm)
    items = _retrieval_candidates(con, args.n_retrieval, args.seed, llm, args.allow_template)
    items += _aggregate_items(con, args.n_aggregate, args.seed)
    items += _unanswerable_items()
    if args.auto_verify:
        for it in items:
            it["verified"] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    by_type = {t: sum(i["type"] == t for i in items) for t in ("retrieval", "aggregate", "unanswerable")}
    pending = sum(not i["verified"] for i in items)
    print(f"Wrote {len(items)} items to {args.out}: {by_type}; {pending} need review "
          f"(run `python -m cfpb_rag.eval.review`)")


if __name__ == "__main__":
    main()
