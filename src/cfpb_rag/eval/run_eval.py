"""Run the evaluation and write results tables.

Always runs (no API key needed):
  * retrieval metrics (hit@k, recall@k, MRR) for dense / bm25 / hybrid [+ rerank]
  * router accuracy and SQL-route accuracy against exact SQL gold

With --with-llm (needs ANTHROPIC_API_KEY, costs tokens, results are cached):
  * grounded answer quality via an LLM judge, citation validity
  * abstention precision/recall and false-abstention rate
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..agent.pipeline import Pipeline
from ..agent.router import LLMRouter, RuleRouter
from ..config import DB_PATH, INDEX_DIR
from ..llm import get_llm
from ..retrieve.retriever import MODES, Retriever
from . import metrics as M
from .judge import judge_answer


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def eval_retrieval(retr: Retriever, items: list[dict], modes: list[str], ks: list[int],
                   rerank: bool) -> pd.DataFrame:
    rows = []
    variants = [(m, False) for m in modes] + ([("hybrid", True)] if rerank else [])
    kmax = max(ks)
    for mode, rr in variants:
        per_q = []
        for it in items:
            hits = retr.search(it["question"], k=kmax, mode=mode, rerank=rr)
            per_q.append(([h.complaint_id for h in hits], it["gold_ids"]))
        row = {"retrieval": mode + ("+rerank" if rr else ""), "n": len(items)}
        for k in ks:
            row[f"hit@{k}"] = M.mean([M.hit_at_k(r, g, k) for r, g in per_q])
            row[f"recall@{k}"] = M.mean([M.recall_at_k(r, g, k) for r, g in per_q])
        row["MRR"] = M.mean([M.reciprocal_rank(r, g) for r, g in per_q])
        rows.append(row)
    return pd.DataFrame(rows)


def eval_routing(items: list[dict], routers: dict) -> pd.DataFrame:
    expect = {"retrieval": "retrieve", "aggregate": "sql"}
    scored = [it for it in items if it["type"] in expect]
    rows = []
    for name, router in routers.items():
        correct = [router.route(it["question"]) == expect[it["type"]] for it in scored]
        sql_ok = [c for c, it in zip(correct, scored) if it["type"] == "aggregate"]
        ret_ok = [c for c, it in zip(correct, scored) if it["type"] == "retrieval"]
        rows.append({"router": name, "accuracy": M.mean(map(float, correct)),
                     "sql_recall": M.mean(map(float, sql_ok)),
                     "retrieve_recall": M.mean(map(float, ret_ok)), "n": len(scored)})
    return pd.DataFrame(rows)


def eval_aggregate(pipe: Pipeline, items: list[dict]) -> dict:
    top1, full, routed = [], [], []
    for it in items:
        out = pipe.ask(it["question"])
        routed.append(out["route"] == "sql")
        res = M.aggregate_match(it["expected"], out.get("rows") or [])
        top1.append(res["top1"] and routed[-1])
        full.append(res["full"] and routed[-1])
    return {"n": len(items), "routed_to_sql": M.mean(map(float, routed)),
            "top1_match": M.mean(map(float, top1)), "full_match": M.mean(map(float, full))}


def eval_generation(pipe: Pipeline, retr_items: list[dict], unans_items: list[dict],
                    max_items: int) -> dict:
    docs = pipe.retriever.docs
    texts = dict(zip(docs["complaint_id"].astype(str), docs["narrative"]))
    abst, cite_valid, grounded, correct, judged = [], [], [], [], 0
    for it in retr_items[:max_items]:
        out = pipe.ask(it["question"])
        abst.append((False, out["abstained"]))
        if out["abstained"]:
            continue
        cite_valid.append(M.citation_validity(out["citations"], out.get("invalid_citations", [])))
        verdict = judge_answer(pipe.llm, it["question"], out["answer"],
                               {c: texts.get(c, "") for c in out["citations"]})
        if verdict:
            judged += 1
            grounded.append(float(verdict["grounded"]))
            correct.append(float(verdict["correct"]))
    for it in unans_items:
        out = pipe.ask(it["question"])
        abst.append((True, out["abstained"]))
    return {**M.abstention_scores(abst),
            "citation_validity": M.mean(cite_valid),
            "judge_grounded": M.mean(grounded), "judge_correct": M.mean(correct),
            "judged_answers": judged, "answered_questions": len(cite_valid)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval, routing and answers")
    ap.add_argument("--gold", type=Path, default=Path("eval/goldset.jsonl"))
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--rerank", action="store_true", help="Add hybrid+cross-encoder reranking")
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--max-llm-items", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument("--label", default="", help="Free-text label saved in the summary (e.g. 'MiniLM, meta_in_text')")
    ap.add_argument("--min-hit-at-5", type=float, default=None,
                    help="Exit non-zero if hybrid hit@5 falls below this (CI regression gate)")
    args = ap.parse_args(argv)

    gold = [json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()]
    unverified = [g for g in gold if not g.get("verified")]
    if unverified:
        print(f"Ignoring {len(unverified)} unverified items.", file=sys.stderr)
    gold = [g for g in gold if g.get("verified")]
    by = lambda t: [g for g in gold if g["type"] == t]
    ks = [int(k) for k in args.ks.split(",")]
    retr = Retriever(args.index_dir)

    summary: dict = {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "label": args.label, "index": retr.info,
                     "counts": {t: len(by(t)) for t in ("retrieval", "aggregate", "unanswerable")}}

    retrieval_df = eval_retrieval(retr, by("retrieval"), args.modes.split(","), ks, args.rerank)
    summary["retrieval"] = retrieval_df.to_dict("records")

    llm = get_llm("anthropic" if args.with_llm else "stub")
    rules = RuleRouter()
    routers = {"rules": rules}
    if args.with_llm:
        routers["llm"] = LLMRouter(llm)
    routing_df = eval_routing(gold, routers)
    summary["routing"] = routing_df.to_dict("records")

    pipe = Pipeline(retr, args.db, llm, rules)
    summary["aggregate"] = eval_aggregate(pipe, by("aggregate"))
    if args.with_llm:
        summary["generation"] = eval_generation(pipe, by("retrieval"), by("unanswerable"),
                                                args.max_llm_items)
        summary["llm"] = llm.name

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    retrieval_df.to_csv(args.out / "retrieval.csv", index=False)

    md = [f"# Evaluation results\n\nRun: {summary['when']}  \nIndex: `{retr.info['embedder']}`, "
          f"{retr.info['n_docs']:,} complaints, meta_in_text={retr.info['meta_in_text']}  \n"
          f"Gold set: {summary['counts']}  \n{('Label: ' + args.label) if args.label else ''}\n",
          "## Retrieval\n", _md_table(retrieval_df), "\n## Routing\n", _md_table(routing_df),
          "\n## SQL route (exact-match against SQL gold)\n",
          "\n".join(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
                    for k, v in summary["aggregate"].items())]
    if "generation" in summary:
        md += ["\n## Generation (LLM judge; hand-check a sample)\n",
               "\n".join(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
                         for k, v in summary["generation"].items())]
    (args.out / "latest.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    if args.min_hit_at_5 is not None:
        hyb = retrieval_df[retrieval_df["retrieval"] == "hybrid"]
        score = float(hyb["hit@5"].iloc[0]) if len(hyb) and "hit@5" in hyb else 0.0
        if score < args.min_hit_at_5:
            print(f"\nFAIL: hybrid hit@5 {score:.3f} < {args.min_hit_at_5}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
