"""Interactive review of retrieval candidates. Accept, edit or reject each question.

Gold labels you did not look at are not gold labels. Review every retrieval item:
the question must be answerable from the shown complaint and must not leak its wording.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from ..config import DB_PATH


def main(argv=None):
    ap = argparse.ArgumentParser(description="Review gold-set candidates")
    ap.add_argument("--candidates", type=Path, default=Path("eval/candidates.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/goldset.jsonl"))
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--decisions", type=Path, default=None,
                    help='JSON {"reviewer": str, "reject": [ids], "edit": {id: question}}; every other '
                         "unverified retrieval item is accepted. Recorded as reviewed_by in the output.")
    args = ap.parse_args(argv)

    items = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    con = duckdb.connect(str(args.db), read_only=True)
    kept: list[dict] = []
    already_verified = [i for i in items if i["verified"]]  # snapshot before items are mutated
    todo = [i for i in items if not i["verified"]]
    if args.decisions:
        dec = json.loads(args.decisions.read_text())
        reject, edit = set(map(str, dec.get("reject", []))), {str(k): v for k, v in dec.get("edit", {}).items()}
        for it in todo:
            cid = it["gold_ids"][0]
            if cid in reject:
                continue
            if cid in edit:
                it["question"] = edit[cid]
            it["verified"] = True
            it["reviewed_by"] = dec.get("reviewer", "unspecified")
            kept.append(it)
        todo = []
    for n, it in enumerate(todo, start=1):
        cid = it["gold_ids"][0]
        row = con.execute(
            "SELECT product_group, issue, company, date_received, narrative FROM complaints "
            "WHERE complaint_id = ?", [cid]).fetchone()
        print(f"\n--- {n}/{len(todo)}  complaint {cid} ---")
        if row:
            print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}\n{row[4]}\n")
        print(f"QUESTION: {it['question']}")
        while True:
            c = input("[a]ccept  [e]dit  [r]eject  [q]uit > ").strip().lower()
            if c in {"a", "e", "r", "q"}:
                break
        if c == "q":
            break
        if c == "r":
            continue
        if c == "e":
            it["question"] = input("New question: ").strip() or it["question"]
        it["verified"] = True
        it["reviewed_by"] = "user"
        kept.append(it)

    final = already_verified + kept
    for idx, it in enumerate([i for i in final if i["type"] == "retrieval"], start=1):
        it["id"] = f"r{idx:03d}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in final:
            f.write(json.dumps(it) + "\n")
    print(f"\nWrote {len(final)} verified items to {args.out} "
          f"({len(kept)} retrieval items accepted this session)")


if __name__ == "__main__":
    main()
