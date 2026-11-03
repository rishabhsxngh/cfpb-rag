"""Command line entry point: cfpb-rag {sample,download,ingest,index,ask}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (DB_PATH, DEFAULT_EMBEDDER, DEFAULT_PRODUCT_GROUPS, DEFAULT_SINCE,
                      INDEX_DIR, RAW_DIR)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="cfpb-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Write a SYNTHETIC CSV for tests/demos (not real data)")
    s.add_argument("--out", type=Path, default=RAW_DIR / "complaints.csv")
    s.add_argument("--n", type=int, default=3000)

    d = sub.add_parser("download", help="Download real complaint data")
    d.add_argument("--source", choices=["hf", "cfpb"], default="hf",
                   help="hf = Parquet mirror on Hugging Face (default, ~1 GB); cfpb = official zip")

    i = sub.add_parser("ingest", help="Clean the CSV into DuckDB and write a quality report")
    i.add_argument("--csv", type=Path, default=None,
                   help="CSV, CFPB .zip, or folder of Parquet files (default: data/raw/hf, then complaints.csv.zip, then .csv)")
    i.add_argument("--products", default=",".join(DEFAULT_PRODUCT_GROUPS))
    i.add_argument("--since", default=DEFAULT_SINCE)
    i.add_argument("--min-words", type=int, default=20)
    i.add_argument("--max-rows", type=int, default=None)

    x = sub.add_parser("index", help="Build the embedding + BM25 index")
    x.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    x.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    x.add_argument("--no-meta-in-text", action="store_true")

    a = sub.add_parser("ask", help="Ask a question from the command line")
    a.add_argument("question")
    a.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--llm", default="auto")
    a.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "sample":
        from .ingest.sample import generate_sample
        print(f"Wrote SYNTHETIC data to {generate_sample(args.out, args.n)}")
    elif args.cmd == "download":
        if args.source == "hf":
            from .ingest.download import download_hf
            download_hf()
        else:
            from .ingest.download import download
            download()
    elif args.cmd == "ingest":
        from .ingest.clean import build_database
        src = args.csv or next(
            (p for p in (RAW_DIR / "hf", RAW_DIR / "complaints.csv.zip", RAW_DIR / "complaints.csv") if p.exists()), None)
        if src is None:
            raise SystemExit(f"No input file found. Put the CFPB zip at {RAW_DIR / 'complaints.csv.zip'} "
                             "(run `cfpb-rag download` for instructions) or pass --csv.")
        print(f"Reading {src} (this can take several minutes on the full dataset)...")
        rep = build_database(src, DB_PATH, [g.strip() for g in args.products.split(",")],
                             args.since, args.min_words, args.max_rows, progress=True)
        print(json.dumps({k: rep[k] for k in ("rows_read", "dropped", "rows_final", "profile")}, indent=2))
    elif args.cmd == "index":
        from .index.store import build_index
        print(json.dumps(build_index(DB_PATH, args.index_dir, args.embedder,
                                     meta_in_text=not args.no_meta_in_text), indent=2))
    elif args.cmd == "ask":
        from .agent.pipeline import build_pipeline
        out = build_pipeline(llm=args.llm, mode=args.mode, k=args.k).ask(args.question)
        if args.json:
            print(json.dumps(out, indent=2, default=str))
        else:
            print(out["answer"])
            print(f"\n[route={out['route']} abstained={out['abstained']} citations={out['citations']}]")


if __name__ == "__main__":
    main()
