"""Load the raw CFPB CSV (or the .zip of it) into DuckDB with explicit quality checks.

Every row that is dropped is counted by reason, and the counts are written to
a JSON quality report next to the database so the cleaning is auditable.

The input can be the plain CSV or the zip from CFPB's "All complaint data"
download. A zip is streamed directly, so it never has to be unzipped to disk.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..config import DEFAULT_PRODUCT_GROUPS, DEFAULT_SINCE

COLMAP = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Company response to consumer": "company_response",
    "Complaint ID": "complaint_id",
}
REQUIRED = ["Date received", "Product", "Issue", "Consumer complaint narrative",
            "Company", "Complaint ID"]

# CFPB redacts personal details as runs of X (e.g. "XXXX", "XX/XX/2024").
_REDACTION_RE = re.compile(r"X{2,}")
_WS_RE = re.compile(r"\s+")


def check_columns(csv_path: Path) -> None:
    """Fail early, with a useful message, if the file cannot support the project."""
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        hint = ""
        if "Consumer complaint narrative" in missing:
            hint = ("\nThis file has no complaint narratives, and the project searches "
                    "narrative text. The filtered 'Export data' CSV on the CFPB site may omit "
                    "them; use the full 'All complaint data' zip instead.")
        raise ValueError(f"{csv_path} is missing required column(s): {missing}.{hint}")


def build_database(
    csv_path: Path,
    db_path: Path,
    product_groups: list[str] | None = None,
    since: str = DEFAULT_SINCE,
    min_words: int = 20,
    max_rows: int | None = None,
    seed: int = 7,
    chunksize: int = 200_000,
    progress: bool = False,
) -> dict:
    """Filter the raw file to the project slice and write a `complaints` table.

    If max_rows is set, a uniform random sample of that size is kept. Sampling happens while
    streaming (random keys, keep the smallest), so memory stays bounded on the full dataset.
    """
    csv_path = Path(csv_path)
    check_columns(csv_path)
    groups = product_groups or DEFAULT_PRODUCT_GROUPS
    since_ts = pd.Timestamp(since)
    rng = np.random.default_rng(seed)
    report: dict = {
        "slice": {"product_groups": groups, "since": since, "min_words": min_words},
        "rows_read": 0,
        "dropped": {
            "outside_product_slice": 0,
            "unparseable_or_old_date": 0,
            "no_narrative": 0,
            "narrative_too_short": 0,
            "duplicate_complaint_id": 0,
            "duplicate_narrative_text": 0,
        },
    }
    seen_ids: set[str] = set()
    seen_text: set[int] = set()
    pool: list[pd.DataFrame] = []
    pooled = 0
    kept_total = 0

    def shrink(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        df = pd.concat(frames, ignore_index=True)
        return [df.nsmallest(max_rows, "_k")]

    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c in COLMAP,
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
    )
    for n_chunk, chunk in enumerate(reader, start=1):
        chunk = chunk.rename(columns=COLMAP)
        report["rows_read"] += len(chunk)

        prod_lower = chunk["product"].str.lower()
        group = pd.Series(None, index=chunk.index, dtype=object)
        for g in groups:  # first matching group wins
            hit = group.isna() & prod_lower.str.contains(g.lower(), regex=False)
            group[hit] = g
        n = len(chunk)
        chunk = chunk.assign(product_group=group)
        chunk = chunk[chunk["product_group"].notna()]
        report["dropped"]["outside_product_slice"] += n - len(chunk)

        n = len(chunk)
        chunk = chunk.assign(date_received=pd.to_datetime(chunk["date_received"], errors="coerce"))
        chunk = chunk[chunk["date_received"] >= since_ts]
        report["dropped"]["unparseable_or_old_date"] += n - len(chunk)

        n = len(chunk)
        chunk = chunk.assign(narrative=chunk["narrative"].map(lambda s: _WS_RE.sub(" ", s).strip()))
        chunk = chunk[chunk["narrative"] != ""]
        report["dropped"]["no_narrative"] += n - len(chunk)

        n = len(chunk)
        chunk = chunk.assign(n_words=chunk["narrative"].str.split().str.len())
        chunk = chunk[chunk["n_words"] >= min_words]
        report["dropped"]["narrative_too_short"] += n - len(chunk)

        n = len(chunk)
        chunk = chunk[~chunk["complaint_id"].duplicated()]
        chunk = chunk[~chunk["complaint_id"].isin(seen_ids)]
        report["dropped"]["duplicate_complaint_id"] += n - len(chunk)
        seen_ids.update(chunk["complaint_id"])

        n = len(chunk)
        hashes = chunk["narrative"].str.lower().map(hash)
        keep = ~hashes.duplicated() & ~hashes.isin(seen_text)
        chunk = chunk[keep]
        report["dropped"]["duplicate_narrative_text"] += n - len(chunk)
        seen_text.update(hashes[keep])

        if len(chunk):
            kept_total += len(chunk)
            if max_rows:
                chunk = chunk.assign(_k=rng.random(len(chunk)))
            pool.append(chunk)
            pooled += len(chunk)
            if max_rows and pooled > 2 * max_rows:
                pool = shrink(pool)
                pooled = len(pool[0])
        if progress and n_chunk % 5 == 0:
            print(f"  read {report['rows_read']:,} rows, kept {kept_total:,}", flush=True)

    if not pool:
        raise ValueError(
            "No rows matched the slice. Check product groups, the --since date, and the file."
        )
    df = shrink(pool)[0] if max_rows else pd.concat(pool, ignore_index=True)
    report["rows_after_cleaning"] = kept_total
    if max_rows and kept_total > max_rows:
        report["sampled_down_from"] = kept_total
    df = df.drop(columns=["_k"], errors="ignore")

    df["xxxx_ratio"] = df["narrative"].map(
        lambda s: len(_REDACTION_RE.findall(s)) / max(1, len(s.split()))
    )
    df = df.sort_values("complaint_id").reset_index(drop=True)
    cols = [
        "complaint_id", "date_received", "product", "product_group", "sub_product",
        "issue", "sub_issue", "company", "state", "company_response",
        "narrative", "n_words", "xxxx_ratio",
    ]
    for c in cols:  # optional columns can be absent from some exports
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    report["rows_final"] = len(df)
    report["profile"] = {
        "date_min": str(df["date_received"].min().date()),
        "date_max": str(df["date_received"].max().date()),
        "by_product_group": df["product_group"].value_counts().to_dict(),
        "distinct_companies": int(df["company"].nunique()),
        "distinct_issues": int(df["issue"].nunique()),
        "empty_sub_issue_rate": round(float((df["sub_issue"] == "").mean()), 4),
        "empty_state_rate": round(float((df["state"] == "").mean()), 4),
        "narrative_words_p10_p50_p90": [
            int(x) for x in df["n_words"].quantile([0.1, 0.5, 0.9]).tolist()
        ],
        "mean_redaction_ratio": round(float(df["xxxx_ratio"].mean()), 4),
    }

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DROP TABLE IF EXISTS complaints")
        con.register("df", df)
        con.execute(
            """
            CREATE TABLE complaints AS
            SELECT complaint_id, CAST(date_received AS DATE) AS date_received, product,
                   product_group, sub_product, issue, sub_issue, company, state,
                   company_response, narrative, CAST(n_words AS INTEGER) AS n_words,
                   CAST(xxxx_ratio AS DOUBLE) AS xxxx_ratio
            FROM df
            """
        )
    finally:
        con.close()

    (db_path.parent / "quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
