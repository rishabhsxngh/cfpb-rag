"""Load the raw CFPB CSV into DuckDB with explicit quality checks.

Every row that is dropped is counted by reason, and the counts are written to
a JSON quality report next to the database so the cleaning is auditable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
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

# CFPB redacts personal details as runs of X (e.g. "XXXX", "XX/XX/2024").
_REDACTION_RE = re.compile(r"X{2,}")
_WS_RE = re.compile(r"\s+")


def _group_for(product: str, groups: list[str]) -> str | None:
    p = (product or "").lower()
    for g in groups:
        if g.lower() in p:
            return g
    return None


def build_database(
    csv_path: Path,
    db_path: Path,
    product_groups: list[str] | None = None,
    since: str = DEFAULT_SINCE,
    min_words: int = 20,
    max_rows: int | None = None,
    seed: int = 7,
    chunksize: int = 200_000,
) -> dict:
    """Filter the raw CSV to the project slice and write a `complaints` table."""
    groups = product_groups or DEFAULT_PRODUCT_GROUPS
    since_ts = pd.Timestamp(since)
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
    frames: list[pd.DataFrame] = []

    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c in COLMAP,
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
    )
    for chunk in reader:
        chunk = chunk.rename(columns=COLMAP)
        report["rows_read"] += len(chunk)

        chunk["product_group"] = chunk["product"].map(lambda p: _group_for(p, groups))
        n = len(chunk)
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
            frames.append(chunk)

    if not frames:
        raise ValueError(
            "No rows matched the slice. Check product groups, the --since date, and the CSV path."
        )
    df = pd.concat(frames, ignore_index=True)
    report["rows_after_cleaning"] = len(df)

    df["xxxx_ratio"] = df["narrative"].map(
        lambda s: len(_REDACTION_RE.findall(s)) / max(1, len(s.split()))
    )

    if max_rows and len(df) > max_rows:
        report["sampled_down_from"] = len(df)
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    df = df.sort_values("complaint_id").reset_index(drop=True)
    cols = [
        "complaint_id", "date_received", "product", "product_group", "sub_product",
        "issue", "sub_issue", "company", "state", "company_response",
        "narrative", "n_words", "xxxx_ratio",
    ]
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
