"""Load raw CFPB complaints into DuckDB with explicit quality checks.

Every row that is dropped is counted by reason, and the counts are written to
a JSON quality report next to the database so the cleaning is auditable.

The input can be the plain CSV, the zip from CFPB's "All complaint data" download (streamed
directly, never unzipped to disk), or a folder / file of Parquet shards from a Hugging Face
mirror (snake_case column names). All three go through the same cleaning code.
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

# Column names used by the Parquet mirror on Hugging Face -> our internal names.
PARQUET_COLMAP = {
    "date_received": "date_received",
    "product": "product",
    "sub_product": "sub_product",
    "issue": "issue",
    "sub_issue": "sub_issue",
    "consumer_complaint_narrative": "narrative",
    "company": "company",
    "state": "state",
    "company_response_to_consumer": "company_response",
    "complaint_id": "complaint_id",
}
PARQUET_REQUIRED = ["date_received", "product", "issue", "consumer_complaint_narrative",
                    "company", "complaint_id"]

# CFPB redacts personal details as runs of X (e.g. "XXXX", "XX/XX/2024").
_REDACTION_RE = re.compile(r"X{2,}")
_WS_RE = re.compile(r"\s+")


def _parquet_files(path: Path) -> list[Path]:
    path = Path(path)
    return sorted(path.rglob("*.parquet")) if path.is_dir() else [path]


def _is_parquet(path: Path) -> bool:
    path = Path(path)
    return path.suffix == ".parquet" or (path.is_dir() and any(path.rglob("*.parquet")))


def check_columns(csv_path: Path) -> None:
    """Fail early, with a useful message, if the file cannot support the project."""
    if _is_parquet(csv_path):
        import pyarrow.parquet as pq
        header = list(pq.ParquetFile(_parquet_files(csv_path)[0]).schema_arrow.names)
        missing = [c for c in PARQUET_REQUIRED if c not in header]
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {missing}.")
        return
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        hint = ""
        if "Consumer complaint narrative" in missing:
            hint = ("\nThis file has no complaint narratives, and the project searches "
                    "narrative text. The filtered 'Export data' CSV on the CFPB site may omit "
                    "them; use the full 'All complaint data' zip instead.")
        raise ValueError(f"{csv_path} is missing required column(s): {missing}.{hint}")


def _iter_chunks(path: Path, chunksize: int):
    """Yield DataFrames of string columns with our internal column names."""
    if _is_parquet(path):
        import pyarrow.parquet as pq
        cols = list(PARQUET_COLMAP)
        for f in _parquet_files(path):
            pf = pq.ParquetFile(f)
            have = [c for c in cols if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=chunksize, columns=have):
                df = batch.to_pandas()
                for c in df.columns:
                    df[c] = df[c].astype("string").fillna("").astype(object)
                yield df.rename(columns=PARQUET_COLMAP)
        return
    reader = pd.read_csv(path, usecols=lambda c: c in COLMAP, dtype=str,
                         chunksize=chunksize, keep_default_na=False)
    for chunk in reader:
        yield chunk.rename(columns=COLMAP)


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

    for n_chunk, chunk in enumerate(_iter_chunks(csv_path, chunksize), start=1):
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
        # Some sources carry a timezone ("...+00:00"); normalize to naive dates.
        dates = pd.to_datetime(chunk["date_received"], errors="coerce", utc=True).dt.tz_localize(None)
        chunk = chunk.assign(date_received=dates)
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
