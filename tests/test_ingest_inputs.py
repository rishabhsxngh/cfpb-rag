import zipfile

import duckdb
import pandas as pd
import pytest

from cfpb_rag.ingest.clean import build_database
from cfpb_rag.ingest.sample import generate_sample


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("inputs")
    return generate_sample(d / "s.csv", n=1200, seed=9)


def _rows(db):
    con = duckdb.connect(str(db), read_only=True)
    return con.execute("SELECT complaint_id FROM complaints ORDER BY 1").fetchall()


def test_zip_input_matches_csv_input(csv, tmp_path):
    z = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv, "complaints.csv")
    a = build_database(csv, tmp_path / "a.duckdb")
    b = build_database(z, tmp_path / "b.duckdb")
    assert a["rows_final"] == b["rows_final"]
    assert _rows(tmp_path / "a.duckdb") == _rows(tmp_path / "b.duckdb")


def test_missing_narrative_column_fails_with_clear_message(csv, tmp_path):
    bad = tmp_path / "no_narrative.csv"
    pd.read_csv(csv, dtype=str).drop(columns=["Consumer complaint narrative"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="no complaint narratives"):
        build_database(bad, tmp_path / "x.duckdb")


def test_max_rows_samples_uniformly_and_reconciles(csv, tmp_path):
    rep = build_database(csv, tmp_path / "s.duckdb", max_rows=100, chunksize=150)
    assert rep["rows_final"] == 100
    assert rep["sampled_down_from"] == rep["rows_after_cleaning"] > 100
    assert rep["rows_read"] - sum(rep["dropped"].values()) == rep["rows_after_cleaning"]
    # sample draws from both products, not just the first chunks
    con = duckdb.connect(str(tmp_path / "s.duckdb"), read_only=True)
    assert con.execute("SELECT COUNT(DISTINCT product_group) FROM complaints").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(DISTINCT year(date_received)) FROM complaints").fetchone()[0] >= 2


def test_max_rows_is_reproducible(csv, tmp_path):
    build_database(csv, tmp_path / "1.duckdb", max_rows=80, seed=5, chunksize=200)
    build_database(csv, tmp_path / "2.duckdb", max_rows=80, seed=5, chunksize=200)
    assert _rows(tmp_path / "1.duckdb") == _rows(tmp_path / "2.duckdb")


def test_parquet_folder_matches_csv_input(csv, tmp_path):
    from cfpb_rag.ingest.clean import COLMAP, PARQUET_COLMAP
    inv = {v: k for k, v in PARQUET_COLMAP.items()}
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df = df.rename(columns={k: inv[v] for k, v in COLMAP.items()})
    d = tmp_path / "hf" / "data"
    d.mkdir(parents=True)
    half = len(df) // 2
    df.iloc[:half].to_parquet(d / "complaints-00000.parquet")
    df.iloc[half:].to_parquet(d / "complaints-00001.parquet")
    a = build_database(csv, tmp_path / "a.duckdb")
    b = build_database(tmp_path / "hf", tmp_path / "b.duckdb")
    assert a["rows_final"] == b["rows_final"]
    assert _rows(tmp_path / "a.duckdb") == _rows(tmp_path / "b.duckdb")


def test_parquet_with_timezone_aware_dates(csv, tmp_path):
    from cfpb_rag.ingest.clean import COLMAP, PARQUET_COLMAP
    inv = {v: k for k, v in PARQUET_COLMAP.items()}
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df = df.rename(columns={k: inv[v] for k, v in COLMAP.items()})
    df["date_received"] = pd.to_datetime(df["date_received"]).dt.strftime("%Y-%m-%dT00:00:00+00:00")
    d = tmp_path / "hf"
    d.mkdir()
    df.to_parquet(d / "complaints-00000.parquet")
    a = build_database(csv, tmp_path / "a.duckdb")
    b = build_database(d, tmp_path / "b.duckdb")
    assert a["rows_final"] == b["rows_final"]
    assert a["profile"]["date_min"] == b["profile"]["date_min"]
