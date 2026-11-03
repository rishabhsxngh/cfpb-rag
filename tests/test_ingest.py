import duckdb


def test_drop_counts_reconcile(env):
    r = env["report"]
    assert r["rows_read"] - sum(r["dropped"].values()) == r["rows_after_cleaning"]
    assert r["rows_final"] == r["rows_after_cleaning"]


def test_every_rule_fired_on_messy_sample(env):
    assert all(v > 0 for v in env["report"]["dropped"].values()), env["report"]["dropped"]


def test_clean_table_invariants(env):
    con = duckdb.connect(str(env["db"]), read_only=True)
    n, n_ids, n_short, n_empty, n_other, min_date = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT complaint_id), SUM(n_words < 20), "
        "SUM(narrative = ''), SUM(product_group NOT IN ('Mortgage','Credit reporting')), "
        "MIN(date_received) FROM complaints").fetchone()
    assert n == n_ids
    assert n_short == 0 and n_empty == 0 and n_other == 0
    assert str(min_date) >= "2024-01-01"


def test_redaction_ratio_computed(env):
    con = duckdb.connect(str(env["db"]), read_only=True)
    assert con.execute("SELECT MAX(xxxx_ratio) FROM complaints").fetchone()[0] > 0
