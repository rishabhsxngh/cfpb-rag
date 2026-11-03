import duckdb
import pytest

from cfpb_rag.agent import sql_tool
from cfpb_rag.agent.sql_tool import AggSpec, SpecError


@pytest.fixture
def con(env):
    return duckdb.connect(str(env["db"]), read_only=True)


def test_group_counts_match_direct_query(con):
    res = sql_tool.run(con, AggSpec("issue", {"product_group": "Mortgage"}, "desc", 10))
    direct = con.execute(
        "SELECT issue, COUNT(*) FROM complaints WHERE product_group='Mortgage' "
        "GROUP BY 1 ORDER BY 2 DESC, 1").fetchall()
    assert [(r["group_value"], r["n"]) for r in res["rows"]] == [(a, b) for a, b in direct][:10]


def test_total_count(con):
    res = sql_tool.run(con, AggSpec(None, {"year": 2025}))
    direct = con.execute("SELECT COUNT(*) FROM complaints WHERE year(date_received)=2025").fetchone()[0]
    assert res["rows"][0]["n"] == direct


@pytest.mark.parametrize("bad", [
    AggSpec(group_by="issue; DROP TABLE complaints"),
    AggSpec(filters={"issue; DROP TABLE complaints --": "x"}),
    AggSpec(filters={"year": "2025 OR 1=1"}),
    AggSpec(filters={"month": "2025-13"}),
    AggSpec(order="sideways"),
    AggSpec(limit=10_000),
    AggSpec(filters={"company": ""}),
])
def test_invalid_specs_rejected(bad):
    with pytest.raises(SpecError):
        sql_tool.compile_sql(bad)


def test_values_are_parameterized_not_interpolated(con):
    evil = "x' OR '1'='1"
    sql, params = sql_tool.compile_sql(AggSpec(filters={"company": evil}))
    assert evil not in sql and params[0] == evil
    assert sql_tool.run(con, AggSpec(filters={"company": evil}))["rows"][0]["n"] == 0
    assert con.execute("SELECT COUNT(*) FROM complaints").fetchone()[0] > 0


def test_rule_parser(con):
    known = sql_tool.known_values(con)
    s = sql_tool.parse_spec_rules("What are the top 3 states by number of Credit reporting complaints?", known)
    assert s.group_by == "state" and s.limit == 3 and s.filters == {"product_group": "Credit reporting"}
    s = sql_tool.parse_spec_rules("How many Mortgage complaints were filed in 2025?", known)
    assert s.group_by is None and s.filters == {"product_group": "Mortgage", "year": 2025}
    s = sql_tool.parse_spec_rules("Which company has the most complaints in 2024?", known)
    assert s.group_by == "company" and s.filters == {"year": 2024}
