"""Constrained aggregation tool over the complaints table.

The model (or the rule parser) never writes SQL. It produces a small structured
`AggSpec`, which is validated against allowlists and compiled to a parameterized
query. That keeps the tool safe and makes its behaviour testable.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

GROUPABLE = {
    "product_group": "product_group",
    "issue": "issue",
    "sub_issue": "sub_issue",
    "company": "company",
    "state": "state",
    "company_response": "company_response",
    "month": "strftime(date_received, '%Y-%m')",
    "year": "year(date_received)",
}
TEXT_FILTERS = {"product_group", "issue", "sub_issue", "company", "state", "company_response"}
OTHER_FILTERS = {"year", "month", "company_contains", "narrative_contains"}
MAX_LIMIT = 50


class SpecError(ValueError):
    pass


@dataclass
class AggSpec:
    group_by: str | None = None
    filters: dict = field(default_factory=dict)
    order: str = "desc"
    limit: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AggSpec":
        return cls(
            group_by=d.get("group_by") or None,
            filters=dict(d.get("filters") or {}),
            order=d.get("order") or "desc",
            limit=int(d.get("limit") or 10),
        )


def validate(spec: AggSpec) -> AggSpec:
    if spec.group_by is not None and spec.group_by not in GROUPABLE:
        raise SpecError(f"group_by must be one of {sorted(GROUPABLE)}")
    if spec.order not in ("asc", "desc"):
        raise SpecError("order must be 'asc' or 'desc'")
    if not 1 <= spec.limit <= MAX_LIMIT:
        raise SpecError(f"limit must be between 1 and {MAX_LIMIT}")
    for k, v in spec.filters.items():
        if k not in TEXT_FILTERS | OTHER_FILTERS:
            raise SpecError(f"unknown filter: {k}")
        if k == "year":
            if not (isinstance(v, int) or str(v).isdigit()) or not 2000 <= int(v) <= 2100:
                raise SpecError("year must be an integer between 2000 and 2100")
        elif k == "month":
            if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(v)):
                raise SpecError("month must look like YYYY-MM")
        else:
            if not isinstance(v, str) or not v.strip() or len(v) > 120:
                raise SpecError(f"filter {k} must be a non-empty string up to 120 chars")
    return spec


def compile_sql(spec: AggSpec) -> tuple[str, list]:
    validate(spec)
    where, params = [], []
    for k, v in spec.filters.items():
        if k in TEXT_FILTERS:
            where.append(f"lower({k}) = lower(?)")
            params.append(v)
        elif k == "company_contains":
            where.append("company ILIKE ?")
            params.append(f"%{v}%")
        elif k == "narrative_contains":
            where.append("narrative ILIKE ?")
            params.append(f"%{v}%")
        elif k == "year":
            where.append("year(date_received) = ?")
            params.append(int(v))
        elif k == "month":
            where.append("strftime(date_received, '%Y-%m') = ?")
            params.append(str(v))
    wh = (" WHERE " + " AND ".join(where)) if where else ""
    if spec.group_by:
        expr = GROUPABLE[spec.group_by]  # allowlisted, never user text
        sql = (
            f"SELECT CAST({expr} AS VARCHAR) AS group_value, COUNT(*) AS n FROM complaints{wh} "
            f"GROUP BY 1 ORDER BY n {spec.order.upper()}, group_value LIMIT ?"
        )
        params.append(spec.limit)
    else:
        sql = f"SELECT 'all' AS group_value, COUNT(*) AS n FROM complaints{wh}"
    return sql, params


def run(con, spec: AggSpec) -> dict:
    sql, params = compile_sql(spec)
    cur = con.cursor() if hasattr(con, "cursor") else con
    rows = cur.execute(sql, params).fetchall()
    return {
        "sql": sql,
        "params": params,
        "rows": [{"group_value": str(g), "n": int(n)} for g, n in rows],
    }


def known_values(con) -> dict:
    cur = con.cursor() if hasattr(con, "cursor") else con
    col = lambda c: [r[0] for r in cur.execute(f"SELECT DISTINCT {c} FROM complaints WHERE {c} <> ''").fetchall()]
    years = [r[0] for r in cur.execute("SELECT DISTINCT year(date_received) FROM complaints").fetchall()]
    return {
        "product_group": col("product_group"),
        "company": col("company"),
        "state": col("state"),
        "issue": col("issue"),
        "year": sorted(int(y) for y in years),
    }


# ---------------------------------------------------------------- rule-based parser
STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
_DIM = r"(issue|compan(?:y|ies)|state|product|month|year)"
_GROUP_PATTERNS = [
    re.compile(rf"\b(?:by|per|each)\s+{_DIM}s?\b", re.I),
    re.compile(rf"\bwhich\s+{_DIM}\b", re.I),
    re.compile(rf"\b(?:top|most common|most frequent|most|least common|fewest)\s+(?:\d+\s+)?{_DIM}s?\b", re.I),
]
_DIM_MAP = {"issue": "issue", "company": "company", "companies": "company",
            "state": "state", "product": "product_group", "month": "month", "year": "year"}


def parse_spec_rules(question: str, known: dict) -> AggSpec:
    q = question.strip()
    ql = q.lower()
    group_by = None
    for pat in _GROUP_PATTERNS:
        m = pat.search(q)
        if m:
            group_by = _DIM_MAP[m.group(1).lower()]
            break

    filters: dict = {}
    for pg in known.get("product_group", []):
        if pg.lower() in ql:
            filters["product_group"] = pg
    for y in re.findall(r"\b(20\d{2})\b", q):
        if int(y) in known.get("year", []):
            filters["year"] = int(y)
    for name, code in STATE_NAMES.items():
        if re.search(rf"\b{name}\b", ql) and code in known.get("state", []):
            filters["state"] = code
    m = re.search(r"\b(?:in|from|for)\s+([A-Z]{2})\b", q)
    if m and m.group(1) in known.get("state", []):
        filters["state"] = m.group(1)
    for comp in sorted(known.get("company", []), key=len, reverse=True):
        if comp.lower() in ql:
            filters["company"] = comp
            break
    m = re.search(r"\b(?:mention(?:s|ing)?|contain(?:s|ing)?)\s+(?:the\s+(?:word|term)\s+)?[\"']?([a-z][a-z \-]{2,30}?)[\"']?(?:\?|$|\s+(?:in|for|during|by)\b)", ql)
    if m:
        filters["narrative_contains"] = m.group(1).strip()

    if group_by in filters:  # asking "which state ..." while also filtering that state makes no sense
        filters.pop(group_by)

    m = re.search(r"\btop\s+(\d+)", ql)
    limit = int(m.group(1)) if m else (5 if group_by else 10)
    order = "asc" if re.search(r"\b(least|fewest)\b", ql) else "desc"
    return validate(AggSpec(group_by=group_by, filters=filters, order=order, limit=min(limit, MAX_LIMIT)))
