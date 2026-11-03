"""End-to-end question answering: route -> (retrieve + grounded answer | SQL aggregate)."""
from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb

from ..config import DB_PATH, INDEX_DIR
from ..llm import LLM, extract_json, get_llm
from ..retrieve.retriever import Filters, Hit, Retriever
from . import sql_tool
from .router import LLMRouter, RuleRouter

CITE_RE = re.compile(r"\[(\d{4,})\]")
MAX_QUESTION_CHARS = 500
ABSTAIN_TEXT = "I can't answer that from the indexed complaints."

SYSTEM_PROMPT = (
    "You answer questions about consumer complaints submitted to the CFPB, using ONLY the "
    "numbered complaint excerpts provided.\n"
    "Rules:\n"
    "1. Support every claim with the complaint ID in square brackets, like [7000123].\n"
    "2. If the excerpts do not contain the answer, reply with exactly: INSUFFICIENT_EVIDENCE\n"
    "3. Complaints are unverified consumer statements. Say what consumers reported; never "
    "state allegations as established facts about a company, and never give legal conclusions.\n"
    "4. Do not guess, speculate, or use outside knowledge. Do not repeat redaction markers "
    "or try to infer redacted details.\n"
    "5. Be concise: at most 4 sentences."
)

SPEC_SYSTEM = (
    "Convert a question about CFPB consumer complaints into a JSON aggregation spec. Reply with "
    "JSON only.\n"
    'Schema: {"group_by": null|"product_group"|"issue"|"company"|"state"|"month"|"year", '
    '"filters": {"product_group": str, "issue": str, "company": str, "state": "2-letter code", '
    '"year": int, "month": "YYYY-MM", "narrative_contains": str}, "order": "desc"|"asc", "limit": int}\n'
    "Include only filters the question actually states. group_by is null for a plain count."
)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    lines = ["TASK: answer", f"QUESTION: {question}", "", "EXCERPTS:"]
    for h in hits:
        m = h.meta
        lines.append(
            f"[{h.complaint_id}] ({m['product_group']} | {m['issue']} | {m['company']} | {m['date']}) "
            f"{h.text[:1200]}"
        )
    return "\n".join(lines)


def _scope(spec: sql_tool.AggSpec) -> str:
    bits = [f"{k.replace('_', ' ')}={v}" for k, v in spec.filters.items()]
    return f" ({', '.join(bits)})" if bits else ""


def format_sql_answer(spec: sql_tool.AggSpec, rows: list[dict]) -> str:
    if spec.group_by:
        head = f"Complaints by {spec.group_by.replace('_', ' ')}{_scope(spec)}:"
        body = "\n".join(f"{i}. {r['group_value']}: {r['n']:,}" for i, r in enumerate(rows, 1))
        return f"{head}\n{body}"
    return f"Total complaints{_scope(spec)}: {rows[0]['n']:,}"


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        db_path: Path = DB_PATH,
        llm: LLM | None = None,
        router=None,
        spec_with_llm: bool = False,
        k: int = 5,
        mode: str = "hybrid",
        min_dense_score: float = 0.0,
    ):
        self.retriever = retriever
        self.llm = llm or get_llm("auto")
        self.router = router or RuleRouter()
        self.spec_with_llm = spec_with_llm
        self.k, self.mode, self.min_dense_score = k, mode, min_dense_score
        self.con = duckdb.connect(str(db_path), read_only=True)
        self.known = sql_tool.known_values(self.con)

    # ----------------------------------------------------------------- public
    def ask(self, question: str, filters: Filters | None = None, k: int | None = None,
            mode: str | None = None) -> dict:
        t0 = time.perf_counter()
        q = " ".join(question.split())[:MAX_QUESTION_CHARS]
        route = self.router.route(q)
        out = None
        if route == "sql":
            out = self._ask_sql(q)
            if out is None:  # could not turn it into a safe, meaningful aggregate
                route = "retrieve"
        if out is None:
            out = self._ask_retrieve(q, filters, k or self.k, mode or self.mode)
        out.update(question=q, route=route, latency_ms=round((time.perf_counter() - t0) * 1000))
        return out

    # ----------------------------------------------------------------- sql path
    def _parse_spec(self, q: str) -> sql_tool.AggSpec | None:
        if self.spec_with_llm:
            try:
                obj = extract_json(self.llm.complete(SPEC_SYSTEM, f"TASK: spec\nQUESTION: {q}", 300))
                if obj:
                    return sql_tool.validate(sql_tool.AggSpec.from_dict(obj))
            except Exception:
                pass  # fall through to the rule parser
        try:
            return sql_tool.parse_spec_rules(q, self.known)
        except sql_tool.SpecError:
            return None

    def _ask_sql(self, q: str) -> dict | None:
        spec = self._parse_spec(q)
        if spec is None or (spec.group_by is None and not spec.filters):
            return None
        result = sql_tool.run(self.con, spec)
        rows = result["rows"]
        total = sum(r["n"] for r in rows)
        base = {"sql": result["sql"], "sql_params": result["params"], "spec": spec.to_dict(),
                "rows": rows, "citations": [], "sources": []}
        if not rows or total == 0:
            return {**base, "answer": "No complaints in the indexed slice match those conditions.",
                    "abstained": True, "abstain_reason": "no_matching_rows"}
        return {**base, "answer": format_sql_answer(spec, rows), "abstained": False,
                "abstain_reason": None}

    # ----------------------------------------------------------------- retrieval path
    def _abstain(self, reason: str, hits: list[Hit]) -> dict:
        return {"answer": ABSTAIN_TEXT, "abstained": True, "abstain_reason": reason,
                "citations": [], "sources": self._sources(hits), "sql": None, "rows": None}

    @staticmethod
    def _sources(hits: list[Hit]) -> list[dict]:
        return [{"complaint_id": h.complaint_id, "rank": h.rank, "score": round(h.score, 4),
                 "dense_score": round(h.dense_score, 4), "meta": h.meta,
                 "snippet": h.text[:400]} for h in hits]

    def _ask_retrieve(self, q: str, filters: Filters | None, k: int, mode: str) -> dict:
        hits = self.retriever.search(q, k=k, mode=mode, filters=filters)
        if not hits:
            return self._abstain("no_results", hits)
        if self.min_dense_score and hits[0].dense_score < self.min_dense_score:
            return self._abstain("low_similarity", hits)

        raw = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(q, hits), max_tokens=500).strip()
        if not raw or "INSUFFICIENT_EVIDENCE" in raw:
            return self._abstain("model_abstained", hits)

        retrieved = {h.complaint_id for h in hits}
        cited = list(dict.fromkeys(CITE_RE.findall(raw)))
        valid = [c for c in cited if c in retrieved]
        if not valid:  # unsourced answers are never shown
            return self._abstain("no_valid_citation", hits)
        answer = raw
        for bad in (c for c in cited if c not in retrieved):  # drop fabricated citations
            answer = answer.replace(f"[{bad}]", "")
        return {"answer": answer.strip(), "abstained": False, "abstain_reason": None,
                "citations": valid, "invalid_citations": [c for c in cited if c not in retrieved],
                "sources": self._sources(hits), "sql": None, "rows": None}


def build_pipeline(
    index_dir: Path = INDEX_DIR,
    db_path: Path = DB_PATH,
    llm: str = "auto",
    router: str = "rules",
    spec_with_llm: bool = False,
    mode: str = "hybrid",
    k: int = 5,
    min_dense_score: float = 0.0,
) -> Pipeline:
    model = get_llm(llm)
    r = LLMRouter(model) if router == "llm" else RuleRouter()
    return Pipeline(Retriever(index_dir), db_path, model, r, spec_with_llm, k, mode, min_dense_score)
