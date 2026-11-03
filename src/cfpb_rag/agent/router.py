"""Decide whether a question needs retrieval over narratives or an aggregate SQL query."""
from __future__ import annotations

import re

from ..llm import LLM

_SQL_CUES = re.compile(
    r"\bhow many\b|\bnumber of\b|\bcount\b|\bmost common\b|\bmost frequent\b|\bleast common\b"
    r"|\btop \d+\b|\bwhich (?:issue|company|companies|state|product|month|year)\b"
    r"|\bmost complaints\b|\bfewest\b|\bby (?:issue|company|state|product|month|year)\b|\btotal\b",
    re.I,
)


class RuleRouter:
    name = "rules"

    def route(self, question: str) -> str:
        return "sql" if _SQL_CUES.search(question) else "retrieve"


ROUTER_SYSTEM = (
    "You route questions about a database of consumer complaints. Reply with exactly one word.\n"
    "SQL: the question asks for counts, rankings, totals or comparisons across many complaints "
    "(e.g. 'how many', 'which company has the most', 'top 3 states').\n"
    "RETRIEVE: the question asks what consumers said, reported or experienced, or asks for "
    "examples, themes or details found in the complaint narratives.\n"
    "Reply SQL or RETRIEVE only."
)


class LLMRouter:
    name = "llm"

    def __init__(self, llm: LLM, fallback: RuleRouter | None = None):
        self.llm = llm
        self.fallback = fallback or RuleRouter()

    def route(self, question: str) -> str:
        try:
            out = self.llm.complete(ROUTER_SYSTEM, f"TASK: route\nQUESTION: {question}", max_tokens=5)
        except Exception:
            out = ""
        word = out.strip().upper()
        if word.startswith("SQL"):
            return "sql"
        if word.startswith("RETRIEVE"):
            return "retrieve"
        return self.fallback.route(question)
