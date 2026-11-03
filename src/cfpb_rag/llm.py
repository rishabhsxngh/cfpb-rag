"""LLM access: Anthropic (cached) and a deterministic stub for tests / CI.

Prompts start with a `TASK: <name>` line so the stub can tell what is being asked.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from typing import Protocol

from .config import CACHE_DIR, DEFAULT_MODEL


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str: ...


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply, or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class AnthropicLLM:
    """Claude via the Anthropic API with an on-disk cache to control cost."""

    def __init__(self, model: str = DEFAULT_MODEL, cache: bool = True):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"
        self.cache_path = None
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.cache_path = CACHE_DIR / "llm_cache.sqlite"
            with sqlite3.connect(self.cache_path) as c:
                c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    def _key(self, system: str, user: str, max_tokens: int) -> str:
        return hashlib.sha256(f"{self.model}\x00{max_tokens}\x00{system}\x00{user}".encode()).hexdigest()

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        key = self._key(system, user, max_tokens)
        if self.cache_path:
            with sqlite3.connect(self.cache_path) as c:
                row = c.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
            if row:
                return row[0]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.cache_path:
            with sqlite3.connect(self.cache_path) as c:
                c.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (key, text))
        return text


_STOP = set(
    "a an the of to in on for and or is are was were be been what which who how why when where do does did "
    "about with from by at as that this these those it its their they them there any can could would should "
    "complaints complaint consumers consumer report reported say said tell me".split()
)
_EXCERPT_RE = re.compile(r"^\[(\d+)\] \((.*?)\) (.*)$", re.M)


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", s.lower()) if t not in _STOP}


class ExtractiveStubLLM:
    """Deterministic test double. NOT a language model.

    For `answer` tasks it abstains when the question shares too little vocabulary with
    the excerpts, otherwise it quotes the opening sentence of the top excerpts with
    citations. For routing / spec / judging / question drafting it returns an empty
    string so callers fall back to rules or mark the result as unavailable.
    Use it for tests and CI only; never report its outputs as model results.
    """

    name = "stub"

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        if not user.startswith("TASK: answer"):
            return ""
        q = re.search(r"^QUESTION: (.*)$", user, flags=re.M)
        question = q.group(1) if q else ""
        excerpts = _EXCERPT_RE.findall(user)
        if not excerpts:
            return "INSUFFICIENT_EVIDENCE"
        qt = _tokens(question)
        if not qt:
            return "INSUFFICIENT_EVIDENCE"
        best = max(len(qt & _tokens(text)) / len(qt) for _, _, text in excerpts)
        if best < 0.34:
            return "INSUFFICIENT_EVIDENCE"
        parts = []
        for cid, _, text in excerpts[:2]:
            first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
            parts.append(f"A consumer reported: {first} [{cid}]")
        return " ".join(parts)


def get_llm(name: str = "auto"):
    if name == "stub":
        return ExtractiveStubLLM()
    if name in ("anthropic", "auto"):
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicLLM()
        if name == "anthropic":
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return ExtractiveStubLLM()
    raise ValueError(f"Unknown LLM backend: {name!r}")
