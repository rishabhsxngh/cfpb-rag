"""LLM-as-judge for groundedness and correctness. Hand-check a sample of its verdicts."""
from __future__ import annotations

from ..llm import LLM, extract_json

JUDGE_SYSTEM = (
    "You grade an answer to a question about consumer complaints. You are given the "
    "question, the answer, and the full text of the complaints the answer cites.\n"
    "grounded: true only if EVERY factual claim in the answer is supported by the cited text.\n"
    "correct: true only if the answer actually addresses the question, using what the cited "
    "text supports.\n"
    'Reply with JSON only: {"grounded": true|false, "correct": true|false, "reason": "<one sentence>"}'
)


def judge_answer(llm: LLM, question: str, answer: str, cited_texts: dict[str, str]) -> dict | None:
    """Returns {'grounded', 'correct', 'reason'} or None if the judge is unavailable."""
    sources = "\n".join(f"[{cid}] {text[:1500]}" for cid, text in cited_texts.items())
    user = f"TASK: judge\nQUESTION: {question}\nANSWER: {answer}\nCITED COMPLAINTS:\n{sources}"
    try:
        obj = extract_json(llm.complete(JUDGE_SYSTEM, user, max_tokens=200))
    except Exception:
        return None
    if not obj or "grounded" not in obj or "correct" not in obj:
        return None
    return {"grounded": bool(obj["grounded"]), "correct": bool(obj["correct"]),
            "reason": str(obj.get("reason", ""))}
