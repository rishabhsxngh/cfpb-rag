"""Pure metric functions (no I/O) so they are easy to unit test."""
from __future__ import annotations


def hit_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """1.0 if any gold id appears in the top k."""
    return float(any(g in ranked_ids[:k] for g in gold_ids))


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Fraction of gold ids found in the top k."""
    if not gold_ids:
        return 0.0
    return sum(g in ranked_ids[:k] for g in gold_ids) / len(gold_ids)


def reciprocal_rank(ranked_ids: list[str], gold_ids: list[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def citation_validity(cited: list[str], invalid: list[str]) -> float | None:
    """Share of emitted citations that point at actually-retrieved complaints."""
    total = len(cited) + len(invalid)
    return None if total == 0 else len(cited) / total


def aggregate_match(expected: list[dict], got: list[dict]) -> dict:
    """Compare SQL-route output to the gold rows.

    top1: first row's group_value and count both match.
    full: all returned gold rows match in order.
    """
    if not got:
        return {"top1": False, "full": False}
    top1 = got[0]["group_value"] == expected[0]["group_value"] and got[0]["n"] == expected[0]["n"]
    full = [(r["group_value"], r["n"]) for r in got[: len(expected)]] == [
        (r["group_value"], r["n"]) for r in expected
    ]
    return {"top1": top1, "full": full}


def abstention_scores(items: list[tuple[bool, bool]]) -> dict:
    """items: (should_abstain, did_abstain). Returns precision/recall of abstaining
    plus the false-abstention rate on answerable questions."""
    tp = sum(s and d for s, d in items)
    fp = sum((not s) and d for s, d in items)
    fn = sum(s and (not d) for s, d in items)
    answerable = sum(not s for s, _ in items)
    return {
        "abstain_precision": tp / (tp + fp) if tp + fp else None,
        "abstain_recall": tp / (tp + fn) if tp + fn else None,
        "false_abstention_rate": fp / answerable if answerable else None,
        "n": len(items),
    }


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
