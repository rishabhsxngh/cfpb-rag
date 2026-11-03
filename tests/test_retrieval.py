import pytest

from cfpb_rag.retrieve.retriever import Filters


def _sample_docs(retriever, n=40):
    d = retriever.docs
    # long, distinctive narratives so self-retrieval is a fair sanity check
    return d[d["n_words"] > 50].sample(n=n, random_state=1)


@pytest.mark.parametrize("mode,floor", [("bm25", 0.9), ("dense", 0.9), ("hybrid", 0.9)])
def test_self_retrieval_finds_the_complaint(retriever, mode, floor):
    """Querying with a complaint's own opening words must put a near-identical text in the top 5.

    The synthetic data is templated, so exact-ID hit rate is not the right bar; we check that
    the retrieved text shares the query's opening sentence (i.e. retrieval is not broken).
    """
    docs = _sample_docs(retriever)
    ok = 0
    for _, row in docs.iterrows():
        words = row["narrative"].split()
        query = " ".join(words[:25])
        hits = retriever.search(query, k=5, mode=mode)
        ok += any(h.text.startswith(" ".join(words[:8])) for h in hits)
    assert ok / len(docs) >= floor


def test_topical_query_returns_topical_text(retriever):
    hits = retriever.search("escrow analysis shortage", k=5, mode="hybrid")
    assert sum("escrow" in h.text.lower() for h in hits) >= 4


def test_filters_are_honored(retriever):
    f = Filters(product_group="Credit reporting", state="CA", date_from="2025-01-01")
    hits = retriever.search("incorrect information on my report", k=10, filters=f)
    assert hits
    for h in hits:
        assert h.meta["product_group"] == "Credit reporting"
        assert h.meta["state"] == "CA"
        assert h.meta["date"] >= "2025-01-01"


def test_empty_filter_result_returns_no_hits(retriever):
    assert retriever.search("anything", filters=Filters(company="No Such Company")) == []


def test_modes_return_ranked_hits(retriever):
    for mode in ("dense", "bm25", "hybrid"):
        hits = retriever.search("late fee after payment", k=5, mode=mode)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_invalid_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="nope")
