import pytest

from doc_copilot.retrieve import Filters, get_chunk, list_filters, or_tsquery, retrieve, rrf_merge


def test_rrf_rewards_agreement():
    scores = rrf_merge([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert max(scores, key=scores.get) == "b"
    # A chunk ranked first by one retriever only loses to one ranked 2nd and 1st.
    solo = rrf_merge([["x"], ["y", "x"]], k=60)
    assert solo["x"] > solo["y"]


def test_or_tsquery_sanitizes():
    assert or_tsquery("What's Acme's revenue in 2024?") == "what | acme | revenue | in | 2024"
    assert or_tsquery("!!!") == ""
    assert "'" not in or_tsquery("O'Reilly; DROP TABLE chunks")


@pytest.mark.db
def test_hybrid_finds_lexical_and_vector_hits(ingested, embedder):
    r = retrieve(ingested, embedder, "How many full-time employees did Acme have?",
                 Filters(company="ACME", year=2024))
    assert r.hits[0].text.count("2,310") or any("2,310" in h.text for h in r.hits[:3])
    top = r.hits[0]
    assert top.vector_rank is not None and top.lexical_rank is not None
    assert r.best_similarity > 0


@pytest.mark.db
def test_company_and_year_filters(ingested, embedder):
    r = retrieve(ingested, embedder, "total revenue", Filters(company="GLBX"))
    assert r.hits and {h.ticker for h in r.hits} == {"GLBX"}
    r = retrieve(ingested, embedder, "total revenue", Filters(company="acme robotics", year=2023))
    assert r.hits and {(h.ticker, h.year) for h in r.hits} == {("ACME", 2023)}


@pytest.mark.db
def test_chunking_strategies_are_separate(ingested, embedder):
    base = retrieve(ingested, embedder, "net income", chunking="baseline")
    tab = retrieve(ingested, embedder, "net income", chunking="table_aware")
    assert all("-b" in h.id for h in base.hits)
    assert all("-t" in h.id for h in tab.hits)


@pytest.mark.db
def test_get_chunk_and_filters(ingested):
    f = list_filters(ingested)
    assert [c["ticker"] for c in f["companies"]] == ["ACME", "GLBX"]
    assert f["years"] == [2023, 2024]
    cid = ingested.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
    assert get_chunk(ingested, cid).id == cid
    assert get_chunk(ingested, "nope") is None
