import httpx
import pytest
from conftest import FILINGS

from doc_copilot.embed import FakeEmbedder
from doc_copilot.ingest.chunk import chunk_filing
from doc_copilot.ingest.download import EdgarClient, UserAgentError, validate_user_agent
from doc_copilot.ingest.parse import extract_blocks, parse_filing
from doc_copilot.ingest.pipeline import ingest_file, load_meta

ACME = FILINGS / "ACME_2024_10-K.html"


def sections():
    return parse_filing(ACME.read_text())


# ---------- parsing ----------

def test_sections_found_in_order():
    items = [s.item for s in sections()]
    assert items == ["0", "1", "1A", "2", "7", "7A", "8"]


def test_toc_duplicate_headings_resolved_to_real_section():
    by_item = {s.item: s for s in sections()}
    # The TOC also mentions "Item 1A. Risk Factors"; we must keep the real content.
    assert "lidar sensors" in " ".join(b.text for b in by_item["1A"].blocks)
    assert by_item["1A"].title == "Risk Factors"
    assert by_item["7"].anchor == "item7"


def test_hidden_xbrl_header_dropped():
    text = " ".join(b.text for b in extract_blocks(ACME.read_text()))
    assert "hidden inline XBRL" not in text


def test_table_cells_merged():
    s = {s.item: s for s in sections()}["8"]
    table = next(b for b in s.blocks if b.kind == "table")
    assert "Revenue | $ 1,587.1 | $ 1,285.3 | $ 1,021.7" in table.text
    assert "Interest expense, net | (12.4) | (14.1) | (15.0)" in table.text
    assert table.header_rows == 1
    assert table.rows[0] == "2024 | 2023 | 2022"


def test_percent_cells_merged():
    s = {s.item: s for s in sections()}["7"]
    table = next(b for b in s.blocks if b.kind == "table")
    assert "Warehouse Automation | $ 1,184.2 | $ 973.5 | 22%" in table.text
    assert table.header_rows == 2


def test_uppercase_h2_headings():
    secs = parse_filing((FILINGS / "GLBX_2024_10-K.html").read_text())
    assert [s.item for s in secs] == ["0", "1", "1A", "7"]


def test_sentence_mentioning_item_is_not_a_heading():
    html = "<p><b>Item 7. MD&amp;A</b></p><p>See Item 8 for details.</p><p>Revenue rose.</p>"
    secs = parse_filing(html)
    assert [s.item for s in secs] == ["7"]


def test_title_on_next_line():
    html = "<p>Item 1A.</p><p>Risk Factors</p><p>Our business has risks.</p>"
    secs = parse_filing(html)
    assert secs[0].title == "Risk Factors"
    assert secs[0].blocks[0].text == "Our business has risks."


# ---------- chunking ----------

def test_chunk_ids_stable_and_unique():
    a = chunk_filing(sections(), "ACME", 2024, "baseline", 80, 20)
    b = chunk_filing(sections(), "ACME", 2024, "baseline", 80, 20)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)
    assert a[0].id.startswith("ACME-2024-0-b")


def test_baseline_windows_overlap():
    chunks = [c for c in chunk_filing(sections(), "ACME", 2024, "baseline", 60, 20) if c.section == "1"]
    assert len(chunks) >= 3
    first_tail = chunks[0].text.split()[-20:]
    assert chunks[1].text.split()[:20] == first_tail


def test_baseline_can_split_table_from_its_header():
    # Small windows show a known weakness: table rows lose their header row.
    chunks = chunk_filing(sections(), "ACME", 2024, "baseline", 40, 5)
    net = [c for c in chunks if "Net income | $ 182.4" in c.text]
    assert net and all("2024 | 2023 | 2022" not in c.text for c in net)


# ---------- downloader (no network: httpx.MockTransport) ----------

def test_user_agent_required():
    with pytest.raises(UserAgentError):
        validate_user_agent("")
    with pytest.raises(UserAgentError):
        validate_user_agent("Your Name you@example.com")
    assert validate_user_agent("Test Runner test-contact") == "Test Runner test-contact"


def test_downloader_picks_fiscal_year_and_writes_sidecar(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["User-Agent"] == "Test Runner test-contact"
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 1234, "ticker": "ACME", "title": "Acme Robotics, Inc."}})
        if "submissions" in url:
            assert url.endswith("CIK0000001234.json")
            return httpx.Response(200, json={"filings": {"recent": {
                "form": ["10-K", "10-K/A", "10-Q", "10-K"],
                "accessionNumber": ["0001-25-000001", "0001-25-000009", "0001-24-000005", "0001-24-000002"],
                "reportDate": ["2024-12-31", "2024-12-31", "2024-09-30", "2023-12-31"],
                "filingDate": ["2025-02-20", "2025-04-01", "2024-11-01", "2024-02-21"],
                "primaryDocument": ["acme-20241231.htm", "a.htm", "q.htm", "acme-20231231.htm"],
            }}})
        if "Archives" in url:
            assert url.endswith("/1234/000125000001/acme-20241231.htm")
            return httpx.Response(200, content=b"<html><p>Item 1. Business</p></html>")
        return httpx.Response(404)

    client = EdgarClient("Test Runner test-contact", http=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)
    path = client.download("acme", 2024, tmp_path)
    assert path.name == "ACME_2024_10-K.html"
    meta = load_meta(path)
    assert (meta.ticker, meta.company, meta.year) == ("ACME", "Acme Robotics, Inc.", 2024)
    assert meta.source_url.endswith("acme-20241231.htm")


# ---------- ingestion into Postgres ----------

@pytest.mark.db
def test_ingest_and_reingest_is_idempotent(conn):
    emb = FakeEmbedder(384)
    first = ingest_file(conn, ACME, emb)
    assert first.chunks_embedded == first.chunks_total > 0
    assert set(first.per_strategy) == {"baseline"}
    n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert n == first.chunks_total

    second = ingest_file(conn, ACME, emb)
    assert second.documents_unchanged == 1
    assert second.chunks_embedded == 0
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == n
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


@pytest.mark.db
def test_changed_file_reembeds_only_changed_chunks(conn, tmp_path):
    emb = FakeEmbedder(384)
    src = tmp_path / "ACME_2024_10-K.html"
    src.write_text(ACME.read_text())
    (tmp_path / "ACME_2024_10-K.json").write_text((FILINGS / "ACME_2024_10-K.json").read_text())
    first = ingest_file(conn, src, emb)
    src.write_text(ACME.read_text().replace("410,000 square foot", "455,000 square foot"))
    second = ingest_file(conn, src, emb)
    assert 0 < second.chunks_embedded < first.chunks_embedded
    text = conn.execute("SELECT text FROM chunks WHERE section='2' AND chunking='baseline'").fetchone()[0]
    assert "455,000" in text


@pytest.mark.db
def test_metadata_stored(ingested):
    rows = ingested.execute(
        "SELECT DISTINCT ticker, company, year FROM chunks ORDER BY 1, 3"
    ).fetchall()
    assert rows == [
        ("ACME", "Acme Robotics, Inc.", 2023),
        ("ACME", "Acme Robotics, Inc.", 2024),
        ("GLBX", "Globex Freight Corporation", 2024),
    ]
    anchor = ingested.execute(
        "SELECT anchor FROM chunks WHERE ticker='ACME' AND year=2024 AND section='1A' LIMIT 1"
    ).fetchone()[0]
    assert anchor == "item1a"
