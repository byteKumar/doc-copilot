"""Download 10-K filings from SEC EDGAR.

SEC's fair-access policy requires a User-Agent that identifies you ("Name contact") and
at most 10 requests per second. This client refuses to run without SEC_USER_AGENT and
waits between requests.

Endpoints used (all free and public):
  https://www.sec.gov/files/company_tickers.json            ticker -> CIK
  https://data.sec.gov/submissions/CIK##########.json        filing index for a company
  https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}  the filing's primary document

"Year" means the fiscal year of the report (the year of `reportDate`), not the filing date.
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

PLACEHOLDERS = ("your name", "your-name", "example.com", "changeme", "<", ">")


class UserAgentError(RuntimeError):
    pass


def validate_user_agent(ua: str) -> str:
    ua = (ua or "").strip()
    if not ua:
        raise UserAgentError(
            "SEC_USER_AGENT is not set. SEC requires a User-Agent with your name and a contact, "
            'e.g. export SEC_USER_AGENT="Jane Doe jane@yourdomain". Refusing to download.'
        )
    if len(ua.split()) < 2 or any(p in ua.lower() for p in PLACEHOLDERS):
        raise UserAgentError(
            f"SEC_USER_AGENT={ua!r} looks like a placeholder. Put your own name and contact in it."
        )
    return ua


@dataclass
class FilingRef:
    ticker: str
    company: str
    cik: int
    accession: str
    report_date: str
    filing_date: str
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document
        )


class EdgarClient:
    def __init__(self, user_agent: str, http: httpx.Client | None = None, min_interval: float = 0.15):
        self.user_agent = validate_user_agent(user_agent)
        self.http = http or httpx.Client(timeout=60, follow_redirects=True)
        self.min_interval = min_interval
        self._last = 0.0
        self._tickers: dict[str, tuple[int, str]] | None = None

    def _get(self, url: str) -> httpx.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self.http.get(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
        r.raise_for_status()
        return r

    def lookup(self, ticker: str) -> tuple[int, str]:
        if self._tickers is None:
            data = self._get(TICKERS_URL).json()
            self._tickers = {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in data.values()}
        try:
            return self._tickers[ticker.upper()]
        except KeyError:
            raise ValueError(f"Ticker {ticker!r} not found in SEC company_tickers.json") from None

    def list_10k(self, ticker: str) -> list[FilingRef]:
        cik, company = self.lookup(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        refs = []
        for i, form in enumerate(recent["form"]):
            if form != "10-K":  # skips 10-K/A amendments on purpose
                continue
            refs.append(
                FilingRef(
                    ticker=ticker.upper(),
                    company=company,
                    cik=cik,
                    accession=recent["accessionNumber"][i],
                    report_date=recent["reportDate"][i],
                    filing_date=recent["filingDate"][i],
                    primary_document=recent["primaryDocument"][i],
                )
            )
        return refs

    def download(self, ticker: str, year: int, out_dir: Path) -> Path:
        matches = [r for r in self.list_10k(ticker) if r.fiscal_year == year]
        if not matches:
            raise ValueError(f"No 10-K for {ticker} with fiscal year {year}")
        ref = sorted(matches, key=lambda r: r.filing_date)[-1]
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{ref.ticker}_{year}_10-K.html"
        if not html_path.exists():
            html_path.write_bytes(self._get(ref.url).content)
        html_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "ticker": ref.ticker,
                    "company": ref.company,
                    "year": year,
                    "form": "10-K",
                    "cik": ref.cik,
                    "accession": ref.accession,
                    "report_date": ref.report_date,
                    "filing_date": ref.filing_date,
                    "source_url": ref.url,
                },
                indent=2,
            )
        )
        return html_path


def load_corpus(path: Path) -> list[tuple[str, int]]:
    """corpus.toml: tickers = [...], years = [...]  ->  every (ticker, year) pair."""
    cfg = tomllib.loads(path.read_text())
    return [(t.upper(), int(y)) for t in cfg["tickers"] for y in cfg["years"]]
