"""Cheap, deterministic checks that an answer is supported by the chunks it cites.

These run after the LLM and cost nothing. They do not prove an answer is right; they catch
the common failure modes: citing a chunk that was never retrieved, quoting a number that
appears in none of the cited passages, and answers whose wording has nothing to do with the
cited text.

Number matching understands scale words and rounding: "$1.2 billion" matches a table cell
"1,184.2" in a table stated in millions (1.2 billion = 1,200 million, and 1,184.2 rounds to
1.2 at one decimal place in billions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM_RE = re.compile(
    r"(?<![\w.])\(?-?\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?\s*(%|(?:percent|billion|million|thousand|bn|mm|m|k)\b)?",
    re.IGNORECASE,
)
SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{3,}")
_STOP = set(
    ["that", "this", "with", "from", "were", "have", "been", "their", "there", "which", "about", "would", "could", "should", "into", "than", "then", "them", "they", "these", "those", "also", "over", "under", "during", "year", "years", "fiscal", "company", "compared", "total", "based", "provided", "filings", "filing", "answer", "according", "reported", "report", "million", "billion", "thousand", "percent", "approximately", "roughly"]
)


@dataclass
class Number:
    value: float  # as written, without scale
    decimals: int
    scale: float  # 1.0 when no scale word
    is_percent: bool
    raw: str


def extract_numbers(text: str) -> list[Number]:
    out = []
    for m in NUM_RE.finditer(text):
        digits, unit = m.group(1), (m.group(2) or "").lower()
        clean = digits.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append(
            Number(
                value=float(clean),
                decimals=decimals,
                scale=SCALE.get(unit, 1.0),
                is_percent=unit in {"%", "percent"},
                raw=m.group(0).strip(),
            )
        )
    return out


def _trivial(n: Number) -> bool:
    # Years, item numbers and tiny counts appear everywhere; checking them adds noise.
    if n.decimals == 0 and n.scale == 1.0 and not n.is_percent:
        return n.value < 10 or 1990 <= n.value <= 2035
    return False


def number_supported(n: Number, evidence: list[Number]) -> bool:
    """True if some evidence number equals n, allowing for unit scale and rounding."""
    target = n.value * n.scale
    tol_answer = 0.5 * 10 ** (-n.decimals) * n.scale  # rounding tolerance of the answer
    for e in evidence:
        if n.is_percent != e.is_percent:
            continue
        # The evidence may be stated in any unit (tables say "in millions" once, at the top).
        for e_scale in {e.scale, 1.0, 1e3, 1e6, 1e9}:
            ev = e.value * e_scale
            if abs(ev - target) <= tol_answer + 1e-9 * max(abs(target), 1):
                return True
    return False


@dataclass
class GroundingResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    term_coverage: float = 1.0


def check_grounding(
    answer: str,
    citations: list[str],
    retrieved: dict[str, str],
    question: str = "",
    min_term_coverage: float = 0.4,
) -> GroundingResult:
    """retrieved maps chunk id -> chunk text for everything that was shown to the model."""
    issues: list[str] = []
    if not citations:
        return GroundingResult(ok=False, issues=["answer has no citations"])
    unknown = [c for c in citations if c not in retrieved]
    if unknown:
        return GroundingResult(ok=False, issues=[f"cites chunks that were not retrieved: {unknown}"])

    cited_text = "\n".join(retrieved[c] for c in citations)
    evidence = extract_numbers(cited_text) + extract_numbers(question)
    unsupported = [
        n.raw for n in extract_numbers(answer) if not _trivial(n) and not number_supported(n, evidence)
    ]
    if unsupported:
        issues.append(f"numbers not found in cited chunks: {unsupported}")

    terms = {w.lower().removesuffix("'s") for w in _WORD.findall(answer)} - _STOP
    cited_lower = (cited_text + " " + question).lower()
    coverage = (sum(1 for t in terms if t in cited_lower) / len(terms)) if terms else 1.0
    if coverage < min_term_coverage:
        issues.append(f"only {coverage:.0%} of the answer's key terms appear in the cited chunks")

    return GroundingResult(
        ok=not issues, issues=issues, unsupported_numbers=unsupported, term_coverage=coverage
    )
