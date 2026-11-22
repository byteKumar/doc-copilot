"""Scoring functions: automatic answer match, LLM judge, citation check, agreement."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from ..grounding import _trivial, extract_numbers, number_supported
from ..llm import LLM, LLMError
from .dataset import EvalQuestion

_NORM = re.compile(r"[^a-z0-9.%$ ]+")


def normalize(text: str) -> str:
    t = text.lower().replace(",", "")
    t = _NORM.sub(" ", t)
    return " ".join(t.split()).strip(" .")


def gold_numbers(gold: str):
    return [n for n in extract_numbers(gold) if not _trivial(n)]


def auto_match(gold: str, answer: str) -> bool:
    """Numbers: every non-trivial number in the gold answer must appear in the model answer
    (unit scale and rounding allowed). No numbers: the normalized gold text must appear in
    the normalized answer. Strict on purpose; the LLM judge covers paraphrases."""
    nums = gold_numbers(gold)
    if nums:
        ans = extract_numbers(answer)
        return all(number_supported(n, ans) for n in nums)
    g = normalize(gold)
    return bool(g) and g in normalize(answer)


def chunk_contains_answer(q: EvalQuestion, chunk_text: str) -> bool:
    if q.gold_evidence and normalize(q.gold_evidence) in normalize(chunk_text):
        return True
    if q.gold_answer:
        nums = gold_numbers(q.gold_answer)
        if nums:
            ev = extract_numbers(chunk_text)
            return any(number_supported(n, ev) for n in nums)
        return normalize(q.gold_answer) in normalize(chunk_text)
    return False


def section_match(q: EvalQuestion, cited: list[tuple[str, int, str]]) -> bool:
    gold = {(g.ticker, g.year, g.section) for g in q.gold_sections}
    return any((t.upper(), y, s.upper()) in gold for t, y, s in cited)


JUDGE_SYSTEM = """You grade answers to questions about company 10-K filings.
Compare the model answer with the gold answer. Mark it correct if it states the same facts
(same numbers, allowing different units or rounding the gold answer would accept, and the
right company and year). Extra correct detail is fine. A wrong number, wrong year, wrong
company, a missing part of a multi-part answer, or a refusal is incorrect."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "explanation": {"type": "string"}},
    "required": ["correct", "explanation"],
    "additionalProperties": False,
}


@dataclass
class JudgeResult:
    correct: bool | None
    explanation: str
    input_tokens: int = 0
    output_tokens: int = 0


def judge(llm: LLM, question: str, gold: str, answer: str) -> JudgeResult:
    user = (
        f"<question>\n{question}\n</question>\n<gold_answer>\n{gold}\n</gold_answer>\n"
        f"<model_answer>\n{answer}\n</model_answer>"
    )
    try:
        r = llm.complete_json(JUDGE_SYSTEM, user, JUDGE_SCHEMA, "judge")
    except LLMError as e:
        return JudgeResult(None, f"judge error: {e}")
    return JudgeResult(bool(r.data["correct"]), r.data.get("explanation", ""), r.input_tokens, r.output_tokens)


# ---------------------------------------------------------------- human spot check

SPOTCHECK_FIELDS = [
    "id", "question", "gold_answer", "model_answer", "citations",
    "auto_correct", "judge_correct", "human_correct", "human_notes",
]
_YES = {"y", "yes", "1", "true", "t", "correct"}
_NO = {"n", "no", "0", "false", "f", "incorrect", "wrong"}


def write_spotcheck_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPOTCHECK_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SPOTCHECK_FIELDS})


@dataclass
class Agreement:
    n: int
    agree: int
    rate: float | None
    kappa: float | None
    skipped: int


def agreement_from_csv(path: Path) -> Agreement:
    """Agreement between your human_correct column and judge_correct, plus Cohen's kappa
    (agreement corrected for chance; above ~0.6 is usually called substantial)."""
    pairs, skipped = [], 0
    with path.open() as f:
        for row in csv.DictReader(f):
            h = (row.get("human_correct") or "").strip().lower()
            j = (row.get("judge_correct") or "").strip().lower()
            if h in _YES | _NO and j in _YES | _NO:
                pairs.append((h in _YES, j in _YES))
            else:
                skipped += 1
    n = len(pairs)
    if n == 0:
        return Agreement(0, 0, None, None, skipped)
    agree = sum(h == j for h, j in pairs)
    po = agree / n
    ph, pj = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = None if pe == 1 else (po - pe) / (1 - pe)
    return Agreement(n, agree, po, kappa, skipped)
