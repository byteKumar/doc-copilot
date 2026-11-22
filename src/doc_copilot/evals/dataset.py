"""Eval question format, loader and validator.

One JSON object per line (JSONL). Fields:

  id             unique string, e.g. "q001"
  question       the question as a user would ask it
  answerable     true if the filings contain the answer
  answer_type    "number" | "text" | "comparison" | "unanswerable"
  gold_answer    the correct answer as you would write it (null when unanswerable)
  gold_sections  where the answer lives: [{"ticker": "AAPL", "year": 2024, "section": "7"}]
                 (empty list when unanswerable; two entries for most comparisons)
  gold_evidence  optional short verbatim phrase from the filing that contains the answer
  filters        optional {"company": "AAPL", "year": 2024} sent with the question
  split          "dev" (allowed for tuning) or "test" (never look at it while tuning)
  notes          optional free text
  format_example optional; true marks a format example, which the harness never scores
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

ANSWER_TYPES = {"number", "text", "comparison", "unanswerable"}
MIN_QUESTIONS = 120
TARGET_ANSWERABLE = 90
TARGET_UNANSWERABLE = 30


@dataclass
class GoldSection:
    ticker: str
    year: int
    section: str


@dataclass
class EvalQuestion:
    id: str
    question: str
    answerable: bool
    answer_type: str
    gold_answer: str | None
    gold_sections: list[GoldSection] = field(default_factory=list)
    gold_evidence: str | None = None
    filters: dict = field(default_factory=dict)
    split: str = "test"
    notes: str | None = None
    format_example: bool = False


@dataclass
class ValidationReport:
    questions: list[EvalQuestion]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse(obj: dict, where: str, errors: list[str], warnings: list[str]) -> EvalQuestion | None:
    required = ["id", "question", "answerable", "answer_type", "gold_answer", "gold_sections"]
    missing = [k for k in required if k not in obj]
    if missing:
        errors.append(f"{where}: missing fields {missing}")
        return None
    unknown = set(obj) - {*required, "gold_evidence", "filters", "split", "notes", "format_example"}
    if unknown:
        errors.append(f"{where}: unknown fields {sorted(unknown)}")
    q = obj
    if not isinstance(q["id"], str) or not q["id"].strip():
        errors.append(f"{where}: id must be a non-empty string")
    if not isinstance(q["question"], str) or len(q["question"].strip()) < 8:
        errors.append(f"{where}: question is missing or too short")
    if not isinstance(q["answerable"], bool):
        errors.append(f"{where}: answerable must be true or false")
    if q["answer_type"] not in ANSWER_TYPES:
        errors.append(f"{where}: answer_type must be one of {sorted(ANSWER_TYPES)}")
    if q.get("split", "test") not in {"dev", "test"}:
        errors.append(f"{where}: split must be 'dev' or 'test'")

    sections = []
    if not isinstance(q["gold_sections"], list):
        errors.append(f"{where}: gold_sections must be a list")
    else:
        for s in q["gold_sections"]:
            if not (isinstance(s, dict) and {"ticker", "year", "section"} <= set(s)):
                errors.append(f"{where}: each gold_section needs ticker, year, section")
                continue
            sections.append(GoldSection(str(s["ticker"]).upper(), int(s["year"]), str(s["section"]).upper()))

    if q["answerable"] is True:
        if not (isinstance(q["gold_answer"], str) and q["gold_answer"].strip()):
            errors.append(f"{where}: answerable question needs a gold_answer")
        if not sections:
            errors.append(f"{where}: answerable question needs at least one gold_section")
        if q["answer_type"] == "unanswerable":
            errors.append(f"{where}: answerable question cannot have answer_type 'unanswerable'")
        if q["answer_type"] == "comparison" and len(sections) < 2:
            warnings.append(f"{where}: comparison usually needs a gold_section for each side")
    elif q["answerable"] is False:
        if q["gold_answer"] not in (None, ""):
            errors.append(f"{where}: unanswerable question must have gold_answer null")
        if sections:
            errors.append(f"{where}: unanswerable question must have empty gold_sections")
        if q["answer_type"] != "unanswerable":
            errors.append(f"{where}: unanswerable question must have answer_type 'unanswerable'")

    return EvalQuestion(
        id=str(q["id"]),
        question=str(q["question"]),
        answerable=bool(q["answerable"]),
        answer_type=str(q["answer_type"]),
        gold_answer=q["gold_answer"],
        gold_sections=sections,
        gold_evidence=q.get("gold_evidence"),
        filters=q.get("filters") or {},
        split=q.get("split", "test"),
        notes=q.get("notes"),
        format_example=bool(q.get("format_example", False)),
    )


def load_questions(path: Path) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    questions: list[EvalQuestion] = []
    if not path.exists():
        return ValidationReport([], [f"{path} does not exist. See evals/HOW_TO_WRITE_THE_EVAL_SET.md"], [])
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: invalid JSON ({e.msg})")
            continue
        q = _parse(obj, f"line {n} (id={obj.get('id')!r})", errors, warnings)
        if q:
            questions.append(q)

    ids = [q.id for q in questions]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        errors.append(f"duplicate ids: {dups}")
    seen: dict[str, str] = {}
    for q in questions:
        key = " ".join(q.question.lower().split())
        if key in seen:
            warnings.append(f"{q.id}: same question text as {seen[key]}")
        seen[key] = q.id

    real = [q for q in questions if not q.format_example]
    if real:
        a = sum(q.answerable for q in real)
        u = len(real) - a
        if len(real) < MIN_QUESTIONS:
            warnings.append(
                f"only {len(real)} questions; the spec asks for at least {MIN_QUESTIONS} "
                f"(~{TARGET_ANSWERABLE} answerable, ~{TARGET_UNANSWERABLE} unanswerable)"
            )
        if u < 0.15 * len(real) or u > 0.35 * len(real):
            warnings.append(f"{a} answerable / {u} unanswerable; aim for about 3:1")
        if not any(q.split == "dev" for q in real):
            warnings.append("no 'dev' questions: tuning would have to look at test questions")
    return ValidationReport(questions, errors, warnings)


def is_format_examples(path: Path, questions: list[EvalQuestion]) -> bool:
    return "FORMAT_EXAMPLES" in path.name.upper() or any(q.format_example for q in questions)


def select_subset(questions: list[EvalQuestion], n: int) -> list[EvalQuestion]:
    """Deterministic subset that keeps the answerable/unanswerable ratio (for CI)."""
    if n >= len(questions):
        return list(questions)

    def key(q: EvalQuestion) -> str:
        return hashlib.sha256(q.id.encode()).hexdigest()

    ans = sorted((q for q in questions if q.answerable), key=key)
    una = sorted((q for q in questions if not q.answerable), key=key)
    n_una = round(n * len(una) / len(questions))
    if una and n_una == 0:
        n_una = 1
    picked = ans[: n - n_una] + una[:n_una]
    order = {q.id: i for i, q in enumerate(questions)}
    return sorted(picked, key=lambda q: order[q.id])
