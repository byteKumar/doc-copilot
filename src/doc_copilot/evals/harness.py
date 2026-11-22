"""Run the eval set through the full pipeline and score it.

Metrics (all computed from this run, never estimated):
  answer_accuracy_auto   answerable questions whose answer matches gold (numbers / normalized
                         text). Refusals count as wrong. Denominator: all answerable questions.
  answer_accuracy_judge  same, but graded by the LLM judge (only with --judge).
  citation_accuracy      answered (not refused) answerable questions where at least one cited
                         chunk contains the gold answer.
  citation_section_match answered answerable questions citing at least one gold section.
  correct_refusal_rate   unanswerable questions that were refused.
  false_refusal_rate     answerable questions that were refused.
  latency p50 / p95      end-to-end answer latency (retrieval + LLM + checks), ms.
  cost_per_1000_queries  mean answering cost per question x 1000 (judge cost reported apart).
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psycopg

from ..answer import SYSTEM_PROMPT, Answerer, build_prompt
from ..config import Settings
from ..llm import LLM
from ..retrieve import Filters, get_chunk, retrieve
from .dataset import EvalQuestion
from .scoring import (
    JUDGE_SYSTEM,
    auto_match,
    chunk_contains_answer,
    judge,
    section_match,
    write_spotcheck_csv,
)

# Used only by the dry-run estimate. Deliberately pessimistic (English is ~4 chars/token).
CHARS_PER_TOKEN = 3.5
EST_ANSWER_OUTPUT_TOKENS = 250
EST_JUDGE_INPUT_TOKENS = 450
EST_JUDGE_OUTPUT_TOKENS = 80


@dataclass
class QuestionResult:
    id: str
    question: str
    answerable: bool
    split: str
    gold_answer: str | None
    model_answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[str]
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    auto_correct: bool | None = None
    judge_correct: bool | None = None
    judge_explanation: str | None = None
    citation_contains_answer: bool | None = None
    citation_section_match: bool | None = None


@dataclass
class Summary:
    label: str
    chunking: str
    model: str
    n: int
    n_answerable: int
    n_unanswerable: int
    answer_accuracy_auto: float | None
    answer_accuracy_judge: float | None
    citation_accuracy: float | None
    citation_section_match: float | None
    correct_refusal_rate: float | None
    false_refusal_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    cost_per_1000_queries_usd: float
    answer_cost_usd: float
    judge_cost_usd: float
    warnings: list[str] = field(default_factory=list)


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def run_questions(
    conn: psycopg.Connection,
    questions: list[EvalQuestion],
    answerer: Answerer,
    settings: Settings,
    judge_llm: LLM | None = None,
    progress: bool = False,
) -> tuple[list[QuestionResult], float]:
    results, judge_cost = [], 0.0
    for i, q in enumerate(questions, start=1):
        f = Filters(q.filters.get("company"), q.filters.get("year"))
        r = answerer.answer(conn, q.question, f, log=False)
        qr = QuestionResult(
            id=q.id, question=q.question, answerable=q.answerable, split=q.split,
            gold_answer=q.gold_answer, model_answer=r.answer, refused=r.refused,
            refusal_reason=r.refusal_reason, citations=[c.id for c in r.citations],
            latency_ms=r.latency_ms, cost_usd=r.cost_usd,
            input_tokens=r.input_tokens, output_tokens=r.output_tokens,
        )
        if q.answerable:
            qr.auto_correct = (not r.refused) and auto_match(q.gold_answer or "", r.answer)
            if not r.refused:
                cited = [get_chunk(conn, cid) for cid in qr.citations]
                cited = [c for c in cited if c]
                qr.citation_contains_answer = any(chunk_contains_answer(q, c.text) for c in cited)
                qr.citation_section_match = section_match(q, [(c.ticker, c.year, c.section) for c in cited])
            if judge_llm is not None:
                if r.refused:
                    qr.judge_correct, qr.judge_explanation = False, "refused"
                else:
                    j = judge(judge_llm, q.question, q.gold_answer or "", r.answer)
                    qr.judge_correct, qr.judge_explanation = j.correct, j.explanation
                    judge_cost += settings.cost_usd(j.input_tokens, j.output_tokens)
        results.append(qr)
        if progress:
            mark = "REFUSED" if r.refused else ("ok" if qr.auto_correct else "-")
            print(f"[{i}/{len(questions)}] {q.id} {mark} {r.latency_ms:.0f}ms")
    return results, judge_cost


def summarize(
    results: list[QuestionResult], label: str, chunking: str, model: str, judge_cost: float,
    judged: bool, warnings: list[str] | None = None,
) -> Summary:
    ans = [r for r in results if r.answerable]
    una = [r for r in results if not r.answerable]
    answered = [r for r in ans if not r.refused]
    lat = sorted(r.latency_ms for r in results)
    answer_cost = sum(r.cost_usd for r in results)
    return Summary(
        label=label, chunking=chunking, model=model, n=len(results),
        n_answerable=len(ans), n_unanswerable=len(una),
        answer_accuracy_auto=_rate(sum(bool(r.auto_correct) for r in ans), len(ans)),
        answer_accuracy_judge=_rate(sum(bool(r.judge_correct) for r in ans), len(ans)) if judged else None,
        citation_accuracy=_rate(sum(bool(r.citation_contains_answer) for r in answered), len(answered)),
        citation_section_match=_rate(sum(bool(r.citation_section_match) for r in answered), len(answered)),
        correct_refusal_rate=_rate(sum(r.refused for r in una), len(una)),
        false_refusal_rate=_rate(sum(r.refused for r in ans), len(ans)),
        latency_p50_ms=percentile(lat, 50),
        latency_p95_ms=percentile(lat, 95),
        cost_per_1000_queries_usd=(answer_cost / len(results) * 1000) if results else 0.0,
        answer_cost_usd=answer_cost,
        judge_cost_usd=judge_cost,
        warnings=list(warnings or []),
    )


@dataclass
class CostEstimate:
    questions: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    refused_before_llm: int


def estimate_cost(
    conn: psycopg.Connection, questions: list[EvalQuestion], settings: Settings, embedder,
    chunkings: list[str], with_judge: bool,
) -> CostEstimate:
    """Builds the real prompts (retrieval is local and free) and counts characters.
    No API call is made."""
    calls = tin = tout = pre_refused = 0
    for chunking in chunkings:
        for q in questions:
            r = retrieve(
                conn, embedder, q.question, Filters(q.filters.get("company"), q.filters.get("year")),
                chunking=chunking, top_k=settings.top_k,
                candidates=settings.candidates_per_retriever, rrf_k=settings.rrf_k,
            )
            if not r.hits or r.best_similarity < settings.min_similarity:
                pre_refused += 1
                continue
            calls += 1
            tin += int(len(SYSTEM_PROMPT + build_prompt(q.question, r.hits)) / CHARS_PER_TOKEN)
            tout += EST_ANSWER_OUTPUT_TOKENS
            if with_judge and q.answerable:
                calls += 1
                tin += EST_JUDGE_INPUT_TOKENS + int(len(JUDGE_SYSTEM) / CHARS_PER_TOKEN)
                tout += EST_JUDGE_OUTPUT_TOKENS
    return CostEstimate(
        questions=len(questions) * len(chunkings), llm_calls=calls, input_tokens=tin,
        output_tokens=tout, usd=settings.cost_usd(tin, tout), refused_before_llm=pre_refused,
    )


def spotcheck_rows(results: list[QuestionResult], n: int = 30, seed: int = 0) -> list[dict]:
    """Random sample of answered, answerable questions for the human check of the judge."""
    pool = [r for r in results if r.answerable and r.judge_correct is not None]
    random.Random(seed).shuffle(pool)
    return [
        {
            "id": r.id, "question": r.question, "gold_answer": r.gold_answer,
            "model_answer": r.model_answer, "citations": " ".join(r.citations),
            "auto_correct": r.auto_correct, "judge_correct": r.judge_correct,
            "human_correct": "", "human_notes": "",
        }
        for r in pool[:n]
    ]


def write_run(out_dir: Path, summary: Summary, results: list[QuestionResult], spotcheck: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
    with (out_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    if spotcheck:
        write_spotcheck_csv(spotcheck_rows(results), out_dir / "judge_spotcheck.csv")
    return out_dir


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.1f}"


METRIC_ROWS = [
    ("answer accuracy (auto)", "answer_accuracy_auto", True),
    ("answer accuracy (judge)", "answer_accuracy_judge", True),
    ("citation accuracy", "citation_accuracy", True),
    ("citation section match", "citation_section_match", True),
    ("correct refusal (unanswerable)", "correct_refusal_rate", True),
    ("false refusal (answerable)", "false_refusal_rate", True),
    ("latency p50 ms", "latency_p50_ms", False),
    ("latency p95 ms", "latency_p95_ms", False),
]


def render_table(summaries: list[Summary]) -> str:
    head = f"{'metric':34}" + "".join(f"{s.label:>16}" for s in summaries)
    lines = [head, "-" * len(head)]
    for name, attr, pct in METRIC_ROWS:
        lines.append(f"{name:34}" + "".join(f"{fmt(getattr(s, attr), pct):>16}" for s in summaries))
    lines.append(
        f"{'cost per 1,000 queries':34}"
        + "".join(f"{'$' + format(s.cost_per_1000_queries_usd, '.4f'):>16}" for s in summaries)
    )
    lines.append(f"{'n (answerable/unanswerable)':34}" + "".join(
        f"{f'{s.n_answerable}/{s.n_unanswerable}':>16}" for s in summaries))
    return "\n".join(lines)


def flips(a: list[QuestionResult], b: list[QuestionResult]) -> list[str]:
    """Questions whose auto-correctness changed between two runs (for the before/after story)."""
    bmap = {r.id: r for r in b}
    out = []
    for r in a:
        o = bmap.get(r.id)
        if o and r.answerable and bool(r.auto_correct) != bool(o.auto_correct):
            out.append(f"{r.id}: {'wrong->right' if o.auto_correct else 'right->wrong'}")
    return out


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
