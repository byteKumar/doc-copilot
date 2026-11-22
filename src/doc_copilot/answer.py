"""Question -> retrieve -> (maybe) LLM -> grounding check -> answer with citations.

Three ways to refuse, cheapest first:
1. Retrieval is weak (best vector similarity under MIN_SIMILARITY): refuse without calling
   the LLM. Saves money on obviously off-topic questions.
2. The model itself says the passages don't contain the answer (refused=true).
3. The model answers, but the grounding check fails: refuse and log why.
"""

from __future__ import annotations

import html
import time
import uuid
from dataclasses import asdict, dataclass, field

import psycopg

from .config import Settings
from .embed import Embedder
from .grounding import check_grounding
from .llm import LLM, LLMError
from .retrieve import Filters, Hit, retrieve

REFUSAL_TEXT = "I don't know. The filings I searched don't support an answer to that."

SYSTEM_PROMPT = """You answer questions about companies using ONLY the 10-K excerpts provided.

Rules:
- Use only facts stated in the <chunk> passages. Do not use outside knowledge, even if you are sure.
- Cite every chunk you used by its id in "citations". Cite only ids that appear in the passages.
- Copy numbers exactly as written in the passages, with their units (for example "$1,587.1 million").
  Say which fiscal year a number belongs to.
- If the passages do not contain the answer, set "refused" to true, leave "citations" empty and
  explain in "reason" what is missing. A partial answer is better than a guess only if every part
  you state is supported; otherwise refuse.
- For comparisons, cite a chunk for each side of the comparison.
- Keep the answer to one to three sentences."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The answer, or a short refusal."},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Chunk ids that support the answer.",
        },
        "refused": {"type": "boolean"},
        "reason": {"type": "string", "description": "Why refused, or which passages were used."},
    },
    "required": ["answer", "citations", "refused", "reason"],
    "additionalProperties": False,
}


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        parts.append(
            f'<chunk id="{h.id}" company="{html.escape(h.company)}" year="{h.year}" '
            f'section="Item {h.section}. {html.escape(h.section_title)}">\n{h.text}\n</chunk>'
        )
    return "<passages>\n" + "\n".join(parts) + f"\n</passages>\n\n<question>\n{question}\n</question>"


@dataclass
class Citation:
    id: str
    company: str
    ticker: str
    year: int
    section: str
    section_title: str
    snippet: str


@dataclass
class AnswerResponse:
    query_id: str
    question: str
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None
    latency_ms: float
    retrieval_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_called: bool = False
    grounding_issues: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    best_similarity: float = 0.0
    model_answer: str | None = None  # what the model said, even if we refused it

    def to_dict(self) -> dict:
        return asdict(self)


class Answerer:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLM, chunking: str | None = None):
        self.s = settings
        self.embedder = embedder
        self.llm = llm
        self.chunking = chunking or settings.chunking

    def answer(
        self, conn: psycopg.Connection, question: str, filters: Filters | None = None, log: bool = True
    ) -> AnswerResponse:
        t0 = time.perf_counter()
        filters = filters or Filters()
        r = retrieve(
            conn, self.embedder, question, filters, chunking=self.chunking,
            top_k=self.s.top_k, candidates=self.s.candidates_per_retriever, rrf_k=self.s.rrf_k,
        )
        resp = AnswerResponse(
            query_id=str(uuid.uuid4()), question=question, answer=REFUSAL_TEXT, citations=[],
            refused=True, refusal_reason=None, latency_ms=0.0, retrieval_ms=r.latency_ms,
            retrieved_ids=[h.id for h in r.hits], best_similarity=r.best_similarity,
        )

        if not r.hits or r.best_similarity < self.s.min_similarity:
            resp.refusal_reason = (
                f"weak evidence: best similarity {r.best_similarity:.2f} < {self.s.min_similarity:.2f}"
            )
            return self._finish(conn, resp, t0, filters, log)

        try:
            out = self.llm.complete_json(SYSTEM_PROMPT, build_prompt(question, r.hits), ANSWER_SCHEMA, "answer")
        except LLMError as e:
            resp.refusal_reason = f"llm error: {e}"
            return self._finish(conn, resp, t0, filters, log)

        resp.llm_called = True
        resp.input_tokens, resp.output_tokens = out.input_tokens, out.output_tokens
        resp.cost_usd = self.s.cost_usd(out.input_tokens, out.output_tokens)
        data = out.data
        resp.model_answer = data.get("answer", "")

        if data.get("refused"):
            resp.refusal_reason = f"model: {data.get('reason', '')}".strip()
            return self._finish(conn, resp, t0, filters, log)

        by_id = {h.id: h for h in r.hits}
        cites = list(dict.fromkeys(data.get("citations") or []))
        evidence = {k: f"{v.company} {v.ticker} {v.section_title}\n{v.text}" for k, v in by_id.items()}
        g = check_grounding(resp.model_answer, cites, evidence, question)
        if not g.ok:
            resp.grounding_issues = g.issues
            resp.refusal_reason = "grounding check failed: " + "; ".join(g.issues)
            return self._finish(conn, resp, t0, filters, log)

        resp.refused = False
        resp.answer = resp.model_answer
        resp.citations = [
            Citation(
                id=c, company=by_id[c].company, ticker=by_id[c].ticker, year=by_id[c].year,
                section=by_id[c].section, section_title=by_id[c].section_title,
                snippet=by_id[c].text[:280],
            )
            for c in cites
        ]
        return self._finish(conn, resp, t0, filters, log)

    def _finish(self, conn, resp: AnswerResponse, t0: float, filters: Filters, log: bool) -> AnswerResponse:
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        if log:
            log_query(conn, resp, filters)
        return resp


def log_query(conn: psycopg.Connection, r: AnswerResponse, filters: Filters) -> None:
    from psycopg.types.json import Jsonb

    conn.execute(
        """INSERT INTO query_log (id, question, filters, answer, citations, refused, refusal_reason,
               latency_ms, input_tokens, output_tokens, cost_usd, llm_called)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            r.query_id, r.question, Jsonb(filters.as_dict()), r.answer,
            Jsonb([c.id for c in r.citations]), r.refused, r.refusal_reason, r.latency_ms,
            r.input_tokens, r.output_tokens, r.cost_usd, r.llm_called,
        ),
    )
