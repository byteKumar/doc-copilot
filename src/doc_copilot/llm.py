"""LLM clients behind one interface: `complete_json(system, user, schema, task)`.

- AnthropicLLM: the real Claude client (paid). Built only when LLM_PROVIDER=anthropic.
- FakeLLM: deterministic and offline. Used by tests, CI and local development. It answers
  by picking the retrieved sentence with the most word overlap with the question, which is
  dumb but exercises every code path (citations, refusals, grounding, judging, cost math).

Structured output: the request carries a JSON schema (output_config.format), so the reply
is valid JSON matching the schema. We still check stop_reason before parsing, because a
refusal or a max_tokens cut-off can leave the text incomplete.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


class LLMError(RuntimeError):
    """Base error. `retryable` tells the caller whether trying again makes sense."""

    retryable = False


class LLMRateLimited(LLMError):
    retryable = True


class LLMUnavailable(LLMError):
    retryable = True


class LLMBadResponse(LLMError):
    pass


@dataclass
class LLMResult:
    data: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    model: str

    def complete_json(self, system: str, user: str, schema: dict, task: str) -> LLMResult: ...


class AnthropicLLM:
    def __init__(self, settings: Settings):
        if os.environ.get("DOC_COPILOT_TESTING") == "1":
            raise RuntimeError("Refusing to build the paid Anthropic client inside the test suite.")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY in the environment.")
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=2)
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens

    def complete_json(self, system: str, user: str, schema: dict, task: str) -> LLMResult:
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            )
        except a.RateLimitError as e:
            raise LLMRateLimited(f"rate limited: {e}") from e
        except a.APIStatusError as e:
            err = LLMUnavailable if e.status_code >= 500 or e.status_code == 529 else LLMError
            raise err(f"API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMUnavailable(f"connection error: {e}") from e

        usage = resp.usage
        base = dict(
            input_tokens=(usage.input_tokens or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0),
            output_tokens=usage.output_tokens or 0,
            model=resp.model,
            stop_reason=resp.stop_reason,
        )
        if resp.stop_reason == "refusal":
            raise LLMBadResponse("model declined to answer (stop_reason=refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMBadResponse("answer cut off at max_tokens")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMBadResponse("no text block in response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMBadResponse(f"invalid JSON from model: {e}") from e
        return LLMResult(data=data, **base)


# ---------------------------------------------------------------- fake

_WORD = re.compile(r"[a-z0-9][a-z0-9.,%$]*")
_STOP = set(
    ["a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "was", "were", "be", "by", "with", "what", "which", "how", "much", "many", "did", "does", "do", "as", "at", "from", "that", "this", "its", "it", "their", "our", "we", "us", "than", "vs", "versus", "fiscal", "year", "report", "10-k", "company"]
)
_CHUNK_RE = re.compile(r'<chunk id="(?P<id>[^"]+)"[^>]*>\n(?P<text>.*?)\n</chunk>', re.S)
_QUESTION_RE = re.compile(r"<question>\n(?P<q>.*?)\n</question>", re.S)


def _content_words(text: str) -> set[str]:
    return {w.strip(".,") for w in _WORD.findall(text.lower()) if w.strip(".,") not in _STOP}


class FakeLLM:
    """Offline stand-in. `scripted` lets a test queue exact responses for a task."""

    model = "fake-llm"

    def __init__(self, min_overlap: float = 0.5, tokens_per_char: float = 0.25):
        self.min_overlap = min_overlap
        self.tokens_per_char = tokens_per_char
        self.scripted: dict[str, list[dict]] = {}
        self.calls: list[dict] = []

    def script(self, task: str, *responses: dict) -> None:
        self.scripted.setdefault(task, []).extend(responses)

    def complete_json(self, system: str, user: str, schema: dict, task: str) -> LLMResult:
        self.calls.append({"task": task, "system": system, "user": user})
        if self.scripted.get(task):
            data = self.scripted[task].pop(0)
        elif task == "answer":
            data = self._answer(user)
        elif task == "judge":
            data = self._judge(user)
        else:
            raise LLMBadResponse(f"FakeLLM has no behaviour for task {task!r}")
        return LLMResult(
            data=data,
            input_tokens=int(len(system + user) * self.tokens_per_char),
            output_tokens=int(len(json.dumps(data)) * self.tokens_per_char),
            model=self.model,
            stop_reason="end_turn",
        )

    def _answer(self, user: str) -> dict:
        qm = _QUESTION_RE.search(user)
        question = qm.group("q") if qm else user
        q = _content_words(question)
        best = (0.0, None, "")
        for m in _CHUNK_RE.finditer(user):
            for line in re.split(r"(?<=[.!?])\s+|\n", m.group("text")):
                overlap = len(q & _content_words(line)) / max(len(q), 1)
                if overlap > best[0]:
                    best = (overlap, m.group("id"), line.strip())
        score, cid, sentence = best
        if cid is None or score < self.min_overlap:
            return {
                "answer": "I don't know based on the provided filings.",
                "citations": [],
                "refused": True,
                "reason": f"no passage overlaps the question enough (best overlap {score:.2f})",
            }
        return {"answer": sentence, "citations": [cid], "refused": False, "reason": "extractive match"}

    def _judge(self, user: str) -> dict:
        gold = re.search(r"<gold_answer>\n(.*?)\n</gold_answer>", user, re.S)
        model = re.search(r"<model_answer>\n(.*?)\n</model_answer>", user, re.S)
        g = _content_words(gold.group(1) if gold else "")
        a = _content_words(model.group(1) if model else "")
        ok = bool(g) and len(g & a) / len(g) >= 0.6
        return {"correct": ok, "explanation": "fake judge: word overlap with gold answer"}


def make_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    raise ValueError(f"Unknown LLM_PROVIDER={settings.llm_provider!r} (use 'anthropic' or 'fake')")
