import os

import pytest

from doc_copilot.answer import ANSWER_SCHEMA, Answerer, build_prompt
from doc_copilot.config import Settings
from doc_copilot.grounding import check_grounding, extract_numbers
from doc_copilot.llm import AnthropicLLM, FakeLLM, LLMRateLimited, make_llm
from doc_copilot.retrieve import Filters, Hit

CHUNKS = {
    "ACME-2024-8-t000": "2024 | 2023\nRevenue | $ 1,587.1 | $ 1,285.3\nNet income | $ 182.4 | $ 77.0",
    "ACME-2024-1-b000": "As of December 31, 2024, we had 2,310 full-time employees.",
}


# ---------- grounding ----------

def test_grounding_accepts_supported_numbers():
    g = check_grounding("Acme's revenue was $1,587.1 million in fiscal 2024.", ["ACME-2024-8-t000"], CHUNKS)
    assert g.ok, g.issues


def test_grounding_accepts_scaled_and_rounded_numbers():
    g = check_grounding("Revenue was about $1.6 billion in 2024.", ["ACME-2024-8-t000"], CHUNKS)
    assert g.ok, g.issues


def test_grounding_rejects_made_up_number():
    g = check_grounding("Revenue was $1,612.0 million in 2024.", ["ACME-2024-8-t000"], CHUNKS)
    assert not g.ok and g.unsupported_numbers == ["$1,612.0 million"]


def test_grounding_rejects_unknown_citation():
    g = check_grounding("Revenue was $1,587.1 million.", ["ACME-2024-9-b999"], CHUNKS)
    assert not g.ok and "not retrieved" in g.issues[0]


def test_grounding_rejects_no_citations():
    assert not check_grounding("Revenue was $1,587.1 million.", [], CHUNKS).ok


def test_grounding_rejects_unrelated_wording():
    g = check_grounding(
        "Management expects aggressive expansion into Brazilian agriculture drones.",
        ["ACME-2024-1-b000"], CHUNKS,
    )
    assert not g.ok and "key terms" in g.issues[0]


def test_grounding_number_from_wrong_chunk_rejected():
    # 2,310 is in a retrieved chunk, but not in the one cited.
    g = check_grounding("Acme had 2,310 employees.", ["ACME-2024-8-t000"], CHUNKS)
    assert not g.ok


def test_extract_numbers_units():
    nums = extract_numbers("Revenue rose 23% to $1.6 billion; interest was (12.4).")
    assert [(n.value, n.scale, n.is_percent) for n in nums] == [
        (23.0, 1.0, True), (1.6, 1e9, False), (12.4, 1.0, False)
    ]


# ---------- prompt / schema ----------

def test_schema_is_strict():
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == set(ANSWER_SCHEMA["properties"])


def test_prompt_contains_chunk_ids_and_question():
    h = Hit("ACME-2024-1-b000", "ACME", "Acme <Robotics>", 2024, "1", "Business", None, "text", "body")
    p = build_prompt("How many employees?", [h])
    assert '<chunk id="ACME-2024-1-b000"' in p and "Acme &lt;Robotics&gt;" in p
    assert p.endswith("<question>\nHow many employees?\n</question>")


# ---------- LLM factory safety ----------

def test_default_provider_is_fake_and_paid_client_blocked_in_tests():
    assert isinstance(make_llm(Settings()), FakeLLM)
    assert not any(k.startswith("ANTHROPIC_") for k in os.environ)
    with pytest.raises(RuntimeError, match="test suite"):
        AnthropicLLM(Settings(llm_provider="anthropic"))


# ---------- end to end over the fixture corpus ----------

def answerer(settings, llm=None, **kw):
    from doc_copilot.embed import FakeEmbedder

    return Answerer(Settings(**{**settings.__dict__, **kw}), FakeEmbedder(384), llm or FakeLLM())


@pytest.mark.db
def test_answers_with_valid_citation_and_logs(ingested, settings):
    r = answerer(settings).answer(ingested, "How many full-time employees did Acme have as of December 31, 2024?",
                                  Filters(company="ACME", year=2024))
    assert not r.refused, r.refusal_reason
    assert "2,310" in r.answer
    assert r.citations and r.citations[0].id in r.retrieved_ids
    assert r.cost_usd > 0 and r.input_tokens > 0
    row = ingested.execute("SELECT refused, llm_called, cost_usd FROM query_log WHERE id=%s", (r.query_id,)).fetchone()
    assert row[0] is False and row[1] is True and row[2] > 0


@pytest.mark.db
def test_weak_retrieval_refuses_without_llm(ingested, settings):
    llm = FakeLLM()
    r = answerer(settings, llm, min_similarity=0.99).answer(ingested, "What is the capital of Peru?")
    assert r.refused and r.refusal_reason.startswith("weak evidence")
    assert llm.calls == [] and r.cost_usd == 0


@pytest.mark.db
def test_model_refusal_passed_through(ingested, settings):
    llm = FakeLLM()
    llm.script("answer", {"answer": "Not in the filings.", "citations": [], "refused": True, "reason": "no CEO salary"})
    r = answerer(settings, llm, min_similarity=0.0).answer(ingested, "What was the CEO's salary at Acme?")
    assert r.refused and "no CEO salary" in r.refusal_reason and r.llm_called


@pytest.mark.db
def test_hallucinated_citation_rejected(ingested, settings):
    llm = FakeLLM()
    llm.script("answer", {"answer": "Revenue was $1,587.1 million.", "citations": ["ACME-2024-9-b999"],
                          "refused": False, "reason": ""})
    r = answerer(settings, llm).answer(ingested, "What was Acme's revenue in 2024?")
    assert r.refused and "grounding" in r.refusal_reason
    assert r.model_answer == "Revenue was $1,587.1 million."


@pytest.mark.db
def test_hallucinated_number_rejected(ingested, settings):
    llm = FakeLLM()
    resp = answerer(settings, llm).answer(ingested, "What was Acme's total revenue in 2024?", log=False)
    cid = resp.retrieved_ids[0]
    llm.script("answer", {"answer": "Revenue was $9,999.9 million.", "citations": [cid], "refused": False, "reason": ""})
    r = answerer(settings, llm).answer(ingested, "What was Acme's total revenue in 2024?")
    assert r.refused and "9,999.9" in r.refusal_reason


@pytest.mark.db
def test_llm_error_becomes_refusal(ingested, settings):
    class Boom(FakeLLM):
        def complete_json(self, *a, **k):
            raise LLMRateLimited("slow down")

    r = answerer(settings, Boom()).answer(ingested, "What was Acme's revenue?")
    assert r.refused and "rate limited" not in r.answer and "slow down" in r.refusal_reason
