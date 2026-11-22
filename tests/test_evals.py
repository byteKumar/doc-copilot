import csv
import json
from pathlib import Path

import pytest
from conftest import FIXTURES

from doc_copilot.cli import main
from doc_copilot.evals.dataset import load_questions, select_subset
from doc_copilot.evals.harness import percentile
from doc_copilot.evals.scoring import agreement_from_csv, auto_match, judge, write_spotcheck_csv
from doc_copilot.llm import FakeLLM

ROOT = Path(__file__).parent.parent
FORMAT = ROOT / "evals" / "FORMAT_EXAMPLES.jsonl"
HARNESS_Q = FIXTURES / "harness_questions.jsonl"


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def good(i, answerable=True, **kw):
    row = {
        "id": f"q{i}", "question": f"Question number {i} about revenue?", "answerable": answerable,
        "answer_type": "number" if answerable else "unanswerable",
        "gold_answer": "$5 million" if answerable else None,
        "gold_sections": [{"ticker": "ACME", "year": 2024, "section": "7"}] if answerable else [],
        "split": "dev" if i % 4 == 0 else "test",
    }
    row.update(kw)
    return row


# ---------- dataset ----------

def test_real_eval_set_is_not_written_for_chaman():
    assert not (ROOT / "evals" / "questions.jsonl").exists()


def test_format_examples_are_valid_and_marked():
    rep = load_questions(FORMAT)
    assert rep.ok, rep.errors
    assert len(rep.questions) == 3 and all(q.format_example for q in rep.questions)
    assert {q.answer_type for q in rep.questions} == {"number", "comparison", "unanswerable"}


def test_validator_catches_errors(tmp_path):
    p = tmp_path / "q.jsonl"
    write_jsonl(p, [
        good(1),
        good(1),  # duplicate id
        good(2, gold_answer=None),  # answerable without gold
        good(3, answerable=False, gold_answer="x"),  # unanswerable with gold
        {"id": "q4", "question": "short"},  # missing fields
    ])
    p.write_text(p.read_text() + "{not json\n")
    rep = load_questions(p)
    text = "\n".join(rep.errors)
    assert "duplicate ids" in text
    assert "needs a gold_answer" in text
    assert "must have gold_answer null" in text
    assert "missing fields" in text
    assert "invalid JSON" in text


def test_validator_warns_when_set_is_small(tmp_path):
    p = tmp_path / "q.jsonl"
    write_jsonl(p, [good(i) for i in range(9)] + [good(100 + i, answerable=False) for i in range(3)])
    rep = load_questions(p)
    assert rep.ok
    assert any("at least 120" in w for w in rep.warnings)


def test_validator_quiet_on_full_size_set(tmp_path):
    p = tmp_path / "q.jsonl"
    write_jsonl(p, [good(i) for i in range(90)] + [good(100 + i, answerable=False) for i in range(30)])
    rep = load_questions(p)
    assert rep.ok and rep.warnings == []


def test_subset_is_stratified_and_deterministic(tmp_path):
    p = tmp_path / "q.jsonl"
    write_jsonl(p, [good(i) for i in range(90)] + [good(100 + i, answerable=False) for i in range(30)])
    qs = load_questions(p).questions
    a, b = select_subset(qs, 20), select_subset(qs, 20)
    assert [q.id for q in a] == [q.id for q in b]
    assert len(a) == 20 and sum(not q.answerable for q in a) == 5


# ---------- scoring ----------

@pytest.mark.parametrize("gold,answer,ok", [
    ("$1,587.1 million", "Revenue was $1,587.1 million in fiscal 2024.", True),
    ("$1,587.1 million", "About $1.6 billion.", False),  # less precise than gold: not a match
    ("$1.6 billion", "Revenue was $1,587.1 million.", True),  # more precise than gold: fine
    ("23%", "Revenue grew 23%.", True),
    ("23%", "Revenue grew 22%.", False),
    ("Memphis, Tennessee", "Globex is headquartered in Memphis, Tennessee.", True),
    ("Memphis, Tennessee", "I don't know.", False),
])
def test_auto_match(gold, answer, ok):
    assert auto_match(gold, answer) is ok


def test_judge_uses_llm_and_returns_tokens():
    llm = FakeLLM()
    llm.script("judge", {"correct": True, "explanation": "same number"})
    j = judge(llm, "q?", "$5 million", "five million dollars")
    assert j.correct is True and j.input_tokens > 0
    assert "<gold_answer>\n$5 million\n</gold_answer>" in llm.calls[0]["user"]


def test_agreement_and_kappa(tmp_path):
    p = tmp_path / "spot.csv"
    rows = []
    for i in range(10):
        judge_ok = i < 6
        human_ok = judge_ok if i not in (0, 9) else not judge_ok  # 2 disagreements
        rows.append({"id": f"q{i}", "judge_correct": judge_ok, "human_correct": "yes" if human_ok else "no"})
    rows.append({"id": "blank", "judge_correct": True, "human_correct": ""})
    write_spotcheck_csv(rows, p)
    a = agreement_from_csv(p)
    assert (a.n, a.agree, a.skipped) == (10, 8, 1)
    assert a.rate == 0.8
    assert a.kappa == pytest.approx(0.5833, abs=1e-3)
    with p.open() as f:
        assert "human_correct" in next(csv.reader(f))


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([float(i) for i in range(1, 101)], 50) == pytest.approx(50.5)


# ---------- CLI guards ----------

def test_harness_refuses_format_examples(capsys):
    rc = main(["eval", "run", "--questions", str(FORMAT)])
    assert rc == 2
    assert "FORMAT EXAMPLES" in capsys.readouterr().err


def test_validate_command(capsys):
    assert main(["eval", "validate", "--questions", str(FORMAT)]) == 0
    assert main(["eval", "validate", "--questions", "evals/does-not-exist.jsonl"]) == 2


# ---------- full harness over the fixture corpus (FakeLLM, Postgres) ----------

@pytest.mark.db
def test_harness_end_to_end(ingested, db_url, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("MIN_SIMILARITY", "0.1")
    rc = main(["eval", "run", "--questions", str(HARNESS_Q), "--judge", "--out", str(tmp_path)])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "not results" in out.out  # FakeLLM banner
    assert "at least 120" in out.err  # small-set warning
    run_dir = next(tmp_path.iterdir()) / "baseline"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n"] == 6 and summary["n_answerable"] == 4
    for k in ("answer_accuracy_auto", "citation_accuracy", "correct_refusal_rate", "false_refusal_rate",
              "latency_p50_ms", "latency_p95_ms", "cost_per_1000_queries_usd"):
        assert summary[k] is not None
    assert summary["answer_accuracy_auto"] >= 0.5  # the extractive fake finds the easy facts
    assert (run_dir / "judge_spotcheck.csv").exists()
    assert len((run_dir / "results.jsonl").read_text().splitlines()) == 6


@pytest.mark.db
def test_dry_run_estimates_without_calling(ingested, db_url, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)
    rc = main(["eval", "run", "--questions", str(HARNESS_Q), "--dry-run", "--judge"])
    out = capsys.readouterr().out
    assert rc == 0 and "Estimate:" in out and "LLM calls" in out
    assert "Results written" not in out


@pytest.mark.db
def test_paid_provider_requires_yes(ingested, db_url, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    rc = main(["eval", "run", "--questions", str(HARNESS_Q), "--subset", "3"])
    assert rc == 3  # stopped before building any client
    assert "--yes" in capsys.readouterr().err


@pytest.mark.db
def test_smoke_run_on_format_examples_has_no_metrics(ingested, db_url, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)
    rc = main(["eval", "run", "--questions", str(FORMAT), "--smoke", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "no metrics reported" in out
    assert "answer accuracy" not in out
    assert not any(tmp_path.iterdir())


@pytest.mark.db
def test_compare_baseline_and_improved(ingested, db_url, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("MIN_SIMILARITY", "0.1")
    rc = main(["eval", "run", "--questions", str(HARNESS_Q), "--compare", "baseline,improved",
               "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    header = out.splitlines()[[i for i, line in enumerate(out.splitlines()) if line.startswith("metric")][0]]
    assert "baseline" in header and "improved" in header
    assert "changed between baseline and improved" in out
    run = next(tmp_path.iterdir())
    s = json.loads((run / "improved" / "summary.json").read_text())
    assert s["chunking"] == "table_aware"
    ids = [json.loads(line)["citations"] for line in (run / "improved" / "results.jsonl").read_text().splitlines()]
    assert all("-t" in c for cites in ids for c in cites)
