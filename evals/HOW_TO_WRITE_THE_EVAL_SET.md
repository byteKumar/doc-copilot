# How to write the eval set

The eval set is yours. Every number in RESULTS.md and on your resume comes from it, so you
write it and check it by hand. This guide covers what to write, the file format, and the
judge spot check.

`evals/FORMAT_EXAMPLES.jsonl` shows the format with three examples that point at the
synthetic test fixture. The harness refuses to score that file. Your real set goes in
`evals/questions.jsonl` (it does not exist yet).

## 1. Before you start

1. Download and ingest the corpus (see README, "Run it on real filings").
2. Open the filings in a browser (`data/filings/*.html`) or on sec.gov. Write questions from
   the documents, not from memory or the news. If you know a fact but can't find it in the
   10-K, it is not an answerable question.
3. Do not run the system on a question before you have written its gold answer. Writing the
   gold answer after seeing the model's answer biases the set toward what the model says.

## 2. What to write: about 120 questions

| Kind | How many | What it looks like |
|---|---|---|
| Answerable, single number | ~45 | "What was Apple's total net sales in fiscal 2024?" Answer is in one table or sentence. |
| Answerable, text fact | ~20 | "Where is Nike headquartered?", "What are JPM's reportable segments?" |
| Answerable, comparison | ~25 | Across two companies, or one company across two years. Needs two passages. |
| Unanswerable | ~30 | Plausible questions whose answer is **not** in the filings. |

Mix the sections: Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (financial
statements). Put at least 30 questions on tables (Item 7 and Item 8), because that is where
chunking breaks and where the improvement is measured.

Good unanswerable questions are close misses, not nonsense:
- asks for a year you didn't ingest ("...in fiscal 2019?")
- asks for a company you didn't ingest
- asks for something 10-Ks don't usually contain (CEO pay is in the proxy, not the 10-K;
  stock price targets; next quarter's guidance)
- asks for a breakdown the filing doesn't give (revenue by a city when only regions are given)

Also write ~5 questions with false premises ("Why did Microsoft's revenue fall in fiscal 2024?"
when it rose). Mark them unanswerable and note "false premise" in `notes`.

## 3. Dev / test split

Mark about 30 questions `"split": "dev"` (about 22 answerable, 8 unanswerable) and the rest
`"split": "test"`.

- Use only `--split dev` while you change anything (chunking, prompt, MIN_SIMILARITY, TOP_K).
- Run the test split once per configuration you report. If you change something after
  looking at test results, say so in RESULTS.md.

## 4. Fields

One JSON object per line:

```json
{"id": "q001", "question": "What was Apple's total net sales in fiscal 2024?", "answerable": true, "answer_type": "number", "gold_answer": "$391,035 million", "gold_sections": [{"ticker": "AAPL", "year": 2024, "section": "8"}], "gold_evidence": "Total net sales", "filters": {}, "split": "test", "notes": "income statement"}
```

(The number above shows the format only. Check every gold answer against the filing yourself.)

- `gold_answer`: write it the way the filing states it, with units. For numbers the scorer
  checks that every number in your gold answer appears in the model's answer (unit and
  rounding differences are allowed: "$1,587.1 million" matches "$1,587.1 million" and "$1.6
  billion" gold accepts "$1,587.1 million"). Don't put extra numbers in the gold answer that
  the model doesn't need to say.
- `gold_sections`: `ticker`, fiscal `year`, and `section` = the Item number as printed ("1",
  "1A", "7", "8"). Comparisons need one entry per side.
- `gold_evidence` (optional but helps citation scoring): a short phrase copied from the
  passage that holds the answer, e.g. the table row label "Total net sales".
- `filters` (optional): `{"company": "AAPL", "year": 2024}` if the question should be asked
  with the UI filters set. Leave it empty for most questions so retrieval is tested too.
- Unanswerable: `"answerable": false`, `"answer_type": "unanswerable"`, `"gold_answer": null`,
  `"gold_sections": []`.

Check the file:

```bash
uv run doc-copilot eval validate
```

It reports errors (must fix) and warnings (fewer than 120 questions, a ratio far from 3:1, no
dev split, duplicate question text).

## 5. Running it

```bash
# Free: prints projected tokens and dollars, calls nothing
uv run doc-copilot eval run --dry-run --judge
# Paid: needs LLM_PROVIDER=anthropic and --yes
uv run --env-file .env doc-copilot eval run --judge --yes
# Before/after for the improvement (runs both chunking strategies on the same questions)
uv run --env-file .env doc-copilot eval run --compare baseline,improved --judge --yes
```

Each run writes `evals/runs/<timestamp>/<label>/` with `summary.json`, `results.jsonl` (one
line per question) and `judge_spotcheck.csv`.

## 6. Judge spot check (at least 30 answers)

The LLM judge grades paraphrased answers that exact matching can't. You check the judge.

1. Open `judge_spotcheck.csv` from a full run (30 random answered questions).
2. For each row, read the question, gold answer and model answer, and open the cited chunk if
   needed (`uv run doc-copilot search ...` or the UI). Fill `human_correct` with `yes` or `no`
   **before** looking at `judge_correct`. Cover that column if you need to.
3. Run `uv run doc-copilot eval agreement evals/runs/<timestamp>/<label>/judge_spotcheck.csv`.
   It prints raw agreement and Cohen's kappa (agreement corrected for chance).
4. Report both in RESULTS.md. If agreement is under ~85%, read the disagreements, fix the
   judge prompt in `src/doc_copilot/evals/scoring.py` on dev questions, and re-check.

## 7. Cost

With about 120 questions, a few runs with the judge, and the comparison run, the dry run will
show the projected spend. Keep the whole project under about $15: run `--dry-run` first, use
`--split dev` while iterating, and do full runs only for numbers you will report.
