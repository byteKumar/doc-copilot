# doc-copilot: customer brief

_One page, written as for a client. Results are TBD until the eval runs; nothing below is a
measured claim unless it says so._

## The problem

Analysts spend hours searching long annual reports (10-Ks run 80-150 pages, much of it tables)
for specific facts: a segment's revenue, a risk disclosure, a year-over-year change. General
chatbots answer fluently but can't show where an answer came from, and they invent numbers.
For financial work an unsupported number is worse than no answer.

## What we built

A question-answering assistant over [N] companies' 10-K filings ([years]) that:

- answers only from the filings and cites the exact passages; one click opens the source text;
- says "I don't know" when the filings don't support an answer, instead of guessing;
- checks every answer automatically before showing it: citations must point at passages it was
  given, and every number in the answer must appear in a cited passage;
- lets users filter by company and year, and rate answers (thumbs up/down are stored for review);
- shows usage, latency and cost on a usage page, with an access code, rate limit and daily cap.

## Results

Measured on [N] questions written and checked by hand ([A] answerable, [U] deliberately
unanswerable), never used for tuning:

| | Result |
|---|---|
| Correct answers | TBD |
| Answers whose cited passage contains the answer | TBD |
| Unanswerable questions correctly declined | TBD |
| Answerable questions wrongly declined | TBD |
| Response time (median / 95th percentile) | TBD |
| Cost per 1,000 questions | TBD |
| Improvement from table-aware processing | TBD (before -> after) |

The automated grader agreed with a human reviewer on TBD of 30 spot-checked answers.

## Limits

- Covers only the filings loaded; it does not know news, prices or later filings.
- Questions that need arithmetic across many tables (e.g. five-year growth rates) are weaker
  than direct lookups. [Confirm with the eval breakdown.]
- The automatic check catches invented numbers and bad citations; it does not prove an answer is
  complete. Wrongly declined questions are the price of that caution: TBD%.
- Demo access is a shared code, not individual accounts.

## Next steps

1. Load the client's own document set and write 50+ questions with their analysts.
2. Add re-ranking if retrieval misses are the main error type in the eval breakdown.
3. Review thumbs-down answers weekly and add them to the eval set.
4. Single sign-on and per-user audit logs before wider rollout.
