"""Command line entry point: `uv run doc-copilot <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .db import connection, init_db


def cmd_init_db(args, s: Settings) -> int:
    init_db(s.database_url, s.embedding_dim)
    print("schema ready")
    return 0


def cmd_download(args, s: Settings) -> int:
    from .ingest.download import EdgarClient, UserAgentError, load_corpus

    try:
        client = EdgarClient(s.sec_user_agent)
    except UserAgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    pairs = load_corpus(Path(args.corpus))
    out = Path(args.out)
    for ticker, year in pairs:
        try:
            path = client.download(ticker, year, out)
            print(f"ok   {ticker} {year} -> {path}")
        except Exception as e:  # keep going; report every failure
            print(f"FAIL {ticker} {year}: {e}", file=sys.stderr)
    return 0


def cmd_ingest(args, s: Settings) -> int:
    from .embed import make_embedder
    from .ingest.chunk import STRATEGIES
    from .ingest.pipeline import ingest_dir

    strategies = STRATEGIES if args.chunking == "all" else (args.chunking,)
    init_db(s.database_url, s.embedding_dim)
    embedder = make_embedder(s)
    with connection(s.database_url) as conn:
        stats = ingest_dir(
            conn, Path(args.dir), embedder, strategies=strategies,
            chunk_words=s.chunk_words, overlap_words=s.chunk_overlap_words,
        )
    print(json.dumps(stats.__dict__, indent=2))
    return 0


def cmd_search(args, s: Settings) -> int:
    from .embed import make_embedder
    from .retrieve import Filters, retrieve

    with connection(s.database_url) as conn:
        r = retrieve(
            conn, make_embedder(s), args.question, Filters(args.company, args.year),
            chunking=args.chunking or s.chunking, top_k=args.k,
            candidates=s.candidates_per_retriever, rrf_k=s.rrf_k,
        )
    print(f"best_similarity={r.best_similarity:.3f} lexical_matches={r.lexical_matches} "
          f"latency_ms={r.latency_ms:.1f}")
    for h in r.hits:
        print(f"{h.rrf_score:.4f} vec#{h.vector_rank} lex#{h.lexical_rank} {h.id} "
              f"[{h.section_title[:40]}] {h.text[:100]!r}")
    return 0


def cmd_ask(args, s: Settings) -> int:
    from .answer import Answerer
    from .embed import make_embedder
    from .llm import make_llm
    from .retrieve import Filters

    if s.llm_provider == "fake":
        print("note: LLM_PROVIDER=fake (offline extractive stand-in, not Claude)", file=sys.stderr)
    answerer = Answerer(s, make_embedder(s), make_llm(s), chunking=args.chunking)
    with connection(s.database_url) as conn:
        r = answerer.answer(conn, args.question, Filters(args.company, args.year))
    if args.json:
        print(json.dumps(r.to_dict(), indent=2, default=str))
        return 0
    print(r.answer)
    for c in r.citations:
        print(f"  [{c.id}] {c.company} {c.year} Item {c.section}. {c.section_title}")
    if r.refused:
        print(f"  (refused: {r.refusal_reason})")
    print(f"  latency {r.latency_ms:.0f} ms, tokens {r.input_tokens}/{r.output_tokens}, ${r.cost_usd:.6f}")
    return 0


CHUNKING_ALIASES = {"improved": "table_aware"}  # label -> chunking strategy


def cmd_eval_validate(args, s: Settings) -> int:
    from .evals.dataset import load_questions

    rep = load_questions(Path(args.questions))
    for e in rep.errors:
        print(f"ERROR   {e}")
    for w in rep.warnings:
        print(f"WARNING {w}")
    real = [q for q in rep.questions if not q.format_example]
    a = sum(q.answerable for q in real)
    print(f"{len(rep.questions)} questions parsed: {a} answerable, {len(real) - a} unanswerable, "
          f"{sum(q.split == 'dev' for q in real)} dev / {sum(q.split == 'test' for q in real)} test")
    return 0 if rep.ok else 2


def cmd_eval_run(args, s: Settings) -> int:
    from .answer import Answerer
    from .embed import make_embedder
    from .evals.dataset import is_format_examples, load_questions, select_subset
    from .evals.harness import (
        estimate_cost,
        flips,
        render_table,
        run_id,
        run_questions,
        summarize,
        write_run,
    )
    from .llm import make_llm

    path = Path(args.questions)
    rep = load_questions(path)
    if not rep.ok:
        for e in rep.errors:
            print(f"ERROR {e}", file=sys.stderr)
        return 2
    for w in rep.warnings:
        print(f"WARNING {w}", file=sys.stderr)
    smoke = is_format_examples(path, rep.questions)
    if smoke and not args.smoke:
        print("Refusing: this file holds FORMAT EXAMPLES, not the eval set. Headline numbers come only "
              "from evals/questions.jsonl that you wrote. Use --smoke to check the plumbing.", file=sys.stderr)
        return 2

    qs = [q for q in rep.questions if args.split == "all" or q.split == args.split]
    if args.subset:
        qs = select_subset(qs, args.subset)
    labels = args.compare.split(",") if args.compare else [args.chunking or s.chunking]
    chunkings = [CHUNKING_ALIASES.get(x, x) for x in labels]
    embedder = make_embedder(s)

    with connection(s.database_url) as conn:
        est = estimate_cost(conn, qs, s, embedder, chunkings, with_judge=args.judge)
        print(f"Plan: {len(qs)} questions x {len(chunkings)} config(s), model {s.llm_model} "
              f"(provider {s.llm_provider})")
        print(f"Estimate: {est.llm_calls} LLM calls, ~{est.input_tokens:,} input + ~{est.output_tokens:,} "
              f"output tokens = ~${est.usd:.4f} at ${s.price_input_per_mtok}/${s.price_output_per_mtok} "
              f"per 1M tokens ({est.refused_before_llm} refused before the LLM by the similarity floor)")
        if args.dry_run:
            return 0
        if s.llm_provider != "fake" and not args.yes:
            print("This run calls a paid API. Re-run with --yes to proceed.", file=sys.stderr)
            return 3
        if s.llm_provider == "fake":
            print("NOTE: LLM_PROVIDER=fake. These numbers test the plumbing; they are not results.")

        llm = make_llm(s)
        out_root = Path(args.out) / run_id()
        summaries, all_results = [], []
        for label, chunking in zip(labels, chunkings, strict=True):
            answerer = Answerer(s, embedder, llm, chunking=chunking)
            results, judge_cost = run_questions(
                conn, qs, answerer, s, judge_llm=llm if args.judge else None, progress=args.verbose,
            )
            all_results.append(results)
            if smoke:
                for r in results:
                    state = "REFUSED" if r.refused else "answered"
                    print(f"  {r.id} [{state}] {r.model_answer[:90]!r} cites={r.citations}")
                continue
            summ = summarize(results, label, chunking, llm.model, judge_cost, args.judge, rep.warnings)
            summaries.append(summ)
            write_run(out_root / label, summ, results, spotcheck=args.judge)

    if smoke:
        print("SMOKE RUN on format examples: no metrics reported.")
        return 0
    print(render_table(summaries))
    if len(all_results) == 2:
        changed = flips(all_results[0], all_results[1])
        print(f"\n{len(changed)} answerable questions changed between {labels[0]} and {labels[1]}:")
        for c in changed:
            print("  " + c)
    print(f"\nResults written to {out_root}/")
    if args.judge:
        print("Fill the human_correct column in judge_spotcheck.csv, then run: "
              "doc-copilot eval agreement <csv>")
    return 0


def cmd_eval_agreement(args, s: Settings) -> int:
    from .evals.scoring import agreement_from_csv

    a = agreement_from_csv(Path(args.csv))
    if a.n == 0:
        print(f"No rows with both human_correct and judge_correct filled ({a.skipped} skipped).")
        return 2
    kappa = "n/a" if a.kappa is None else f"{a.kappa:.2f}"
    print(f"Spot-checked {a.n} answers; agreement with the LLM judge {a.agree}/{a.n} = {a.rate:.1%}; "
          f"Cohen's kappa {kappa}; {a.skipped} rows skipped")
    if a.n < 30:
        print("WARNING: the spec asks for at least 30 spot-checked answers.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="doc-copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="create tables and indexes").set_defaults(fn=cmd_init_db)

    d = sub.add_parser("download", help="download 10-Ks from SEC EDGAR (needs SEC_USER_AGENT)")
    d.add_argument("--corpus", default="corpus.toml")
    d.add_argument("--out", default="data/filings")
    d.set_defaults(fn=cmd_download)

    i = sub.add_parser("ingest", help="parse, chunk, embed and store filings")
    i.add_argument("--dir", default="data/filings")
    i.add_argument("--chunking", default="all", choices=["all", "baseline", "table_aware"])
    i.set_defaults(fn=cmd_ingest)

    q = sub.add_parser("search", help="hybrid retrieval only (no LLM)")
    q.add_argument("question")
    q.add_argument("--company")
    q.add_argument("--year", type=int)
    q.add_argument("--chunking")
    q.add_argument("-k", type=int, default=8)
    q.set_defaults(fn=cmd_search)

    a = sub.add_parser("ask", help="answer a question with citations")
    a.add_argument("question")
    a.add_argument("--company")
    a.add_argument("--year", type=int)
    a.add_argument("--chunking")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)

    ev = sub.add_parser("eval", help="eval set tools").add_subparsers(dest="eval_cmd", required=True)
    v = ev.add_parser("validate", help="check an eval JSONL file")
    v.add_argument("--questions", default="evals/questions.jsonl")
    v.set_defaults(fn=cmd_eval_validate)

    r = ev.add_parser("run", help="run and score the eval set")
    r.add_argument("--questions", default="evals/questions.jsonl")
    r.add_argument("--split", default="all", choices=["all", "dev", "test"])
    r.add_argument("--subset", type=int, help="deterministic stratified subset (CI uses 20)")
    r.add_argument("--chunking", help="baseline | table_aware | improved")
    r.add_argument("--compare", help="e.g. baseline,improved")
    r.add_argument("--judge", action="store_true", help="also grade with the LLM judge")
    r.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    r.add_argument("--yes", action="store_true", help="confirm a paid run")
    r.add_argument("--smoke", action="store_true", help="allow FORMAT_EXAMPLES (no metrics)")
    r.add_argument("--out", default="evals/runs")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_eval_run)

    g = ev.add_parser("agreement", help="human vs judge agreement from a filled spot-check CSV")
    g.add_argument("csv")
    g.set_defaults(fn=cmd_eval_agreement)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
