"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { AnswerResponse, ApiError, ChunkDetail, FilterOptions, api } from "@/lib/api";

type Turn =
  | { kind: "user"; text: string }
  | { kind: "answer"; data: AnswerResponse; vote?: 1 | -1 }
  | { kind: "error"; text: string };

const EXAMPLES = [
  "What was total revenue in the most recent fiscal year?",
  "What are the main risk factors related to suppliers?",
  "How many full-time employees does the company have?",
];

export default function Chat() {
  const [filters, setFilters] = useState<FilterOptions | null>(null);
  const [company, setCompany] = useState("");
  const [year, setYear] = useState("");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<ChunkDetail | null>(null);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.filters().then(setFilters).catch(() => setFilters({ companies: [], years: [] }));
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const years = company
    ? filters?.companies.find((c) => c.ticker === company)?.years ?? []
    : filters?.years ?? [];

  async function submit(e?: FormEvent, text?: string) {
    e?.preventDefault();
    const q = (text ?? question).trim();
    if (!q || busy) return;
    setQuestion("");
    setTurns((t) => [...t, { kind: "user", text: q }]);
    setBusy(true);
    try {
      const data = await api.ask(q, company || undefined, year ? Number(year) : undefined);
      setTurns((t) => [...t, { kind: "answer", data }]);
    } catch (err) {
      const msg =
        err instanceof ApiError && err.status === 429
          ? "Too many questions. Please wait a minute and try again."
          : `Request failed: ${err instanceof Error ? err.message : String(err)}`;
      setTurns((t) => [...t, { kind: "error", text: msg }]);
    } finally {
      setBusy(false);
    }
  }

  async function openSource(id: string) {
    setSourceError(null);
    setSource(null);
    try {
      setSource(await api.chunk(id));
    } catch (err) {
      setSourceError(err instanceof Error ? err.message : String(err));
    }
  }

  async function vote(index: number, rating: 1 | -1) {
    const turn = turns[index];
    if (turn.kind !== "answer" || turn.vote) return;
    try {
      await api.feedback(turn.data.query_id, rating);
      setTurns((t) => t.map((x, i) => (i === index && x.kind === "answer" ? { ...x, vote: rating } : x)));
    } catch {
      // Feedback is best effort; the answer is still useful without it.
    }
  }

  return (
    <div className={`layout ${source || sourceError ? "with-panel" : ""}`}>
      <section className="chat">
        <div className="filters">
          <label>
            Company
            <select
              value={company}
              onChange={(e) => {
                setCompany(e.target.value);
                setYear("");
              }}
            >
              <option value="">All companies</option>
              {filters?.companies.map((c) => (
                <option key={c.ticker} value={c.ticker}>
                  {c.name} ({c.ticker})
                </option>
              ))}
            </select>
          </label>
          <label>
            Fiscal year
            <select value={year} onChange={(e) => setYear(e.target.value)}>
              <option value="">All years</option>
              {years.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="turns">
          {turns.length === 0 && (
            <div className="empty">
              <p>Ask a question about the filings. Every answer cites the passages it used; click a citation to read the source.</p>
              <div className="examples">
                {EXAMPLES.map((ex) => (
                  <button key={ex} type="button" onClick={() => submit(undefined, ex)}>
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t, i) => {
            if (t.kind === "user") return <div key={i} className="turn user">{t.text}</div>;
            if (t.kind === "error") return <div key={i} className="turn error">{t.text}</div>;
            const a = t.data;
            return (
              <div key={i} className={`turn answer ${a.refused ? "refused" : ""}`}>
                <p>{a.answer}</p>
                {a.refused && a.refusal_reason && <p className="reason">Why: {a.refusal_reason}</p>}
                {a.citations.length > 0 && (
                  <div className="citations">
                    {a.citations.map((c, n) => (
                      <button key={c.id} type="button" className="cite" onClick={() => openSource(c.id)} title={c.snippet}>
                        [{n + 1}] {c.ticker} {c.year} · Item {c.section}
                      </button>
                    ))}
                  </div>
                )}
                <div className="meta">
                  <span>
                    {Math.round(a.latency_ms)} ms{a.llm_called ? ` · $${a.cost_usd.toFixed(5)}` : " · no LLM call"}
                  </span>
                  <span className="votes">
                    <button type="button" aria-label="Helpful" className={t.vote === 1 ? "on" : ""} disabled={!!t.vote} onClick={() => vote(i, 1)}>
                      Helpful
                    </button>
                    <button type="button" aria-label="Not helpful" className={t.vote === -1 ? "on" : ""} disabled={!!t.vote} onClick={() => vote(i, -1)}>
                      Not helpful
                    </button>
                  </span>
                </div>
              </div>
            );
          })}
          {busy && <div className="turn answer pending">Searching the filings…</div>}
          <div ref={bottom} />
        </div>

        <form className="ask" onSubmit={submit}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What was Apple's total net sales in fiscal 2024?"
            maxLength={1000}
            aria-label="Question"
          />
          <button type="submit" disabled={busy || !question.trim()}>
            Ask
          </button>
        </form>
      </section>

      {(source || sourceError) && (
        <aside className="panel">
          <div className="panel-head">
            <strong>Source passage</strong>
            <button type="button" onClick={() => { setSource(null); setSourceError(null); }} aria-label="Close">
              ×
            </button>
          </div>
          {sourceError && <p className="error">{sourceError}</p>}
          {source && (
            <>
              <p className="src-meta">
                {source.company} · FY{source.year} 10-K · Item {source.section}. {source.section_title}
                <br />
                <code>{source.id}</code>
                {source.source_url && (
                  <>
                    {" · "}
                    <a href={source.source_url} target="_blank" rel="noreferrer">
                      open on sec.gov
                    </a>
                  </>
                )}
              </p>
              <pre className="src-text">{source.text}</pre>
            </>
          )}
        </aside>
      )}
    </div>
  );
}
