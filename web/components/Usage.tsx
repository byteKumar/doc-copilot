"use client";

import { useEffect, useState } from "react";
import { Usage as UsageData, api } from "@/lib/api";
import AccessGate from "@/components/AccessGate";

const ms = (v: number | null) => (v == null ? "n/a" : `${Math.round(v)} ms`);
const usd = (v: number | null, d = 4) => (v == null ? "n/a" : `$${v.toFixed(d)}`);
const pct = (v: number | null) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function UsageView() {
  const [data, setData] = useState<UsageData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.usage().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="empty">Loading...</p>;
  return (
    <>
      <h2>Usage</h2>
      <div className="usage-grid">
        <Stat label="Queries (all time)" value={String(data.queries_total)} />
        <Stat label="Queries (24 h)" value={String(data.queries_24h)} />
        <Stat label="LLM calls today / cap" value={`${data.llm_calls_24h} / ${data.daily_cap}`} />
        <Stat label="Refusal rate" value={pct(data.refusal_rate)} />
        <Stat label="Latency p50" value={ms(data.latency_p50_ms)} />
        <Stat label="Latency p95" value={ms(data.latency_p95_ms)} />
        <Stat label="Total cost" value={usd(data.cost_total_usd)} />
        <Stat label="Cost per 1,000 queries" value={usd(data.cost_per_1000_queries_usd)} />
        <Stat label="Feedback up / down" value={`${data.feedback_up} / ${data.feedback_down}`} />
      </div>
      <h3>Last 7 days</h3>
      <table className="recent">
        <thead>
          <tr>
            <th>Day</th>
            <th>Queries</th>
            <th>Cost</th>
            <th>p50</th>
          </tr>
        </thead>
        <tbody>
          {data.by_day.map((d) => (
            <tr key={d.day}>
              <td>{d.day}</td>
              <td className="num">{d.queries}</td>
              <td className="num">{usd(d.cost_usd)}</td>
              <td className="num">{ms(d.latency_p50_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>Recent questions</h3>
      <table className="recent">
        <thead>
          <tr>
            <th>When</th>
            <th>Question</th>
            <th>Result</th>
            <th>Latency</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {data.recent.map((r) => (
            <tr key={r.at + r.question}>
              <td>{new Date(r.at).toLocaleString()}</td>
              <td>{r.question}</td>
              <td>{r.refused ? "refused" : "answered"}</td>
              <td className="num">{ms(r.latency_ms)}</td>
              <td className="num">{usd(r.cost_usd, 5)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export default function Usage() {
  return (
    <AccessGate>
      <UsageView />
    </AccessGate>
  );
}
