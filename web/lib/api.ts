export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

export type Citation = {
  id: string;
  company: string;
  ticker: string;
  year: number;
  section: string;
  section_title: string;
  snippet: string;
};

export type AnswerResponse = {
  query_id: string;
  question: string;
  answer: string;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string | null;
  latency_ms: number;
  cost_usd: number;
  llm_called: boolean;
};

export type ChunkDetail = {
  id: string;
  company: string;
  ticker: string;
  year: number;
  section: string;
  section_title: string;
  kind: string;
  text: string;
  source_url: string | null;
};

export type FilterOptions = {
  companies: { ticker: string; name: string; years: number[] }[];
  years: number[];
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  filters: () => request<FilterOptions>("/filters"),
  ask: (question: string, company?: string, year?: number) =>
    request<AnswerResponse>("/ask", {
      method: "POST",
      body: JSON.stringify({ question, company: company || null, year: year || null }),
    }),
  chunk: (id: string) => request<ChunkDetail>(`/chunks/${encodeURIComponent(id)}`),
  feedback: (query_id: string, rating: 1 | -1) =>
    request<{ ok: boolean }>("/feedback", { method: "POST", body: JSON.stringify({ query_id, rating }) }),
};
