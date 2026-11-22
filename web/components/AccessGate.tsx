"use client";

import { FormEvent, ReactNode, useEffect, useState } from "react";
import { ApiError, api, setAccessCode } from "@/lib/api";

type State = "checking" | "ok" | "locked" | "down";

function stateFor(err: unknown): State {
  return err instanceof ApiError && err.status === 401 ? "locked" : "down";
}

export default function AccessGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>("checking");
  const [code, setCode] = useState("");
  const [wrong, setWrong] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .check()
      .then(() => !cancelled && setState("ok"))
      .catch((err) => !cancelled && setState(stateFor(err)));
    return () => {
      cancelled = true;
    };
  }, []);

  async function retry() {
    try {
      await api.check();
      setState("ok");
    } catch (err) {
      setState(stateFor(err));
      setWrong(true);
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setAccessCode(code.trim());
    await retry();
  }

  if (state === "ok") return <>{children}</>;
  if (state === "checking") return <p className="empty">Connecting...</p>;
  if (state === "down")
    return (
      <div className="gate">
        <p>The API is not reachable. Is it running?</p>
        <button type="button" onClick={retry}>
          Retry
        </button>
      </div>
    );
  return (
    <form className="gate" onSubmit={submit}>
      <label htmlFor="code">This demo needs an access code.</label>
      <input id="code" type="password" value={code} onChange={(e) => setCode(e.target.value)} autoFocus />
      {wrong && <span className="reason">That code did not work.</span>}
      <button type="submit">Enter</button>
    </form>
  );
}
