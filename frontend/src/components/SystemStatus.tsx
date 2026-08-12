import { useCallback, useEffect, useState } from "react";
import { api } from "../services/api";

type State = "loading" | "ready" | "unavailable";
const modules = ["models", "documents", "retrieval", "rag"];

export function SystemStatus() {
  const [states, setStates] = useState<Record<string, State>>({ backend: "loading" });
  const refresh = useCallback(async () => {
    setStates({ backend: "loading" });
    const health = await api.health().then(() => true).catch(() => false);
    if (!health) { setStates({ backend: "unavailable" }); return; }
    setStates({ backend: "ready", ...Object.fromEntries(modules.map((name) => [name, "loading"])) });
    const results = await Promise.allSettled(modules.map((name) => api.status(name)));
    setStates((current) => ({ ...current, ...Object.fromEntries(results.map((result, index) => [modules[index], result.status === "fulfilled" && result.value.status === "available" ? "ready" : "unavailable"])) }));
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  const labels: Record<string, string> = { backend: "Backend", models: "Models", documents: "Documents", retrieval: "Retrieval", rag: "RAG" };
  return <section className="status-bar" aria-label="系統狀態">
    <div><p className="eyebrow">SYSTEM HEALTH</p><h2>系統狀態</h2></div>
    <div className="status-items">{Object.entries(labels).map(([key, label]) => <div className="status-item" key={key}><span className={`dot ${states[key] ?? "loading"}`} /><span><small>{label}</small><strong>{states[key] === "ready" ? "Ready" : states[key] === "unavailable" ? "Unavailable" : "Loading"}</strong></span></div>)}</div>
    <button className="button ghost compact" onClick={() => void refresh()}>重新檢查</button>
  </section>;
}
