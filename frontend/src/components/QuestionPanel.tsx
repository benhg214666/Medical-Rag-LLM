import { useState } from "react";
import { ApiError, api } from "../services/api";
import type { RagResponse } from "../types/api";

export function QuestionPanel() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<RagResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const ask = async () => {
    const query = question.trim();
    if (!query) { setError("請先輸入問題。"); return; }
    setLoading(true); setError(""); setAnswer(undefined);
    try {
      const result = await api.ask(query);
      if (typeof result.answer !== "string" || !Array.isArray(result.sources)) throw new ApiError("伺服器回傳了無效的回答格式。");
      setAnswer(result);
    } catch (e) { setError(e instanceof ApiError ? e.message : "目前無法取得回答。"); }
    finally { setLoading(false); }
  };
  return <main className="panel qa-panel">
    <div className="section-heading"><div><p className="eyebrow">GROUNDED Q&amp;A</p><h2>醫療紀錄問答</h2></div></div>
    <label htmlFor="question">針對已索引的文件提問</label>
    <div className="question-box"><textarea id="question" rows={4} value={question} onChange={(e) => { setQuestion(e.target.value); setError(""); }} onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") void ask(); }} placeholder="例如：病人目前的狀況是什麼？" /><div className="question-actions"><small>Ctrl / ⌘ + Enter 送出</small><button className="button primary" disabled={loading || !question.trim()} onClick={() => void ask()}>{loading ? "檢索與生成中…" : "送出問題 →"}</button></div></div>
    {error && <p className="error banner" role="alert">{error}</p>}
    {!answer && !loading && <div className="answer-placeholder"><div>✦</div><h3>答案會顯示在這裡</h3><p>先上傳並索引文件，再提出與內容相關的問題。</p></div>}
    {loading && <div className="answer-placeholder" aria-live="polite"><div className="spinner" /><h3>正在查找相關證據</h3><p>本機模型生成答案可能需要一些時間。</p></div>}
    {answer && <section className="answer" aria-live="polite"><div className="answer-header"><p className="eyebrow">ANSWER</p><span className="model">模型：{answer.model}</span></div><div className="answer-text">{answer.answer}</div><div className="sources-heading"><div><p className="eyebrow">EVIDENCE</p><h3>來源追溯</h3></div><span className="count">{answer.sources.length}</span></div>{answer.sources.length === 0 ? <p className="muted">此回答沒有附帶檢索來源。</p> : <div className="sources">{answer.sources.map((source) => {
      const filename = typeof source.metadata.file_name === "string" ? source.metadata.file_name : typeof source.metadata.source === "string" ? source.metadata.source : undefined;
      const page = source.metadata.page_number ?? source.metadata.page;
      return <details className="source-card" key={`${source.chunk_id}-${source.source_number}`}><summary><span className="source-number">{source.source_number}</span><span><strong>{filename || `來源片段 ${source.source_number}`}</strong><small>{page != null ? `第 ${String(page)} 頁 · ` : ""}{source.score != null ? `相關度 ${(source.score * 100).toFixed(1)}%` : `距離 ${source.distance.toFixed(3)}`}</small></span><span className="expand">＋</span></summary><div className="source-body"><p>{source.text}</p><dl><div><dt>Document ID</dt><dd>{source.document_id || "—"}</dd></div><div><dt>Chunk ID</dt><dd>{source.chunk_id}</dd></div><div><dt>距離指標</dt><dd>{source.distance_metric}</dd></div>{Object.entries(source.metadata).filter(([, value]) => value != null && typeof value !== "object").slice(0, 6).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>)}</dl></div></details>;
    })}</div>}</section>}
  </main>;
}
