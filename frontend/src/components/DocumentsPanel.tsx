import { useRef, useState } from "react";
import { ApiError, api } from "../services/api";
import type { SessionDocument } from "../types/api";

interface Props { documents: SessionDocument[]; setDocuments: React.Dispatch<React.SetStateAction<SessionDocument[]>> }
const allowed = [".pdf", ".docx", ".txt"];

export function DocumentsPanel({ documents, setDocuments }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File>();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const choose = (selected?: File) => {
    setError("");
    if (!selected) return setFile(undefined);
    if (!allowed.some((ext) => selected.name.toLowerCase().endsWith(ext))) { setFile(undefined); setError("僅支援 PDF、DOCX 或 TXT 文件。"); return; }
    setFile(selected);
  };
  const upload = async () => {
    if (!file) return;
    setUploading(true); setError("");
    try {
      const result = await api.upload(file);
      setDocuments((items) => [{ ...result, indexState: "uploaded" }, ...items.filter((item) => item.document_id !== result.document_id)]);
      setFile(undefined); if (input.current) input.current.value = "";
    } catch (e) { setError(e instanceof ApiError ? e.message : "文件上傳失敗。"); }
    finally { setUploading(false); }
  };
  const index = async (id: string) => {
    setDocuments((items) => items.map((item) => item.document_id === id ? { ...item, indexState: "indexing", indexError: undefined } : item));
    try {
      const result = await api.index(id);
      setDocuments((items) => items.map((item) => item.document_id === id ? { ...item, indexState: "indexed", indexResult: result } : item));
    } catch (e) {
      const message = e instanceof ApiError ? e.message : "建立索引失敗。";
      setDocuments((items) => items.map((item) => item.document_id === id ? { ...item, indexState: "error", indexError: message } : item));
    }
  };
  return <aside className="panel documents-panel">
    <div className="section-heading"><div><p className="eyebrow">KNOWLEDGE BASE</p><h2>文件管理</h2></div><span className="count">{documents.length}</span></div>
    <div className="upload-box">
      <label htmlFor="document-file">選擇醫療文件</label>
      <input ref={input} id="document-file" type="file" accept=".pdf,.docx,.txt" onChange={(e) => choose(e.target.files?.[0])} />
      <p className="file-name">{file ? file.name : "尚未選擇文件"}</p><small>PDF / DOCX / TXT</small>
      <button className="button primary full" disabled={!file || uploading} onClick={() => void upload()}>{uploading ? "處理文件中…" : "上傳文件"}</button>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
    <div className="document-list">
      {documents.length === 0 && <div className="empty"><span>＋</span><p>本次工作階段尚無文件</p></div>}
      {documents.map((doc) => <article className="document-card" key={doc.document_id}>
        <div className="document-top"><div className="file-icon">{doc.file_type.replace(".", "").toUpperCase()}</div><div><h3>{doc.file_name}</h3><code>{doc.document_id.slice(0, 12)}…</code></div></div>
        <div className="document-meta"><span>{doc.statistics.chunk_count} 個片段</span><span className={`pill ${doc.indexState}`}>{doc.indexState === "indexed" ? "已索引" : doc.indexState === "indexing" ? "索引中" : doc.indexState === "error" ? "索引失敗" : "已上傳"}</span></div>
        {doc.indexError && <p className="error" role="alert">{doc.indexError}</p>}
        {doc.indexState !== "indexed" && <button className="button secondary full" disabled={doc.indexState === "indexing"} onClick={() => void index(doc.document_id)}>{doc.indexState === "indexing" ? "建立索引中…" : doc.indexState === "error" ? "重試索引" : "建立索引"}</button>}
      </article>)}
    </div>
  </aside>;
}
