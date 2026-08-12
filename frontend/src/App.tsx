import { useState } from "react";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { QuestionPanel } from "./components/QuestionPanel";
import { SystemStatus } from "./components/SystemStatus";
import type { SessionDocument } from "./types/api";

export default function App() {
  const [documents, setDocuments] = useState<SessionDocument[]>([]);
  return <><header className="app-header"><div className="brand-mark">M</div><div><h1>Medical RAG</h1><p>Clinical Knowledge Assistant</p></div><div className="privacy">LOCAL · PRIVATE</div></header><div className="shell"><SystemStatus /><div className="workspace"><DocumentsPanel documents={documents} setDocuments={setDocuments} /><QuestionPanel /></div><footer>回答僅依據已索引文件生成，不能取代專業醫療判斷。</footer></div></>;
}
