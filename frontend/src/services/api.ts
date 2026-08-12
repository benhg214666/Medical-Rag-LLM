import type { DocumentUploadResponse, IndexResponse, ModuleStatus, RagResponse } from "../types/api";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const TIMEOUT_MS = 65_000;

export class ApiError extends Error {
  constructor(message: string, public status?: number) { super(message); }
}

async function request<T>(path: string, init?: RequestInit, timeout = TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = body && typeof body === "object" && "detail" in body ? String(body.detail) : `請求失敗 (${response.status})`;
      throw new ApiError(detail, response.status);
    }
    if (!body || typeof body !== "object") throw new ApiError("伺服器回傳了無效的資料格式。");
    return body as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") throw new ApiError("請求逾時，請確認後端服務狀態後再試。");
    throw new ApiError("無法連線至後端服務。");
  } finally { window.clearTimeout(timer); }
}

export const api = {
  health: () => request<{ status: string }>("/health", undefined, 8_000),
  status: (module: string) => request<ModuleStatus>(`/api/${module}/status`, undefined, 8_000),
  upload: (file: File) => {
    const data = new FormData();
    data.append("file", file);
    return request<DocumentUploadResponse>("/api/documents/upload", { method: "POST", body: data });
  },
  index: (id: string) => request<IndexResponse>(`/api/documents/${encodeURIComponent(id)}/index`, { method: "POST" }),
  ask: (query: string) => request<RagResponse>("/api/rag/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  }),
};
