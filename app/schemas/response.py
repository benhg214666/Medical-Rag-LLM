"""API 回應（response）資料結構。

這些 model 定義的是「對外契約」—— FastAPI 會依此產生 OpenAPI 文件，
也會在回傳時做驗證。刻意與 ingestion 內部的資料模型分離：
內部模型（DocumentChunk）含有完整文字，對外只回傳預覽，避免大量醫療內容經由 HTTP 外流。
"""

from typing import Any

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """GET / 的回應內容。"""

    project: str
    version: str
    status: str


class HealthResponse(BaseModel):
    """GET /health 的回應內容。"""

    status: str


class ModuleStatusResponse(BaseModel):
    """各功能模組 status endpoint 的回應內容。"""

    module: str
    status: str


class ChunkPreview(BaseModel):
    """單一 chunk 的預覽資訊。

    只含截斷後的文字片段，不含完整內容。用途是讓上傳者確認切塊結果合理，
    而不是傳遞資料 —— 完整內容請讀取 processed JSON。
    """

    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    paragraph_number: int | None = None
    text_preview: str


class IngestionStatistics(BaseModel):
    """一次匯入的摘要統計。"""

    loaded_units: int
    cleaned_units: int
    chunk_count: int
    total_characters: int


class DocumentUploadResponse(BaseModel):
    """POST /api/documents/upload 成功時的回應內容。

    output_file 一律為相對路徑（例如 data/processed/xxx.json），
    不含主機絕對路徑，避免洩漏伺服器目錄結構。
    """

    status: str
    document_id: str
    file_name: str
    file_type: str
    statistics: IngestionStatistics
    output_file: str
    chunk_previews: list[ChunkPreview] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """錯誤回應的統一格式。

    只包含可安全對外揭露的訊息；stack trace 與絕對路徑僅記錄在伺服器 log。
    """

    detail: str


class IndexDocumentResponse(BaseModel):
    """POST /api/documents/{document_id}/index 的安全摘要。"""

    status: str
    document_id: str
    collection_name: str
    indexed_chunks: int
    embedding_model: str
    embedding_dimension: int


class RetrievalResultItem(BaseModel):
    """單筆檢索結果的對外表示。

    與內部的 RetrievalResult 欄位相同，但刻意分開定義：
    這是對外契約，不應因內部模型調整而被動改變。

    這裡**不包含 embedding 向量**，也不包含任何 vector store 內部欄位。
    向量對使用者沒有意義，回傳它只會放大 payload 並洩漏實作細節。

    關於 distance 與 score：
        distance 越小越相似（cosine 時範圍 [0, 2]）。
        score 越大越相似，僅在 metric 為 cosine 時提供，
        其值即為 cosine similarity；其他 metric 為 null，
        此時請以 distance 排序。
    """

    chunk_id: str
    document_id: str | None = None
    text: str
    distance: float
    score: float | None = None
    distance_metric: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalSearchResponse(BaseModel):
    """POST /api/retrieval/search 的回應內容。

    results 已由最相關排到最不相關。查無結果時 results 為空陣列、
    result_count 為 0，這是正常回應而非錯誤 —— 向量資料庫沒有資料，
    或沒有語意相近的片段，都屬於合理狀態。

    Attributes:
        query: 回傳呼叫端送出的查詢，方便非同步情境對應請求。
        top_k: 本次實際生效的 top_k（未指定時為伺服器預設值）。
        result_count: results 的筆數。
        results: 已排序的檢索結果。
    """

    query: str
    top_k: int
    result_count: int
    results: list[RetrievalResultItem] = Field(default_factory=list)
