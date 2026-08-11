"""API 請求（request）資料結構。

Phase 1 的 endpoint 皆為 GET，不接受參數；Phase 2 的
POST /api/documents/upload 使用 multipart/form-data，由 FastAPI 的
UploadFile 直接處理，兩者都不需要 Pydantic request model。

Phase 4 起才出現第一個 JSON request body：檢索查詢。
"""

from pydantic import BaseModel, Field, StrictInt


class RetrievalSearchRequest(BaseModel):
    """POST /api/retrieval/search 的請求內容。

    這裡只做「型別與範圍」層級的驗證，例如 top_k 必須是正整數；
    實際的業務規則（例如上限是多少、空白查詢如何處理）
    仍由 Retriever 依設定判斷，避免把設定值寫死在 schema 裡。

    Attributes:
        query: 使用者的自然語言問題，不可為空白。
        top_k: 最多回傳幾筆結果；未提供時採用 RETRIEVAL_TOP_K 設定值。
    """

    query: str = Field(
        ...,
        min_length=1,
        description="自然語言查詢內容，不可為空白。",
    )
    # 使用 StrictInt 而非 int：Pydantic 在寬鬆模式下會做型別轉換，
    # 使得 true -> 1、1.0 -> 1、"1" -> 1 都被接受。這會讓明顯錯誤的
    # 請求靜默通過（例如 top_k=true 變成只取一筆），呼叫端不會察覺。
    # StrictInt 要求必須是真正的 JSON integer，其餘一律回 422。
    top_k: StrictInt | None = Field(
        default=None,
        ge=1,
        description=(
            "最多回傳幾筆結果；必須是整數，未提供時採用伺服器設定的預設值。"
        ),
    )


class RAGAskRequest(BaseModel):
    """POST /api/rag/ask request."""

    query: str = Field(..., min_length=1, description="不可為空白的醫療紀錄問題。")
    top_k: StrictInt | None = Field(default=None, ge=1)
