"""Retrieval 子系統的自訂例外。

與 app/ingestion/exceptions.py 相同的設計理由：集中定義例外種類，
讓上層（API router）能依「錯誤類型」決定 HTTP 狀態碼，
而不需要去解析錯誤訊息字串。

分成兩類的用意：
  - RetrievalValidationError 是「呼叫端給的輸入不合法」→ HTTP 400
  - RetrievalBackendError 是「後端元件出錯」→ HTTP 500
上層若只想攔截所有檢索相關錯誤，捕捉 RetrievalError 即可。
"""


class RetrievalError(RuntimeError):
    """所有 retrieval 相關錯誤的基底類別。

    繼承 RuntimeError 是為了與 Phase 3 的 EmbeddingError、
    VectorStoreError 保持一致的例外風格。
    """


class RetrievalValidationError(RetrievalError):
    """查詢輸入不合法，例如空白 query 或超出範圍的 top_k。"""


class RetrievalBackendError(RetrievalError):
    """embedding 或 vector store 在檢索過程中失敗。"""
