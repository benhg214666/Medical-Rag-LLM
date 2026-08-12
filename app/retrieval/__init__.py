"""檢索（retrieval）子系統。

Phase 4 完成的部分：擴大語意候選池，再以通用精確匹配訊號輕量重排。

    query 文字 -> EmbeddingBackend.embed_query -> 查詢向量
              -> VectorStore.search_by_vector  -> VectorMatch
              -> VectorRetriever -> RetrievalPipeline -> RetrievalResult（已排序）

各模組職責：
  - vector_retriever  第一階段：驗證輸入、向量化、候選查詢與轉換
  - reranker          第二階段：日期、字詞與查詢詞精確匹配的確定性重排
  - pipeline          擴大候選池並將結果截回呼叫端要求的 top_k
  - models            RetrievalResult，專案層級的檢索結果契約
  - exceptions        自訂例外，讓上層依錯誤種類決定 HTTP 狀態碼
  - dependencies      FastAPI DI，共用 Phase 3 的 embedding backend

Hybrid search 仍未實作；RAG 答案生成由 app.rag 負責。
"""

from app.retrieval.exceptions import (
    RetrievalBackendError,
    RetrievalError,
    RetrievalValidationError,
)
from app.retrieval.models import RetrievalResult
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import LightweightReranker
from app.retrieval.vector_retriever import VectorRetriever

__all__ = [
    "RetrievalBackendError",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalPipeline",
    "RetrievalValidationError",
    "VectorRetriever",
    "LightweightReranker",
]
