"""檢索（retrieval）子系統。

Phase 4 完成的部分：以語意相似度從 vector store 取回相關文件片段。

    query 文字 -> EmbeddingBackend.embed_query -> 查詢向量
              -> VectorStore.search_by_vector  -> VectorMatch
              -> VectorRetriever               -> RetrievalResult（已排序）

各模組職責：
  - vector_retriever  檢索主體：驗證輸入、向量化、查詢、轉換與排序
  - models            RetrievalResult，專案層級的檢索結果契約
  - exceptions        自訂例外，讓上層依錯誤種類決定 HTTP 狀態碼
  - dependencies      FastAPI DI，共用 Phase 3 的 embedding backend

尚未實作：hybrid search、reranker、LLM 答案生成（Phase 5 以後）。
"""

from app.retrieval.exceptions import (
    RetrievalBackendError,
    RetrievalError,
    RetrievalValidationError,
)
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever

__all__ = [
    "RetrievalBackendError",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalValidationError",
    "VectorRetriever",
]
