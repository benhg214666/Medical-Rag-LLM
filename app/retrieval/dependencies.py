"""FastAPI 可覆寫的 Phase 4 dependency providers。

從 app.embeddings.dependencies 取得共享的 embedding backend，
而不是自行初始化，也不依賴 indexing 子系統的內部細節。相依方向為：

        app.embeddings.dependencies
              ↑                ↑
        app.indexing      app.retrieval

這保證 document 與 query 使用完全相同的模型、revision 與維度 ——
兩者若不在同一個 embedding space，檢索結果會靜默劣化而不報錯。
"""

from fastapi import Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingError
from app.embeddings.dependencies import get_embedding_backend_for
from app.retrieval.exceptions import RetrievalBackendError
from app.retrieval.vector_retriever import VectorRetriever
from app.vector_store.base import VectorStoreError
from app.vector_store.factory import create_vector_store


def get_vector_retriever(
    settings: Settings = Depends(get_settings),
) -> VectorRetriever:
    """建立 request-scoped retriever，共用 application-scoped embedding backend。

    建立時即驗證 embedding 相容性（見 VectorRetriever.ensure_ready），
    讓不相容的組態在第一個 request 就明確失敗，
    而不是回傳語意上錯誤但外觀正常的檢索結果。
    """
    try:
        retriever = VectorRetriever(
            embedding_backend=get_embedding_backend_for(settings),
            vector_store=create_vector_store(settings),
            default_top_k=settings.retrieval_top_k,
            max_top_k=settings.retrieval_max_top_k,
        )
        retriever.ensure_ready()
        return retriever
    except (
        EmbeddingError,
        VectorStoreError,
        RetrievalBackendError,
        ValueError,
    ) as exc:
        # 對外不揭露 collection 名稱、模型路徑或堆疊。
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器無法初始化本地檢索服務。",
        ) from exc
