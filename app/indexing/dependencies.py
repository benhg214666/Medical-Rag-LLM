"""FastAPI 可覆寫的 Phase 3 dependency providers。"""

from fastapi import Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingError
from app.embeddings.dependencies import get_embedding_backend_for
from app.indexing.pipeline import IndexingPipeline
from app.vector_store.base import VectorStoreError
from app.vector_store.factory import create_vector_store


def get_indexing_pipeline(
    settings: Settings = Depends(get_settings),
) -> IndexingPipeline:
    """建立 request-scoped pipeline，共用 application-scoped embedding backend。"""
    try:
        embedding_backend = get_embedding_backend_for(settings)
        return IndexingPipeline(
            embedding_backend=embedding_backend,
            vector_store=create_vector_store(settings),
            batch_size=settings.embedding_batch_size,
        )
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器無法初始化本地 indexing 服務。",
        ) from exc
