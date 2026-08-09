"""FastAPI 可覆寫的 Phase 3 dependency providers。"""

from functools import lru_cache

from fastapi import Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingBackend, EmbeddingError
from app.embeddings.factory import create_embedding_backend
from app.indexing.pipeline import IndexingPipeline
from app.vector_store.base import VectorStoreError
from app.vector_store.factory import create_vector_store


@lru_cache(maxsize=4)
def _get_cached_embedding_backend(
    provider: str,
    model_name: str,
    device: str,
) -> EmbeddingBackend:
    """依 embedding 設定共用 backend，避免每個 request 重新載入模型。"""
    backend_settings = Settings(
        embedding_provider=provider,
        embedding_model_name=model_name,
        embedding_device=device,
    )
    return create_embedding_backend(backend_settings)


def get_indexing_pipeline(
    settings: Settings = Depends(get_settings),
) -> IndexingPipeline:
    """建立 request-scoped pipeline，共用 application-scoped embedding backend。"""
    try:
        embedding_backend = _get_cached_embedding_backend(
            settings.embedding_provider.lower(),
            settings.embedding_model_name,
            settings.embedding_device.lower(),
        )
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