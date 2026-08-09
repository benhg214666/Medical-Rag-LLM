"""Embedding backend factory。"""

from app.core.config import Settings
from app.embeddings.base import EmbeddingBackend, EmbeddingError
from app.embeddings.local_embedding import LocalEmbeddingBackend


def create_embedding_backend(settings: Settings) -> EmbeddingBackend:
    """依設定建立 embedding backend。"""
    if settings.embedding_provider.lower() == "local":
        return LocalEmbeddingBackend(
            model_name=settings.embedding_model_name,
            device=settings.embedding_device,
        )
    raise EmbeddingError(
        f"不支援的 embedding provider: {settings.embedding_provider}"
    )
