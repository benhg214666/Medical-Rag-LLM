"""Vector store factory。"""

from app.core.config import Settings
from app.vector_store.base import VectorStore, VectorStoreError
from app.vector_store.chroma_store import ChromaStore


def create_vector_store(settings: Settings) -> VectorStore:
    """依設定建立本地 vector store。"""
    if settings.vector_store_provider.lower() == "chroma":
        return ChromaStore(
            persist_directory=settings.vector_db_dir,
            collection_name=settings.chroma_collection_name,
        )
    raise VectorStoreError(
        f"不支援的 vector store provider: {settings.vector_store_provider}"
    )
