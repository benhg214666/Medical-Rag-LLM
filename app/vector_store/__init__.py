"""向量資料庫（vector store）子系統。Phase 1 尚未實作。"""
"""Vector store abstraction 與 Chroma 實作。"""

from app.vector_store.base import VectorStore, VectorStoreError

__all__ = ["VectorStore", "VectorStoreError"]
