"""Vector store 的共同介面。"""

from abc import ABC, abstractmethod

from app.ingestion.models import DocumentChunk


class VectorStoreError(RuntimeError):
    """向量資料庫操作失敗。"""


class VectorStore(ABC):
    """Phase 3 所需的最小向量儲存介面。"""

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """目前 collection 名稱。"""

    @abstractmethod
    def ensure_embedding_compatibility(
        self,
        model_name: str,
        dimension: int,
        normalized: bool,
    ) -> None:
        """建立或驗證 collection 的 embedding 相容性合約。"""

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        document_id: str,
    ) -> None:
        """以 chunk_id 為 key 新增或更新 records。"""

    @abstractmethod
    def count(self) -> int:
        """回傳 collection 中的 record 數。"""

    @abstractmethod
    def delete_collection(self) -> None:
        """刪除 collection 與其中資料。"""

    @abstractmethod
    def collection_exists(self) -> bool:
        """回傳 collection 是否存在。"""
