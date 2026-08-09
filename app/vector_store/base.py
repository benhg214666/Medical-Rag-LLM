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

    @abstractmethod
    def delete_stale_chunks(
        self,
        document_id: str,
        keep_chunk_ids: set[str],
    ) -> int:
        """刪除同一 document 中不在 keep_chunk_ids 的舊 records。"""

    

    @abstractmethod
    def get_document_chunk_ids(
        self,
        document_id: str,
    ) -> set[str]:
        """取得指定 document 目前已有的全部 chunk IDs。"""

    @abstractmethod
    def delete_chunks_by_ids(
        self,
        chunk_ids: set[str],
    ) -> int:
        """依 IDs 刪除 chunks，回傳實際刪除數量。"""