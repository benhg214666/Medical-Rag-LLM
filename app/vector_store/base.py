"""Vector store 的共同介面。"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.ingestion.models import DocumentChunk


class VectorStoreError(RuntimeError):
    """向量資料庫操作失敗。"""


class VectorMatch(BaseModel):
    """相似度搜尋的單筆命中結果（vector store 中性格式）。

    這是 Phase 4 新增的「翻譯層」資料結構。存在的理由是：
    Chroma 的 query() 回傳的是平行 list 組成的巢狀 dict
    （ids / documents / metadatas / distances 各自一個 list of list），
    這是 Chroma 專屬格式。若讓上層直接消費，換掉 vector store 時
    Retriever 與 API 都得跟著改寫。

    因此由各 VectorStore 實作負責把自家格式轉成本模型，
    上層只認得 VectorMatch。

    Attributes:
        chunk_id: 對應 DocumentChunk.chunk_id，也是 vector store 的 record ID。
        text: chunk 原文。
        distance: 距離值，語意由 distance_metric 決定；
            cosine 時為 1 - cosine_similarity，範圍 [0, 2]，**越小越相似**。
        metadata: 寫入時保存的扁平化 metadata（含 document_id）。
    """

    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    """Phase 3 所需的最小向量儲存介面；Phase 4 追加相似度搜尋。"""

    @property
    @abstractmethod
    def distance_metric(self) -> str:
        """collection 實際使用的距離度量，例如 "cosine"。

        Retriever 依此決定 distance 能否安全換算成 similarity，
        避免對不同 metric 套用錯誤的數學公式。
        """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """目前 collection 名稱。"""

    @abstractmethod
    def ensure_embedding_compatibility(
        self,
        model_name: str,
        model_revision: str,
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
    def search_by_vector(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[VectorMatch]:
        """以查詢向量取回最相近的 records。

        實作必須把 vector store 專屬的回傳格式轉成 VectorMatch，
        並依 distance 由小到大（最相似在前）排序。

        Args:
            embedding: 查詢向量；維度必須與 collection 一致。
            top_k: 最多回傳幾筆；必須大於 0。

        Returns:
            已排序的命中結果；collection 為空時回傳空 list，不視為錯誤。

        Raises:
            VectorStoreError: 參數不合法或底層查詢失敗。
        """

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
