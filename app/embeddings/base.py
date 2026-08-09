"""Embedding 後端的共同介面。"""

from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    """Embedding 模型載入或推論失敗。"""


class EmbeddingBackend(ABC):
    """可替換的文字向量化後端。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """將文件文字轉換為向量。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """將查詢文字轉換為向量。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量維度。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型識別名稱。"""
    
    @property
    @abstractmethod
    def normalizes_embeddings(self) -> bool:
        """回傳輸出的向量是否已做 L2 normalization。"""
