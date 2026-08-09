"""Phase 3 離線測試替身；production factory 不會載入此模組。"""

import hashlib

from app.embeddings.base import EmbeddingBackend


class FakeEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    @property
    def model_revision(self) -> str:
        return "test-revision"

    @property
    def normalizes_embeddings(self) -> bool:
        return False

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index] / 255.0 for index in range(self._dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
