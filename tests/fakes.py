"""Phase 3 / Phase 4 離線測試替身；production factory 不會載入此模組。"""

import hashlib
import math

from app.embeddings.base import EmbeddingBackend
from app.ingestion.models import DocumentChunk
from app.vector_store.base import VectorMatch, VectorStore


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


class MemoryVectorStore(VectorStore):
    """In-memory upsert store shared by indexing orchestration unit tests."""

    def __init__(self, collection_name: str = "memory_test") -> None:
        self._collection_name = collection_name
        self.records: dict[
            str,
            tuple[DocumentChunk, list[float], str],
        ] = {}

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def distance_metric(self) -> str:
        return "cosine"

    def ensure_embedding_compatibility(
        self,
        model_name: str,
        model_revision: str,
        dimension: int,
        normalized: bool,
    ) -> None:
        return None

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        document_id: str,
    ) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.records[chunk.chunk_id] = (chunk, embedding, document_id)

    def delete_stale_chunks(
        self,
        document_id: str,
        keep_chunk_ids: set[str],
    ) -> int:
        stale_ids = {
            chunk_id
            for chunk_id, record in self.records.items()
            if record[2] == document_id and chunk_id not in keep_chunk_ids
        }
        return self.delete_chunks_by_ids(stale_ids)

    def get_document_chunk_ids(self, document_id: str) -> set[str]:
        return {
            chunk_id
            for chunk_id, record in self.records.items()
            if record[2] == document_id
        }

    def delete_chunks_by_ids(self, chunk_ids: set[str]) -> int:
        existing_ids = set(self.records).intersection(chunk_ids)
        for chunk_id in existing_ids:
            del self.records[chunk_id]
        return len(existing_ids)

    def count(self) -> int:
        return len(self.records)

    def delete_collection(self) -> None:
        self.records.clear()

    def collection_exists(self) -> bool:
        return True

    def search_by_vector(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[VectorMatch]:
        return search_records_by_vector(self.records, embedding, top_k)


def cosine_distance(left: list[float], right: list[float]) -> float:
    """以 Chroma 的 cosine space 定義計算距離：1 - cosine_similarity。

    測試替身需要與 ChromaStore 相同的 distance 語意，
    否則驗證 score 換算的測試會失去意義。
    """
    if len(left) != len(right):
        raise ValueError("向量維度不一致")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - (dot / (left_norm * right_norm))


def search_records_by_vector(
    records: dict[str, tuple[DocumentChunk, list[float], str]],
    embedding: list[float],
    top_k: int,
) -> list[VectorMatch]:
    """以純 Python 重現 cosine 相似度搜尋，供記憶體型 fake store 共用。

    records 的形狀刻意與既有 MemoryVectorStore 一致：
    chunk_id -> (chunk, embedding, document_id)。
    """
    if top_k <= 0:
        raise ValueError("top_k 必須大於 0")

    matches: list[VectorMatch] = []
    for chunk_id, (chunk, vector, document_id) in records.items():
        metadata = dict(chunk.metadata)
        metadata.update(
            chunk.model_dump(exclude={"chunk_id", "text", "metadata"})
        )
        metadata["document_id"] = document_id

        matches.append(
            VectorMatch(
                chunk_id=chunk_id,
                text=chunk.text,
                distance=cosine_distance(embedding, vector),
                metadata=metadata,
            )
        )

    matches.sort(key=lambda match: match.distance)
    return matches[:top_k]
