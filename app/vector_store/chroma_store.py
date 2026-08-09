"""本地持久化 Chroma vector store。"""

import math
from pathlib import Path
from typing import Any

from app.ingestion.models import DocumentChunk
from app.vector_store.base import VectorStore, VectorStoreError

def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """扁平化 metadata，只保留 Chroma 支援的 scalar 值。"""
    result: dict[str, str | int | float | bool] = {}

    def visit(prefix: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            result[prefix] = value
        elif isinstance(value, (str, int)):
            result[prefix] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[prefix] = value
        elif isinstance(value, dict):
            for key, nested in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), nested)

    for key, value in metadata.items():
        visit(str(key), value)
    return result


class ChromaStore(VectorStore):
    """使用 chromadb.PersistentClient 的本地向量資料庫。"""

    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        if not collection_name.strip():
            raise ValueError("Chroma collection name 不可為空白")
        self.persist_directory = Path(persist_directory)
        self._collection_name = collection_name
        try:
            import chromadb

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError("無法初始化本地 Chroma vector store") from exc

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def collection_exists(self) -> bool:
        try:
            return any(
                collection.name == self._collection_name
                for collection in self._client.list_collections()
            )
        except Exception as exc:
            raise VectorStoreError("無法讀取 Chroma collection 清單") from exc

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError("無法計算 Chroma records") from exc

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        document_id: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunk 與 embedding 數量不一致")
        if not chunks:
            return
        metadatas = []
        for chunk in chunks:
            raw = chunk.model_dump(exclude={"chunk_id", "text", "metadata"})
            raw["document_id"] = document_id
            raw.update(chunk.metadata)
            metadatas.append(sanitize_metadata(raw))
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError("寫入 Chroma records 失敗") from exc

    def delete_collection(self) -> None:
        try:
            if self.collection_exists():
                self._client.delete_collection(self._collection_name)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("刪除 Chroma collection 失敗") from exc

    def get_records(self, ids: list[str] | None = None) -> dict[str, Any]:
        """診斷與測試用讀取；Phase 3 不提供語意搜尋。"""
        try:
            return self._collection.get(ids=ids, include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError("讀取 Chroma records 失敗") from exc
