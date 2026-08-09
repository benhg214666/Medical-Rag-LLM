"""本地持久化 Chroma vector store。"""

import math
from pathlib import Path
from typing import Any

from app.ingestion.models import DocumentChunk
from app.vector_store.base import VectorStore, VectorStoreError
_COLLECTION_SCHEMA_VERSION = 1
_DISTANCE_METRIC = "cosine"

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
                metadata={
                    # Chroma 的 hnsw:space 建立後不可修改。
                    "hnsw:space": _DISTANCE_METRIC,
                    "distance_metric": _DISTANCE_METRIC,
                    "schema_version": _COLLECTION_SCHEMA_VERSION,
                },
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

    def ensure_embedding_compatibility(
        self,
        model_name: str,
        dimension: int,
        normalized: bool,
    ) -> None:
        """建立或驗證 collection 的 embedding 相容性合約。"""
        if not model_name.strip():
            raise VectorStoreError("embedding model name 不可為空白")
        if dimension <= 0:
            raise VectorStoreError("embedding dimension 必須大於 0")

        expected = {
            "distance_metric": _DISTANCE_METRIC,
            "schema_version": _COLLECTION_SCHEMA_VERSION,
            "embedding_model": model_name,
            "embedding_dimension": dimension,
            "embedding_normalized": normalized,
        }

        try:
            current = dict(self._collection.metadata or {})
            missing_fields = [
                key for key in expected if key not in current
            ]

            if missing_fields:
                if self.count() != 0:
                    raise VectorStoreError(
                        "既有 collection 含有 records，但缺少 embedding "
                        "相容性資訊；請建立新 collection 或重新建立索引"
                    )

                # hnsw:space 是 Chroma 的 immutable 建立參數，
                # 不可再次傳入 modify()。
                modifiable_current = {
                    key: value
                    for key, value in current.items()
                    if key != "hnsw:space"
                }

                self._collection.modify(
                    metadata={**modifiable_current, **expected}
                )
                return

            mismatches = [
                key
                for key, expected_value in expected.items()
                if current.get(key) != expected_value
            ]

            if mismatches:
                fields = ", ".join(mismatches)
                raise VectorStoreError(
                    f"collection embedding 設定不相容：{fields}"
                )

        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "無法驗證 Chroma collection embedding 設定"
            ) from exc

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
