"""本地持久化 Chroma vector store。"""

import math
from pathlib import Path
from typing import Any

from app.ingestion.models import DocumentChunk
from app.vector_store.base import VectorMatch, VectorStore, VectorStoreError
_COLLECTION_SCHEMA_VERSION = 1
_DISTANCE_METRIC = "cosine"
_RESERVED_CHUNK_METADATA_KEYS = frozenset(
    {
        "document_id",
        "chunk_id",
        "text",
        "source",
        "file_name",
        "file_type",
        "page_number",
        "paragraph_number",
        "chunk_index",
        "start_char",
        "end_char",
        "metadata",
    }
)

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

    @property
    def distance_metric(self) -> str:
        """本 store 建立 collection 時固定使用的距離度量。

        以模組常數為準而非讀取 collection metadata：hnsw:space 是
        Chroma 的 immutable 建立參數，這裡的宣告即為事實來源。
        """
        return _DISTANCE_METRIC

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
        model_revision: str,
        dimension: int,
        normalized: bool,
    ) -> None:
        """建立或驗證 collection 的 embedding 相容性合約。"""
        if not model_name.strip():
            raise VectorStoreError("embedding model name 不可為空白")
        if not model_revision.strip():
            raise VectorStoreError(
                "embedding model revision 不可為空白"
            )
        if dimension <= 0:
            raise VectorStoreError("embedding dimension 必須大於 0")

        expected = {
            "distance_metric": _DISTANCE_METRIC,
            "schema_version": _COLLECTION_SCHEMA_VERSION,
            "embedding_model": model_name,
            "embedding_model_revision": model_revision,
            "embedding_dimension": dimension,
            "embedding_normalized": normalized,
        }

        try:
            current = dict(self._collection.metadata or {})
            current_distance = current.get("hnsw:space")
            if current_distance not in {None, _DISTANCE_METRIC}:
                raise VectorStoreError(
                    "collection embedding 設定不相容：hnsw:space"
                )

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
            conflicting_keys = sorted(
                _RESERVED_CHUNK_METADATA_KEYS.intersection(
                    chunk.metadata
                )
            )

            if conflicting_keys:
                fields = ", ".join(conflicting_keys)
                raise VectorStoreError(
                    "chunk metadata 不可覆蓋系統欄位："
                    f"{fields}"
                )

            raw = chunk.model_dump(
                exclude={"chunk_id", "text", "metadata"}
            )
            raw.update(chunk.metadata)

            # 系統 document_id 最後設定，作為額外防護。
            raw["document_id"] = document_id

            metadatas.append(sanitize_metadata(raw))
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                "寫入 Chroma records 失敗"
            ) from exc

    def search_by_vector(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[VectorMatch]:
        """以查詢向量做 cosine 相似度搜尋，回傳中性的 VectorMatch。

        Chroma 的 query() 回傳的是「平行 list 的 list」結構，例如
        {"ids": [[...]], "documents": [[...]], "distances": [[...]]}，
        外層 list 對應每一個查詢向量。本方法一次只送一個向量，
        因此固定取索引 0，並在此把 Chroma 專屬格式翻譯成 VectorMatch，
        不讓上層看到任何 Chroma 內部結構。
        """
        if top_k <= 0:
            raise VectorStoreError("top_k 必須大於 0")
        if not embedding:
            raise VectorStoreError("查詢向量不可為空")

        # Chroma 在 n_results 超過 record 數時會自行截斷，
        # 但先讀 count 可在 collection 為空時直接短路，省下一次查詢。
        record_count = self.count()
        if record_count == 0:
            return []

        try:
            response = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, record_count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Chroma 相似度查詢失敗") from exc

        def first_row(key: str) -> list[Any]:
            """取出對應第一個查詢向量的那一列。

            Chroma 對每個 include 的欄位都回傳「list of list」，
            外層對應查詢向量。本方法只送一個向量，故取索引 0。
            """
            rows = response.get(key)
            if rows is None:
                raise VectorStoreError(
                    f"Chroma 查詢回應缺少必要欄位：{key}"
                )
            if not rows or rows[0] is None:
                return []
            return list(rows[0])

        ids = first_row("ids")
        documents = first_row("documents")
        metadatas = first_row("metadatas")
        distances = first_row("distances")

        if not ids:
            return []

        # 這四個是平行陣列，長度必須一致。長度不符代表回應結構異常，
        # 此時若繼續以位置對應，會把 A chunk 的文字配上 B chunk 的距離——
        # 產生看似正常、實際錯誤的檢索結果。醫療情境下寧可明確失敗。
        expected = len(ids)
        for name, column in (
            ("documents", documents),
            ("metadatas", metadatas),
            ("distances", distances),
        ):
            if len(column) != expected:
                raise VectorStoreError(
                    "Chroma 查詢回應結構不完整："
                    f"{name} 長度與 ids 不一致"
                )

        matches: list[VectorMatch] = []
        for position, chunk_id in enumerate(ids):
            if chunk_id is None:
                raise VectorStoreError("Chroma 回傳的 record ID 為空")

            raw_distance = distances[position]
            if raw_distance is None or isinstance(raw_distance, bool):
                raise VectorStoreError("Chroma 回傳無效的 distance 值")
            try:
                distance = float(raw_distance)
            except (TypeError, ValueError) as exc:
                raise VectorStoreError(
                    "Chroma 回傳無法解析的 distance 值"
                ) from exc
            if not math.isfinite(distance):
                raise VectorStoreError("Chroma 回傳無效的 distance 值")

            # chunk 內容是 answer traceability 的基礎。若有 chunk_id
            # 卻沒有對應文字，回傳空字串會讓上層以為「這段就是空的」，
            # 進而把空內容當成佐證。這種情況必須明確失敗。
            text = documents[position]
            if not isinstance(text, str):
                raise VectorStoreError(
                    "Chroma 回傳的 record 缺少文字內容"
                )

            metadata = metadatas[position]
            if metadata is not None and not isinstance(metadata, dict):
                raise VectorStoreError(
                    "Chroma 回傳的 metadata 結構異常"
                )

            matches.append(
                VectorMatch(
                    chunk_id=str(chunk_id),
                    text=text,
                    distance=distance,
                    # metadata 為 None 是合法的（寫入時可能沒有額外欄位），
                    # 與「結構異常」不同，故轉成空 dict 而非報錯。
                    metadata=dict(metadata or {}),
                )
            )

        # Chroma 已依距離排序，這裡再排一次是為了讓「最相似在前」
        # 成為本抽象層的明確保證，而不是依賴底層實作的附帶行為。
        matches.sort(key=lambda match: match.distance)
        return matches

    def delete_stale_chunks(
        self,
        document_id: str,
        keep_chunk_ids: set[str],
    ) -> int:
        """刪除指定 document 已不再存在的舊 chunks。"""
        if not document_id.strip():
            raise VectorStoreError("document_id 不可為空白")

        try:
            records = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            stale_ids = [
                record_id
                for record_id in records["ids"]
                if record_id not in keep_chunk_ids
            ]

            if stale_ids:
                self._collection.delete(ids=stale_ids)

            return len(stale_ids)

        except Exception as exc:
            raise VectorStoreError(
                "刪除過期 Chroma records 失敗"
            ) from exc

    def get_document_chunk_ids(
        self,
        document_id: str,
    ) -> set[str]:
        if not document_id.strip():
            raise VectorStoreError("document_id 不可為空白")

        try:
            records = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            return set(records["ids"])
        except Exception as exc:
            raise VectorStoreError(
                "讀取 document chunk IDs 失敗"
            ) from exc

    def delete_chunks_by_ids(
        self,
        chunk_ids: set[str],
    ) -> int:
        if not chunk_ids:
            return 0

        try:
            records = self._collection.get(
                ids=sorted(chunk_ids),
                include=["metadatas"],
            )
            existing_ids = set(records["ids"])

            if existing_ids:
                self._collection.delete(
                    ids=sorted(existing_ids)
                )

            return len(existing_ids)
        except Exception as exc:
            raise VectorStoreError(
                "依 IDs 刪除 Chroma records 失敗"
            ) from exc


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
