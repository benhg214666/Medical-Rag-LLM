"""協調 processed JSON、embedding backend 與 vector store。"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.embeddings.base import EmbeddingBackend, EmbeddingError
from app.ingestion.models import IngestionResult
from app.vector_store.base import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class IndexingError(RuntimeError):
    """processed data 無效或 indexing 失敗。"""


class ProcessedDocumentError(IndexingError):
    """processed JSON 不可讀或不符合 Phase 2 schema。"""


class IndexingResult(BaseModel):
    status: str = "indexed"
    document_id: str
    collection_name: str
    indexed_chunks: int
    embedding_model: str
    embedding_dimension: int


class IndexingPipeline:
    """以固定大小 batch 執行 embedding 與 Chroma upsert。"""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("embedding batch_size 必須大於 0")
        self.embedding_backend = embedding_backend
        self.vector_store = vector_store
        self.batch_size = batch_size

    @staticmethod
    def _validate_payload(payload: Any) -> IngestionResult:
        if not isinstance(payload, dict):
            raise ProcessedDocumentError("processed JSON 根節點必須是 object")
        if not payload.get("document_id"):
            raise ProcessedDocumentError("processed JSON 缺少 document_id")
        if "chunks" not in payload or not isinstance(payload["chunks"], list):
            raise ProcessedDocumentError("processed JSON 的 chunks 必須是 list")
        if not payload["chunks"]:
            raise ProcessedDocumentError("processed JSON 沒有可 index 的 chunks")
        for index, chunk in enumerate(payload["chunks"]):
            if not isinstance(chunk, dict) or not chunk.get("chunk_id"):
                raise ProcessedDocumentError(f"chunk {index} 缺少 chunk_id")
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ProcessedDocumentError(f"chunk {index} 的 text 不可為空白")
            if "metadata" in chunk and not isinstance(chunk["metadata"], dict):
                raise ProcessedDocumentError(f"chunk {index} 的 metadata 必須是 object")
        try:
            return IngestionResult.model_validate(payload)
        except ValidationError as exc:
            raise ProcessedDocumentError(
                "processed JSON 結構不符合 Phase 2 schema"
            ) from exc

    def index_payload(self, payload: Any) -> IndexingResult:
        ingestion = self._validate_payload(payload)
        started_at = datetime.now(timezone.utc)
        completed = 0
        dimension = 0
        batch_count = (len(ingestion.chunks) + self.batch_size - 1) // self.batch_size
        logger.info(
            "開始 indexing：document_id=%s chunks=%d model=%s batches=%d collection=%s",
            ingestion.document_id,
            len(ingestion.chunks),
            self.embedding_backend.model_name,
            batch_count,
            self.vector_store.collection_name,
        )
        try:
            for start in range(0, len(ingestion.chunks), self.batch_size):
                batch = ingestion.chunks[start : start + self.batch_size]
                vectors = self.embedding_backend.embed_documents(
                    [chunk.text for chunk in batch]
                )
                if len(vectors) != len(batch):
                    raise IndexingError("embedding backend 回傳的向量數量不正確")
                if vectors:
                    current_dimension = len(vectors[0])
                    if current_dimension <= 0 or any(
                        len(vector) != current_dimension for vector in vectors
                    ):
                        raise IndexingError("embedding backend 回傳無效的向量維度")
                    if dimension not in {0, current_dimension}:
                        raise IndexingError("不同 batch 的 embedding 維度不一致")
                    dimension = current_dimension
                self.vector_store.add_chunks(batch, vectors, ingestion.document_id)
                completed += len(batch)
        except (EmbeddingError, VectorStoreError, IndexingError) as exc:
            logger.exception(
                "Indexing 失敗：document_id=%s completed_chunks=%d；前批次可能已寫入",
                ingestion.document_id,
                completed,
            )
            if isinstance(exc, IndexingError):
                raise
            raise IndexingError(
                f"indexing 失敗；已有 {completed} 個 chunk 可能完成 upsert"
            ) from exc

        elapsed_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
        logger.info(
            "Indexing 完成：document_id=%s chunks=%d dimension=%d 耗時=%.1fms",
            ingestion.document_id,
            completed,
            dimension,
            elapsed_ms,
        )
        return IndexingResult(
            document_id=ingestion.document_id,
            collection_name=self.vector_store.collection_name,
            indexed_chunks=completed,
            embedding_model=self.embedding_backend.model_name,
            embedding_dimension=dimension,
        )

    def index_processed_document(
        self,
        path: Path,
        expected_document_id: str | None = None,
    ) -> IndexingResult:
        """讀取一個 UTF-8 processed JSON 並 index。"""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProcessedDocumentError("無法讀取有效的 processed JSON") from exc
        if expected_document_id is not None and (
            not isinstance(payload, dict)
            or payload.get("document_id") != expected_document_id
        ):
            raise ProcessedDocumentError(
                "processed JSON 的 document_id 與檔名不一致"
            )
        return self.index_payload(payload)
