"""Indexing API：dependency injection 確保不下載真實模型。"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.indexing.dependencies import get_indexing_pipeline
from app.indexing.pipeline import IndexingPipeline
from app.ingestion.models import DocumentChunk
from app.main import app
from app.vector_store.base import VectorStore
from tests.fakes import FakeEmbeddingBackend, search_records_by_vector
from tests.test_indexing_pipeline import payload


class MemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self.records: dict[
            str,
            tuple[DocumentChunk, list[float], str],
        ] = {}

    @property
    def collection_name(self) -> str:
        return "api_test"

    def ensure_embedding_compatibility(
        self,
        model_name: str,
        model_revision: str,
        dimension: int,
        normalized: bool,
    ) -> None:
        return None
    def add_chunks(self, chunks, embeddings, document_id) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.records[chunk.chunk_id] = (
                chunk,
                embedding,
                document_id,
            )
    def delete_stale_chunks(
        self,
        document_id: str,
        keep_chunk_ids: set[str],
    ) -> int:
        stale_ids = [
            chunk_id
            for chunk_id, record in self.records.items()
            if record[2] == document_id
            and chunk_id not in keep_chunk_ids
        ]

        for chunk_id in stale_ids:
            del self.records[chunk_id]

        return len(stale_ids)




    def get_document_chunk_ids(
        self,
        document_id: str,
    ) -> set[str]:
        return {
            chunk_id
            for chunk_id, record in self.records.items()
            if record[2] == document_id
        }

    def delete_chunks_by_ids(
        self,
        chunk_ids: set[str],
    ) -> int:
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

    # --- Phase 4 追加：VectorStore 新增了搜尋介面 ---
    # 這個 fake 只服務 indexing API 測試，不會被呼叫到搜尋，
    # 但 ABC 有 abstract member 就必須實作，否則無法實例化。

    @property
    def distance_metric(self) -> str:
        return "cosine"

    def search_by_vector(self, embedding, top_k):
        return search_records_by_vector(
            records=self.records,
            embedding=embedding,
            top_k=top_k,
        )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, Settings]]:
    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vector_db_dir=tmp_path / "db",
    )
    pipeline = IndexingPipeline(FakeEmbeddingBackend(), MemoryVectorStore())
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_indexing_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client, settings
    app.dependency_overrides.clear()


def test_existing_processed_document_indexes_without_vectors_in_response(
    client: tuple[TestClient, Settings],
) -> None:
    test_client, settings = client
    settings.processed_data_dir.mkdir(parents=True)
    path = settings.processed_data_dir / "0123456789abcdef.json"
    path.write_text(json.dumps(payload(), ensure_ascii=False), encoding="utf-8")

    response = test_client.post("/api/documents/0123456789abcdef/index")
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "0123456789abcdef"
    assert body["indexed_chunks"] == 1
    assert body["embedding_dimension"] == 4
    assert "embeddings" not in body
    assert str(settings.processed_data_dir.resolve()) not in response.text


def test_missing_processed_document_returns_404(
    client: tuple[TestClient, Settings],
) -> None:
    response = client[0].post("/api/documents/0123456789abcdef/index")
    assert response.status_code == 404


def test_malformed_processed_document_returns_400(
    client: tuple[TestClient, Settings],
) -> None:
    test_client, settings = client
    settings.processed_data_dir.mkdir(parents=True)
    (settings.processed_data_dir / "0123456789abcdef.json").write_text(
        "{broken", encoding="utf-8"
    )
    response = test_client.post("/api/documents/0123456789abcdef/index")
    assert response.status_code == 400
    assert str(settings.processed_data_dir.resolve()) not in response.text


def test_invalid_document_id_does_not_allow_path_traversal(
    client: tuple[TestClient, Settings],
) -> None:
    response = client[0].post("/api/documents/not-a-valid-id/index")
    assert response.status_code == 404
