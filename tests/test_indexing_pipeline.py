"""processed JSON 到 Chroma 的整合測試（fake embedding）。"""

from pathlib import Path

import pytest

from app.indexing.pipeline import (
    IndexingError,
    IndexingPipeline,
    ProcessedDocumentError,
)
from app.vector_store.chroma_store import ChromaStore
from tests.fakes import FakeEmbeddingBackend
from app.embeddings.base import EmbeddingError


def payload(text: str = "病患主訴胸痛。") -> dict:
    return {
        "document_id": "0123456789abcdef",
        "source_file": "note.txt",
        "file_type": "txt",
        "created_at": "2026-08-08T00:00:00+00:00",
        "processing": {"chunk_size": 500, "chunk_overlap": 100, "min_chunk_size": 50},
        "statistics": {"loaded_units": 1, "cleaned_units": 1, "chunk_count": 1, "total_characters": len(text)},
        "chunks": [{
            "chunk_id": "abcdef0123456789",
            "text": text,
            "source": "note.txt",
            "file_name": "note.txt",
            "file_type": "txt",
            "page_number": None,
            "paragraph_number": None,
            "chunk_index": 0,
            "start_char": 0,
            "end_char": len(text),
            "metadata": {"language": "zh-TW"},
        }],
        "output_file": "data/processed/0123456789abcdef.json",
    }


def test_indexing_and_duplicate_are_correct(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "db", "medical_test")
    pipeline = IndexingPipeline(FakeEmbeddingBackend(), store, batch_size=1)
    first = pipeline.index_payload(payload())
    second = pipeline.index_payload(payload())

    assert first.status == "indexed"
    assert first.document_id == "0123456789abcdef"
    assert first.indexed_chunks == 1
    assert first.embedding_dimension == 4
    assert second.indexed_chunks == 1
    assert store.count() == 1
    metadata = store.get_records()["metadatas"][0]
    assert metadata["language"] == "zh-TW"


@pytest.mark.parametrize(
    "bad_payload",
    [
        "not-object",
        {},
        {"document_id": "id"},
        {"document_id": "id", "chunks": "bad"},
        {"document_id": "id", "chunks": []},
        {"document_id": "id", "chunks": [{"chunk_id": "x", "text": " "}]},
    ],
)
def test_malformed_payload_fails_before_writing(
    tmp_path: Path, bad_payload: object
) -> None:
    store = ChromaStore(tmp_path / "db", "medical_test")
    pipeline = IndexingPipeline(FakeEmbeddingBackend(), store)
    with pytest.raises(ProcessedDocumentError):
        pipeline.index_payload(bad_payload)
    assert store.count() == 0


def test_reindex_removes_stale_chunks(
    tmp_path: Path,
) -> None:
    store = ChromaStore(
        tmp_path / "stale-pipeline-db",
        "medical_test",
    )
    pipeline = IndexingPipeline(
        FakeEmbeddingBackend(),
        store,
        batch_size=1,
    )

    first_payload = payload("相同文件內容。")
    second_old_chunk = dict(first_payload["chunks"][0])
    second_old_chunk.update(
        {
            "chunk_id": "2222222222222222",
            "chunk_index": 1,
        }
    )
    first_payload["chunks"].append(second_old_chunk)
    first_payload["statistics"]["chunk_count"] = 2
    first_payload["statistics"]["total_characters"] *= 2

    pipeline.index_payload(first_payload)

    assert store.count() == 2

    replacement_payload = payload("相同文件內容。")
    replacement_payload["chunks"][0]["chunk_id"] = (
        "3333333333333333"
    )

    pipeline.index_payload(replacement_payload)

    assert store.count() == 1
    assert store.get_records()["ids"] == [
        "3333333333333333"
    ]
def test_second_batch_failure_rolls_back_new_chunks(
    tmp_path: Path,
) -> None:
    class FailingSecondBatchBackend(FakeEmbeddingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def embed_documents(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            self.calls += 1

            if self.calls == 2:
                raise EmbeddingError(
                    "模擬第二個 batch embedding 失敗"
                )

            return super().embed_documents(texts)

    store = ChromaStore(
        tmp_path / "rollback-pipeline-db",
        "medical_test",
    )

    good_pipeline = IndexingPipeline(
        FakeEmbeddingBackend(),
        store,
        batch_size=1,
    )
    good_pipeline.index_payload(payload("既有內容。"))

    existing_ids = set(store.get_records()["ids"])
    assert len(existing_ids) == 1

    failing_payload = payload("更新內容。")
    failing_payload["chunks"][0]["chunk_id"] = (
        "1111111111111111"
    )

    second_chunk = dict(failing_payload["chunks"][0])
    second_chunk.update(
        {
            "chunk_id": "2222222222222222",
            "chunk_index": 1,
        }
    )
    failing_payload["chunks"].append(second_chunk)
    failing_payload["statistics"]["chunk_count"] = 2
    failing_payload["statistics"]["total_characters"] *= 2

    failing_pipeline = IndexingPipeline(
        FailingSecondBatchBackend(),
        store,
        batch_size=1,
    )

    with pytest.raises(IndexingError, match="已有 1 個"):
        failing_pipeline.index_payload(failing_payload)

    assert set(store.get_records()["ids"]) == existing_ids

@pytest.mark.parametrize(
    "invalid_vectors",
    [
        [[float("nan"), 0.0, 0.0, 0.0]],
        [[float("inf"), 0.0, 0.0, 0.0]],
        [["invalid", 0.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.3]],
    ],
    ids=[
        "nan",
        "infinity",
        "non-numeric",
        "wrong-dimension",
    ],
)
def test_invalid_vectors_are_rejected_before_write(
    tmp_path: Path,
    invalid_vectors: list[list[object]],
) -> None:
    class InvalidVectorBackend(FakeEmbeddingBackend):
        def embed_documents(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return invalid_vectors  # type: ignore[return-value]

    store = ChromaStore(
        tmp_path / "invalid-vector-db",
        "medical_test",
    )
    pipeline = IndexingPipeline(
        InvalidVectorBackend(),
        store,
    )

    with pytest.raises(
        IndexingError,
        match="數值|維度",
    ):
        pipeline.index_payload(payload())

    assert store.count() == 0