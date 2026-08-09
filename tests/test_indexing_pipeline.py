"""processed JSON 到 Chroma 的整合測試（fake embedding）。"""

from pathlib import Path

import pytest

from app.indexing.pipeline import IndexingPipeline, ProcessedDocumentError
from app.vector_store.chroma_store import ChromaStore
from tests.fakes import FakeEmbeddingBackend


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
