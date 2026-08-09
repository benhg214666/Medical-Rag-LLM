"""Persistent Chroma store 測試。"""

from pathlib import Path

from app.ingestion.models import DocumentChunk
from app.vector_store.chroma_store import ChromaStore


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-1",
        text="病患主訴胸痛。",
        source="note.txt",
        file_name="note.txt",
        file_type="txt",
        page_number=None,
        paragraph_number=None,
        chunk_index=0,
        start_char=0,
        end_char=7,
        metadata={"nested": {"ward": "A"}, "tags": ["private"], "none": None},
    )


def test_persistent_upsert_metadata_and_delete(tmp_path: Path) -> None:
    db_dir = tmp_path / "chroma"
    store = ChromaStore(db_dir, "test_collection")
    assert store.collection_exists()
    assert store.count() == 0

    store.add_chunks([make_chunk()], [[0.1, 0.2, 0.3, 0.4]], "doc-1")
    store.add_chunks([make_chunk()], [[0.4, 0.3, 0.2, 0.1]], "doc-1")
    assert store.count() == 1

    record = store.get_records(["chunk-1"])
    assert record["documents"] == ["病患主訴胸痛。"]
    metadata = record["metadatas"][0]
    assert metadata["document_id"] == "doc-1"
    assert metadata["nested.ward"] == "A"
    assert "none" not in metadata
    assert "tags" not in metadata

    reopened = ChromaStore(db_dir, "test_collection")
    assert reopened.count() == 1
    reopened.delete_collection()
    assert not reopened.collection_exists()
