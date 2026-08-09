"""Persistent Chroma store 測試。"""
from pathlib import Path

import pytest


from app.vector_store.base import VectorStoreError
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


def test_embedding_contract_persists_after_reopen(
    tmp_path: Path,
) -> None:
    db_dir = tmp_path / "contract-db"

    store = ChromaStore(db_dir, "contract_test")
    store.ensure_embedding_compatibility(
        model_name="intfloat/multilingual-e5-small",
        dimension=384,
        normalized=True,
    )

    reopened = ChromaStore(db_dir, "contract_test")
    reopened.ensure_embedding_compatibility(
        model_name="intfloat/multilingual-e5-small",
        dimension=384,
        normalized=True,
    )

    metadata = reopened._collection.metadata
    assert metadata["embedding_model"] == (
        "intfloat/multilingual-e5-small"
    )
    assert metadata["embedding_dimension"] == 384
    assert metadata["embedding_normalized"] is True
    assert metadata["distance_metric"] == "cosine"
    assert metadata["schema_version"] == 1


@pytest.mark.parametrize(
    ("model_name", "dimension", "normalized"),
    [
        ("another/model", 384, True),
        ("intfloat/multilingual-e5-small", 768, True),
        ("intfloat/multilingual-e5-small", 384, False),
    ],
)
def test_incompatible_embedding_contract_is_rejected(
    tmp_path: Path,
    model_name: str,
    dimension: int,
    normalized: bool,
) -> None:
    db_dir = tmp_path / "incompatible-db"

    store = ChromaStore(db_dir, "contract_test")
    store.ensure_embedding_compatibility(
        model_name="intfloat/multilingual-e5-small",
        dimension=384,
        normalized=True,
    )

    reopened = ChromaStore(db_dir, "contract_test")

    with pytest.raises(VectorStoreError, match="不相容"):
        reopened.ensure_embedding_compatibility(
            model_name=model_name,
            dimension=dimension,
            normalized=normalized,
        )


def test_nonempty_legacy_collection_without_contract_is_rejected(
    tmp_path: Path,
) -> None:
    store = ChromaStore(
        tmp_path / "legacy-db",
        "legacy_collection",
    )
    store.add_chunks(
        [make_chunk()],
        [[0.1, 0.2, 0.3, 0.4]],
        "doc-1",
    )

    with pytest.raises(VectorStoreError, match="缺少"):
        store.ensure_embedding_compatibility(
            model_name="intfloat/multilingual-e5-small",
            dimension=4,
            normalized=True,
        )

def test_delete_stale_chunks_keeps_only_current_document_chunks(
    tmp_path: Path,
) -> None:
    store = ChromaStore(
        tmp_path / "stale-db",
        "stale_test",
    )
    store.ensure_embedding_compatibility(
        model_name="test/model",
        dimension=4,
        normalized=False,
    )

    old_first = make_chunk()
    old_second = make_chunk().model_copy(
        update={
            "chunk_id": "chunk-2",
            "chunk_index": 1,
        }
    )
    new_chunk = make_chunk().model_copy(
        update={
            "chunk_id": "chunk-new",
            "chunk_index": 0,
        }
    )

    store.add_chunks(
        [old_first, old_second],
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ],
        "doc-1",
    )
    store.add_chunks(
        [new_chunk],
        [[0.4, 0.3, 0.2, 0.1]],
        "doc-1",
    )

    assert store.count() == 3

    deleted = store.delete_stale_chunks(
        document_id="doc-1",
        keep_chunk_ids={"chunk-new"},
    )

    assert deleted == 2
    assert store.count() == 1
    assert store.get_records()["ids"] == ["chunk-new"]

def test_get_document_ids_and_delete_by_ids(
    tmp_path: Path,
) -> None:
    store = ChromaStore(
        tmp_path / "rollback-db",
        "rollback_test",
    )
    store.ensure_embedding_compatibility(
        model_name="test/model",
        dimension=4,
        normalized=False,
    )

    first = make_chunk()
    second = make_chunk().model_copy(
        update={
            "chunk_id": "chunk-2",
            "chunk_index": 1,
        }
    )
    other_document = make_chunk().model_copy(
        update={
            "chunk_id": "other-1",
            "chunk_index": 0,
        }
    )

    store.add_chunks(
        [first, second],
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ],
        "doc-1",
    )
    store.add_chunks(
        [other_document],
        [[0.4, 0.3, 0.2, 0.1]],
        "doc-2",
    )

    assert store.get_document_chunk_ids("doc-1") == {
        "chunk-1",
        "chunk-2",
    }

    deleted = store.delete_chunks_by_ids(
        {"chunk-2", "does-not-exist"}
    )

    assert deleted == 1
    assert store.get_document_chunk_ids("doc-1") == {
        "chunk-1"
    }
    assert store.get_document_chunk_ids("doc-2") == {
        "other-1"
    }

    