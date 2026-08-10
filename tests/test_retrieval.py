"""Phase 4 Retriever 的離線單元測試。"""

from pathlib import Path

import pytest

from app.embeddings.base import EmbeddingBackend, EmbeddingError
from app.ingestion.models import DocumentChunk
from app.retrieval.exceptions import RetrievalBackendError, RetrievalValidationError
from app.retrieval.vector_retriever import VectorRetriever
from app.vector_store.base import VectorStore, VectorStoreError
from app.vector_store.chroma_store import ChromaStore
from tests.fakes import FakeEmbeddingBackend, search_records_by_vector


class MemorySearchStore(VectorStore):
    def __init__(self, distance_metric: str = "cosine", collection_name: str = "retrieval_test") -> None:
        self.records: dict[str, tuple[DocumentChunk, list[float], str]] = {}
        self._distance_metric = distance_metric
        self._collection_name = collection_name
        self.last_top_k: int | None = None
        self.compatibility_calls: list[tuple[str, str, int, bool]] = []

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def ensure_embedding_compatibility(self, model_name, model_revision, dimension, normalized) -> None:
        self.compatibility_calls.append((model_name, model_revision, dimension, normalized))

    def add_chunks(self, chunks, embeddings, document_id) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.records[chunk.chunk_id] = (chunk, embedding, document_id)

    def search_by_vector(self, embedding, top_k):
        self.last_top_k = top_k
        return search_records_by_vector(self.records, embedding, top_k)

    def count(self) -> int:
        return len(self.records)

    def delete_collection(self) -> None:
        self.records.clear()

    def collection_exists(self) -> bool:
        return True

    def delete_stale_chunks(self, document_id, keep_chunk_ids) -> int:
        stale = [key for key, value in self.records.items() if value[2] == document_id and key not in keep_chunk_ids]
        for key in stale:
            del self.records[key]
        return len(stale)

    def get_document_chunk_ids(self, document_id) -> set[str]:
        return {key for key, value in self.records.items() if value[2] == document_id}

    def delete_chunks_by_ids(self, chunk_ids) -> int:
        existing = set(self.records).intersection(chunk_ids)
        for key in existing:
            del self.records[key]
        return len(existing)


class FailingEmbeddingBackend(FakeEmbeddingBackend):
    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("測試 embedding 失敗")


class FailingSearchStore(MemorySearchStore):
    def search_by_vector(self, embedding, top_k):
        raise VectorStoreError("測試 store 失敗")


class IncompatibleSearchStore(MemorySearchStore):
    def ensure_embedding_compatibility(self, model_name, model_revision, dimension, normalized) -> None:
        raise VectorStoreError("collection embedding 設定不相容：embedding_model")


def make_chunk(chunk_id: str, text: str, chunk_index: int, metadata=None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id, text=text, source="synthetic_note.txt",
        file_name="synthetic_note.txt", file_type="txt", chunk_index=chunk_index,
        start_char=0, end_char=len(text), metadata=metadata or {},
    )


def build_retriever(store=None, backend=None, default_top_k=5, max_top_k=50):
    store = store or MemorySearchStore()
    backend = backend or FakeEmbeddingBackend()
    return VectorRetriever(backend, store, default_top_k, max_top_k), store


def populate(store, backend, texts: dict[str, str], document_id="doc-synthetic-1") -> None:
    chunks = [make_chunk(key, text, index) for index, (key, text) in enumerate(texts.items())]
    store.add_chunks(chunks, backend.embed_documents([chunk.text for chunk in chunks]), document_id)


class TestQueryValidation:
    @pytest.mark.parametrize("query", ["", "   ", "\n", "\t", " \n\t "])
    def test_blank_query(self, query) -> None:
        retriever, _ = build_retriever()
        with pytest.raises(RetrievalValidationError, match="空白"):
            retriever.retrieve(query)

    def test_non_string_query(self) -> None:
        retriever, _ = build_retriever()
        with pytest.raises(RetrievalValidationError, match="字串"):
            retriever.retrieve(None)  # type: ignore[arg-type]

    def test_query_is_stripped(self) -> None:
        backend = FakeEmbeddingBackend()
        retriever, store = build_retriever(backend=backend)
        populate(store, backend, {"c1": "hypertension"})
        assert retriever.retrieve("  hypertension  ")[0].distance == retriever.retrieve("hypertension")[0].distance


class TestTopKValidation:
    @pytest.mark.parametrize("top_k", [0, -1, -100])
    def test_nonpositive(self, top_k) -> None:
        retriever, _ = build_retriever()
        with pytest.raises(RetrievalValidationError, match="大於 0"):
            retriever.retrieve("query", top_k)

    def test_above_maximum(self) -> None:
        retriever, _ = build_retriever(max_top_k=10)
        with pytest.raises(RetrievalValidationError, match="不可超過 10"):
            retriever.retrieve("query", 11)

    def test_equal_to_maximum(self) -> None:
        retriever, store = build_retriever(max_top_k=10)
        assert retriever.retrieve("query", 10) == []
        assert store.last_top_k == 10

    def test_bool_rejected(self) -> None:
        retriever, _ = build_retriever()
        with pytest.raises(RetrievalValidationError, match="整數"):
            retriever.retrieve("query", True)

    def test_none_uses_default(self) -> None:
        retriever, store = build_retriever(default_top_k=3)
        retriever.retrieve("query")
        assert store.last_top_k == 3

    def test_invalid_constructor_values(self) -> None:
        with pytest.raises(ValueError):
            build_retriever(default_top_k=0)
        with pytest.raises(ValueError, match="不可大於"):
            build_retriever(default_top_k=11, max_top_k=10)


class TestResultShaping:
    def test_results_and_metadata(self) -> None:
        backend = FakeEmbeddingBackend()
        retriever, store = build_retriever(backend=backend)
        populate(store, backend, {"c1": "one", "c2": "two", "c3": "three"})
        results = retriever.retrieve("one", 2)
        assert len(results) == 2
        assert [item.distance for item in results] == sorted(item.distance for item in results)
        assert results[0].document_id == "doc-synthetic-1"
        assert "document_id" not in results[0].metadata
        assert {"source", "file_name", "file_type", "chunk_index"} <= results[0].metadata.keys()
        assert "embedding" not in results[0].model_dump()
        assert "vector" not in results[0].model_dump()

    def test_counts_and_empty_store(self) -> None:
        backend = FakeEmbeddingBackend()
        retriever, store = build_retriever(backend=backend)
        assert retriever.retrieve("query", 2) == []
        populate(store, backend, {"c1": "one", "c2": "two"})
        assert len(retriever.retrieve("one", 1)) == 1
        assert len(retriever.retrieve("one", 20)) == 2


class TestDistanceSemantics:
    def test_cosine_score_and_identical_text(self) -> None:
        backend = FakeEmbeddingBackend()
        retriever, store = build_retriever(backend=backend)
        populate(store, backend, {"c1": "same text"})
        result = retriever.retrieve("same text", 1)[0]
        assert result.distance == pytest.approx(0.0)
        assert result.score == pytest.approx(1.0 - result.distance)

    def test_l2_has_no_score(self) -> None:
        backend = FakeEmbeddingBackend()
        retriever, store = build_retriever(store=MemorySearchStore("l2"), backend=backend)
        populate(store, backend, {"c1": "same"})
        assert retriever.retrieve("same", 1)[0].score is None


class TestErrorTranslation:
    def test_embedding_failure(self) -> None:
        retriever, _ = build_retriever(backend=FailingEmbeddingBackend())
        with pytest.raises(RetrievalBackendError, match="向量表示") as caught:
            retriever.retrieve("query")
        assert isinstance(caught.value.__cause__, EmbeddingError)

    def test_store_failure(self) -> None:
        retriever, _ = build_retriever(store=FailingSearchStore())
        with pytest.raises(RetrievalBackendError, match="向量資料庫") as caught:
            retriever.retrieve("query")
        assert isinstance(caught.value.__cause__, VectorStoreError)


class TestDuplicateIngestion:
    def test_upsert(self) -> None:
        backend = FakeEmbeddingBackend()
        retriever, store = build_retriever(backend=backend)
        for _ in range(3):
            populate(store, backend, {"c1": "one"})
        assert len(retriever.retrieve("one")) == 1


class TestEmbeddingCompatibility:
    def test_contract_and_cache(self) -> None:
        backend = FakeEmbeddingBackend()
        retriever, store = build_retriever(backend=backend)
        for _ in range(3):
            retriever.retrieve("query")
        assert store.compatibility_calls == [(backend.model_name, backend.model_revision, backend.dimension, backend.normalizes_embeddings)]

    def test_incompatible_store_blocks_search(self) -> None:
        retriever, _ = build_retriever(store=IncompatibleSearchStore())
        with pytest.raises(RetrievalBackendError, match="不相容") as caught:
            retriever.retrieve("query")
        assert isinstance(caught.value.__cause__, VectorStoreError)

    def test_same_dimension_different_model_is_rejected(self, tmp_path: Path) -> None:
        store = ChromaStore(tmp_path / "db", "different_model")
        store.ensure_embedding_compatibility("vendor/model-A", "revision-a", 4, True)
        backend = FakeEmbeddingBackend(4)
        store.add_chunks([make_chunk("c1", "one", 0)], [[1.0, 0.0, 0.0, 0.0]], "doc-1")
        retriever, _ = build_retriever(store=store, backend=backend)
        with pytest.raises(RetrievalBackendError, match="不相容"):
            retriever.retrieve("one")

    def test_matching_model_is_accepted(self, tmp_path: Path) -> None:
        backend = FakeEmbeddingBackend(4)
        store = ChromaStore(tmp_path / "db", "matching_model")
        store.ensure_embedding_compatibility(backend.model_name, backend.model_revision, 4, backend.normalizes_embeddings)
        chunk = make_chunk("c1", "one", 0)
        store.add_chunks([chunk], backend.embed_documents([chunk.text]), "doc-1")
        retriever, _ = build_retriever(store=store, backend=backend)
        assert retriever.retrieve("one", 1)[0].chunk_id == "c1"


class TestChromaSearchContract:
    def test_empty_and_validation(self, tmp_path: Path) -> None:
        store = ChromaStore(tmp_path / "db", "empty_search")
        assert store.search_by_vector([1.0, 0.0], 2) == []
        with pytest.raises(VectorStoreError, match="top_k"):
            store.search_by_vector([1.0], 0)
        with pytest.raises(VectorStoreError, match="查詢向量"):
            store.search_by_vector([], 1)
        assert store.distance_metric == "cosine"

    def test_order_and_limit(self, tmp_path: Path) -> None:
        store = ChromaStore(tmp_path / "db", "ordered_search")
        chunks = [make_chunk(f"c{i}", str(i), i) for i in range(2)]
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        store.add_chunks(chunks, vectors, "doc-1")
        matches = store.search_by_vector(vectors[0], 2)
        assert [match.chunk_id for match in matches] == ["c0", "c1"]
        assert matches[0].distance < matches[1].distance
        assert matches[0].metadata["document_id"] == "doc-1"

        extra_chunks = [make_chunk(f"c{i}", str(i), i) for i in range(2, 4)]
        extra_vectors = [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        store.add_chunks(extra_chunks, extra_vectors, "doc-1")
        assert len(store.search_by_vector(vectors[0], 2)) == 2


def store_with_response(tmp_path: Path, response: dict) -> ChromaStore:
    store = ChromaStore(tmp_path / "db", "malformed")

    class StubCollection:
        def count(self): return 1
        def query(self, **kwargs): return response

    store._collection = StubCollection()
    return store


class TestChromaMalformedResponse:
    @pytest.mark.parametrize("response,match", [
        ({"ids": [["c1"]], "documents": [["x"]], "metadatas": [[{}]]}, "缺少必要欄位"),
        ({"ids": [["c1", "c2"]], "documents": [["x"]], "metadatas": [[{}, {}]], "distances": [[0.0, 1.0]]}, "長度與 ids 不一致"),
        ({"ids": [["c1"]], "documents": [[None]], "metadatas": [[{}]], "distances": [[0.0]]}, "缺少文字內容"),
        ({"ids": [["c1"]], "documents": [["x"]], "metadatas": [[{}]], "distances": [[float("nan")]]}, "distance"),
    ])
    def test_malformed(self, tmp_path, response, match) -> None:
        with pytest.raises(VectorStoreError, match=match):
            store_with_response(tmp_path, response).search_by_vector([1.0], 1)

    def test_none_metadata_is_valid(self, tmp_path) -> None:
        response = {"ids": [["c1"]], "documents": [["x"]], "metadatas": [[None]], "distances": [[0.0]]}
        assert store_with_response(tmp_path, response).search_by_vector([1.0], 1)[0].metadata == {}


class TestPersistence:
    def test_reopen(self, tmp_path: Path) -> None:
        backend = FakeEmbeddingBackend()
        writer = ChromaStore(tmp_path / "db", "persistent_retrieval")
        writer.ensure_embedding_compatibility(backend.model_name, backend.model_revision, backend.dimension, backend.normalizes_embeddings)
        chunk = make_chunk("c1", "persistent", 0)
        writer.add_chunks([chunk], backend.embed_documents([chunk.text]), "doc-1")
        reopened = ChromaStore(tmp_path / "db", "persistent_retrieval")
        retriever, _ = build_retriever(store=reopened, backend=backend)
        result = retriever.retrieve("persistent", 1)[0]
        assert (result.chunk_id, result.document_id, result.text) == ("c1", "doc-1", "persistent")
