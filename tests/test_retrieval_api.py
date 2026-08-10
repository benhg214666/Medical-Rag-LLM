"""Phase 4 Retrieval API 離線測試。"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.dependencies import get_vector_retriever
from app.retrieval.vector_retriever import VectorRetriever
from tests.fakes import FakeEmbeddingBackend
from tests.test_retrieval import IncompatibleSearchStore, MemorySearchStore, make_chunk


@pytest.fixture
def client() -> Iterator[tuple[TestClient, MemorySearchStore]]:
    backend = FakeEmbeddingBackend()
    store = MemorySearchStore()
    chunks = [
        make_chunk("c1", "Patient takes metformin.", 0, {"page": 1}),
        make_chunk("c2", "Patient had surgery.", 1),
        make_chunk("c3", "No known allergies.", 2),
    ]
    store.add_chunks(chunks, backend.embed_documents([chunk.text for chunk in chunks]), "doc-api")
    retriever = VectorRetriever(backend, store, default_top_k=2, max_top_k=3)
    app.dependency_overrides[get_vector_retriever] = lambda: retriever
    with TestClient(app) as test_client:
        yield test_client, store
    app.dependency_overrides.clear()


class TestModuleStatus:
    def test_retrieval_status_and_existing_endpoints(self, client) -> None:
        test_client, _ = client
        assert test_client.get("/api/retrieval/status").json() == {"module": "retrieval", "status": "available"}
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/api/query/status").status_code == 200
        assert test_client.get("/api/documents/status").status_code == 200
        assert test_client.get("/api/models/status").status_code == 200


class TestSearchSuccess:
    def test_sorted_response_contract(self, client) -> None:
        response = client[0].post("/api/retrieval/search", json={"query": "Patient takes metformin.", "top_k": 2})
        assert response.status_code == 200
        body = response.json()
        assert body["top_k"] == 2
        assert body["result_count"] == 2
        assert set(body["results"][0]) == {"chunk_id", "document_id", "text", "distance", "score", "distance_metric", "metadata"}
        assert "embedding" not in response.text
        assert "query_embeddings" not in response.text
        assert body["results"][0]["score"] == pytest.approx(1.0 - body["results"][0]["distance"])
        assert body["results"][0]["metadata"]["page"] == 1

    @pytest.mark.parametrize("payload,expected", [({"query": "query"}, 2), ({"query": "query", "top_k": 1}, 1), ({"query": "query", "top_k": 3}, 3)])
    def test_default_custom_and_total(self, client, payload, expected) -> None:
        body = client[0].post("/api/retrieval/search", json=payload).json()
        assert body["top_k"] == expected
        assert body["result_count"] == expected


class TestSearchValidation:
    @pytest.mark.parametrize("query", ["   ", "\n", "\t"])
    def test_blank_is_400(self, client, query) -> None:
        assert client[0].post("/api/retrieval/search", json={"query": query}).status_code == 400

    @pytest.mark.parametrize("payload", [{"query": ""}, {}, {"query": "q", "top_k": 0}, {"query": "q", "top_k": -1}, {"query": "q", "top_k": -50}])
    def test_schema_validation_is_422(self, client, payload) -> None:
        assert client[0].post("/api/retrieval/search", json=payload).status_code == 422

    def test_over_maximum_is_400(self, client) -> None:
        assert client[0].post("/api/retrieval/search", json={"query": "q", "top_k": 4}).status_code == 400

    @pytest.mark.parametrize("top_k", [True, False, 1.0, 2.5, "1", "5"])
    def test_strict_integer(self, client, top_k) -> None:
        assert client[0].post("/api/retrieval/search", json={"query": "q", "top_k": top_k}).status_code == 422

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_valid_integer(self, client, top_k) -> None:
        response = client[0].post("/api/retrieval/search", json={"query": "q", "top_k": top_k})
        assert response.status_code == 200
        assert response.json()["top_k"] == top_k


class TestIncompatibleEmbedding:
    def test_safe_500(self) -> None:
        retriever = VectorRetriever(FakeEmbeddingBackend(), IncompatibleSearchStore(), 2, 3)
        app.dependency_overrides[get_vector_retriever] = lambda: retriever
        try:
            with TestClient(app) as test_client:
                response = test_client.post("/api/retrieval/search", json={"query": "q"})
            assert response.status_code == 500
            assert "collection" not in response.json()["detail"]
            assert "fake-embedding" not in response.text
            assert "Traceback" not in response.text
        finally:
            app.dependency_overrides.clear()


class TestEmptyVectorStore:
    def test_empty_is_success(self) -> None:
        retriever = VectorRetriever(FakeEmbeddingBackend(), MemorySearchStore(), 2, 3)
        app.dependency_overrides[get_vector_retriever] = lambda: retriever
        try:
            with TestClient(app) as test_client:
                response = test_client.post("/api/retrieval/search", json={"query": "q"})
            assert response.status_code == 200
            assert response.json()["result_count"] == 0
            assert response.json()["results"] == []
        finally:
            app.dependency_overrides.clear()
