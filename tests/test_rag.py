"""Phase 5 prompt, orchestration, local-provider, and API tests."""

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMError
from app.llm.local_backend import OpenAICompatibleLLM
from app.main import app
from app.prompts.prompt_builder import build_rag_prompt
from app.rag.dependencies import get_rag_service
from app.rag.service import NO_CONTEXT_ANSWER, RAGService
from app.retrieval.models import RetrievalResult


def make_result(text: str = "Patient takes metformin.") -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        text=text,
        distance=0.12,
        score=0.88,
        distance_metric="cosine",
        metadata={"page_number": 2, "file_name": "record.txt"},
    )


class FakeRetriever:
    default_top_k = 5
    max_top_k = 10

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not query.strip():
            from app.retrieval.exceptions import RetrievalValidationError

            raise RetrievalValidationError("查詢內容不可為空白")
        if top_k is not None and top_k > self.max_top_k:
            from app.retrieval.exceptions import RetrievalValidationError

            raise RetrievalValidationError("top_k 不可超過 10")
        self.calls.append((query, top_k))
        return self.results


class FakeLLM:
    model_name = "fake-local-model"

    def __init__(self, answer: str = "The patient takes metformin [1].") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


def test_prompt_contains_question_text_and_consistent_numbers() -> None:
    prompt = build_rag_prompt(
        "What medications?", [make_result(), make_result("No allergies.")]
    )
    assert "What medications?" in prompt.user
    assert "[Source 1]\nPatient takes metformin." in prompt.user
    assert "[Source 2]\nNo allergies." in prompt.user
    assert "untrusted reference data" in prompt.system


def test_service_retrieves_generates_and_preserves_sources() -> None:
    retriever = FakeRetriever([make_result()])
    llm = FakeLLM()
    result = RAGService(retriever, llm).answer("What medications?", 3)
    assert retriever.calls == [("What medications?", 3)]
    assert "Patient takes metformin." in llm.calls[0][1]
    assert result.answer == "The patient takes metformin [1]."
    assert result.model == "fake-local-model"
    assert result.sources[0].source_number == 1
    assert result.sources[0].metadata == {"page_number": 2, "file_name": "record.txt"}
    assert result.sources[0].distance == pytest.approx(0.12)


def test_empty_or_blank_context_skips_llm() -> None:
    llm = FakeLLM()
    result = RAGService(FakeRetriever([make_result("   ")]), llm).answer("Question")
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert llm.calls == []


def test_local_provider_payload_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"][0] == {"role": "system", "content": "system"}
        return httpx.Response(200, json={"choices": [{"message": {"content": " answer "}}]})

    client = httpx.Client(
        base_url="http://local.test/v1", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLM(
        base_url="http://unused", model_name="model", temperature=0, max_tokens=10,
        timeout=1, client=client,
    )
    assert provider.generate(system_prompt="system", user_prompt="question") == "answer"


def test_internal_local_client_does_not_inherit_proxy_environment() -> None:
    provider = OpenAICompatibleLLM(
        base_url="http://127.0.0.1:8001/v1",
        model_name="model",
        temperature=0,
        max_tokens=10,
        timeout=1,
    )
    try:
        assert provider._client._trust_env is False
    finally:
        provider._client.close()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"error": "internal"}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, content=b"not-json"),
    ],
)
def test_local_provider_converts_bad_responses(response: httpx.Response) -> None:
    client = httpx.Client(
        base_url="http://local.test/v1",
        transport=httpx.MockTransport(lambda request: response),
    )
    provider = OpenAICompatibleLLM(
        base_url="http://unused", model_name="model", temperature=0, max_tokens=10,
        timeout=1, client=client,
    )
    with pytest.raises(LLMError):
        provider.generate(system_prompt="system", user_prompt="question")


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("model timed out"),
    ],
)
def test_local_provider_converts_connection_and_timeout_errors(
    failure: httpx.RequestError,
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        failure.request = request
        raise failure

    client = httpx.Client(
        base_url="http://local.test/v1", transport=httpx.MockTransport(fail)
    )
    provider = OpenAICompatibleLLM(
        base_url="http://unused", model_name="model", temperature=0, max_tokens=10,
        timeout=1, client=client,
    )
    with pytest.raises(LLMError):
        provider.generate(system_prompt="system", user_prompt="question")


@pytest.fixture
def rag_client() -> Iterator[tuple[TestClient, FakeLLM]]:
    llm = FakeLLM()
    service = RAGService(FakeRetriever([make_result()]), llm)  # type: ignore[arg-type]
    app.dependency_overrides[get_rag_service] = lambda: service
    with TestClient(app) as client:
        yield client, llm
    app.dependency_overrides.clear()


def test_rag_api_success_and_status(rag_client) -> None:
    client, _ = rag_client
    response = client.post("/api/rag/ask", json={"query": "What medications?", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["answer"] == "The patient takes metformin [1]."
    assert response.json()["sources"][0]["chunk_id"] == "chunk-1"
    status_body = client.get("/api/rag/status").json()
    assert status_body["module"] == "rag"
    assert "base_url" not in status_body


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"query": ""}, 422),
        ({"query": "   "}, 400),
        ({"query": "q", "top_k": 0}, 422),
        ({"query": "q", "top_k": 11}, 400),
        ({"query": "q", "top_k": True}, 422),
    ],
)
def test_rag_api_validation(rag_client, payload, code) -> None:
    assert rag_client[0].post("/api/rag/ask", json=payload).status_code == code


def test_rag_api_converts_llm_error() -> None:
    class FailingLLM(FakeLLM):
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            raise LLMError("secret runtime details")

    service = RAGService(FakeRetriever([make_result()]), FailingLLM())  # type: ignore[arg-type]
    app.dependency_overrides[get_rag_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post("/api/rag/ask", json={"query": "question"})
        assert response.status_code == 503
        assert "secret" not in response.text
        assert "Traceback" not in response.text
    finally:
        app.dependency_overrides.clear()
