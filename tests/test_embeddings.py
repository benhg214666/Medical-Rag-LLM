"""Embedding abstraction 的純離線測試。"""

import sys
from types import SimpleNamespace
from app.indexing.dependencies import _get_cached_embedding_backend
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pytest

from app.core.config import Settings
from app.embeddings.base import EmbeddingBackend, EmbeddingError
from app.embeddings.factory import create_embedding_backend
from app.embeddings.local_embedding import LocalEmbeddingBackend
from tests.fakes import FakeEmbeddingBackend


def test_base_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingBackend()  # type: ignore[abstract]


def test_fake_backend_is_deterministic() -> None:
    backend = FakeEmbeddingBackend()
    assert backend.embed_query("中文病歷") == backend.embed_query("中文病歷")
    assert len(backend.embed_query("中文病歷")) == 4


def test_local_backend_is_lazy() -> None:
    backend = LocalEmbeddingBackend("intfloat/multilingual-e5-small")
    assert backend._model is None
    assert backend.model_name == "intfloat/multilingual-e5-small"


def test_local_backend_rejects_empty_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        LocalEmbeddingBackend("test/model", revision="  ")


def test_factory_creates_local_backend() -> None:
    settings = Settings(embedding_device="cpu")
    backend = create_embedding_backend(settings)

    assert isinstance(backend, LocalEmbeddingBackend)
    assert backend._model is None
    assert backend.model_revision == settings.embedding_model_revision


def test_factory_rejects_unsupported_provider() -> None:
    with pytest.raises(EmbeddingError, match="不支援"):
        create_embedding_backend(Settings(embedding_provider="external"))


def test_empty_documents_do_not_load_model() -> None:
    backend = LocalEmbeddingBackend("intfloat/multilingual-e5-small")
    assert backend.embed_documents([]) == []
    assert backend._model is None


def test_encode_uses_e5_prefix_and_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    class MockModel:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            calls["texts"] = texts
            calls["kwargs"] = kwargs
            return [[1.0, 0.0] for _ in texts]

        def get_sentence_embedding_dimension(self) -> int:
            return 2

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: MockModel()),
    )
    backend = LocalEmbeddingBackend("intfloat/multilingual-e5-small")
    vectors = backend.embed_documents(["胸痛", "fever"])

    assert len(vectors) == 2
    assert calls["texts"] == ["passage: 胸痛", "passage: fever"]
    assert calls["kwargs"]["normalize_embeddings"] is True
    assert backend.embed_query("治療") == [1.0, 0.0]
    assert calls["texts"] == ["query: 治療"]


def test_non_e5_model_does_not_add_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_texts: list[list[str]] = []

    class MockModel:
        def encode(
            self,
            texts: list[str],
            **kwargs: object,
        ) -> list[list[float]]:
            encoded_texts.append(texts)
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: MockModel()
        ),
    )

    backend = LocalEmbeddingBackend("sentence-transformers/test-model")
    backend.embed_documents(["document"])
    backend.embed_query("question")

    assert encoded_texts == [["document"], ["question"]]

def test_embedding_backend_is_reused_across_requests() -> None:
    _get_cached_embedding_backend.cache_clear()

    try:
        first = _get_cached_embedding_backend(
            "local",
            "intfloat/multilingual-e5-small",
            "revision-a",
            "cpu",
        )
        second = _get_cached_embedding_backend(
            "local",
            "intfloat/multilingual-e5-small",
            "revision-a",
            "cpu",
        )
        different_revision = _get_cached_embedding_backend(
            "local",
            "intfloat/multilingual-e5-small",
            "revision-b",
            "cpu",
        )

        assert first is second
        assert first is not different_revision
        assert first._model is None
    finally:
        _get_cached_embedding_backend.cache_clear()

def test_model_is_loaded_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls = 0
    calls_lock = Lock()

    class MockModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 2

    def create_mock_model(*args: object, **kwargs: object) -> MockModel:
        nonlocal constructor_calls

        with calls_lock:
            constructor_calls += 1

        # 放大 concurrency race，確保沒有 lock 時測試會失敗。
        time.sleep(0.05)
        return MockModel()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=create_mock_model),
    )

    backend = LocalEmbeddingBackend("intfloat/multilingual-e5-small")

    with ThreadPoolExecutor(max_workers=8) as executor:
        dimensions = list(executor.map(lambda _: backend.dimension, range(8)))

    assert dimensions == [2] * 8
    assert constructor_calls == 1

def test_backends_report_normalization_contract() -> None:
    local_backend = LocalEmbeddingBackend(
        "intfloat/multilingual-e5-small"
    )
    fake_backend = FakeEmbeddingBackend()

    assert local_backend.normalizes_embeddings is True
    assert fake_backend.normalizes_embeddings is False

def test_model_revision_is_passed_to_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_kwargs: dict[str, object] = {}

    class MockModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 2

    def create_mock_model(
        *args: object,
        **kwargs: object,
    ) -> MockModel:
        constructor_kwargs.update(kwargs)
        return MockModel()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=create_mock_model
        ),
    )

    backend = LocalEmbeddingBackend(
        "intfloat/multilingual-e5-small",
        device="cpu",
        revision="fixed-test-revision",
    )

    assert backend.dimension == 2
    assert backend.model_revision == "fixed-test-revision"
    assert constructor_kwargs["revision"] == (
        "fixed-test-revision"
    )
    assert constructor_kwargs["device"] == "cpu"


def test_dimension_prefers_current_sentence_transformers_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class MockModel:
        def get_embedding_dimension(self) -> int:
            calls.append("current")
            return 384

        def get_sentence_embedding_dimension(self) -> int:
            calls.append("deprecated")
            return 384

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: MockModel()
        ),
    )

    backend = LocalEmbeddingBackend("intfloat/multilingual-e5-small")

    assert backend.dimension == 384
    assert calls == ["current"]
