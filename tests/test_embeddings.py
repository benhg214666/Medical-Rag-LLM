"""Embedding abstraction 的純離線測試。"""

import sys
from types import SimpleNamespace

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


def test_factory_creates_local_backend() -> None:
    backend = create_embedding_backend(Settings(embedding_device="cpu"))
    assert isinstance(backend, LocalEmbeddingBackend)
    assert backend._model is None


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
