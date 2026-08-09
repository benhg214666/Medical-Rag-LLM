"""以 Sentence Transformers 執行的 lazy-loading 本地 embedding。"""

from typing import Any

from app.embeddings.base import EmbeddingBackend, EmbeddingError


class LocalEmbeddingBackend(EmbeddingBackend):
    """本地模型後端；直到第一次向量化才載入模型。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        if not model_name.strip():
            raise ValueError("embedding model_name 不可為空白")
        if device not in {"cpu", "auto", "cuda"}:
            raise ValueError("embedding device 僅支援 cpu、auto 或 cuda")
        self._model_name = model_name
        self.device = device
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        model = self._load_model()
        dimension = model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise EmbeddingError("embedding 模型回傳無效的向量維度")
        return dimension

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._resolve_device(),
            )
        except Exception as exc:
            raise EmbeddingError(
                f"無法載入本地 embedding 模型 '{self._model_name}'"
            ) from exc
        return self._model

    def _uses_e5_prefix(self) -> bool:
        return "e5" in self._model_name.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._load_model().encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [[float(value) for value in vector] for vector in vectors]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("本地 embedding 推論失敗") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("文件文字不可為空白")
        prepared = [f"passage: {text}" for text in texts] if self._uses_e5_prefix() else texts
        return self._encode(prepared)

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("查詢文字不可為空白")
        prepared = f"query: {text}" if self._uses_e5_prefix() else text
        return self._encode([prepared])[0]
