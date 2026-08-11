"""LLM provider contract and project-level failures."""

from typing import Protocol


class LLMError(RuntimeError):
    """A configured model provider could not generate a valid response."""


class LLMProvider(Protocol):
    """Minimal boundary between RAG orchestration and model serving."""

    @property
    def model_name(self) -> str: ...

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...
