"""Factory for configured LLM providers."""

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.local_backend import OpenAICompatibleLLM


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider != "openai_compatible":
        raise ValueError(f"不支援的 LLM provider: {settings.llm_provider}")
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
