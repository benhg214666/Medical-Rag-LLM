"""Factory for configured LLM providers."""

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.local_backend import (
    OllamaOpenAICompatibleLLM,
    OpenAICompatibleLLM,
)
from app.llm.network import validate_local_llm_base_url


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider != "openai_compatible":
        raise ValueError(f"不支援的 LLM provider: {settings.llm_provider}")
    base_url = validate_local_llm_base_url(
        settings.llm_base_url,
        allow_private_network=settings.llm_allow_private_network,
    )
    provider_class = (
        OllamaOpenAICompatibleLLM
        if settings.llm_compatibility_mode == "ollama"
        else OpenAICompatibleLLM
    )
    return provider_class(
        base_url=base_url,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
