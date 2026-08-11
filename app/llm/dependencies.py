"""FastAPI dependencies for the shared LLM provider."""

from functools import lru_cache

from fastapi import Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider


@lru_cache(maxsize=1)
def _cached_provider() -> LLMProvider:
    return create_llm_provider(get_settings())


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    try:
        if settings is get_settings():
            return _cached_provider()
        return create_llm_provider(settings)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器無法初始化本地 LLM 服務。",
        ) from exc
