"""Models API：未來提供 embedding 與 LLM 模型的查詢與切換 endpoint。

Phase 1 僅提供 status endpoint，明確標示此模組尚未實作。
"""

import logging

from fastapi import APIRouter

from app.schemas.response import ModuleStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/status", response_model=ModuleStatusResponse)
def get_models_status() -> ModuleStatusResponse:
    """回報 models 模組的實作狀態。"""
    return ModuleStatusResponse(module="models", status="not_implemented")
