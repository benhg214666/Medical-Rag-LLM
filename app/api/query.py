"""Query API：未來提供醫療問答（RAG 查詢）的 endpoint。

Phase 1 僅提供 status endpoint，明確標示此模組尚未實作。
"""

import logging

from fastapi import APIRouter

from app.schemas.response import ModuleStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


@router.get("/status", response_model=ModuleStatusResponse)
def get_query_status() -> ModuleStatusResponse:
    """回報 query 模組的實作狀態。"""
    return ModuleStatusResponse(module="query", status="not_implemented")
