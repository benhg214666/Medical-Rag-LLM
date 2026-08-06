"""Documents API：未來提供醫療文件上傳與管理的 endpoint。

Phase 1 僅提供 status endpoint，明確標示此模組尚未實作。
"""

import logging

from fastapi import APIRouter

from app.schemas.response import ModuleStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/status", response_model=ModuleStatusResponse)
def get_documents_status() -> ModuleStatusResponse:
    """回報 documents 模組的實作狀態。"""
    return ModuleStatusResponse(module="documents", status="not_implemented")
