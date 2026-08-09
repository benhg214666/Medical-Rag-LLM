"""Models API：回報 embedding 模組的設定狀態。"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelsStatusResponse(BaseModel):
    module: str
    status: str
    embedding_provider: str
    embedding_model: str
    embedding_model_revision: str
    embedding_device: str


@router.get("/status", response_model=ModelsStatusResponse)
def get_models_status(
    settings: Settings = Depends(get_settings),
) -> ModelsStatusResponse:
    """只回報設定，不載入或下載 embedding 模型。"""
    return ModelsStatusResponse(
        module="models",
        status="available",
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model_name,
        embedding_model_revision=(
            settings.embedding_model_revision
        ),
        embedding_device=settings.embedding_device,
    )
