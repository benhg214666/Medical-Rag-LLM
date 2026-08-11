"""FastAPI 應用程式進入點。

啟動方式：
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api import documents, models, query, rag, retrieval
from app.core.config import settings
from app.core.logging import setup_logging
from app.schemas.response import HealthResponse, RootResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """管理應用程式的啟動與關閉流程。"""
    setup_logging()
    logger.info("Medical RAG Started")
    yield
    logger.info("Medical RAG Stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(query.router)
app.include_router(documents.router)
app.include_router(models.router)
app.include_router(retrieval.router)
app.include_router(rag.router)


@app.get("/", response_model=RootResponse, tags=["system"])
def read_root() -> RootResponse:
    """回傳專案基本資訊。"""
    return RootResponse(
        project=settings.app_name,
        version=settings.app_version,
        status="running",
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """健康檢查 endpoint，供監控或部署流程使用。"""
    return HealthResponse(status="healthy")
