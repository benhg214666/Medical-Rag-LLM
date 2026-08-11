"""Thin HTTP layer for retrieval-augmented answer generation."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.llm.base import LLMError
from app.rag.dependencies import get_rag_service
from app.rag.service import RAGService
from app.retrieval.exceptions import RetrievalBackendError, RetrievalValidationError
from app.schemas.request import RAGAskRequest
from app.schemas.response import RAGAskResponse, RAGSourceItem, RAGStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.get("/status", response_model=RAGStatusResponse)
def get_rag_status(settings: Settings = Depends(get_settings)) -> RAGStatusResponse:
    return RAGStatusResponse(
        module="rag", status="available", model=settings.llm_model_name
    )


@router.post("/ask", response_model=RAGAskResponse)
def ask(
    payload: RAGAskRequest,
    service: RAGService = Depends(get_rag_service),
) -> RAGAskResponse:
    try:
        result = service.answer(payload.query, payload.top_k)
    except RetrievalValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RetrievalBackendError, LLMError) as exc:
        logger.exception("RAG 回答產生失敗")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="伺服器目前無法產生回答，請稍後再試。",
        ) from exc
    return RAGAskResponse(
        answer=result.answer,
        model=result.model,
        sources=[RAGSourceItem(**source.model_dump()) for source in result.sources],
    )
