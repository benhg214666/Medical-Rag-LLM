"""Retrieval API：以語意相似度檢索醫療文件片段。

分層原則與 documents.py 一致：本模組只處理 HTTP 層 —— 驗證輸入、
決定狀態碼、組裝回應。實際檢索邏輯全部委派給 app.retrieval，
讓同一套檢索能被未來的 RAG 層或 CLI 工具直接重用，不必經過 HTTP。

錯誤對應：
  400 查詢空白、top_k 不合法或超出上限
  500 embedding 或 vector store 失敗

注意本階段只做「檢索」，不產生任何自然語言答案。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.retrieval.dependencies import get_vector_retriever
from app.retrieval.exceptions import (
    RetrievalBackendError,
    RetrievalValidationError,
)
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.request import RetrievalSearchRequest
from app.schemas.response import (
    ModuleStatusResponse,
    RetrievalResultItem,
    RetrievalSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.get("/status", response_model=ModuleStatusResponse)
def get_retrieval_status() -> ModuleStatusResponse:
    """回報 retrieval 模組的實作狀態。

    延續 Phase 1 的語意：這裡表示「程式模組是否已實作」，
    不是 runtime 健康檢查，因此不會去連線 vector store 或載入模型。
    """
    return ModuleStatusResponse(module="retrieval", status="available")


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="以語意相似度檢索相關文件片段",
)
def search(
    payload: RetrievalSearchRequest,
    retriever: VectorRetriever = Depends(get_vector_retriever),
) -> RetrievalSearchResponse:
    """執行 query -> embedding -> 向量搜尋 -> 排序結果。

    查無結果會回傳 200 與空陣列，而不是 404：
    「沒有語意相近的片段」是一個有效的檢索結果，
    與「資源不存在」是不同的語意。
    """
    try:
        results = retriever.retrieve(
            query=payload.query,
            top_k=payload.top_k,
        )
    except RetrievalValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RetrievalBackendError as exc:
        # 內部保留完整 traceback，對外只給概括訊息，
        # 不洩漏模型路徑、collection 名稱或堆疊。
        logger.exception("檢索失敗：後端元件錯誤")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器執行檢索時發生錯誤，請稍後再試。",
        ) from exc

    effective_top_k = (
        payload.top_k
        if payload.top_k is not None
        else retriever.default_top_k
    )

    return RetrievalSearchResponse(
        query=payload.query,
        top_k=effective_top_k,
        result_count=len(results),
        results=[
            RetrievalResultItem(**result.model_dump())
            for result in results
        ],
    )
