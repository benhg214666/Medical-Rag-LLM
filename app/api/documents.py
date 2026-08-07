"""Documents API：醫療文件上傳與前處理。

分層原則：本模組只負責 HTTP 層的事情 —— 驗證輸入、決定狀態碼、組裝回應。
實際的 load / clean / chunk / save 全部委派給 app.ingestion，
這樣同一套匯入邏輯未來也能被 CLI 腳本或批次工具重用，不必經過 HTTP。

錯誤對應：
  400 空檔案、內容損毀、抽不到文字（含掃描式 PDF）
  413 檔案超過上限
  415 副檔名不支援
  500 未預期錯誤或寫檔失敗
"""

import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, ensure_data_directories, get_settings
from app.ingestion.exceptions import (
    DocumentLoadError,
    EmptyDocumentError,
    IngestionError,
    InvalidChunkConfigError,
    UnsupportedFileTypeError,
)
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.models import DocumentChunk
from app.ingestion.pipeline import ingest_document, sanitize_filename
from app.schemas.response import (
    ChunkPreview,
    DocumentUploadResponse,
    IngestionStatistics,
    ModuleStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 回應中最多附上幾個 chunk 預覽，以及每個預覽的字元上限。
# 刻意保守：預覽只為了讓上傳者確認切塊合理，不是資料傳遞管道。
_MAX_PREVIEW_CHUNKS = 3
_PREVIEW_CHAR_LIMIT = 200

def _build_document_id(content: bytes) -> str:
    """依檔案內容產生穩定的 SHA-256 文件識別碼。"""
    return hashlib.sha256(content).hexdigest()


def _build_raw_file_name(safe_name: str, document_id: str) -> str:
    """建立不易覆蓋的原始檔名。

    例如：
        note.txt -> note_a31f829c.txt
    """
    path = Path(safe_name)
    return f"{path.stem}_{document_id[:8]}{path.suffix.lower()}"

@router.get("/status", response_model=ModuleStatusResponse)
def get_documents_status() -> ModuleStatusResponse:
    """回報 documents 模組的實作狀態。"""
    return ModuleStatusResponse(module="documents", status="available")


def _validate_extension(file_name: str) -> str:
    """檢查副檔名是否受支援。

    刻意不信任 Content-Type：那是由客戶端宣告的，可以任意偽造。
    以副檔名為準，並在後續由實際的 parser 再做一次內容驗證。

    Raises:
        HTTPException: 415，副檔名不支援。
    """
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"不支援的檔案格式 '{suffix or '(無副檔名)'}'，"
                f"目前僅支援：{supported}"
            ),
        )
    return suffix


def _build_previews(chunks: list[DocumentChunk]) -> list[ChunkPreview]:
    """從完整 chunk 清單擷取少量預覽。"""
    previews: list[ChunkPreview] = []
    for chunk in chunks[:_MAX_PREVIEW_CHUNKS]:
        text = chunk.text[:_PREVIEW_CHAR_LIMIT]
        if len(chunk.text) > _PREVIEW_CHAR_LIMIT:
            text += "..."
        previews.append(
            ChunkPreview(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                paragraph_number=chunk.paragraph_number,
                text_preview=text,
            )
        )
    return previews


def _remove_file_quietly(path: Path) -> None:
    """刪除檔案，忽略失敗。

    用於失敗路徑的清理：此時已經要回報錯誤了，
    清理本身再拋例外只會遮蔽真正的錯誤原因。
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("清理暫存檔失敗：%s（%s）", path.name, type(exc).__name__)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="上傳醫療文件並執行前處理",
)
async def upload_document(
    file: UploadFile = File(..., description="支援 .txt / .pdf / .docx"),
    settings: Settings = Depends(get_settings),
) -> DocumentUploadResponse:
    """上傳文件，執行 load -> clean -> chunk -> JSON 流程。

    上傳的原始檔會保存到 RAW_DATA_DIR，處理結果 JSON 寫入 PROCESSED_DATA_DIR。
    相同內容重複上傳會覆寫同一個 JSON（依內容 hash 命名），不會產生重複資料。
    """
    original_name = file.filename or "document"
    _validate_extension(original_name)

    safe_name = sanitize_filename(original_name)

    # 讀取整份內容以取得實際大小。Content-Length 由客戶端提供，同樣不可信任。
    content = await file.read()
    size_bytes = len(content)

    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="上傳的檔案是空的，沒有任何內容可以處理。",
        )

    if size_bytes > settings.max_upload_size_bytes:
        raise HTTPException(
            # 413：新版 Starlette 改名為 CONTENT_TOO_LARGE，直接用數字最穩定
            status_code=413,
            detail=(
                f"檔案大小 {size_bytes / 1024 / 1024:.1f} MB 超過上限 "
                f"{settings.max_upload_size_mb} MB。"
            ),
        )

    document_id = _build_document_id(content)
    raw_file_name = _build_raw_file_name(safe_name, document_id)

    ensure_data_directories(settings)
    raw_path = settings.raw_data_dir / raw_file_name

    try:
        raw_path.write_bytes(content)
    except OSError as exc:
        logger.exception("儲存上傳檔案失敗：%s", raw_file_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器無法儲存上傳的檔案，請稍後再試或聯絡管理員。",
        ) from exc

    try:
        result = ingest_document(raw_path, settings)
    except UnsupportedFileTypeError as exc:
        _remove_file_quietly(raw_path)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except (EmptyDocumentError, DocumentLoadError, InvalidChunkConfigError) as exc:
        _remove_file_quietly(raw_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except IngestionError as exc:
        # 內部 log 保留完整 traceback，對外只給概括訊息，不洩漏路徑或堆疊
        logger.exception("處理文件時發生匯入錯誤：%s", raw_file_name)
        _remove_file_quietly(raw_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器處理文件時發生錯誤，請稍後再試或聯絡管理員。",
        ) from exc
    except Exception as exc:
        logger.exception("處理文件時發生未預期錯誤：%s", raw_file_name)
        _remove_file_quietly(raw_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器發生未預期的錯誤，請稍後再試或聯絡管理員。",
        ) from exc

    return DocumentUploadResponse(
        status="processed",
        document_id=result.document_id,
        file_name=safe_name,
        file_type=result.file_type,
        statistics=IngestionStatistics(
            loaded_units=result.statistics.loaded_units,
            cleaned_units=result.statistics.cleaned_units,
            chunk_count=result.statistics.chunk_count,
            total_characters=result.statistics.total_characters,
        ),
        output_file=result.output_file or "",
        chunk_previews=_build_previews(result.chunks),
    )
