"""Ingestion pipeline：串接 load -> clean -> chunk -> save 的完整流程。

這是 ingestion 子系統的對外主要入口。API router 只需呼叫 ingest_document()，
不需要知道底下有幾個階段、各階段怎麼實作 —— 這讓 HTTP 層與業務邏輯徹底分離。

冪等性（idempotency）策略：
輸出檔名為 <safe_stem>_<document_id 前 8 碼>.json，而 document_id 由「檔案內容」
的 SHA-256 決定。因此相同內容重複處理會覆寫同一個檔案，不會產生重複資料。
這個設計讓管線可以安全地重跑，是後續接上向量資料庫時的重要前提。
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.cleaner import clean_text
from app.ingestion.exceptions import EmptyDocumentError, IngestionError
from app.ingestion.loaders import load_document
from app.ingestion.models import (
    IngestionResult,
    IngestionStatistics,
    LoadedDocument,
    ProcessingConfig,
)

logger = logging.getLogger(__name__)

# 檔名安全化：只保留英數、底線、連字號、點，其餘一律換成底線。
# 這能同時擋掉 path traversal（../）、shell 特殊字元與空白造成的問題。
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")

_MAX_STEM_LENGTH = 64


def sanitize_filename(file_name: str) -> str:
    """將任意檔名轉換成安全、可跨平台使用的檔名。

    處理項目：
      - 只取 basename，移除任何目錄成分（擋 path traversal）
      - 非英數字元替換為底線（中文檔名會變成底線，但 document_id 仍可區分不同檔案）
      - 移除開頭的點（避免產生 Linux 隱藏檔）
      - 限制長度，避免超出檔案系統上限

    Args:
        file_name: 原始檔名，可能來自使用者上傳，不可信任。

    Returns:
        安全的檔名。輸入完全無有效字元時回傳 "document"。
    """
    # PurePosixPath 與 PureWindowsPath 分隔符不同，直接用字串處理最保險
    base_name = file_name.replace("\\", "/").split("/")[-1]

    suffix = Path(base_name).suffix.lower()
    stem = Path(base_name).stem

    safe_stem = _UNSAFE_FILENAME_CHARS.sub("_", stem).lstrip(".")
    safe_stem = safe_stem[:_MAX_STEM_LENGTH].strip("_")

    if not safe_stem:
        safe_stem = "document"

    safe_suffix = _UNSAFE_FILENAME_CHARS.sub("", suffix)
    return f"{safe_stem}{safe_suffix}"


def compute_document_id(file_path: Path) -> str:
    """依「檔案內容」計算確定性的文件識別碼。

    使用內容而非檔名，因此同一份文件換個檔名上傳，仍會得到相同 ID，
    可用來偵測重複匯入。以 8 KB 為單位串流讀取，避免大檔一次載入記憶體。

    Returns:
        SHA-256 的前 16 個十六進位字元。
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()[:16]


def _clean_documents(documents: list[LoadedDocument]) -> list[LoadedDocument]:
    """清理每個文件單位，並剔除清理後變成空白的單位。

    會變成空白的常見情況：PDF 某頁只有頁碼或頁首頁尾，清理後不剩實質內容。
    """
    cleaned: list[LoadedDocument] = []
    for document in documents:
        cleaned_text = clean_text(document.text)
        if not cleaned_text:
            continue
        cleaned.append(document.model_copy(update={"text": cleaned_text}))
    return cleaned


def to_display_path(path: Path) -> str:
    """把輸出路徑轉成可安全對外顯示的相對路徑。

    絕不回傳主機絕對路徑：那會洩漏伺服器的目錄結構（例如使用者名稱、掛載點）。
    優先嘗試相對於目前工作目錄；若路徑不在工作目錄底下（例如測試用的 tmp_path），
    退而只保留「父目錄名/檔名」兩層，仍足以辨識而不暴露完整路徑。
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"{path.parent.name}/{path.name}"


def _write_result_json(result: IngestionResult, output_path: Path) -> None:
    """將結果寫成 UTF-8 JSON。

    ensure_ascii=False 是關鍵：預設 True 會把中文轉成 \\uXXXX escape，
    檔案雖然合法但人類無法直接閱讀，除錯時非常痛苦。

    Raises:
        IngestionError: 寫檔失敗（權限不足、磁碟空間不足等）。
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise IngestionError(
            f"寫入處理結果失敗：{type(exc).__name__}。"
            "請確認輸出目錄存在且具有寫入權限。"
        ) from exc


def ingest_document(
    file_path: Path,
    settings: Settings,
    write_output: bool = True,
    metadata: dict[str, Any] | None = None,
    source: str | None = None,
) -> IngestionResult:
    """執行完整的文件匯入流程。

    流程：
      1. 計算 document_id（依檔案內容）
      2. 載入 -> list[LoadedDocument]
      3. 逐單位清理，剔除空白單位
      4. 切塊 -> list[DocumentChunk]
      5. 產生統計
      6. 寫出 JSON 至 settings.processed_data_dir
      7. 回傳結構化結果

    Args:
        file_path: 待處理的檔案路徑。
        settings: 設定實例，提供切塊參數與輸出目錄。
        write_output: 是否寫出 JSON。測試時可設為 False 以避免產生檔案。

    Returns:
        IngestionResult，含全部 chunk 與統計資訊。

    Raises:
        UnsupportedFileTypeError: 副檔名不支援。
        DocumentLoadError: 檔案不存在或無法讀取。
        EmptyDocumentError: 抽不到文字，或清理後無有效內容。
        InvalidChunkConfigError: 切塊參數不合法。
        IngestionError: 寫檔失敗。
    """
    started_at = datetime.now(timezone.utc)

    document_id = compute_document_id(file_path)
    file_type = file_path.suffix.lower().lstrip(".")

    loaded_documents = load_document(file_path)
    if metadata or source:
        loaded_documents = [
            document.model_copy(
                update={
                    "source": source or document.source,
                    "metadata": {**document.metadata, **(metadata or {})},
                }
            )
            for document in loaded_documents
        ]
    cleaned_documents = _clean_documents(loaded_documents)

    if not cleaned_documents:
        raise EmptyDocumentError(
            f"'{file_path.name}' 清理後沒有任何有效內容，"
            "文件可能只包含格式標記或空白字元。"
        )

    chunks = chunk_documents(
        cleaned_documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
        document_id=document_id,
    )

    if not chunks:
        raise EmptyDocumentError(
            f"'{file_path.name}' 未能產生任何有效的文字片段。"
        )

    statistics = IngestionStatistics(
        loaded_units=len(loaded_documents),
        cleaned_units=len(cleaned_documents),
        chunk_count=len(chunks),
        total_characters=sum(len(chunk.text) for chunk in chunks),
    )

    result = IngestionResult(
        document_id=document_id,
        # 只記錄檔名，不記錄絕對路徑，避免洩漏主機目錄結構
        source_file=file_path.name,
        file_type=file_type,
        created_at=started_at.isoformat(),
        processing=ProcessingConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
        ),
        statistics=statistics,
        chunks=chunks,
        output_file=None,
    )

    if write_output:
        output_name = f"{document_id}.json"
        output_path = settings.processed_data_dir / output_name

        # 先設定 output_file 再寫檔，讓 JSON 內容包含自己的相對路徑
        result.output_file = to_display_path(output_path)
        _write_result_json(result, output_path)

    elapsed_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
    # 只記錄統計數字與檔名，絕不記錄文件正文或 chunk 內容
    logger.info(
        "匯入完成：file=%s type=%s units=%d->%d chunks=%d chars=%d 耗時=%.1fms",
        sanitize_filename(file_path.name),
        file_type,
        statistics.loaded_units,
        statistics.cleaned_units,
        statistics.chunk_count,
        statistics.total_characters,
        elapsed_ms,
    )

    return result
