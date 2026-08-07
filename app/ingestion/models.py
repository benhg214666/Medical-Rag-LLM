"""Ingestion 子系統的資料模型。

這裡定義文件在管線中流動時的兩種形態：

    檔案 --load--> LoadedDocument --clean--> LoadedDocument --chunk--> DocumentChunk

LoadedDocument 代表「一個文件單位」：PDF 的一頁、DOCX 的一個段落、TXT 的整份檔案。
DocumentChunk 代表「一個可被檢索的片段」，是未來 embedding 與向量檢索的最小單位。

使用 Pydantic BaseModel 而非 dataclass，是為了直接取得 JSON 序列化與型別驗證，
與專案其他部分（FastAPI schemas）保持一致。
"""

from typing import Any

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """從原始檔案抽取出來的一個文件單位。

    Attributes:
        text: 抽取出的純文字內容。
        source: 來源識別字串，使用相對路徑或檔名，不含本機絕對路徑。
        file_name: 原始檔名。
        file_type: 副檔名（不含點），例如 "pdf"。
        page_number: PDF 頁碼，從 1 開始；非 PDF 為 None。
        paragraph_number: DOCX 段落編號，從 1 開始；非 DOCX 為 None。
        metadata: 額外資訊，保留給未來擴充。
    """

    text: str
    source: str
    file_name: str
    file_type: str
    page_number: int | None = None
    paragraph_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """切塊後的文字片段，是未來 embedding 與檢索的最小單位。

    Attributes:
        chunk_id: 確定性（deterministic）識別碼，見 chunker.build_chunk_id。
        text: 片段文字。
        source: 繼承自 LoadedDocument。
        file_name: 繼承自 LoadedDocument。
        file_type: 繼承自 LoadedDocument。
        page_number: 繼承自 LoadedDocument。
        paragraph_number: 繼承自 LoadedDocument。
        chunk_index: 在「整份文件」中的序號，從 0 開始且連續。
        start_char: 在所屬 LoadedDocument.text 中的起始位置（含）。
        end_char: 在所屬 LoadedDocument.text 中的結束位置（不含）。
        metadata: 繼承自 LoadedDocument 的額外資訊。
    """

    chunk_id: str
    text: str
    source: str
    file_name: str
    file_type: str
    page_number: int | None = None
    paragraph_number: int | None = None
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingConfig(BaseModel):
    """本次處理實際採用的切塊參數，會一併寫入輸出 JSON 以利重現。"""

    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int


class IngestionStatistics(BaseModel):
    """一次匯入的摘要統計。

    Attributes:
        loaded_units: 載入的文件單位數（清理前）。
        cleaned_units: 清理後仍有內容的文件單位數。
        chunk_count: 產生的 chunk 總數。
        total_characters: 所有 chunk 的字元總數（含 overlap 重複部分）。
    """

    loaded_units: int
    cleaned_units: int
    chunk_count: int
    total_characters: int


class IngestionResult(BaseModel):
    """ingest_document 的完整回傳結果，同時也是輸出 JSON 的結構。

    Attributes:
        document_id: 依「檔案內容」計算的 SHA-256 截短值，內容相同則 ID 相同。
        source_file: 相對路徑或檔名，不含本機絕對路徑。
        file_type: 副檔名（不含點）。
        created_at: ISO 8601 格式的處理時間。
        processing: 本次使用的切塊參數。
        statistics: 摘要統計。
        chunks: 全部 chunk。
        output_file: 寫出的 JSON 相對路徑；未寫檔時為 None。
    """

    document_id: str
    source_file: str
    file_type: str
    created_at: str
    processing: ProcessingConfig
    statistics: IngestionStatistics
    chunks: list[DocumentChunk] = Field(default_factory=list)
    output_file: str | None = None
