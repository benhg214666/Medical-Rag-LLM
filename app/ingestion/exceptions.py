"""Ingestion 子系統的自訂例外。

集中定義例外的目的，是讓上層（API router）能依照錯誤「種類」決定 HTTP 狀態碼，
而不需要去解析錯誤訊息字串。所有例外都繼承自 IngestionError，
上層若只想「攔截所有匯入相關錯誤」，捕捉 IngestionError 即可。
"""


class IngestionError(Exception):
    """所有 ingestion 相關錯誤的基底類別。"""


class UnsupportedFileTypeError(IngestionError):
    """副檔名不在支援清單內（目前支援 .txt / .pdf / .docx）。"""


class DocumentLoadError(IngestionError):
    """文件存在但無法正確讀取，例如編碼錯誤、檔案損毀。"""


class EmptyDocumentError(IngestionError):
    """文件可以讀取，但抽取不到任何有效文字內容。"""


class InvalidChunkConfigError(IngestionError):
    """切塊參數不合法，例如 chunk_overlap >= chunk_size。"""
