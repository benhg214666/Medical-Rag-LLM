"""文件匯入（ingestion）子系統。

負責把原始醫療文件轉換成可供未來檢索的文字片段：

    檔案 -> loaders -> cleaner -> chunker -> pipeline -> JSON

各模組職責：
  - loaders    依格式（TXT / PDF / DOCX）抽取文字，統一成 LoadedDocument
  - cleaner    純函式，只做格式正規化，絕不改寫醫療內容
  - chunker    以字元數切塊（非 token），產生確定性的 chunk_id
  - pipeline   串接上述階段並輸出 UTF-8 JSON
  - models     資料模型（LoadedDocument / DocumentChunk / IngestionResult）
  - exceptions 自訂例外，讓上層能依錯誤種類決定 HTTP 狀態碼
"""
