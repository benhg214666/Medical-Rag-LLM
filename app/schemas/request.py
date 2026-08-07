"""API 請求（request）資料結構。

目前所有 endpoint 都不使用 JSON request body，因此這裡刻意保持空白：

  - Phase 1 的 endpoint 皆為 GET，不接受參數。
  - Phase 2 的 POST /api/documents/upload 使用 multipart/form-data 上傳檔案，
    由 FastAPI 的 UploadFile 直接處理，不需要 Pydantic request model。

未來若加入查詢 API（例如 POST /api/query 帶 question 與 top_k），
才會在此定義對應的 model。此時不預先建立空殼 schema，避免無用的抽象。
"""
