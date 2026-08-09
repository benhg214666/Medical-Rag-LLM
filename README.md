# Medical Local RAG（醫療本地 RAG 系統）

## 一、專案目的

本專案的最終目標，是打造一套**完全在本地端執行**的醫療 RAG（Retrieval-Augmented Generation，檢索增強生成）系統。醫師或醫療人員可以把院內文件、指引、病歷等資料匯入系統，系統會將文件切分、轉成向量並存進本地向量資料庫；當使用者提出問題時，系統先檢索出最相關的段落，再交由本地 LLM 依據這些段落產生回答，並附上來源引用。

之所以強調「本地」，是因為醫療資料具有高度敏感性，不適合送到外部雲端服務。整套流程留在本機，資料就不會離開機器。

**本專案採分階段（Phase）開發，目前已完成 Phase 3。**

## Phase 3：Embedding + Vector Store

目前資料流程為：

```text
Document → Chunk → Embedding → Vector → ChromaDB
```

Embedding 將文字片段轉成可比較語意的數值向量；Vector Store 則把向量連同原文、
來源 metadata 與確定性的 `chunk_id` 保存。Phase 3 使用本地
`chromadb.PersistentClient`，預設資料目錄是 `vector_db/`，collection 是
`medical_documents`。重複 index 相同文件時採 upsert，不會持續增加重複 records。

預設模型是 `intfloat/multilingual-e5-small`。文件與查詢所需的 `passage:` / `query:`
前綴由 backend 自動處理，向量也會正規化。模型採 lazy loading：啟動 API、查看 models
status 或執行正常 pytest 都不下載模型；第一次真正 index 時才可能從 Hugging Face 下載，
之後使用本機 cache。模型檔與 `vector_db/` runtime data 都不可 commit。

上傳與 index 是兩個獨立步驟：

```text
POST /api/documents/upload
→ data/processed/<document_id>.json
→ POST /api/documents/<document_id>/index
→ local ChromaDB
```

Phase 3 預設 CPU 執行。未來 AMD ROCm 環境會安裝對應 ROCm 版本的 PyTorch。專案
embedding backend 不假設 NVIDIA GPU；ROCm、GPU、網路與 Hugging Face 都不是 pytest
的必要條件。

目前仍不是完整 RAG：尚未實作 Retriever、語意搜尋、LLM 或答案生成。Phase 4 才會加入
Retriever，本階段不提供 Top-K search API。

主要執行與部署環境為 Linux（公司 Lab 遠端主機），未來可使用 AMD GPU 與 ROCm。但 Phase 3 預設 CPU，純 CPU 環境即可啟動 API 與執行全部測試。

## 二、Phase 2 完成內容

Phase 2 完成了整條**文件前處理管線**：把原始醫療文件轉換成帶有完整來源資訊的文字片段，並存成結構化 JSON。

已完成的項目包括：支援 TXT、PDF、DOCX 三種格式的文字抽取；文字清理與格式正規化；依字元數切塊並保留重疊；完整的 metadata（來源、頁碼、段落、chunk index、字元位置）；確定性的 chunk_id 與 document_id；UTF-8 JSON 輸出至 `data/processed/`；以及 `POST /api/documents/upload` 上傳端點。

Phase 1 的所有功能（FastAPI 應用、`GET /`、`GET /health`、三個模組 status endpoint、設定系統、集中式 logging）完全保留且未受影響。

## 三、尚未完成內容

以下功能**目前完全沒有實作**，相關模組只有一行 docstring 說明未來用途：

Embedding（文字轉向量）、向量資料庫（ChromaDB / FAISS）、Retriever（向量檢索）、Hybrid Search、Reranker、LLM 與 Ollama 整合、RAG 問答功能、前端介面、Docker 實際部署、評估指標。

另外**本階段不支援掃描式 PDF 的 OCR**。掃描式 PDF 的內容是影像而非文字圖層，系統無法抽取文字，會明確回報錯誤訊息告知這可能是掃描式 PDF 且本階段不支援 OCR，而不是靜默回傳空結果。

`configs/default.yaml` 目前**尚未接入程式執行流程**。實際生效的設定來源只有系統環境變數、`.env` 與 `app/core/config.py` 的預設值；YAML 僅作為參數設計的整體參考。

## 四、處理流程

```
上傳檔案
   ↓
[loaders]   依格式抽取文字 → LoadedDocument（保留頁碼 / 段落編號）
   ↓
[cleaner]   格式正規化（不改寫任何醫療內容）
   ↓
[chunker]   依字元數切塊，相鄰塊保留 overlap → DocumentChunk
   ↓
[pipeline]  產生統計 → 寫出 UTF-8 JSON 至 data/processed/
   ↓
回傳摘要、統計與少量 chunk 預覽
```

各模組職責：**loaders** 隔離格式差異，把 TXT / PDF / DOCX 統一成相同的資料結構；**cleaner** 是純函式，只做格式正規化；**chunker** 負責切塊與產生確定性 ID，不依賴 LangChain；**pipeline** 串接上述階段並輸出 JSON；**API router** 只處理 HTTP 層（驗證、狀態碼、組裝回應），實際處理邏輯全部委派給 ingestion 模組。

### 文件單位的切法

不同格式保留的來源資訊不同，目的是讓未來檢索到某個 chunk 時，能精確告訴醫師「這段話出自哪裡」：

| 格式 | 文件單位 | 保留的定位資訊 |
| --- | --- | --- |
| PDF | 每頁一個單位 | `page_number`（從 1 開始，與閱讀器顯示一致） |
| DOCX | 每個非空段落一個單位 | `paragraph_number`（空段落跳過但編號不前移） |
| TXT | 整份檔案一個單位 | 僅來源檔名（純文字無內建結構） |

### Metadata 說明

每個 chunk 都帶有以下欄位：

`chunk_id` 確定性識別碼（SHA-256 截短 16 字元）、`text` 片段文字、`source` 與 `file_name` 來源檔名、`file_type` 格式、`page_number` 頁碼、`paragraph_number` 段落編號、`chunk_index` 在整份文件中的序號（從 0 開始且連續）、`start_char` 與 `end_char` 在原文中的精確字元位置、`metadata` 額外資訊。

`start_char` 與 `end_char` 是實際位置而非估算值 —— 用它們去原文取子字串，會精確等於 `text` 欄位的內容。

### 切塊參數

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `CHUNK_SIZE` | 500 | 每塊的目標**字元數**上限 |
| `CHUNK_OVERLAP` | 100 | 相鄰塊之間重疊的字元數 |
| `MIN_CHUNK_SIZE` | 50 | 最後一塊若短於此值，會併入前一塊 |

**重要：本階段明確以「字元數」計量，不是 token。** token 是 embedding 模型的 tokenizer 切出來的單位，中英文的字元/token 比率差異很大（英文約 4 字元 1 token，中文約 1 字元 1 token）。要等 Phase 3 引入實際模型後才有辦法談 token 數，現在宣稱是 token 會造成誤導。

切塊會優先在自然語意邊界斷開，優先順序為：空白行、單一換行、中文句末標點（。！？；）、英文句末標點（. ! ? ;）。找不到合理邊界時才硬切。

### 重複處理的策略

`document_id` 依**檔案內容**的 SHA-256 計算，輸出檔名為 `<安全化檔名>_<document_id 前 8 碼>.json`。因此**相同內容重複處理會覆寫同一個 JSON 檔，不會累積重複資料**。同一份文件改名後上傳，仍會得到相同的 `document_id`，可用來偵測重複匯入。這個冪等（idempotent）特性讓管線可以安全重跑，是後續接上向量資料庫的重要前提。

## 五、專案資料夾簡介

```
Medical-Rag-LLM/
├── app/
│   ├── main.py           FastAPI 進入點
│   ├── api/              HTTP 層（query / documents / models）
│   ├── core/             config.py 設定、logging.py 日誌
│   ├── ingestion/        ★ Phase 2 主要實作
│   │   ├── loaders.py    TXT / PDF / DOCX 文字抽取
│   │   ├── cleaner.py    文字清理（純函式）
│   │   ├── chunker.py    字元數切塊
│   │   ├── pipeline.py   串接流程並輸出 JSON
│   │   ├── models.py     LoadedDocument / DocumentChunk / IngestionResult
│   │   └── exceptions.py 自訂例外
│   ├── schemas/          API 請求 / 回應結構
│   ├── embeddings/       （未實作）
│   ├── retrieval/        （未實作）
│   ├── llm/              （未實作）
│   ├── vector_store/     （未實作）
│   ├── prompts/          （未實作）
│   └── evaluation/       （未實作）
├── data/
│   ├── raw/              上傳的原始檔（不進版控）
│   ├── processed/        處理後的 JSON（不進版控）
│   └── evaluation/       評估資料集（未使用）
├── configs/default.yaml  參數設計參考（尚未接入程式）
├── tests/                pytest 測試
├── requirements.txt
├── .env.example
└── README.md
```

## 六、Linux 安裝與執行步驟

以下指令**全部都必須在專案根目錄執行**，也就是能看到 `requirements.txt` 的那一層。假設透過 SSH 或遠端連線操作主機，全程使用終端機，不需要任何桌面 GUI。

### 1. 進入專案目錄

```bash
cd Medical-Rag-LLM
```

### 2. 建立並啟用虛擬環境

虛擬環境會把這個專案的套件裝在專案自己的資料夾裡，不污染系統 Python。

```bash
python3 -m venv .venv
source .venv/bin/activate
```

啟用成功後，提示字元最前面通常會出現 `(.venv)`。**看到 `(.venv)` 才代表虛擬環境已啟用**。

### 3. 安裝依賴

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**不要使用 `sudo pip install`。** 在虛擬環境內安裝不需要管理員權限，用 sudo 反而會把套件裝到系統目錄、造成權限混亂。

Phase 2 使用 `pypdf`、`python-docx` 與 `python-multipart`；Phase 3 新增 `sentence-transformers` 與 `chromadb`。requirements 不指定任何 NVIDIA CUDA wheel，也不要求 ROCm 或 GPU。

### 4. 建立 .env

```bash
cp .env.example .env
```

預設值即可直接使用。若想看更詳細的日誌，可把 `LOG_LEVEL` 改成 `DEBUG`。

### 5. 啟動 FastAPI

只在本機測試時，綁定 `127.0.0.1` 最安全：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

若需要由**允許的**遠端環境存取，才改用 `0.0.0.0`，並務必遵守公司網路與防火牆規範：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

開發時可加 `--reload`，程式碼一存檔就自動重啟：

```bash
uvicorn app.main:app --reload
```

啟動成功後終端機會出現 `Medical RAG Started` 的日誌。使用 **Ctrl+C** 停止伺服器，此時會輸出 `Medical RAG Stopped`。

## 七、如何使用 API

### Swagger UI（推薦）

瀏覽器開啟：

```
http://127.0.0.1:8000/docs
```

這是 FastAPI 自動產生的互動式文件。上傳文件的步驟是：展開 `POST /api/documents/upload` → 點「Try it out」→ 在 `file` 欄位選擇檔案 → 點「Execute」→ 下方會顯示處理結果。

### curl 上傳範例

```bash
curl -X POST "http://127.0.0.1:8000/api/documents/upload" \
  -H "accept: application/json" \
  -F "file=@data/raw/sample.txt"
```

### PowerShell 上傳範例（若從 Windows 端測試）

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/documents/upload" -F "file=@data/raw/sample.txt"
```

### 各 endpoint 一覽

| Endpoint | 說明 |
| --- | --- |
| `GET /` | 專案基本資訊 |
| `GET /health` | 健康檢查 |
| `GET /api/documents/status` | documents 模組狀態 |
| `POST /api/documents/upload` | **上傳並處理文件** |
| `POST /api/documents/{document_id}/index` | **將 processed JSON 建立本地向量索引** |
| `GET /api/query/status` | query 模組狀態（尚未實作） |
| `GET /api/models/status` | embedding provider/model/device 設定（不載入模型） |

### 回應範例

```json
{
  "status": "processed",
  "document_id": "7b9e152edbcdaca0",
  "file_name": "sample.txt",
  "file_type": "txt",
  "statistics": {
    "loaded_units": 1,
    "cleaned_units": 1,
    "chunk_count": 10,
    "total_characters": 1650
  },
  "output_file": "data/processed/sample_7b9e152e.json",
  "chunk_previews": [
    {
      "chunk_id": "91cd58637845199d",
      "chunk_index": 0,
      "page_number": null,
      "paragraph_number": null,
      "text_preview": "病患主訴：胸痛。..."
    }
  ]
}
```

回應**只包含少量 chunk 預覽**，不會回傳全部 chunk 文字。完整結果請讀取 `output_file` 指向的 JSON。

### HTTP 狀態碼

| 狀態碼 | 情境 |
| --- | --- |
| 200 | 處理成功 |
| 400 | 空檔案、內容損毀、無法抽取文字（含掃描式 PDF） |
| 413 | 檔案超過 `MAX_UPLOAD_SIZE_MB` |
| 415 | 不支援的格式 |
| 500 | 未預期的處理或寫檔錯誤 |

### 處理結果的位置

處理後的 JSON 寫入 `data/processed/`，檔名格式為 `<安全化檔名>_<document_id 前 8 碼>.json`。原始上傳檔保存在 `data/raw/`。

這兩個目錄都已列入 `.gitignore`，不會進入版本控制。

## 八、如何執行測試

測試不需要啟動伺服器，全部離線執行，不依賴外部模型、網路或預先準備的測試檔（DOCX 與 PDF 都在測試中動態建立）。

```bash
pytest
```

預期看到全部測試 passed。所有測試輸出都導向 pytest 的暫存目錄，**不會污染專案的 `data/` 目錄**。

## 九、常見錯誤排除

**`ModuleNotFoundError: No module named 'app'`**
目前所在目錄不是專案根目錄。`cd` 到看得到 `requirements.txt` 的那一層再執行。

**`uvicorn: command not found`**
虛擬環境沒有啟用（提示字元前沒有 `(.venv)`），或依賴尚未安裝。也可以改用 `python -m uvicorn app.main:app`。

**上傳 PDF 得到 400 且訊息提到「掃描式 PDF」**
這份 PDF 的內容是影像而非文字圖層，無法抽取文字。本階段不支援 OCR。請改用含文字圖層的 PDF，或先用其他工具做 OCR 轉換後再上傳。

**上傳得到 415**
副檔名不在支援清單內。目前只支援 `.txt`、`.pdf`、`.docx`。系統以副檔名為準而非 Content-Type，因為後者由客戶端宣告、可以偽造。

**上傳得到 413**
檔案超過 `MAX_UPLOAD_SIZE_MB`（預設 20 MB）。可在 `.env` 調整，但要考量主機記憶體。

**DOCX 上傳得到 400 且訊息提到「無法開啟」**
檔案可能損毀，或副檔名是 `.docx` 但實際是舊版 `.doc` 格式。`python-docx` 只支援 `.docx`（Office 2007 以後的 XML 格式）。請用 Word 另存新檔為 `.docx`。

**權限問題（Permission denied）**
優先檢查專案目錄的擁有者與虛擬環境是否正確，例如 `ls -ld data/raw`。**不要用 `sudo` 或 `chmod 777` 放寬成不安全的全域權限。** 若目錄擁有者不對，用 `chown` 改成目前使用者即可。

**`Address already in use`**
8000 埠已被占用。換一個埠：`uvicorn app.main:app --port 8001`。

**修改 `.env` 沒有生效**
設定只在啟動時讀取一次且有快取，請完全停止伺服器後重新啟動。另外注意優先順序：**系統環境變數會覆蓋 `.env`**。

**檔名大小寫問題**
Linux 檔名**大小寫敏感**，`Documents.py` 與 `documents.py` 是兩個不同的檔案。在 Windows 上能跑的 import，搬到 Linux 可能因大小寫不符而失敗。

**換行符問題**
Git 內統一使用 LF 換行（已由 `.gitattributes` 強制）。若 shell script 混入 CRLF，Linux 執行時會出現 `bad interpreter` 這類難以理解的錯誤。

## 十、醫療資料安全提醒

**請只使用公開、合成或已去識別化的資料進行測試與開發。**

不要把真實病人的個資送到任何公開服務或外部 API。本系統設計為完全本地執行正是為了這個理由，但如果在測試過程中把真實資料上傳到其他工具，這層保護就失效了。

`data/raw/` 與 `data/processed/` 已列入 `.gitignore`，但仍請留意：一旦敏感檔案被 commit 並推送到遠端（尤其是公開 repo），**即使之後刪除檔案，commit 歷史中仍然找得到**。commit 前養成執行 `git status` 檢查的習慣。

系統的 log 只記錄檔名、格式、單位數、chunk 數與耗時等統計資訊，**不會記錄文件正文、完整病歷或 chunk 全文**。

**本系統不是醫療器材，不能取代醫師的專業判斷。** 系統的輸出僅供資訊參考，任何臨床決策都必須由具備資格的醫療專業人員依據完整的臨床脈絡做出。

## 十一、目前的能力界線

Phase 2 完成的是「文件前處理」，也就是把文件變成結構良好、帶有完整來源資訊的文字片段。

**系統目前還無法回答任何問題。** Phase 3 已能產生 Embedding 並保存至 ChromaDB，但向量檢索、Retriever 與 LLM 生成尚未實作。Phase 4 才會加入 Retriever；現階段可以驗證文件前處理、向量化與持久化 indexing。
