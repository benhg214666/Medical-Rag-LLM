# Medical Local RAG（醫療本地 RAG 系統）

## 一、專案目的

本專案的最終目標，是打造一套**完全在本地端執行**的醫療 RAG（Retrieval-Augmented Generation，檢索增強生成）系統。醫師或醫療人員可以把院內文件、指引、病歷等資料匯入系統，系統會將文件切分、轉成向量並存進本地向量資料庫；當使用者提出問題時，系統先檢索出最相關的段落，再交由本地 LLM 依據這些段落產生回答，並附上來源引用。

之所以強調「本地」，是因為醫療資料具有高度敏感性，不適合送到外部雲端服務。整套流程留在本機，資料就不會離開機器。

完整系統規劃的能力如下：匯入醫療文件、清理與切分、建立 Embedding、儲存至本地向量資料庫、依問題檢索、由本地 LLM 生成回答、附上來源與引用、支援模型替換、支援評估與前端介面。

**本專案採分階段（Phase）開發。目前處於 Phase 1。**

## 二、Phase 1 目前完成內容

Phase 1 只做「專案骨架與基礎 API」，也就是把整個系統的地基與資料夾結構先立起來，並確保一個最小可運行的 FastAPI 服務能正常啟動、能通過測試。具體完成項目包括：可用 `uvicorn` 啟動的 FastAPI 應用程式、使用 lifespan 管理啟動與關閉並寫入 log、`GET /` 與 `GET /health` 兩個系統 endpoint、query / documents / models 三個模組各自的 APIRouter 與 status endpoint、以 pydantic-settings 實作的設定系統、集中式 logging 設定、`configs/default.yaml` 設定範例、pytest 測試、requirements.txt、`.env.example`、`.gitignore` 以及本份 README。

## 三、尚未完成內容（Phase 1 不做）

以下功能**目前完全沒有實作**，相關資料夾中的 Python 檔案只有一行 docstring 說明未來用途，不含任何可執行邏輯：PDF / DOCX / TXT 文件讀取、文字清理、Chunking、Embedding、ChromaDB、FAISS、Retriever、Reranker、LLM、Ollama、LangChain、LlamaIndex、Transformers、PyTorch、前端介面、Docker 實際部署、以及最重要的 —— **真正的 RAG 問答功能**。

因此，`/api/query/status` 等 endpoint 回傳的 `"status": "not_implemented"` 是刻意的設計，用來誠實標示模組尚未完成，而不是回傳假資料假裝功能已可使用。

另外請注意：**`configs/default.yaml` 在 Phase 1 尚未接入程式執行流程**。程式實際讀取的設定來源只有系統環境變數、`.env` 與 `config.py` 中的預設值；YAML 目前純粹是未來參數的設計草稿。

## 四、專案資料夾簡介

```
medical-rag/
├── app/                  應用程式主要程式碼
│   ├── main.py           FastAPI 進入點，註冊所有 router
│   ├── api/              HTTP API 層（query / documents / models）
│   ├── core/             核心基礎設施：config.py 設定、logging.py 日誌
│   ├── ingestion/        文件匯入：讀取、清理、切分（未實作）
│   ├── embeddings/       文字轉向量（未實作）
│   ├── retrieval/        檢索與重新排序（未實作）
│   ├── llm/              本地 LLM 後端（未實作）
│   ├── vector_store/     向量資料庫存取層（未實作）
│   ├── prompts/          Prompt 模板與組裝器（未實作）
│   ├── evaluation/       檢索與回答品質評估（未實作）
│   └── schemas/          Pydantic 請求 / 回應資料結構
├── frontend/             前端介面（未實作）
├── data/                 資料目錄：raw 原始、processed 處理後、evaluation 評估集
├── vector_db/            本地向量資料庫持久化目錄（尚未使用）
├── configs/default.yaml  未來設定範例（尚未接入程式）
├── scripts/              批次腳本（尚未使用）
├── tests/                pytest 測試
├── requirements.txt      Python 依賴
├── .env.example          環境變數範本
├── .gitignore
├── docker-compose.yml    僅含說明註解，Docker 尚未啟用
└── README.md
```

## 五、Windows 安裝與執行步驟

以下指令**全部都必須在 `medical-rag` 專案根目錄下執行**，也就是能看到 `requirements.txt` 的那一層。指令以 Windows PowerShell 為主。

### 1. 進入專案目錄

```powershell
cd medical-rag
```

### 2. 建立虛擬環境

虛擬環境（virtual environment）會把這個專案用到的套件裝在專案自己的資料夾裡，不會污染系統的 Python。

```powershell
python -m venv .venv
```

執行後專案下會多出一個 `.venv` 資料夾。這個資料夾已被 `.gitignore` 忽略，不會進版控。

### 3. 啟用虛擬環境

```powershell
.\.venv\Scripts\Activate.ps1
```

啟用成功後，PowerShell 提示字元最前面會出現 `(.venv)`。**看到 `(.venv)` 才代表虛擬環境已啟用**，後續指令才會裝進 / 跑在正確的環境。

### 4. 升級 pip 並安裝依賴

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. 建立 .env

`.env.example` 是範本，實際使用的是 `.env`（已被 `.gitignore` 忽略，不會被提交）。

```powershell
Copy-Item .env.example .env
```

Phase 1 的預設值即可直接使用，不需修改。若想看更詳細的日誌，可把 `.env` 裡的 `LOG_LEVEL` 改成 `DEBUG`。

### 6. 啟動 FastAPI 伺服器

```powershell
uvicorn app.main:app --reload
```

`--reload` 表示程式碼一存檔就自動重啟，適合開發階段使用。啟動成功後終端機會出現 `Medical RAG Started` 的日誌，以及 `Uvicorn running on http://127.0.0.1:8000`。

## 六、如何測試各 API

### Swagger UI（推薦）

瀏覽器打開：

```
http://127.0.0.1:8000/docs
```

這是 FastAPI 自動產生的互動式 API 文件，可以直接在網頁上按「Try it out」送出請求，不必自己打指令。

### 直接用瀏覽器或 PowerShell

| Endpoint | 預期回傳 |
| --- | --- |
| `GET /` | `{"project": "Medical Local RAG", "version": "0.1.0", "status": "running"}` |
| `GET /health` | `{"status": "healthy"}` |
| `GET /api/query/status` | `{"module": "query", "status": "not_implemented"}` |
| `GET /api/documents/status` | `{"module": "documents", "status": "not_implemented"}` |
| `GET /api/models/status` | `{"module": "models", "status": "not_implemented"}` |

PowerShell 範例（請另開一個視窗，讓伺服器繼續執行）：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/query/status
```

## 七、如何執行測試

測試不需要先啟動伺服器，FastAPI 的 `TestClient` 會在測試過程中自行載入應用程式。

```powershell
pytest
```

預期會看到全部測試 passed。測試內容涵蓋 `GET /` 與 `GET /health` 的狀態碼與回傳內容，以及三個模組 status endpoint。

## 八、如何停止伺服器

在執行 `uvicorn` 的那個終端機視窗按 **Ctrl + C**。此時日誌會輸出 `Medical RAG Stopped`。

離開虛擬環境則輸入：

```powershell
deactivate
```

## 九、常見錯誤排除

**`.\.venv\Scripts\Activate.ps1` 出現「因為這個系統上已停用指令碼執行」**
這是 PowerShell 的執行原則限制。以目前使用者身分放寬即可：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

之後重新執行啟用指令。

**`ModuleNotFoundError: No module named 'app'`**
代表目前所在目錄不是專案根目錄。請 `cd` 到看得到 `requirements.txt` 的那一層再執行 `uvicorn` 或 `pytest`。

**`'uvicorn' 不是內部或外部命令` / `command not found`**
通常是虛擬環境沒有啟用（提示字元前沒有 `(.venv)`），或 `pip install -r requirements.txt` 尚未執行成功。也可以改用 `python -m uvicorn app.main:app --reload` 這種寫法。

**`[Errno 10048] address already in use`（連接埠被占用）**
8000 埠已被其他程式使用。換一個埠即可：

```powershell
uvicorn app.main:app --reload --port 8001
```

**修改了 `.env` 但設定沒有生效**
設定只在應用程式啟動時讀取一次，且會被快取。請完全停止伺服器（Ctrl + C）後重新啟動。另外請注意優先順序：**系統環境變數會覆蓋 `.env`**，若系統層級已設定同名變數，`.env` 的值不會生效。

**`pytest` 顯示找不到 `httpx`**
`TestClient` 依賴 `httpx`。請確認虛擬環境已啟用並重新執行 `pip install -r requirements.txt`。

## 十、重要提醒

**Phase 1 尚未具備任何真正的醫療文件問答能力。** 目前的系統只是一個能正常啟動、能回報狀態、能通過測試的 API 骨架。所有 RAG 相關功能（文件匯入、Embedding、向量檢索、LLM 生成）都會在後續 Phase 才逐步實作。請不要將現階段的系統用於任何實際醫療判讀用途。
