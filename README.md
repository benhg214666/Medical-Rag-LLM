# Medical Local RAG（醫療本地 RAG 系統）

## Phase 8：可重現 demo

Phase 8 提供固定 synthetic/de-identified 資料、隔離的
`medical_demo_v1` collection、真實 RAG pipeline runner，以及 JSON/Markdown
evaluation 匯出。資料與輸出皆 **not for clinical use**。

```bash
python scripts/demo.py preflight --allow-unseeded
python scripts/demo.py seed --reset
python scripts/demo.py run
python scripts/demo.py evaluate --output-dir reports/demo
```

完整啟動、5–10 分鐘講稿、停止與復原方式請見
[`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md)。

## 一、專案目的

本專案的最終目標，是打造一套使用**本地/私有基礎設施**的醫療 RAG（Retrieval-Augmented Generation，檢索增強生成）系統。醫師或醫療人員可以把院內文件、指引、病歷等資料匯入系統，系統會將文件切分、轉成向量並存進本地向量資料庫；當使用者提出問題時，系統先檢索出最相關的段落，再交由配置的本地 LLM 伺服器產生回答，並附上來源引用。

預設 `LLM_ALLOW_PRIVATE_NETWORK=false` 且 endpoint 為 loopback，檢索 context 留在同一主機。只有明確啟用 `LLM_ALLOW_PRIVATE_NETWORK=true` 時，應用才可將 context 送到核准的 Lab 私有 IP；此時資料會離開應用主機並經過私有網路。私有 IP 不代表傳輸自動安全，部署者仍須依組織政策處理網路隔離、傳輸保護與存取控制。

**本專案採分階段（Phase）開發：Phase 1–7 系統與 Lab ROCm runtime 已完成，Phase 8 demo/presentation-readiness tooling 已實作。**

## Phase 7：ROCm 本地推論整合

Phase 7 保留既有 OpenAI-compatible HTTP 邊界，讓 FastAPI RAG 應用與
AMD GPU/ROCm 模型伺服器分離。vLLM 是文件中的主要 serving 範例，
不是 production application dependency；實際相容性取決於 Lab 的 GPU、gfx、
ROCm 與 runtime 版本。

倉庫提供三個工具：

```bash
python scripts/check_runtime.py
python scripts/smoke_test_llm.py
python scripts/smoke_test_rag.py --query "What medication is in the indexed synthetic record?"
```

單元測試不需 GPU、ROCm、Docker、vLLM 或網路；真實端到端驗證需要
已啟動的本地模型伺服器與已索引的合成/去識別文件。詳細步驟見
[`docs/rocm_local_llm.md`](docs/rocm_local_llm.md)。

## Phase 5：本地 LLM + RAG 回答

完整問答管線為：

```text
Question → Retriever → Numbered Context → Local LLM → Answer + Sources
```

RAG 服務只依賴專案的 LLM provider 介面，實際模型由另一個本地
OpenAI-compatible chat-completions server 提供，因此 FastAPI 不會直接載入
Hugging Face 模型，也不綁定 CUDA、ROCm 或雲端 API。

| 設定 | 預設 | 用途 |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai_compatible` | LLM provider |
| `LLM_BASE_URL` | `http://127.0.0.1:8001/v1` | 本地推論服務 |
| `LLM_MODEL_NAME` | `local-medical-model` | 服務器公開的模型名 |
| `LLM_TEMPERATURE` | `0.0` | 生成溫度 |
| `LLM_MAX_TOKENS` | `512` | 最大回答 tokens |
| `LLM_TIMEOUT` | `60.0` | HTTP timeout（秒） |
| `LLM_ALLOW_PRIVATE_NETWORK` | `false` | 是否明確允許 Lab 私有 IP endpoint |

```bash
curl -X POST http://127.0.0.1:8000/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What medications is this patient taking?","top_k":5}'
```

沒有可用檢索內容時會直接回傳資料不足與空 `sources`，不會呼叫 LLM。

## Phase 6：RAG 評估與品質驗證

Phase 6 以 JSONL gold dataset 重用 Phase 4 Retriever 與 Phase 5 RAGService，
提供可重現的 retrieval-only 與 end-to-end RAG 評估：

- Hit@K：Top-K 是否至少命中一個預期來源。
- Recall@K：Top-K 命中的預期來源比例。
- MRR：第一個相關來源排名的倒數平均。
- Citation validity：引用編號是否實際存在於返回 sources。
- Citation relevance：被引用 source 是否符合 gold document/chunk ID。
- Abstention accuracy：不可回答案例是否回傳專案明確定義的資料不足回應；
  不只以沒有檢索 sources 當判斷依據。
- Reference token F1：非 CJK 字母數字保留為連續 token，CJK 漢字以單字為
  可重現的字面單位（不是中文分詞）。此 F1 不代表語意等價、事實正確、
  醫療正確或臨床安全。

`data/evaluation/example_cases.jsonl` 是無 PHI 的合成範本；其
`REPLACE_WITH_INDEXED_DOCUMENT_ID` 必須替換為自行匯入且已知正確性的測試文件 ID。
實際 gold set 應由人工從合成或去識別文件標註 document/chunk ID。

```bash
# 只評估檢索，不呼叫 LLM
python -m app.evaluation.cli --dataset data/evaluation/cases.jsonl --mode retrieval

# 端到端 RAG（需本地 LLM server）並儲存 JSON report
python -m app.evaluation.cli --dataset data/evaluation/cases.jsonl --mode rag \
  --output data/evaluation/results.json
```

聚合指標只使用適用案例當分母：沒有 relevance label 的案例不會被當成
retrieval 失敗，沒有 reference answer 的案例也不會被當成 F1=0。

## Phase 4：Retriever（檢索層）

目前完整管線為：

```text
Document → Loader → Cleaning → Chunking → Embedding → Vector Store → Retriever
```

查詢時的資料流為：

```text
query → embed_query → search_by_vector → VectorMatch → RetrievalResult
```

分層上，API 與未來 RAG 層只依賴 `VectorRetriever`；Retriever 依賴
`VectorStore` 抽象，由 `ChromaStore` 負責把 Chroma 專屬回傳格式翻譯成
中性的 `VectorMatch`：

```text
API / RAG → VectorRetriever → VectorStore 抽象 → ChromaStore
```

| 欄位 | 語意 | 排序 |
| --- | --- | --- |
| `distance` | cosine 時為 `1 - cosine_similarity` | 越小越相似 |
| `score` | cosine 時為 `1 - distance` | 越大越相似 |

`score` 的換算不依賴向量事先正規化，因為 cosine similarity 的定義
本身就會除以向量模長。非 cosine metric 下 `score` 為 `null`，以原始
`distance` 判斷。

### Embedding 相容性檢查

Retriever 搜尋前會比對 model name、revision、dimension、normalization 與
distance metric。只比對維度不足夠：兩個模型即使維度相同，仍可能處於
完全不同的 embedding space。若放行會形成沒有例外、但結果無意義的
silent retrieval corruption。

| 設定 | 預設 | 用途 |
| --- | --- | --- |
| `RETRIEVAL_TOP_K` | 5 | 未指定 `top_k` 時的回傳筆數 |
| `RETRIEVAL_MAX_TOP_K` | 50 | 允許的上限，超過時回 HTTP 400 |

```bash
curl -X POST http://127.0.0.1:8000/api/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{"query":"第二型糖尿病的用藥是什麼？","top_k":3}'
```

```json
{
  "query": "第二型糖尿病的用藥是什麼？",
  "top_k": 3,
  "result_count": 1,
  "results": [{
    "chunk_id": "0123456789abcdef",
    "document_id": "fedcba9876543210",
    "text": "Metformin 500 mg twice daily.",
    "distance": 0.12,
    "score": 0.88,
    "distance_metric": "cosine",
    "metadata": {"source": "note.txt", "chunk_index": 0}
  }]
}
```

**Retriever 同時供獨立語意搜尋 API 與 Phase 5 RAG 服務重用。**

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

Phase 3 預設 CPU 執行，embedding 模型固定為
`intfloat/multilingual-e5-small` revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`。一般 pytest 使用 mock/fake，
不下載模型，也不需要 GPU、ROCm、網路或 Hugging Face 連線。真模型 smoke test
必須另外明確執行。

目前 Phase 3 索引、Phase 4 兩階段檢索與 Phase 5 RAG 答案生成
已串接完成。

主要執行與部署環境為 Linux（公司 Lab 遠端主機），未來可使用 AMD GPU 與 ROCm。但 Phase 3 預設 CPU，純 CPU 環境即可啟動 API 與執行全部測試。

## 二、Phase 2 完成內容

Phase 2 完成了整條**文件前處理管線**：把原始醫療文件轉換成帶有完整來源資訊的文字片段，並存成結構化 JSON。

已完成的項目包括：支援 TXT、PDF、DOCX 三種格式的文字抽取；文字清理與格式正規化；依字元數切塊並保留重疊；完整的 metadata（來源、頁碼、段落、chunk index、字元位置）；確定性的 chunk_id 與 document_id；UTF-8 JSON 輸出至 `data/processed/`；以及 `POST /api/documents/upload` 上傳端點。

Phase 1 的所有功能（FastAPI 應用、`GET /`、`GET /health`、三個模組 status endpoint、設定系統、集中式 logging）完全保留且未受影響。

## 三、尚未完成內容

以下功能目前尚未實作：Hybrid Search、cross-encoder Reranker、前端介面與
Docker 實際部署。Embedding、ChromaDB indexing、Retriever、RAG 問答與
輕量評估已分別在 Phase 3–6 完成。

Phase 4.1 已加入不需額外模型的輕量 reranker。系統先用既有向量檢索取得較大的
候選池，再綜合原始語意分數、正規化字詞重疊、查詢中的精確日期，以及查詢內
有意義詞彙的精確匹配來重排。候選池大小由 `RETRIEVAL_CANDIDATE_MULTIPLIER`
與 `RETRIEVAL_MIN_CANDIDATE_K` 控制；API 的 `top_k` 仍表示最終回傳及送入 LLM
的片段數，因此 LLM context 不會因候選擴張而無界增加。

Phase 4.2 會在切塊後保留可確定辨識的父層 section／encounter heading，並以
`section_title`、`section_path` metadata 隨 chunk 寫入向量索引。RAG prompt 只顯示
實際保存的結構資訊，不推測相鄰 chunk 的關係。既有索引不會自動補上新 metadata；
要套用此能力，必須重新匯入並重新索引原始文件。沒有可辨識 heading 的文件維持原行為。

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
│   ├── embeddings/       Phase 3 本地 embedding
│   ├── retrieval/        ★ Phase 4 檢索層主要實作
│   ├── llm/              Phase 5 本地 LLM provider
│   ├── rag/              Phase 5 RAG orchestration
│   ├── vector_store/     Phase 3 ChromaDB abstraction
│   ├── prompts/          Phase 5 RAG prompt builder
│   └── evaluation/       Phase 6 品質評估
├── data/
│   ├── raw/              上傳的原始檔（不進版控）
│   ├── processed/        處理後的 JSON（不進版控）
│   └── evaluation/       Phase 6 gold dataset / reports
├── docs/                Phase 7 ROCm runtime 指南
├── configs/default.yaml  參數設計參考（尚未接入程式）
├── tests/                pytest 測試
├── requirements.txt
├── requirements-dev.txt
├── scripts/             embedding/runtime/smoke 工具
├── .env.example
└── README.md
```

## 六、安裝與執行步驟

以下指令**全部都必須在專案根目錄執行**，也就是能看到 `requirements.txt` 的那一層。假設透過 SSH 或遠端連線操作主機，全程使用終端機，不需要任何桌面 GUI。

### 1. 進入專案目錄

```bash
cd Medical-Rag-LLM
```

### 2. 建立並啟用虛擬環境

本專案以 Python 3.12 驗證。請依下一節的平台指令建立 `.venv`；虛擬環境會把
套件隔離在專案內，不污染系統 Python。啟用成功後，提示字元通常會出現
`(.venv)`。

### 3. 安裝依賴

`requirements.txt` 固定 production 的直接依賴版本；`requirements-dev.txt` 再加入
pytest 與 Starlette 目前使用的 `httpx2` 測試工具。PyTorch 是平台專屬套件，刻意不放在
通用 requirements。

#### Windows CPU（PowerShell）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

#### Linux CPU

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

#### Linux AMD ROCm

先依 [AMD ROCm PyTorch 安裝文件](https://rocm.docs.amd.com/en/latest/rocm-for-ai/pytorch.html)
確認 GPU、Linux 發行版、Python、ROCm 與 PyTorch 的相容矩陣，再安裝該組合指定的
ROCm wheel 或 AMD 驗證過的 container。**不要先執行上面的 CPU wheel 指令，也不要用
一般 PyPI 的 `pip install torch` 代替 ROCm wheel。**

安裝 ROCm PyTorch 後，先確認它不是 CPU build：

```bash
python -c "import torch; assert torch.version.hip; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
python -m pip install -r requirements-dev.txt
python -c "import torch; assert torch.version.hip; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

第二次檢查可防止後續 dependency resolution 意外替換 ROCm build。PyTorch 官方也要求
Linux AMD 環境在安裝器選擇 ROCm compute platform；ROCm build 在 Python API 中仍以
`torch.cuda` 檢查裝置。

**不要使用 `sudo pip install`。** 在虛擬環境內安裝不需要管理員權限，用 sudo 反而會把套件裝到系統目錄、造成權限混亂。

Phase 2 使用 `pypdf`、`python-docx` 與 `python-multipart`；Phase 3 新增
`sentence-transformers` 與 `chromadb`。

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
| `GET /api/retrieval/status` | retrieval 模組狀態（已實作） |
| `POST /api/retrieval/search` | **候選擴張、輕量重排後的 Top-K 檢索** |
| `GET /api/rag/status` | RAG 模組狀態與模型名 |
| `POST /api/rag/ask` | **產生有來源引用的 RAG 回答** |
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

處理後的 JSON 寫入 `data/processed/`，檔名格式為 `<document_id>.json`；原始上傳檔
則以 `<安全化檔名>_<document_id 前 8 碼>.<副檔名>` 保存在 `data/raw/`。

這兩個目錄都已列入 `.gitignore`，不會進入版本控制。

## 八、如何執行測試

測試不需要啟動伺服器，全部離線執行，不依賴外部模型、網路或預先準備的測試檔（DOCX 與 PDF 都在測試中動態建立）。

Linux：

```bash
python -m pytest
```

Windows 的 Chroma 檔案可能在 pytest 清理時仍被占用，因此將 basetemp 放在專案外：

```powershell
.\.venv\Scripts\python.exe -m pytest -vv --basetemp ..\pytest-medical-rag
```

預期看到全部測試 passed。所有測試輸出都導向 pytest 的暫存目錄，**不會污染專案的 `data/` 目錄**。

Phase 4 的離線測試可單獨執行：

```bash
python -m pytest -m "not integration"
```

真實 embedding + Chroma 的端對端檢索測試使用 `integration` marker：

```bash
python -m pytest -m integration -v
```

首次執行可能需下載固定 revision 的模型。只有在能明確辨識為網路不可用、
模型或 revision 取得失敗、本地 cache 缺檔等環境問題時才會 skip；
其他 embedding 實作錯誤一律視為測試失敗。

真模型 CPU smoke test 會在首次執行時下載模型，不屬於一般單元測試。取得下載許可後才執行：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_embedding.py
```

它會驗證固定 revision、384 維、所有值 finite、L2 normalized，以及 E5 的
`passage:`／`query:` prefix。

## 後端 Corpus 批次匯入

`data/corpus/` 是永久、人工維護的資料來源；`data/raw/`、`data/processed/`
與 `vector_db/` 則是可重建的執行期資料。建議結構：

```text
data/corpus/
├── patient_records/
│   └── P001/
│       └── 2026-06-10_SOAP.txt
└── medical_knowledge/
    └── diabetes/
        └── hba1c_targets.txt
```

從專案根目錄批次匯入所有 `.txt`、`.pdf` 與 `.docx`：

```powershell
python scripts/ingest_corpus.py
```

腳本會沿用上傳 API 的 loader、清理、切塊、section context、embedding 與
vector-store upsert 流程。病歷路徑會自動加入 `patient_id`、`source_type`，符合
日期與類型命名時也會加入 `encounter_date`、`document_type`。

如需清除並重建執行期資料（不會刪除 `data/corpus/`）：

```powershell
python scripts/reset_rag_data.py
```

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
Retriever 的 log 也只記錄查詢長度、候選／最終 `top_k`、結果數與 chunk ID，
不記錄查詢全文、向量或醫療文件正文。

**本系統不是醫療器材，不能取代醫師的專業判斷。** 系統的輸出僅供資訊參考，任何臨床決策都必須由具備資格的醫療專業人員依據完整的臨床脈絡做出。

## 十一、目前的能力界線

Phase 2 完成文件匯入與前處理；Phase 3 完成 Embedding 與 ChromaDB
索引；Phase 4 完成候選擴張與確定性輕量重排；Phase 5 已可透過另行啟動的
本地 LLM，依檢索內容產生自然語言回答，並保留編號來源與 metadata
以供追溯；Phase 6 提供 retrieval-only 與 end-to-end RAG 的可重現品質評估。

Hybrid retrieval、模型式 reranking、大型 benchmark / LLM judge、前端、對話記憶、
進階醫療 guardrails 與 production deployment 仍屬後續工作。
