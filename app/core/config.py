"""應用程式設定（settings）。

使用 pydantic-settings 從「系統環境變數 -> .env -> 程式預設值」的優先順序讀取設定，
並提供一個可重複使用的 settings 實例供全專案 import。

設計原則：本模組在 import 階段只做「讀取設定」，不建立目錄、不開檔、不連線。
需要目錄時由呼叫端明確呼叫 ensure_data_directories()，避免 import 產生意外副作用
（例如跑測試時在專案目錄留下垃圾）。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全域設定。

    欄位使用 Python 慣例的小寫命名，但透過 case_sensitive=False 對應到大寫環境變數，
    例如環境變數 APP_NAME 會寫入欄位 app_name。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Phase 1：應用程式基本資訊 ---
    app_name: str = "Medical Local RAG"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Phase 2：資料目錄 ---
    # 使用相對路徑，讓專案可以搬到任何機器（含 Linux Lab 主機）而不需修改設定。
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")

    # --- Phase 2：上傳限制 ---
    max_upload_size_mb: int = 20

    # --- Phase 2：切塊參數（單位為「字元數」，不是 token）---
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50

    # --- Phase 3：Embedding 與本地向量資料庫 ---
    embedding_provider: str = "local"
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_model_revision: str = (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    )
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    vector_store_provider: str = "chroma"
    vector_db_dir: Path = Path("vector_db")
    chroma_collection_name: str = "medical_documents"

    # --- Phase 4：檢索（retrieval）---
    # retrieval_top_k 是未指定 top_k 時的預設回傳筆數。
    # 5 是 RAG 常見起點：足以涵蓋一個問題的多個佐證段落，
    # 又不會在 Phase 5 把過長的 context 塞進 LLM。
    retrieval_top_k: int = 5
    # 上限的用意是防止呼叫端要求極大的 top_k，
    # 造成不必要的記憶體與序列化負擔（等同一種 DoS 防護）。
    retrieval_max_top_k: int = 50

    # --- Phase 5：本地 OpenAI-compatible LLM ---
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_model_name: str = "local-medical-model"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    llm_timeout: float = 60.0
    # 預設僅允許 loopback。Lab 內網推論伺服器必須明確 opt in。
    llm_allow_private_network: bool = False

    @property
    def max_upload_size_bytes(self) -> int:
        """上傳大小上限，換算成 bytes 以便與檔案長度直接比較。"""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """回傳快取後的 Settings 實例，避免重複解析 .env。

    注意：因為有快取，修改 .env 後必須重新啟動程式才會生效。
    測試若需覆蓋設定，請使用 FastAPI 的 dependency_overrides 注入，
    不要依賴這個函式的回傳值。
    """
    return Settings()


def ensure_data_directories(settings: Settings) -> None:
    """建立資料目錄（若不存在）。

    刻意設計成需要明確呼叫，而不是在 import 時自動執行。
    parents=True 讓多層目錄一次建立完成，exist_ok=True 讓重複呼叫是安全的。
    """
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)


settings: Settings = get_settings()
