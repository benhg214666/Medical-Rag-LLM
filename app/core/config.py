"""應用程式設定（settings）。

使用 pydantic-settings 從「系統環境變數 -> .env -> 程式預設值」的優先順序讀取設定，
並提供一個可重複使用的 settings 實例供全專案 import。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全域設定。

    欄位使用 Python 慣例的小寫命名，但透過 alias 對應到大寫環境變數，
    例如環境變數 APP_NAME 會寫入欄位 app_name。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Medical Local RAG"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """回傳快取後的 Settings 實例，避免重複解析 .env。"""
    return Settings()


settings: Settings = get_settings()
