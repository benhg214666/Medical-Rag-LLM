"""Embedding backend 的共享取得方式。

放在 embeddings 子系統而非任何消費端（indexing / retrieval）之下，
是為了讓相依方向保持單向：

        app.embeddings.dependencies（共享來源）
              ↑                    ↑
        app.indexing          app.retrieval

而不是讓 retrieval 反過來依賴 indexing 的內部細節 ——
那會讓兩個平行的子系統產生不必要的耦合。

為什麼要 cache：embedding 模型可達數百 MB，每個 HTTP request 重新載入
不可行。更重要的是，indexing 與 retrieval 必須使用**完全相同**的模型、
revision 與維度：document 與 query 若不在同一個 embedding space，
檢索結果會靜默劣化而不會拋出任何錯誤。共用同一個 cache key
讓這件事在結構上被保證，而不是靠兩邊各自設定正確。
"""

from functools import lru_cache

from app.core.config import Settings
from app.embeddings.base import EmbeddingBackend
from app.embeddings.factory import create_embedding_backend


@lru_cache(maxsize=4)
def get_cached_embedding_backend(
    provider: str,
    model_name: str,
    model_revision: str,
    device: str,
) -> EmbeddingBackend:
    """依 embedding 設定共用 backend，避免每個 request 重新載入模型。

    參數刻意攤平成純量而非直接收 Settings：Settings 不是 hashable，
    無法作為 lru_cache 的 key。攤平後也讓「哪些設定會影響 backend 身分」
    變得明確 —— 只有這四項改變才需要重新載入模型。
    """
    backend_settings = Settings(
        embedding_provider=provider,
        embedding_model_name=model_name,
        embedding_model_revision=model_revision,
        embedding_device=device,
    )
    return create_embedding_backend(backend_settings)


def get_embedding_backend_for(settings: Settings) -> EmbeddingBackend:
    """以目前設定取得共享 backend。

    統一在這裡做大小寫正規化，避免各呼叫端各自處理而產生
    不同的 cache key（例如 "CPU" 與 "cpu" 會載入兩份模型）。
    """
    return get_cached_embedding_backend(
        settings.embedding_provider.lower(),
        settings.embedding_model_name,
        settings.embedding_model_revision,
        settings.embedding_device.lower(),
    )
