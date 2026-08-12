"""向量檢索器：以語意相似度從 vector store 取回文件片段。

這是 Phase 4 的核心。它把 Phase 3 已完成的兩個元件串起來：

    query 文字 --(EmbeddingBackend.embed_query)--> 查詢向量
              --(VectorStore.search_by_vector)--> VectorMatch
              --(本模組轉換)--> RetrievalResult

刻意「不」自己載入 embedding 模型、也「不」直接呼叫 Chroma：
兩者都透過建構子注入的抽象取得。這讓 Retriever 可以在測試中
搭配 fake backend 快速執行，也讓未來替換 vector store 時
不需要修改任何檢索邏輯。

Retriever 只負責找出相關片段，不產生自然語言答案；
答案生成屬於 Phase 5 的 LLM 層。
"""

import logging

from app.embeddings.base import EmbeddingBackend, EmbeddingError
from app.retrieval.exceptions import (
    RetrievalBackendError,
    RetrievalValidationError,
)
from app.retrieval.models import RetrievalResult
from app.vector_store.base import VectorMatch, VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

# 只有這些 metric 能安全地把 distance 換算成「越大越相似」的 score。
# cosine 在 Chroma 的定義是 1 - cosine_similarity，
# 因此 score = 1 - distance 會還原成 cosine similarity 本身。
# L2 或 inner product 沒有這個性質，換算會產生誤導性數值，
# 故不列入；那些情況下 score 會是 None，呼叫端須改用 distance 排序。
_SIMILARITY_CONVERTIBLE_METRICS = frozenset({"cosine"})

# document_id 在 Phase 3 是寫進 vector store 的 metadata，
# 而不是 DocumentChunk 的欄位，因此這裡以 key 取出後提升為一級欄位。
_DOCUMENT_ID_KEY = "document_id"


class VectorRetriever:
    """以查詢向量在 vector store 中找出最相關的 chunks。"""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_store: VectorStore,
        default_top_k: int,
        max_top_k: int,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("retrieval default_top_k 必須大於 0")
        if max_top_k <= 0:
            raise ValueError("retrieval max_top_k 必須大於 0")
        if default_top_k > max_top_k:
            raise ValueError(
                "retrieval default_top_k 不可大於 max_top_k"
            )

        self.embedding_backend = embedding_backend
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k
        self._compatibility_verified = False

    def ensure_ready(self) -> None:
        """驗證目前 embedding backend 與 persisted collection 相容。

        為什麼這件事非做不可：假設 collection 原本是用 model A 建立的，
        後來設定改成同樣是 384 維的 model B。Chroma 只檢查向量維度，
        因此查詢**依然會成功執行並回傳結果** —— 但 document vectors 與
        query vectors 已不在同一個 embedding space，相似度完全沒有意義。
        這是所謂的 silent retrieval corruption：沒有任何錯誤訊息，
        結果卻是錯的。在醫療情境下，錯誤的檢索會直接變成錯誤的佐證。

        實作上直接重用 Phase 3 已有的 ensure_embedding_compatibility()，
        不另做一套比對邏輯 —— 該方法已涵蓋 model_name、model_revision、
        dimension、normalized 與 distance metric / schema version 的合約。

        呼叫時機由呼叫端決定（dependency 建立時或首次檢索前）。
        驗證成功後會記錄狀態，避免每次檢索重複讀取 collection metadata。

        Raises:
            RetrievalBackendError: 不相容，或 embedding / vector store 失敗。
        """
        if self._compatibility_verified:
            return

        try:
            self.vector_store.ensure_embedding_compatibility(
                model_name=self.embedding_backend.model_name,
                model_revision=self.embedding_backend.model_revision,
                dimension=self.embedding_backend.dimension,
                normalized=self.embedding_backend.normalizes_embeddings,
            )
        except VectorStoreError as exc:
            logger.error(
                "檢索初始化失敗：embedding 設定與既有 collection 不相容"
                "（collection=%s model=%s）",
                self.vector_store.collection_name,
                self.embedding_backend.model_name,
            )
            raise RetrievalBackendError(
                "目前的 embedding 設定與既有向量索引不相容；"
                "請確認設定或重新建立索引"
            ) from exc
        except EmbeddingError as exc:
            logger.exception("檢索初始化失敗：無法取得 embedding 模型資訊")
            raise RetrievalBackendError(
                "無法初始化 embedding 模型"
            ) from exc

        self._compatibility_verified = True

    def resolve_top_k(self, top_k: int | None) -> int:
        """驗證並決定實際使用的 top_k。

        None 代表呼叫端未指定，採用設定檔預設值；
        其餘情況一律驗證，超出範圍直接拒絕而不是靜默截斷 ——
        靜默修正會讓呼叫端誤以為自己的參數生效。
        """
        if top_k is None:
            return self.default_top_k

        # bool 是 int 的子類別，True 會被當成 1，這不是呼叫端的本意。
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RetrievalValidationError("top_k 必須是整數")
        if top_k <= 0:
            raise RetrievalValidationError("top_k 必須大於 0")
        if top_k > self.max_top_k:
            raise RetrievalValidationError(
                f"top_k 不可超過 {self.max_top_k}"
            )
        return top_k

    @staticmethod
    def _validate_query(query: str) -> str:
        """檢查查詢字串並回傳去除首尾空白後的內容。

        空白、純空格、純換行都視為無效查詢：它們無法產生有意義的
        語意向量，若放行只會得到隨機排序的結果。
        """
        if not isinstance(query, str):
            raise RetrievalValidationError("查詢內容必須是字串")

        normalized = query.strip()
        if not normalized:
            raise RetrievalValidationError("查詢內容不可為空白")
        return normalized

    def _to_result(self, match: VectorMatch) -> RetrievalResult:
        """把 vector store 的中性命中結果轉成專案層級的檢索結果。"""
        metadata = dict(match.metadata)

        # document_id 提升為一級欄位，其餘 metadata 原樣保留。
        # 用 pop 避免同一份資訊在兩個地方重複出現。
        raw_document_id = metadata.pop(_DOCUMENT_ID_KEY, None)
        document_id = (
            str(raw_document_id) if raw_document_id is not None else None
        )

        metric = self.vector_store.distance_metric
        score = (
            1.0 - match.distance
            if metric in _SIMILARITY_CONVERTIBLE_METRICS
            else None
        )

        return RetrievalResult(
            chunk_id=match.chunk_id,
            document_id=document_id,
            text=match.text,
            distance=match.distance,
            score=score,
            distance_metric=metric,
            metadata=metadata,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """檢索與查詢語意最相關的 chunks。

        Args:
            query: 使用者的自然語言問題。
            top_k: 最多回傳幾筆；None 表示採用設定檔預設值。

        Returns:
            由最相關到最不相關排序的結果；vector store 為空或
            沒有命中時回傳空 list（這是正常結果，不是錯誤）。

        Raises:
            RetrievalValidationError: query 或 top_k 不合法。
            RetrievalBackendError: embedding 或 vector store 失敗。
        """
        normalized_query = self._validate_query(query)
        resolved_top_k = self.resolve_top_k(top_k)

        # 即使呼叫端忘了呼叫 ensure_ready()（例如直接使用 Retriever
        # 而非透過 FastAPI dependency），也絕不在未驗證的狀態下搜尋。
        # 已驗證時這是一次布林檢查，成本可忽略。
        self.ensure_ready()

        # 醫療查詢可能含個資，因此只記錄長度而非查詢全文。
        logger.info(
            "檢索請求：query_length=%d top_k=%d collection=%s",
            len(normalized_query),
            resolved_top_k,
            self.vector_store.collection_name,
        )

        try:
            embedding = self.embedding_backend.embed_query(
                normalized_query
            )
            matches = self.vector_store.search_by_vector(
                embedding=embedding,
                top_k=resolved_top_k,
            )
        except EmbeddingError as exc:
            logger.exception("檢索失敗：query embedding 產生錯誤")
            raise RetrievalBackendError(
                "無法為查詢建立向量表示"
            ) from exc
        except VectorStoreError as exc:
            logger.exception("檢索失敗：vector store 查詢錯誤")
            raise RetrievalBackendError(
                "向量資料庫查詢失敗"
            ) from exc

        results = [self._to_result(match) for match in matches]

        logger.info(
            "檢索完成：results=%d top_k=%d",
            len(results),
            resolved_top_k,
        )
        return results
