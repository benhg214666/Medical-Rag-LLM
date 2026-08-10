"""Retrieval 子系統的資料模型。

RetrievalResult 是「專案層級」的檢索結果，刻意獨立於 vector store 的
回傳格式。資料在管線中的形態變化為：

    Chroma query dict --(ChromaStore)--> VectorMatch --(Retriever)--> RetrievalResult

多這一層轉換的理由：VectorMatch 只是「向量資料庫查到了什麼」，
RetrievalResult 則是「本專案認定的一筆檢索證據」——它把 document_id
從扁平 metadata 提升為一級欄位，並附上可直接用於排序的 score。
未來 Phase 5 的 RAG 層與前端只依賴 RetrievalResult，
換掉 Chroma 不會影響它們。
"""

from typing import Any

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """單一檢索結果，代表一段被判定與查詢相關的文件片段。

    關於 distance 與 score 的語意（重要，勿混用）：

        distance 直接來自 vector store，語意由 distance_metric 決定。
            cosine 時 distance = 1 - cosine_similarity，範圍 [0, 2]，
            **數值越小代表越相似**。

        score 是「越大越相似」的正向分數，方便排序與門檻判斷。
            僅在 metric 為 cosine 時由 score = 1 - distance 換算，
            結果即為 cosine similarity 本身，範圍 [-1, 1]。
            其他 metric 下無法保證此換算正確，因此 score 會是 None，
            此時請一律以 distance 排序。

    這個換算**不需要**向量事先做 L2 normalization：cosine similarity
    的定義本身就除以兩個向量的模長，因此對向量長度不敏感。
    正規化會影響的是 L2 或 inner product 這類度量，與此處無關。

    兩者同時保留而不是只留一個，是為了避免產生誤導性分數：
    呼叫端若看到 score is None，就知道不能假設任何 similarity 語意。

    Attributes:
        chunk_id: Phase 2 產生的確定性 chunk 識別碼。
        document_id: 所屬文件的識別碼；由 Phase 3 寫入 vector store metadata。
        text: chunk 原文內容。
        distance: 原始距離值，越小越相似（cosine 時）。
        score: 正向相似度分數，越大越相似；無法安全換算時為 None。
        distance_metric: 產生 distance 的度量名稱，例如 "cosine"。
        metadata: 保存於 vector store 的其他 metadata，
            例如 source、file_name、file_type、page_number、
            paragraph_number、chunk_index、start_char、end_char。
            只包含實際存在的欄位，不會補上任何虛構值。
    """

    chunk_id: str
    document_id: str | None = None
    text: str
    distance: float
    score: float | None = None
    distance_metric: str
    metadata: dict[str, Any] = Field(default_factory=dict)
