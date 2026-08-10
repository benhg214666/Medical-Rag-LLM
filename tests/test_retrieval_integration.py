"""Phase 4 端對端整合測試：真實 embedding + 真實 Chroma + Retriever。

為什麼一定要有這一組測試：

單元測試用的 FakeEmbeddingBackend 產生的是 SHA-256 雜湊向量 ——
它是**確定性**的，但**沒有語意**。「metformin」和「aspirin」在它眼中
的距離完全是隨機的。因此 fake 能驗證「排序邏輯有沒有照 distance 執行」，
卻無法驗證「排序結果在語意上是否正確」。

而 Phase 4 的核心正是檢索**正確性**。所以必須有一組測試真的跑完
    文字 -> 真實 embedding -> 真實向量庫 -> 相似度搜尋 -> 排名
才能證明整條路徑可用。

執行成本：會載入 intfloat/multilingual-e5-small（約 470 MB），
首次執行需從 Hugging Face 下載，之後使用本機 cache。
因此標記為 integration，可用 `-m "not integration"` 排除。

測試資料全部為虛構的合成病歷，不含任何真實病人資料。
"""

import socket
from pathlib import Path

import pytest

from app.core.config import Settings
from app.embeddings.base import EmbeddingError
from app.embeddings.factory import create_embedding_backend
from app.ingestion.models import DocumentChunk
from app.retrieval.vector_retriever import VectorRetriever
from app.vector_store.chroma_store import ChromaStore

pytestmark = pytest.mark.integration


# 虛構的合成病歷片段，語意上刻意彼此明顯不同。
SYNTHETIC_NOTES: dict[str, str] = {
    "chunk-diagnosis": "Patient has type 2 diabetes mellitus.",
    "chunk-medication": (
        "Patient is currently taking metformin 500 mg twice daily."
    ),
    "chunk-surgery": "Patient underwent appendectomy in 2018.",
    "chunk-allergy": "Patient has no known drug allergies.",
}


def build_chunk(chunk_id: str, text: str, chunk_index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source="synthetic_patient_note.txt",
        file_name="synthetic_patient_note.txt",
        file_type="txt",
        page_number=None,
        paragraph_number=None,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(text),
        metadata={},
    )


def _is_model_unavailable(exc: BaseException) -> str | None:
    """判斷例外是否確實是「取不到模型」，而非程式本身的 bug。

    為什麼要這麼小心：若對任意 Exception 都 skip，一個真正的
    embedding 實作 regression（例如前綴邏輯寫錯、維度計算錯誤）
    會被當成「環境沒有模型」而靜默跳過，測試永遠是綠的，
    但保護作用完全失效。

    因此只在能明確辨識為環境問題時才允許 skip：
    網路不可用、Hugging Face 上找不到該 revision、本機 cache 缺檔。
    其餘一律讓測試 fail。

    Returns:
        可 skip 時回傳原因字串；應視為真實失敗時回傳 None。
    """
    # 走訪整條 __cause__ / __context__ 鏈：EmbeddingError 會包裝
    # 底層的網路或檔案錯誤，只看最外層會漏判。
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, (OSError, socket.gaierror, socket.timeout)):
            # OSError 涵蓋連線失敗與檔案不存在，兩者都是環境問題。
            return f"無法取得模型檔案或網路不可用：{type(current).__name__}"

        module = type(current).__module__ or ""
        name = type(current).__name__
        # huggingface_hub 的錯誤型別不直接 import，避免在未安裝時
        # 造成 collection error；改以模組名稱與型別名稱辨識。
        if module.startswith("huggingface_hub") or module.startswith("requests"):
            return f"Hugging Face 存取失敗：{name}"
        if name in {
            "LocalEntryNotFoundError",
            "RepositoryNotFoundError",
            "RevisionNotFoundError",
            "OfflineModeIsEnabled",
            "HFValidationError",
        }:
            return f"模型或 revision 不可取得：{name}"

        current = current.__cause__ or current.__context__

    return None


@pytest.fixture(scope="module")
def embedding_backend():
    """真實的本地 embedding backend。

    module scope：模型載入成本高，同一個模組內的測試共用一份。
    這不會造成測試間的狀態污染 —— embedding 是無狀態的純函式，
    真正需要隔離的是 vector store，那個仍然每個測試各自建立。

    skip 政策：只有在明確判定為「環境取不到模型」時才 skip；
    任何 embedding 實作本身的錯誤都必須讓測試失敗（見 _is_model_unavailable）。
    """
    settings = Settings()
    backend = create_embedding_backend(settings)

    try:
        # 觸發 lazy loading。模型下載或載入的問題會在這裡出現。
        vector = backend.embed_query("warm up")
    except EmbeddingError as exc:
        reason = _is_model_unavailable(exc)
        if reason is None:
            # 不是環境問題 —— 這是真正的 regression，必須失敗。
            raise
        pytest.skip(f"整合測試需要真實 embedding 模型；{reason}")

    # 走到這裡代表模型確實可用，之後任何錯誤都是真實失敗。
    # 順帶驗證 backend 的基本契約，讓後續測試的前提明確。
    assert isinstance(vector, list) and vector, "embed_query 應回傳非空向量"
    assert len(vector) == backend.dimension, (
        "embed_query 的維度與 backend 宣告不一致"
    )
    return backend


@pytest.fixture
def retriever(embedding_backend, tmp_path: Path) -> VectorRetriever:
    """每個測試都用全新的 tmp_path Chroma，確保彼此隔離。"""
    store = ChromaStore(tmp_path / "chroma", "integration_retrieval")
    store.ensure_embedding_compatibility(
        model_name=embedding_backend.model_name,
        model_revision=embedding_backend.model_revision,
        dimension=embedding_backend.dimension,
        normalized=embedding_backend.normalizes_embeddings,
    )

    chunks = [
        build_chunk(chunk_id, text, index)
        for index, (chunk_id, text) in enumerate(SYNTHETIC_NOTES.items())
    ]
    embeddings = embedding_backend.embed_documents(
        [chunk.text for chunk in chunks]
    )
    store.add_chunks(chunks, embeddings, "doc-synthetic-patient")

    return VectorRetriever(
        embedding_backend=embedding_backend,
        vector_store=store,
        default_top_k=5,
        max_top_k=50,
    )


def test_medication_query_ranks_medication_chunk_above_surgery(
    retriever: VectorRetriever,
) -> None:
    """核心測試：語意排序必須真的正確。

    詢問「病人在吃什麼藥」時，提到 metformin 的片段應該排在
    提到 appendectomy 的片段之前。注意查詢字串裡完全沒有出現
    "metformin" 這個詞 —— 若只是字串比對是做不到的，
    唯有真正理解語意的 embedding 才能命中。
    """
    results = retriever.retrieve(
        "What medication is the patient taking?", top_k=4
    )

    assert len(results) == 4

    ranking = [result.chunk_id for result in results]
    medication_rank = ranking.index("chunk-medication")
    surgery_rank = ranking.index("chunk-surgery")

    assert medication_rank < surgery_rank, (
        "與用藥相關的 chunk 應排在手術史之前，"
        f"實際排名為 {ranking}"
    )


def test_medication_query_ranks_medication_chunk_first(
    retriever: VectorRetriever,
) -> None:
    """更嚴格的期望：用藥片段應該就是第一名。"""
    results = retriever.retrieve(
        "What medication is the patient taking?", top_k=4
    )

    assert results[0].chunk_id == "chunk-medication"
    assert "metformin" in results[0].text.lower()


def test_surgical_history_query_ranks_surgery_chunk_first(
    retriever: VectorRetriever,
) -> None:
    """反向驗證：換一個主題，排名應該跟著改變。

    這一條防止「某個 chunk 剛好對所有查詢都排第一」的假陽性。
    """
    results = retriever.retrieve(
        "Has the patient had any surgery in the past?", top_k=4
    )

    assert results[0].chunk_id == "chunk-surgery"


def test_scores_are_descending_and_distances_ascending(
    retriever: VectorRetriever,
) -> None:
    """distance 與 score 的語意必須與宣告一致。"""
    results = retriever.retrieve("diabetes diagnosis", top_k=4)

    distances = [result.distance for result in results]
    scores = [result.score for result in results]

    assert distances == sorted(distances), "distance 應由小到大"
    assert scores == sorted(scores, reverse=True), "score 應由大到小"

    for result in results:
        assert result.distance_metric == "cosine"
        assert result.score == pytest.approx(1.0 - result.distance)
        # cosine distance 的定義域為 [0, 2]。
        assert 0.0 <= result.distance <= 2.0


def test_metadata_survives_the_full_pipeline(
    retriever: VectorRetriever,
) -> None:
    """走完整條路徑後，Phase 2 的來源資訊仍必須完整。"""
    result = retriever.retrieve("metformin dosage", top_k=1)[0]

    assert result.document_id == "doc-synthetic-patient"
    assert result.chunk_id in SYNTHETIC_NOTES
    assert result.metadata["source"] == "synthetic_patient_note.txt"
    assert result.metadata["file_name"] == "synthetic_patient_note.txt"
    assert result.metadata["file_type"] == "txt"
    assert "chunk_index" in result.metadata


def test_top_k_controls_result_count(retriever: VectorRetriever) -> None:
    assert len(retriever.retrieve("patient history", top_k=1)) == 1
    assert len(retriever.retrieve("patient history", top_k=2)) == 2
    # 超過實際 chunk 數時回傳全部，不補空值也不報錯。
    assert len(retriever.retrieve("patient history", top_k=20)) == 4


def test_persisted_chunks_are_retrievable_after_reopen(
    embedding_backend, tmp_path: Path
) -> None:
    """證明檢索依賴的是磁碟上的資料，而非 process 記憶體。"""
    db_dir = tmp_path / "persistent-integration"

    writer = ChromaStore(db_dir, "persistent_integration")
    # 與 Phase 3 indexing pipeline 相同：寫入前先建立相容性合約，
    # 之後重新開啟的 Retriever 才能通過 ensure_ready() 驗證。
    writer.ensure_embedding_compatibility(
        model_name=embedding_backend.model_name,
        model_revision=embedding_backend.model_revision,
        dimension=embedding_backend.dimension,
        normalized=embedding_backend.normalizes_embeddings,
    )
    chunks = [
        build_chunk(chunk_id, text, index)
        for index, (chunk_id, text) in enumerate(SYNTHETIC_NOTES.items())
    ]
    writer.add_chunks(
        chunks,
        embedding_backend.embed_documents(
            [chunk.text for chunk in chunks]
        ),
        "doc-synthetic-patient",
    )
    del writer

    reopened = ChromaStore(db_dir, "persistent_integration")
    retriever = VectorRetriever(
        embedding_backend=embedding_backend,
        vector_store=reopened,
        default_top_k=5,
        max_top_k=50,
    )

    results = retriever.retrieve(
        "What medication is the patient taking?", top_k=4
    )

    assert len(results) == 4
    assert results[0].chunk_id == "chunk-medication"


def test_query_prefix_is_applied_for_e5_models(embedding_backend) -> None:
    """E5 是非對稱模型，query 與 passage 必須走不同前綴。

    這裡不重寫前綴邏輯，而是驗證 Phase 3 的 backend 確實區別對待
    兩者 —— 若 Phase 4 自行呼叫 encode() 繞過 embed_query，
    就會漏掉 "query:" 前綴且不會有任何錯誤訊息。
    """
    if "e5" not in embedding_backend.model_name.lower():
        pytest.skip("目前模型不是 E5 系列，不適用前綴檢查")

    text = "Patient is taking metformin."
    as_query = embedding_backend.embed_query(text)
    as_document = embedding_backend.embed_documents([text])[0]

    assert as_query != as_document, (
        "E5 的 query 與 passage 向量應因前綴不同而相異"
    )
