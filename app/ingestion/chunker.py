"""文件切塊（chunking）。

為什麼需要切塊：未來的 embedding 模型有輸入長度上限，而且檢索的精確度也與片段大小有關。
片段太大，一個 chunk 裡混雜多個主題，向量會被「平均」掉而失去鑑別力；
片段太小，又會失去回答問題所需的上下文。切塊就是在這兩者之間取平衡。

overlap（重疊）的用意：若在句子中間硬切，語意會被切斷。讓相鄰 chunk 共用一段文字，
可以確保跨越邊界的資訊至少完整出現在其中一塊裡。

重要界定：本階段一律以「字元數」計量，不是 token。
token 是 embedding 模型的 tokenizer 切出來的單位，中文與英文的字元/token 比率差異很大
（英文約 4 字元 1 token，中文約 1 字元 1 token）。等到 Phase 3 引入實際模型後，
才有辦法談 token 數。現在宣稱是 token 會造成誤導。

本模組不依賴 LangChain 或任何外部切塊套件。
"""

import hashlib
import logging

from app.ingestion.exceptions import InvalidChunkConfigError
from app.ingestion.models import DocumentChunk, LoadedDocument

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 50

# 切分邊界的優先順序：由「語意斷點最強」到最弱。
# 找到高優先的分隔符就採用，不再往下找。
_SEPARATOR_GROUPS: tuple[tuple[str, ...], ...] = (
    ("\n\n",),                        # 空白行：段落結束，最強的語意邊界
    ("\n",),                          # 單一換行
    ("。", "！", "？", "；"),          # 中文句末標點
    (". ", "! ", "? ", "; "),         # 英文句末標點（後接空格，避免誤判小數點與縮寫）
)

# 邊界搜尋的最小可接受比例：只在 chunk 後半段找分隔符。
# 若允許在很前面切，會產生大量過短的 chunk，反而破壞語意完整性。
_MIN_BOUNDARY_RATIO = 0.5


def validate_chunk_config(
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> None:
    """驗證切塊參數合法性。

    Raises:
        InvalidChunkConfigError: 任一參數不合法。
    """
    if chunk_size <= 0:
        raise InvalidChunkConfigError(f"chunk_size 必須大於 0，收到 {chunk_size}")
    if chunk_overlap < 0:
        raise InvalidChunkConfigError(
            f"chunk_overlap 不可為負數，收到 {chunk_overlap}"
        )
    if chunk_overlap >= chunk_size:
        raise InvalidChunkConfigError(
            f"chunk_overlap（{chunk_overlap}）必須小於 chunk_size（{chunk_size}），"
            "否則切塊無法前進"
        )
    if min_chunk_size < 0:
        raise InvalidChunkConfigError(
            f"min_chunk_size 不可為負數，收到 {min_chunk_size}"
        )
    if min_chunk_size > chunk_size:
        raise InvalidChunkConfigError(
            f"min_chunk_size（{min_chunk_size}）不可大於 chunk_size（{chunk_size}）"
        )


def build_chunk_id(
    source: str,
    file_name: str,
    page_number: int | None,
    paragraph_number: int | None,
    chunk_index: int,
    text: str,
    document_id: str | None = None,
) -> str:
    """產生確定性（deterministic）的 chunk 識別碼。

    為什麼不用隨機 UUID：同一份文件重新處理時，若 ID 每次都不同，
    向量資料庫就無法判斷「這筆資料已經存在」，會不斷產生重複項目。
    用內容雜湊當 ID，相同輸入必然得到相同 ID，天然具備去重能力（idempotent）。

    取 SHA-256 前 16 個十六進位字元（64 bits）。以本專案的資料量級，
    碰撞機率可忽略，同時比完整 64 字元易讀許多。

    Returns:
        16 個字元的十六進位字串。
    """
    # 正式 ingestion 流程會傳入 document_id，因此 chunk ID 不受檔名影響。
    # 若是舊測試或單獨使用 chunker，沒有 document_id 時則沿用舊識別方式。
    document_identity = document_id or f"{source}|{file_name}"

    payload = "|".join(
        [
            document_identity,
            str(page_number),
            str(paragraph_number),
            str(chunk_index),
            text,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def _find_boundary(text: str, start: int, end: int) -> int:
    """在 text[start:end] 的後半段尋找最佳切分點。

    由高優先到低優先逐組嘗試，每組使用 rfind 取「最靠近 end」的位置，
    讓 chunk 盡量填滿而不浪費空間。

    Returns:
        切分點的絕對索引（exclusive，即 chunk 的 end_char）。
        找不到合理邊界時回傳 -1，由呼叫端決定硬切。
    """
    window_start = start + int((end - start) * _MIN_BOUNDARY_RATIO)

    for separators in _SEPARATOR_GROUPS:
        best = -1
        for separator in separators:
            index = text.rfind(separator, window_start, end)
            if index != -1:
                # 切分點落在分隔符之後，讓分隔符歸屬於前一個 chunk
                candidate = index + len(separator)
                best = max(best, candidate)
        if best > start:
            return best

    return -1


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """將文字切成 (start_char, end_char) 區間清單。

    這裡刻意只回傳位置而不回傳文字，讓 start_char / end_char 必然與實際內容一致，
    不會出現「位置是估算的」這種不可靠情況。

    Returns:
        區間清單，每個元素為 (start, end)，end 為 exclusive。
    """
    total_length = len(text)
    if total_length == 0:
        return []

    spans: list[tuple[int, int]] = []
    position = 0

    while position < total_length:
        ideal_end = min(position + chunk_size, total_length)

        if ideal_end >= total_length:
            end = total_length
        else:
            boundary = _find_boundary(text, position, ideal_end)
            end = boundary if boundary > position else ideal_end

        if text[position:end].strip():
            spans.append((position, end))

        if end >= total_length:
            break

        # 下一段的起點退回 overlap 個字元。
        # max(..., position + 1) 是防止無限迴圈的保險：
        # 無論 overlap 與邊界如何互動，位置一定嚴格遞增。
        position = max(end - chunk_overlap, position + 1)

    return spans


def _merge_trailing_short_span(
    spans: list[tuple[int, int]],
    min_chunk_size: int,
) -> list[tuple[int, int]]:
    """若最後一段過短，與前一段合併。

    過短的 chunk 缺乏足夠上下文，單獨做 embedding 幾乎沒有檢索價值。
    合併時保留前一段的起點、最後一段的終點，因此內容不會遺失。
    """
    if len(spans) < 2 or min_chunk_size <= 0:
        return spans

    last_start, last_end = spans[-1]
    if (last_end - last_start) >= min_chunk_size:
        return spans

    previous_start, _ = spans[-2]
    return spans[:-2] + [(previous_start, last_end)]


def _merge_docx_group(
    documents: list[LoadedDocument],
) -> LoadedDocument:
    """將一組相鄰 DOCX 段落合併成一個可切塊的文件單位。

    多段合併時，以換行保留原本的段落邊界，
    並在 metadata 中記錄原始段落範圍。
    """
    if len(documents) == 1:
        return documents[0]

    first = documents[0]
    last = documents[-1]

    metadata = dict(first.metadata)
    metadata["paragraph_start"] = first.paragraph_number
    metadata["paragraph_end"] = last.paragraph_number

    return LoadedDocument(
        text="\n".join(document.text for document in documents),
        source=first.source,
        file_name=first.file_name,
        file_type=first.file_type,
        page_number=None,
        # 為了與既有 schema 相容，paragraph_number 保留起始段落。
        paragraph_number=first.paragraph_number,
        metadata=metadata,
    )


def _merge_short_docx_documents(
    documents: list[LoadedDocument],
    min_chunk_size: int,
) -> list[LoadedDocument]:
    """在 chunking 前合併相鄰且過短的 DOCX 段落。

    loader 仍保留原始段落結構；只有在準備 chunk 時才合併，
    因此格式解析與 chunking 的責任不會混在一起。
    """
    if min_chunk_size <= 0 or not documents:
        return documents

    # chunk_documents 正常情況下一次處理同一份文件。
    # 若呼叫端傳入混合格式，就維持原狀，不跨格式合併。
    if any(document.file_type != "docx" for document in documents):
        return documents

    groups: list[list[LoadedDocument]] = []
    buffer: list[LoadedDocument] = []
    buffer_length = 0

    for document in documents:
        if buffer:
            # 合併時會在段落間加入一個換行字元。
            buffer_length += 1

        buffer.append(document)
        buffer_length += len(document.text)

        if buffer_length >= min_chunk_size:
            groups.append(buffer)
            buffer = []
            buffer_length = 0

    # 最後若剩下一小段，併入前一組，避免產生尾端超短 chunk。
    if buffer:
        if groups:
            groups[-1].extend(buffer)
        else:
            # 整份 DOCX 本身就小於 min_chunk_size 時沒有其他內容可合併。
            groups.append(buffer)

    return [_merge_docx_group(group) for group in groups]


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    start_index: int = 0,
    document_id: str | None = None,
) -> list[DocumentChunk]:
    """將單一 LoadedDocument 切成多個 DocumentChunk。

    Args:
        document: 已載入（通常也已清理）的文件單位。
        chunk_size: 每塊的目標字元數上限。
        chunk_overlap: 相鄰塊之間重疊的字元數。
        min_chunk_size: 最後一塊若短於此值，會併入前一塊。
        start_index: chunk_index 的起始值，供 chunk_documents 產生跨單位的連續編號。

    Returns:
        DocumentChunk 清單。文件為空時回傳空清單。

    Raises:
        InvalidChunkConfigError: 參數不合法。
    """
    validate_chunk_config(chunk_size, chunk_overlap, min_chunk_size)

    text = document.text
    if not text.strip():
        return []

    spans = _split_text(text, chunk_size, chunk_overlap)
    spans = _merge_trailing_short_span(spans, min_chunk_size)

    chunks: list[DocumentChunk] = []
    for offset, (start_char, end_char) in enumerate(spans):
        chunk_index = start_index + offset
        chunk_text = text[start_char:end_char]

        chunks.append(
            DocumentChunk(
                chunk_id=build_chunk_id(
                    source=document.source,
                    file_name=document.file_name,
                    page_number=document.page_number,
                    paragraph_number=document.paragraph_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    document_id=document_id,
                ),
                text=chunk_text,
                source=document.source,
                file_name=document.file_name,
                file_type=document.file_type,
                page_number=document.page_number,
                paragraph_number=document.paragraph_number,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata=dict(document.metadata),
            )
        )

    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    document_id: str | None = None,
) -> list[DocumentChunk]:
    """將多個 LoadedDocument 切塊，chunk_index 跨單位連續編號。

    連續編號的意義：chunk_index 反映的是「在整份文件中的順序」，
    而不是「在某一頁中的順序」，這樣未來要還原上下文才有意義。

    Raises:
        InvalidChunkConfigError: 參數不合法。
    """
    validate_chunk_config(chunk_size, chunk_overlap, min_chunk_size)

    prepared_documents = _merge_short_docx_documents(
        documents,
        min_chunk_size,
    )


    all_chunks: list[DocumentChunk] = []
    for document in prepared_documents:
        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            start_index=len(all_chunks),
            document_id=document_id,
        )
        all_chunks.extend(chunks)

    logger.info(
        "切塊完成：%d 個文件單位 -> %d 個 chunk（chunk_size=%d, overlap=%d）",
        len(documents),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
