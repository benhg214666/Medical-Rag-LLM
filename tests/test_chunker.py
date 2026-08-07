"""切塊器（chunker）測試。

切塊是 RAG 品質的第一道關卡：切壞了，後面的 embedding 與檢索再好也救不回來。
因此除了基本行為，這裡特別驗證三件容易出錯的事：
  1. 內容不遺失（每個字元都被某個 chunk 涵蓋）
  2. start_char / end_char 與實際文字精確對應（不是估算值）
  3. 不會無限迴圈（極端參數下仍能終止）
"""

import pytest

from app.ingestion.chunker import (
    build_chunk_id,
    chunk_document,
    chunk_documents,
    validate_chunk_config,
)
from app.ingestion.exceptions import InvalidChunkConfigError
from app.ingestion.models import LoadedDocument


def make_document(text: str, **kwargs) -> LoadedDocument:
    """建立測試用的 LoadedDocument。"""
    defaults = {
        "source": "sample.txt",
        "file_name": "sample.txt",
        "file_type": "txt",
    }
    defaults.update(kwargs)
    return LoadedDocument(text=text, **defaults)


class TestBasicChunking:
    """基本切塊行為。"""

    def test_short_text_yields_single_chunk(self) -> None:
        chunks = chunk_document(make_document("很短的一段話。"), 500, 100, 50)
        assert len(chunks) == 1
        assert chunks[0].text == "很短的一段話。"

    def test_long_text_yields_multiple_chunks(self) -> None:
        text = "".join(f"這是第{i}句話，內容關於病患狀況。" for i in range(60))
        chunks = chunk_document(make_document(text), 200, 50, 30)
        assert len(chunks) > 1

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_document(make_document(""), 500, 100, 50) == []

    def test_whitespace_only_yields_no_chunks(self) -> None:
        assert chunk_document(make_document("   \n\n  "), 500, 100, 50) == []

    def test_no_empty_chunks_produced(self) -> None:
        text = "段落一。\n\n\n段落二。\n\n\n" + "內容。" * 200
        chunks = chunk_document(make_document(text), 100, 20, 10)
        assert all(chunk.text.strip() for chunk in chunks)

    def test_chunk_index_starts_at_zero_and_is_continuous(self) -> None:
        text = "".join(f"句子{i}。" for i in range(200))
        chunks = chunk_document(make_document(text), 100, 20, 10)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


class TestOverlap:
    """重疊行為。"""

    def test_adjacent_chunks_overlap(self) -> None:
        text = "".join(f"病患第{i}次回診記錄。" for i in range(80))
        chunks = chunk_document(make_document(text), 200, 50, 30)
        assert len(chunks) > 1
        for current, following in zip(chunks, chunks[1:]):
            # 下一塊的起點必須早於前一塊的終點，才算有重疊
            assert following.start_char < current.end_char

    def test_zero_overlap_produces_adjacent_spans(self) -> None:
        text = "A" * 500
        chunks = chunk_document(make_document(text), 100, 0, 0)
        for current, following in zip(chunks, chunks[1:]):
            assert following.start_char == current.end_char


class TestContentIntegrity:
    """內容完整性 —— 最重要的一組。"""

    def test_every_character_is_covered(self) -> None:
        text = "".join(f"第{i}次評估，血壓 120/80 mmHg。" for i in range(80))
        chunks = chunk_document(make_document(text), 200, 50, 30)

        covered: set[int] = set()
        for chunk in chunks:
            covered.update(range(chunk.start_char, chunk.end_char))

        assert covered == set(range(len(text))), "有字元未被任何 chunk 涵蓋"

    def test_offsets_match_actual_text(self) -> None:
        text = "".join(f"句子{i}。" for i in range(150))
        chunks = chunk_document(make_document(text), 120, 30, 20)
        for chunk in chunks:
            assert text[chunk.start_char : chunk.end_char] == chunk.text

    def test_offsets_are_within_bounds(self) -> None:
        text = "測試內容。" * 100
        chunks = chunk_document(make_document(text), 100, 20, 10)
        for chunk in chunks:
            assert 0 <= chunk.start_char < chunk.end_char <= len(text)


class TestNaturalBoundaries:
    """在自然語意邊界切分。"""

    def test_chinese_period_boundary(self) -> None:
        # 每句約 20 字元，chunk_size 100 時應在句號處斷開
        text = "".join(f"這是一個完整的中文句子編號{i:03d}。" for i in range(40))
        chunks = chunk_document(make_document(text), 100, 20, 10)
        ends_at_period = sum(1 for c in chunks[:-1] if c.text.rstrip().endswith("。"))
        assert ends_at_period >= len(chunks) // 2, "多數 chunk 應在句號處切分"

    def test_blank_line_is_preferred_boundary(self) -> None:
        first = "第一段內容。" * 8
        second = "第二段內容。" * 8
        chunks = chunk_document(make_document(f"{first}\n\n{second}"), 60, 10, 5)
        assert len(chunks) > 1

    def test_hard_split_when_no_boundary_exists(self) -> None:
        # 完全沒有標點時仍須切開，不可因找不到邊界就放棄
        chunks = chunk_document(make_document("A" * 1000), 200, 50, 30)
        assert len(chunks) > 1
        assert all(chunk.text for chunk in chunks)


class TestDeterministicId:
    """chunk_id 的確定性。"""

    def test_same_input_yields_same_ids(self) -> None:
        document = make_document("內容測試。" * 100)
        first = chunk_document(document, 100, 20, 10)
        second = chunk_document(document, 100, 20, 10)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_different_text_yields_different_id(self) -> None:
        a = build_chunk_id("s.txt", "s.txt", None, None, 0, "文字A")
        b = build_chunk_id("s.txt", "s.txt", None, None, 0, "文字B")
        assert a != b

    def test_different_index_yields_different_id(self) -> None:
        a = build_chunk_id("s.txt", "s.txt", None, None, 0, "相同文字")
        b = build_chunk_id("s.txt", "s.txt", None, None, 1, "相同文字")
        assert a != b

    def test_different_page_yields_different_id(self) -> None:
        a = build_chunk_id("s.pdf", "s.pdf", 1, None, 0, "相同文字")
        b = build_chunk_id("s.pdf", "s.pdf", 2, None, 0, "相同文字")
        assert a != b

    def test_id_is_hex_of_expected_length(self) -> None:
        chunk_id = build_chunk_id("s.txt", "s.txt", None, None, 0, "文字")
        assert len(chunk_id) == 16
        assert all(char in "0123456789abcdef" for char in chunk_id)


class TestMetadataInheritance:
    """metadata 與來源資訊的繼承。"""

    def test_pdf_metadata_inherited(self) -> None:
        document = make_document(
            "頁面內容。" * 60,
            source="report.pdf",
            file_name="report.pdf",
            file_type="pdf",
            page_number=7,
            metadata={"total_pages": 12},
        )
        chunks = chunk_document(document, 100, 20, 10)
        assert chunks
        for chunk in chunks:
            assert chunk.source == "report.pdf"
            assert chunk.file_type == "pdf"
            assert chunk.page_number == 7
            assert chunk.metadata == {"total_pages": 12}

    def test_docx_paragraph_number_inherited(self) -> None:
        document = make_document(
            "段落內容。" * 60,
            source="note.docx",
            file_name="note.docx",
            file_type="docx",
            paragraph_number=3,
        )
        chunks = chunk_document(document, 100, 20, 10)
        assert all(c.paragraph_number == 3 for c in chunks)

    def test_metadata_is_copied_not_shared(self) -> None:
        # 若共用同一個 dict，改動一個 chunk 會污染其他 chunk
        document = make_document("內容。" * 60, metadata={"key": "value"})
        chunks = chunk_document(document, 100, 20, 10)
        chunks[0].metadata["key"] = "changed"
        assert chunks[-1].metadata["key"] == "value"


class TestTrailingShortChunk:
    """最後過短的 chunk 處理。"""

    def test_short_tail_is_merged(self) -> None:
        text = "A" * 205
        chunks = chunk_document(make_document(text), 100, 0, 50)
        # 尾段只有 5 字元，應併入前一塊而非單獨存在
        assert all(len(c.text) >= 50 for c in chunks)

    def test_merge_does_not_lose_content(self) -> None:
        text = "B" * 205
        chunks = chunk_document(make_document(text), 100, 0, 50)
        assert chunks[-1].end_char == len(text)

    def test_single_short_chunk_is_kept(self) -> None:
        # 只有一塊時無處可併，必須保留，否則短文件會整份消失
        chunks = chunk_document(make_document("短。"), 500, 100, 50)
        assert len(chunks) == 1
        assert chunks[0].text == "短。"


class TestConfigValidation:
    """參數驗證。"""

    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap, min_chunk_size",
        [
            (0, 0, 0),        # chunk_size 必須 > 0
            (-1, 0, 0),       # 負數
            (100, 100, 50),   # overlap 等於 size 會導致無法前進
            (100, 150, 50),   # overlap 大於 size
            (100, -1, 50),    # overlap 為負
            (100, 50, -1),    # min_chunk_size 為負
            (100, 50, 200),   # min_chunk_size 大於 size
        ],
    )
    def test_invalid_config_raises(
        self, chunk_size: int, chunk_overlap: int, min_chunk_size: int
    ) -> None:
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(chunk_size, chunk_overlap, min_chunk_size)

    def test_valid_config_passes(self) -> None:
        validate_chunk_config(500, 100, 50)

    def test_chunk_document_rejects_invalid_config(self) -> None:
        with pytest.raises(InvalidChunkConfigError):
            chunk_document(make_document("內容"), 100, 100, 50)


class TestTermination:
    """終止性：極端參數下不可無限迴圈。"""

    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [(10, 9), (2, 1), (500, 499), (3, 2)],
    )
    def test_extreme_overlap_terminates(
        self, chunk_size: int, chunk_overlap: int
    ) -> None:
        # 若位置沒有嚴格遞增，這裡會直接掛住而非失敗，因此是重要的迴歸測試
        chunks = chunk_document(
            make_document("x" * 300), chunk_size, chunk_overlap, 0
        )
        assert len(chunks) > 0
        assert chunks[-1].end_char == 300

    def test_size_one_terminates(self) -> None:
        chunks = chunk_document(make_document("測試文字"), 1, 0, 0)
        assert len(chunks) == 4


class TestChunkDocuments:
    """多文件單位的批次切塊。"""

    def test_index_is_continuous_across_documents(self) -> None:
        documents = [
            make_document("第一頁內容。" * 40, page_number=1, file_type="pdf"),
            make_document("第二頁內容。" * 40, page_number=2, file_type="pdf"),
            make_document("第三頁內容。" * 40, page_number=3, file_type="pdf"),
        ]
        chunks = chunk_documents(documents, 100, 20, 10)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_page_numbers_are_preserved_in_order(self) -> None:
        documents = [
            make_document("甲頁。" * 40, page_number=1, file_type="pdf"),
            make_document("乙頁。" * 40, page_number=2, file_type="pdf"),
        ]
        chunks = chunk_documents(documents, 100, 20, 10)
        pages = [c.page_number for c in chunks]
        assert pages == sorted(pages)
        assert set(pages) == {1, 2}

    def test_empty_list_yields_no_chunks(self) -> None:
        assert chunk_documents([], 500, 100, 50) == []

    def test_blank_documents_are_skipped(self) -> None:
        documents = [
            make_document("有內容。" * 30),
            make_document("   "),
            make_document("也有內容。" * 30),
        ]
        chunks = chunk_documents(documents, 100, 20, 10)
        assert all(c.text.strip() for c in chunks)

    def test_short_docx_paragraphs_are_merged_with_paragraph_range(self) -> None:
        documents = [
            make_document(
                "主訴：病人表示胸痛已持續三天，活動時症狀明顯加劇。",
                source="clinical_note.docx",
                file_name="clinical_note.docx",
                file_type="docx",
                paragraph_number=1,
            ),
            make_document(
                "生命徵象：血壓 120/80 mmHg，心跳每分鐘 78 次。",
                source="clinical_note.docx",
                file_name="clinical_note.docx",
                file_type="docx",
                paragraph_number=2,
            ),
            make_document(
                "處置：給予 Aspirin 100 mg，並安排後續心電圖檢查。",
                source="clinical_note.docx",
                file_name="clinical_note.docx",
                file_type="docx",
                paragraph_number=3,
            ),
        ]

        chunks = chunk_documents(
            documents,
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=50,
            document_id="test-document-id",
        )

        assert len(chunks) == 1

        chunk = chunks[0]

        assert "主訴：" in chunk.text
        assert "生命徵象：" in chunk.text
        assert "處置：" in chunk.text

        assert len(chunk.text) >= 50

        assert chunk.paragraph_number == 1
        assert chunk.metadata["paragraph_start"] == 1
        assert chunk.metadata["paragraph_end"] == 3