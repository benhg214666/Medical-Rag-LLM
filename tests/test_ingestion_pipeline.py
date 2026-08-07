"""Ingestion pipeline 端對端測試。

全部輸出都導向 pytest 的 tmp_path，絕不寫入專案的 data/ 目錄 ——
測試污染正式資料目錄是很難察覺卻很麻煩的問題。
"""

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.ingestion.exceptions import EmptyDocumentError
from app.ingestion.pipeline import (
    compute_document_id,
    ingest_document,
    sanitize_filename,
    to_display_path,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """所有路徑都指向 tmp_path 的測試用設定。"""
    return Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_size=30,
    )


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """含中英文與醫療數值的測試文件。"""
    file_path = tmp_path / "clinical_note.txt"
    content = "病患主訴：胸痛。\n\n" + "".join(
        f"第{i}次評估，血壓 120/80 mmHg，給予 Aspirin 100 mg。" for i in range(30)
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


class TestFullPipeline:
    """完整流程 load -> clean -> chunk -> JSON。"""

    def test_returns_structured_result(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)

        assert result.document_id
        assert result.file_type == "txt"
        assert result.chunks
        assert result.output_file

    def test_json_file_is_created_and_readable(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)

        output_files = list(settings.processed_data_dir.glob("*.json"))
        assert len(output_files) == 1

        data = json.loads(output_files[0].read_text(encoding="utf-8"))
        assert data["document_id"] == result.document_id

    def test_json_contains_required_keys(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        ingest_document(sample_txt, settings)
        output_file = next(settings.processed_data_dir.glob("*.json"))
        data = json.loads(output_file.read_text(encoding="utf-8"))

        for key in (
            "document_id",
            "source_file",
            "file_type",
            "created_at",
            "processing",
            "statistics",
            "chunks",
        ):
            assert key in data

    def test_chinese_is_not_escaped_in_json(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        ingest_document(sample_txt, settings)
        output_file = next(settings.processed_data_dir.glob("*.json"))
        raw_text = output_file.read_text(encoding="utf-8")

        # ensure_ascii=False 生效時，中文應直接可讀而非 \uXXXX
        assert "病患" in raw_text
        assert "\\u75c5" not in raw_text

    def test_processing_config_is_recorded(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)

        assert result.processing.chunk_size == 200
        assert result.processing.chunk_overlap == 50
        assert result.processing.min_chunk_size == 30

    def test_created_at_is_iso8601(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        from datetime import datetime

        result = ingest_document(sample_txt, settings)
        # 能被 fromisoformat 解析即表示格式合法
        assert datetime.fromisoformat(result.created_at)


class TestStatistics:
    """統計數字的正確性。"""

    def test_statistics_match_actual_chunks(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)

        assert result.statistics.chunk_count == len(result.chunks)
        assert result.statistics.total_characters == sum(
            len(chunk.text) for chunk in result.chunks
        )

    def test_txt_yields_one_loaded_unit(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)

        assert result.statistics.loaded_units == 1
        assert result.statistics.cleaned_units == 1

    def test_statistics_in_json_match_result(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)
        output_file = next(settings.processed_data_dir.glob("*.json"))
        data = json.loads(output_file.read_text(encoding="utf-8"))

        assert data["statistics"]["chunk_count"] == result.statistics.chunk_count
        assert len(data["chunks"]) == data["statistics"]["chunk_count"]


class TestDeterminism:
    """確定性與冪等性。"""

    def test_document_id_is_stable(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        first = ingest_document(sample_txt, settings)
        second = ingest_document(sample_txt, settings)

        assert first.document_id == second.document_id

    def test_chunk_ids_are_stable(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        first = ingest_document(sample_txt, settings)
        second = ingest_document(sample_txt, settings)

        assert [c.chunk_id for c in first.chunks] == [
            c.chunk_id for c in second.chunks
        ]

    def test_reprocessing_overwrites_single_file(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        ingest_document(sample_txt, settings)
        ingest_document(sample_txt, settings)

        # 檔名含內容 hash，相同內容應覆寫而非累積
        assert len(list(settings.processed_data_dir.glob("*.json"))) == 1

    def test_same_content_different_name_shares_document_id(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        content = "相同內容。" * 40
        first_path = tmp_path / "name_a.txt"
        second_path = tmp_path / "name_b.txt"
        first_path.write_text(content, encoding="utf-8")
        second_path.write_text(content, encoding="utf-8")

        # document_id 依內容計算，可用來偵測重複上傳
        assert compute_document_id(first_path) == compute_document_id(second_path)

    def test_different_content_yields_different_id(self, tmp_path: Path) -> None:
        first_path = tmp_path / "a.txt"
        second_path = tmp_path / "b.txt"
        first_path.write_text("內容甲", encoding="utf-8")
        second_path.write_text("內容乙", encoding="utf-8")

        assert compute_document_id(first_path) != compute_document_id(second_path)


class TestOutputIsolation:
    """輸出隔離：絕不污染專案正式目錄。"""

    def test_output_stays_in_tmp_path(
        self, sample_txt: Path, settings: Settings, tmp_path: Path
    ) -> None:
        ingest_document(sample_txt, settings)

        for output_file in settings.processed_data_dir.glob("*.json"):
            assert tmp_path in output_file.parents

    def test_project_data_dir_untouched(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        project_processed = Path("data/processed")
        before = (
            set(project_processed.glob("*.json")) if project_processed.exists() else set()
        )

        ingest_document(sample_txt, settings)

        after = (
            set(project_processed.glob("*.json")) if project_processed.exists() else set()
        )
        assert before == after

    def test_write_output_false_creates_no_file(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings, write_output=False)

        assert result.output_file is None
        assert not settings.processed_data_dir.exists() or not list(
            settings.processed_data_dir.glob("*.json")
        )


class TestNoAbsolutePathLeak:
    """不洩漏主機絕對路徑。"""

    def test_source_file_is_name_only(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)

        assert result.source_file == "clinical_note.txt"
        assert "/" not in result.source_file

    def test_output_file_is_not_absolute(
        self, sample_txt: Path, settings: Settings
    ) -> None:
        result = ingest_document(sample_txt, settings)

        assert result.output_file is not None
        assert not Path(result.output_file).is_absolute()
        assert not result.output_file.startswith("/")

    def test_json_contains_no_absolute_path(
        self, sample_txt: Path, settings: Settings, tmp_path: Path
    ) -> None:
        ingest_document(sample_txt, settings)
        output_file = next(settings.processed_data_dir.glob("*.json"))
        raw_text = output_file.read_text(encoding="utf-8")

        assert str(tmp_path) not in raw_text

    def test_to_display_path_handles_outside_cwd(self, tmp_path: Path) -> None:
        display = to_display_path(tmp_path / "processed" / "x.json")

        assert not Path(display).is_absolute()
        assert display == "processed/x.json"


class TestErrorHandling:
    """錯誤情境。"""

    def test_empty_file_raises(self, tmp_path: Path, settings: Settings) -> None:
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding="utf-8")

        with pytest.raises(EmptyDocumentError):
            ingest_document(file_path, settings)

    def test_content_that_cleans_to_nothing_raises(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        file_path = tmp_path / "control.txt"
        # 只有控制字元，清理後不剩任何內容
        file_path.write_text("\x01\x02\x03", encoding="utf-8")

        with pytest.raises(EmptyDocumentError):
            ingest_document(file_path, settings)


class TestSanitizeFilename:
    """檔名安全化。"""

    @pytest.mark.parametrize(
        "raw_name, expected",
        [
            ("../../etc/passwd.txt", "passwd.txt"),
            ("..\\..\\windows\\system.txt", "system.txt"),
            ("normal.txt", "normal.txt"),
            (".hidden.txt", "hidden.txt"),
            ("with space.txt", "with_space.txt"),
            ("semi;colon.txt", "semi_colon.txt"),
            ("UPPER.TXT", "UPPER.txt"),
        ],
    )
    def test_dangerous_names_are_neutralized(
        self, raw_name: str, expected: str
    ) -> None:
        assert sanitize_filename(raw_name) == expected

    def test_path_separators_are_removed(self) -> None:
        result = sanitize_filename("../../etc/passwd.txt")

        assert "/" not in result
        assert ".." not in result

    def test_chinese_name_falls_back_safely(self) -> None:
        result = sanitize_filename("病歷.txt")

        # 中文會被替換，但仍是合法檔名；document_id 才是真正的識別依據
        assert result.endswith(".txt")
        assert result

    def test_empty_name_gets_default(self) -> None:
        assert sanitize_filename("") == "document"

    def test_very_long_name_is_truncated(self) -> None:
        result = sanitize_filename("a" * 300 + ".txt")

        assert len(result) < 100
