"""Documents API 測試。

關鍵手法：用 FastAPI 的 dependency_overrides 把 get_settings 換成指向 tmp_path 的設定，
讓上傳與輸出都落在測試暫存目錄，不會污染專案的 data/ 目錄。
每個測試結束後都會清除 override，避免測試之間互相影響。
"""

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """所有路徑導向 tmp_path 的設定。"""
    return Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        max_upload_size_mb=1,
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_size=30,
    )


@pytest.fixture
def client(test_settings: Settings) -> Iterator[TestClient]:
    """已注入測試設定的 TestClient。"""
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_txt_upload(
    content: str = "病患主訴：胸痛。\n" + "評估內容。" * 100,
    filename: str = "note.txt",
) -> dict:
    """建立 multipart 上傳用的 files 參數。"""
    return {"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")}


class TestPhase1Compatibility:
    """Phase 1 的 endpoint 不可被破壞。"""

    def test_root_still_works(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["status"] == "running"

    def test_health_still_works(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    @pytest.mark.parametrize("module", ["query", "documents", "models"])
    def test_module_status_endpoints_still_work(
        self, client: TestClient, module: str
    ) -> None:
        response = client.get(f"/api/{module}/status")

        assert response.status_code == 200
        assert response.json() == {"module": module, "status": "not_implemented"}


class TestSuccessfulUpload:
    """成功上傳的行為。"""

    def test_txt_upload_returns_200(self, client: TestClient) -> None:
        response = client.post("/api/documents/upload", files=make_txt_upload())

        assert response.status_code == 200

    def test_response_contains_document_id(self, client: TestClient) -> None:
        response = client.post("/api/documents/upload", files=make_txt_upload())
        body = response.json()

        assert body["document_id"]
        assert body["status"] == "processed"

    def test_response_contains_statistics(self, client: TestClient) -> None:
        response = client.post("/api/documents/upload", files=make_txt_upload())
        statistics = response.json()["statistics"]

        assert statistics["loaded_units"] >= 1
        assert statistics["cleaned_units"] >= 1
        assert statistics["chunk_count"] >= 1
        assert statistics["total_characters"] > 0

    def test_response_contains_chunk_previews(self, client: TestClient) -> None:
        response = client.post("/api/documents/upload", files=make_txt_upload())
        previews = response.json()["chunk_previews"]

        assert previews
        assert previews[0]["chunk_index"] == 0
        assert previews[0]["chunk_id"]
        assert previews[0]["text_preview"]

    def test_previews_are_limited_in_count(self, client: TestClient) -> None:
        long_content = "這是一段很長的病歷內容。" * 500
        response = client.post(
            "/api/documents/upload", files=make_txt_upload(content=long_content)
        )
        body = response.json()

        # 預覽不是資料傳遞管道，數量必須遠少於實際 chunk 數
        assert len(body["chunk_previews"]) <= 3
        assert body["statistics"]["chunk_count"] > len(body["chunk_previews"])

    def test_raw_file_saved_in_test_directory(
        self, client: TestClient, test_settings: Settings
    ) -> None:
        client.post("/api/documents/upload", files=make_txt_upload())

        saved = list(test_settings.raw_data_dir.glob("*.txt"))
        assert len(saved) == 1

    def test_processed_json_created_in_test_directory(
        self, client: TestClient, test_settings: Settings
    ) -> None:
        client.post("/api/documents/upload", files=make_txt_upload())

        outputs = list(test_settings.processed_data_dir.glob("*.json"))
        assert len(outputs) == 1

    def test_project_data_dir_not_polluted(self, client: TestClient) -> None:
        project_raw = Path("data/raw")
        before = set(project_raw.iterdir()) if project_raw.exists() else set()

        client.post("/api/documents/upload", files=make_txt_upload())

        after = set(project_raw.iterdir()) if project_raw.exists() else set()
        assert before == after

    def test_docx_upload_succeeds(self, client: TestClient) -> None:
        import docx

        document = docx.Document()
        for index in range(20):
            document.add_paragraph(f"第 {index} 段病歷記錄，血壓 120/80 mmHg。")
        buffer = io.BytesIO()
        document.save(buffer)
        buffer.seek(0)

        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "note.docx",
                    buffer,
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document",
                )
            },
        )

        assert response.status_code == 200
        assert response.json()["file_type"] == "docx"


class TestNoAbsolutePathLeak:
    """回應不可洩漏主機絕對路徑。"""

    def test_output_file_is_relative(self, client: TestClient) -> None:
        response = client.post("/api/documents/upload", files=make_txt_upload())
        output_file = response.json()["output_file"]

        assert not output_file.startswith("/")
        assert not Path(output_file).is_absolute()

    def test_response_body_has_no_tmp_path(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        response = client.post("/api/documents/upload", files=make_txt_upload())

        assert str(tmp_path) not in response.text


class TestUnsupportedFileType:
    """415：不支援的格式。"""

    @pytest.mark.parametrize("filename", ["data.csv", "sheet.xlsx", "script.exe"])
    def test_unsupported_extension_returns_415(
        self, client: TestClient, filename: str
    ) -> None:
        response = client.post(
            "/api/documents/upload",
            files={"file": (filename, io.BytesIO(b"content"), "text/plain")},
        )

        assert response.status_code == 415

    def test_content_type_spoofing_is_rejected(self, client: TestClient) -> None:
        # 宣告 text/plain 但副檔名是 .exe，必須以副檔名為準而非 Content-Type
        response = client.post(
            "/api/documents/upload",
            files={"file": ("malware.exe", io.BytesIO(b"MZ"), "text/plain")},
        )

        assert response.status_code == 415


class TestEmptyFile:
    """400：空檔案。"""

    def test_empty_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )

        assert response.status_code == 400

    def test_whitespace_only_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("blank.txt", io.BytesIO(b"   \n\n  "), "text/plain")},
        )

        assert response.status_code == 400

    def test_failed_upload_cleans_up_raw_file(
        self, client: TestClient, test_settings: Settings
    ) -> None:
        client.post(
            "/api/documents/upload",
            files={"file": ("blank.txt", io.BytesIO(b"   \n  "), "text/plain")},
        )

        # 處理失敗時不可留下半成品檔案
        if test_settings.raw_data_dir.exists():
            assert list(test_settings.raw_data_dir.glob("*.txt")) == []


class TestFileTooLarge:
    """413：檔案過大。"""

    def test_oversized_file_returns_413(self, client: TestClient) -> None:
        # test_settings 的上限是 1 MB
        oversized = b"a" * (2 * 1024 * 1024)
        response = client.post(
            "/api/documents/upload",
            files={"file": ("big.txt", io.BytesIO(oversized), "text/plain")},
        )

        assert response.status_code == 413

    def test_oversized_file_is_not_saved(
        self, client: TestClient, test_settings: Settings
    ) -> None:
        oversized = b"a" * (2 * 1024 * 1024)
        client.post(
            "/api/documents/upload",
            files={"file": ("big.txt", io.BytesIO(oversized), "text/plain")},
        )

        if test_settings.raw_data_dir.exists():
            assert list(test_settings.raw_data_dir.glob("*.txt")) == []


class TestScannedPdf:
    """400：掃描式 PDF 無法抽取文字。"""

    def test_scanned_pdf_returns_400_with_ocr_hint(self, client: TestClient) -> None:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)
        buffer.seek(0)

        response = client.post(
            "/api/documents/upload",
            files={"file": ("scanned.pdf", buffer, "application/pdf")},
        )

        assert response.status_code == 400
        assert "OCR" in response.json()["detail"]


class TestFilenameSanitization:
    """上傳檔名的安全化。"""

    def test_path_traversal_name_is_neutralized(
        self, client: TestClient, test_settings: Settings
    ) -> None:
        response = client.post(
            "/api/documents/upload",
            files=make_txt_upload(filename="../../evil.txt"),
        )

        assert response.status_code == 200
        assert response.json()["file_name"] == "evil.txt"
        # 檔案必須留在指定目錄內，不可跳脫
        assert (test_settings.raw_data_dir / "evil.txt").exists()


class TestIdempotency:
    """重複上傳相同內容。"""

    def test_same_content_yields_same_document_id(self, client: TestClient) -> None:
        first = client.post("/api/documents/upload", files=make_txt_upload())
        second = client.post("/api/documents/upload", files=make_txt_upload())

        assert first.json()["document_id"] == second.json()["document_id"]

    def test_repeated_upload_does_not_accumulate_files(
        self, client: TestClient, test_settings: Settings
    ) -> None:
        client.post("/api/documents/upload", files=make_txt_upload())
        client.post("/api/documents/upload", files=make_txt_upload())

        assert len(list(test_settings.processed_data_dir.glob("*.json"))) == 1
