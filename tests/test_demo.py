"""Phase 8 offline demo safety, orchestration, CLI, and export tests."""

import json
import subprocess
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core.config import Settings
from app.demo.cli import build_parser, main
from app.demo.data import CLINICAL_USE_MARKER, SYNTHETIC_MARKER, load_demo_cases, validate_demo_data
from app.demo.report import write_evaluation_artifacts
from app.demo.service import (
    DEMO_COLLECTION_NAME,
    DEMO_REFERENCE_F1_MIN,
    DemoError,
    create_demo_rag_service,
    demo_settings,
    evaluate_demo_cases,
    reset_demo_collection,
    run_demo_cases,
    seed_demo_collection,
    validate_demo_index,
)
from app.embeddings.base import EmbeddingError
from app.ingestion.models import DocumentChunk
from app.rag.models import RAGAnswer, RAGSource
from app.vector_store.base import VectorMatch, VectorStore
from tests.fakes import FakeEmbeddingBackend, search_records_by_vector


class MemoryDemoStore(VectorStore):
    def __init__(self, name: str = DEMO_COLLECTION_NAME) -> None:
        self.name = name
        self.records: dict[str, tuple[DocumentChunk, list[float], str]] = {}
        self.deleted = False

    @property
    def distance_metric(self) -> str:
        return "cosine"

    @property
    def collection_name(self) -> str:
        return self.name

    def ensure_embedding_compatibility(self, model_name, model_revision, dimension, normalized) -> None:
        return None

    def add_chunks(self, chunks, embeddings, document_id) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.records[chunk.chunk_id] = (chunk, embedding, document_id)

    def search_by_vector(self, embedding, top_k) -> list[VectorMatch]:
        return search_records_by_vector(self.records, embedding, top_k)

    def count(self) -> int:
        return len(self.records)

    def delete_collection(self) -> None:
        self.deleted = True
        self.records.clear()

    def collection_exists(self) -> bool:
        return bool(self.records)

    def delete_stale_chunks(self, document_id, keep_chunk_ids) -> int:
        stale = {
            chunk_id
            for chunk_id, (_, _, owner) in self.records.items()
            if owner == document_id and chunk_id not in keep_chunk_ids
        }
        return self.delete_chunks_by_ids(stale)

    def get_document_chunk_ids(self, document_id) -> set[str]:
        return {
            chunk_id
            for chunk_id, (_, _, owner) in self.records.items()
            if owner == document_id
        }

    def delete_chunks_by_ids(self, chunk_ids) -> int:
        existing = set(self.records) & set(chunk_ids)
        for chunk_id in existing:
            del self.records[chunk_id]
        return len(existing)


class CaseAwareRAG:
    def __init__(self) -> None:
        self.cases = {case.query: case for case in load_demo_cases()}
        self.calls: list[tuple[str, int | None]] = []

    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
        self.calls.append((query, top_k))
        case = self.cases[query]
        return RAGAnswer(
            answer=f"{case.reference_answer} [1]",
            model="fake-local",
            sources=[
                RAGSource(
                    source_number=1,
                    chunk_id=case.expected_chunk_ids[0],
                    document_id=case.expected_document_ids[0],
                    text="Synthetic test evidence.",
                    distance=0.1,
                    score=0.9,
                    distance_metric="cosine",
                    metadata={"source": "DEMO-P-test.txt"},
                )
            ],
        )


def test_demo_data_is_deterministic_synthetic_and_cross_referenced() -> None:
    records, cases = validate_demo_data()
    assert len(records) == 4
    assert [case.id for case in cases] == ["DEMO-C001", "DEMO-C002", "DEMO-C003", "DEMO-C004"]
    assert len({case.id for case in cases}) == len(cases)
    for record in records:
        text = record.read_text(encoding="utf-8")
        assert SYNTHETIC_MARKER in text
        assert CLINICAL_USE_MARKER in text
    assert all(case.synthetic and case.not_for_clinical_use for case in cases)
    assert all(case.expected_facts for case in cases)


def test_reset_refuses_non_demo_namespace() -> None:
    production_like = MemoryDemoStore("medical_documents")
    with pytest.raises(DemoError, match="refusing destructive operation"):
        reset_demo_collection(production_like)
    assert production_like.deleted is False


def test_seed_is_idempotent_and_uses_ingestion_ids() -> None:
    settings = Settings(chroma_collection_name=DEMO_COLLECTION_NAME)
    store = MemoryDemoStore()
    backend = FakeEmbeddingBackend()
    first = seed_demo_collection(settings, backend, store)
    first_ids = set(store.records)
    first_snapshot = {
        chunk_id: (chunk.model_dump(), vector, document_id)
        for chunk_id, (chunk, vector, document_id) in store.records.items()
    }
    second = seed_demo_collection(settings, backend, store)
    assert first.record_count == 4
    assert first.chunk_count == 4
    assert second.chunk_count == first.chunk_count
    assert set(store.records) == first_ids
    assert {
        chunk_id: (chunk.model_dump(), vector, document_id)
        for chunk_id, (chunk, vector, document_id) in store.records.items()
    } == first_snapshot
    assert set(store.records) == {
        chunk_id for case in load_demo_cases() for chunk_id in case.expected_chunk_ids
    }


def test_demo_chunking_is_pinned_against_environment_overrides() -> None:
    resolved = demo_settings(Settings(chunk_size=80, chunk_overlap=20, min_chunk_size=10))
    assert (resolved.chunk_size, resolved.chunk_overlap, resolved.min_chunk_size) == (500, 100, 50)


def test_missing_seed_fails_before_model_initialization(monkeypatch) -> None:
    monkeypatch.setattr("app.demo.service.create_vector_store", lambda settings: MemoryDemoStore())
    monkeypatch.setattr(
        "app.demo.service.get_embedding_backend_for",
        lambda settings: pytest.fail("embedding backend must not load for an empty collection"),
    )
    with pytest.raises(DemoError, match="seed --reset"):
        create_demo_rag_service(demo_settings(Settings()))


def test_partial_or_stale_demo_index_is_rejected() -> None:
    settings = demo_settings(Settings())
    store = MemoryDemoStore()
    seed_demo_collection(settings, FakeEmbeddingBackend(), store)
    removed_id = next(iter(store.records))
    del store.records[removed_id]
    with pytest.raises(DemoError, match="does not match"):
        validate_demo_index(store, load_demo_cases())

    seed_demo_collection(settings, FakeEmbeddingBackend(), store)
    chunk, vector, document_id = next(iter(store.records.values()))
    stale = chunk.model_copy(update={"chunk_id": "stale-demo-chunk"})
    del store.records[chunk.chunk_id]
    store.records[stale.chunk_id] = (stale, vector, document_id)
    with pytest.raises(DemoError, match="missing or stale"):
        validate_demo_index(store, load_demo_cases())


def test_runner_delegates_every_case_to_service() -> None:
    cases = load_demo_cases()[:2]
    rag = CaseAwareRAG()
    results = run_demo_cases(rag, cases)  # type: ignore[arg-type]
    assert [call[0] for call in rag.calls] == [case.query for case in cases]
    assert all(result.answer is not None and result.error is None for result in results)


def test_runner_rejects_missing_expected_evidence() -> None:
    class NoEvidenceRAG:
        def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
            return RAGAnswer(answer="Insufficient context.", model="fake", sources=[])

    result = run_demo_cases(
        NoEvidenceRAG(),  # type: ignore[arg-type]
        load_demo_cases()[:1],
    )[0]
    assert result.answer is not None
    assert "expected demo evidence" in (result.error or "")


def test_evaluation_json_and_markdown_include_required_fields(tmp_path: Path) -> None:
    cases = load_demo_cases()[:2]
    artifact = evaluate_demo_cases(
        CaseAwareRAG(),  # type: ignore[arg-type]
        cases,
        model_name="fake-local",
        now=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )
    json_path, markdown_path = write_evaluation_artifacts(artifact, tmp_path / "nested")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["totals"] == {"case_count": 2, "pass_count": 2, "fail_count": 0, "error_count": 0}
    assert loaded["cases"][0]["question"]
    assert loaded["cases"][0]["answer"]
    assert loaded["cases"][0]["sources"]
    assert loaded["cases"][0]["expected_facts"]
    assert loaded["cases"][0]["metrics"]
    assert loaded["quality_gate"]["reference_token_f1_min"] == DEMO_REFERENCE_F1_MIN
    assert loaded["cases"][0]["quality_gate"]["reference_token_f1_pass"] is True
    assert loaded["cases"][0]["status"] == "pass"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "not for clinical use" in markdown
    assert "DEMO-C001" in markdown
    assert "lexical overlap only" in markdown
    assert f">= {DEMO_REFERENCE_F1_MIN:.2f}" in markdown
    with pytest.raises(FileExistsError):
        write_evaluation_artifacts(artifact, tmp_path / "nested")


def test_evaluation_does_not_pass_an_uncited_answer() -> None:
    class UncitedRAG(CaseAwareRAG):
        def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
            answer = super().answer(query, top_k)
            return answer.model_copy(update={"answer": "An answer without a citation."})

    artifact = evaluate_demo_cases(
        UncitedRAG(),  # type: ignore[arg-type]
        load_demo_cases()[:1],
        model_name="fake-local",
    )
    assert artifact["totals"]["pass_count"] == 0
    assert artifact["totals"]["fail_count"] == 1
    assert artifact["cases"][0]["status"] == "fail"


def test_reasonable_paraphrase_with_expected_citation_passes() -> None:
    class ParaphraseRAG(CaseAwareRAG):
        def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
            answer = super().answer(query, top_k)
            return answer.model_copy(
                update={
                    "answer": (
                        "The record lists metformin 500 mg two times per day; "
                        "the latest HbA1c is 7.2 percent. [1]"
                    )
                }
            )

    artifact = evaluate_demo_cases(
        ParaphraseRAG(),  # type: ignore[arg-type]
        load_demo_cases()[:1],
        model_name="fake-local",
    )
    row = artifact["cases"][0]
    assert row["quality_gate"]["reference_token_f1"] >= DEMO_REFERENCE_F1_MIN
    assert row["quality_gate"]["reference_token_f1_pass"] is True
    assert row["status"] == "pass"


def test_clearly_wrong_answer_with_valid_expected_citation_fails() -> None:
    class WrongAnswerRAG(CaseAwareRAG):
        def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
            answer = super().answer(query, top_k)
            return answer.model_copy(
                update={"answer": "The patient takes aspirin 999 mg. [1]"}
            )

    artifact = evaluate_demo_cases(
        WrongAnswerRAG(),  # type: ignore[arg-type]
        load_demo_cases()[:1],
        model_name="fake-local",
    )
    row = artifact["cases"][0]
    assert row["quality_gate"]["retrieval_hit_pass"] is True
    assert row["quality_gate"]["citation_validity_pass"] is True
    assert row["quality_gate"]["citation_relevance_pass"] is True
    assert row["quality_gate"]["reference_token_f1"] < DEMO_REFERENCE_F1_MIN
    assert row["status"] == "fail"


def test_low_reference_f1_alone_fails_otherwise_valid_case() -> None:
    class LowOverlapRAG(CaseAwareRAG):
        def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
            answer = super().answer(query, top_k)
            return answer.model_copy(update={"answer": "Metformin. [1]"})

    artifact = evaluate_demo_cases(
        LowOverlapRAG(),  # type: ignore[arg-type]
        load_demo_cases()[:1],
        model_name="fake-local",
    )
    row = artifact["cases"][0]
    assert row["quality_gate"]["citation_presence_pass"] is True
    assert row["quality_gate"]["citation_validity_pass"] is True
    assert row["quality_gate"]["citation_relevance_pass"] is True
    assert row["quality_gate"]["reference_token_f1_pass"] is False
    assert row["status"] == "fail"


def test_reference_f1_is_explicitly_not_applicable_without_reference_answer() -> None:
    case = load_demo_cases()[0].model_copy(update={"reference_answer": None})
    artifact = evaluate_demo_cases(
        CaseAwareRAG(),  # type: ignore[arg-type]
        [case],
        model_name="fake-local",
    )
    gate = artifact["cases"][0]["quality_gate"]
    assert gate["reference_token_f1_applicable"] is False
    assert gate["reference_token_f1"] is None
    assert gate["reference_token_f1_pass"] is None
    assert artifact["cases"][0]["status"] == "pass"


def test_cli_help_and_reset_confirmation(capsys) -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "--case", "DEMO-C001"]).case_id == "DEMO-C001"
    assert parser.parse_args(["preflight", "--allow-unseeded"]).allow_unseeded is True
    assert parser.parse_args(["seed", "--reset"]).reset is True
    assert parser.parse_args(["evaluate", "--output-dir", "out"]).output_dir == Path("out")
    assert parser.parse_args(["all", "--require-api"]).require_api is True
    assert main(["reset"]) == 1
    assert "requires --yes" in capsys.readouterr().err


def test_documented_script_help_runs_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/demo.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "preflight" in completed.stdout


def test_cli_run_uses_service_and_reports_failure(monkeypatch, capsys) -> None:
    rag = CaseAwareRAG()
    monkeypatch.setattr("app.demo.cli.create_demo_rag_service", lambda settings: rag)
    assert main(["run", "--case", "DEMO-C001"]) == 0
    assert "SOURCES:" in capsys.readouterr().out

    class BrokenRAG:
        def answer(self, query, top_k=None):
            raise RuntimeError("local model service unavailable")

    monkeypatch.setattr("app.demo.cli.create_demo_rag_service", lambda settings: BrokenRAG())
    assert main(["run", "--case", "DEMO-C001"]) == 1
    assert "local model service unavailable" in capsys.readouterr().out


class FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _seeded_memory_store() -> MemoryDemoStore:
    store = MemoryDemoStore()
    seed_demo_collection(demo_settings(Settings()), FakeEmbeddingBackend(), store)
    return store


def _isolated_preflight_settings(*, llm_model_name: str = "preflight-test-model") -> Settings:
    """Return preflight settings that cannot inherit relevant local overrides."""
    return Settings(
        _env_file=None,
        vector_store_provider="chroma",
        embedding_provider="local",
        embedding_model_name="test-embedding-model",
        embedding_model_revision="test-revision",
        embedding_batch_size=1,
        llm_provider="openai_compatible",
        llm_base_url="http://127.0.0.1:8001/v1",
        llm_model_name=llm_model_name,
        llm_allow_private_network=False,
    )


def test_preflight_checks_configured_model_and_optional_api(monkeypatch, capsys) -> None:
    store = _seeded_memory_store()
    settings = _isolated_preflight_settings()
    monkeypatch.setattr("app.demo.cli.get_settings", lambda: settings)
    monkeypatch.setattr("app.demo.cli.create_vector_store", lambda settings: store)
    monkeypatch.setattr(
        "app.demo.cli.get_embedding_backend_for",
        lambda settings: FakeEmbeddingBackend(),
    )

    def fake_urlopen(request, timeout):
        url = request.full_url
        if url.endswith("/models"):
            return FakeHTTPResponse(
                {"object": "list", "data": [{"id": settings.llm_model_name}]}
            )
        raise urllib.error.URLError("FastAPI not running")

    monkeypatch.setattr("app.demo.cli._LOCAL_URL_OPENER.open", fake_urlopen)
    assert main(["preflight"]) == 0
    output = capsys.readouterr().out
    assert "local vLLM model" in output
    assert "embedding model" in output
    assert "[warn] FastAPI service" in output

    assert main(["preflight", "--require-api"]) == 1
    assert "FastAPI service" in capsys.readouterr().err


def test_preflight_rejects_wrong_served_model_and_blank_setting(monkeypatch, capsys) -> None:
    store = _seeded_memory_store()
    settings = _isolated_preflight_settings()
    monkeypatch.setattr("app.demo.cli.get_settings", lambda: settings)
    monkeypatch.setattr("app.demo.cli.create_vector_store", lambda settings: store)
    monkeypatch.setattr(
        "app.demo.cli.get_embedding_backend_for",
        lambda settings: FakeEmbeddingBackend(),
    )
    monkeypatch.setattr(
        "app.demo.cli._LOCAL_URL_OPENER.open",
        lambda request, timeout: FakeHTTPResponse({"data": [{"id": "other-model"}]}),
    )
    assert main(["preflight"]) == 1
    assert "is not served" in capsys.readouterr().err

    monkeypatch.setattr(
        "app.demo.cli.get_settings",
        lambda: _isolated_preflight_settings(llm_model_name=""),
    )
    assert main(["preflight"]) == 1
    assert "LLM_MODEL_NAME must not be blank" in capsys.readouterr().err


def test_preflight_embedding_failure_is_actionable(monkeypatch, capsys) -> None:
    store = _seeded_memory_store()
    monkeypatch.setattr(
        "app.demo.cli.get_settings", _isolated_preflight_settings
    )
    monkeypatch.setattr("app.demo.cli.create_vector_store", lambda settings: store)

    class BrokenEmbedding(FakeEmbeddingBackend):
        @property
        def dimension(self) -> int:
            raise EmbeddingError("model is not cached")

    monkeypatch.setattr(
        "app.demo.cli.get_embedding_backend_for",
        lambda settings: BrokenEmbedding(),
    )
    assert main(["preflight"]) == 1
    assert "pre-cache the configured model" in capsys.readouterr().err


def test_cli_evaluate_writes_artifacts_and_error_is_nonzero(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        "app.demo.cli.create_demo_rag_service",
        lambda settings: CaseAwareRAG(),
    )
    assert main(
        ["evaluate", "--case", "DEMO-C001", "--output-dir", str(tmp_path)]
    ) == 0
    assert len(list(tmp_path.glob("evaluation-*.json"))) == 1
    assert len(list(tmp_path.glob("evaluation-*.md"))) == 1
    assert "pass=1" in capsys.readouterr().out

    class BrokenRAG:
        def answer(self, query, top_k=None):
            raise RuntimeError("vLLM stopped")

    second_output = tmp_path / "error"
    monkeypatch.setattr(
        "app.demo.cli.create_demo_rag_service",
        lambda settings: BrokenRAG(),
    )
    assert main(
        ["evaluate", "--case", "DEMO-C001", "--output-dir", str(second_output)]
    ) == 1
    error_report = json.loads(next(second_output.glob("evaluation-*.json")).read_text(encoding="utf-8"))
    assert error_report["totals"]["error_count"] == 1
    assert error_report["cases"][0]["status"] == "error"


def test_all_runs_stages_in_order_and_stops_on_run_failure(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("app.demo.cli.preflight", lambda **kwargs: calls.append("preflight"))
    monkeypatch.setattr("app.demo.cli._seed", lambda **kwargs: calls.append("seed"))
    monkeypatch.setattr(
        "app.demo.cli._display_run",
        lambda *args, **kwargs: calls.append("run") or False,
    )
    monkeypatch.setattr(
        "app.demo.cli._evaluate",
        lambda *args, **kwargs: calls.append("evaluate") or True,
    )
    assert main(["all"]) == 1
    assert calls == ["preflight", "seed", "run"]
