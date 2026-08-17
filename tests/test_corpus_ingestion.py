from pathlib import Path

from app.core.config import Settings
from app.core.runtime_data import reset_rag_data
from app.indexing.pipeline import IndexingPipeline
from app.ingestion.corpus import ingest_corpus, metadata_from_corpus_path, scan_corpus
from tests.fakes import FakeEmbeddingBackend, MemoryVectorStore


def test_patient_metadata_and_soap_filename(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    path = corpus / "patient_records" / "P001" / "2026-06-10_SOAP.txt"
    metadata = metadata_from_corpus_path(path, corpus)
    assert metadata == {
        "source_type": "patient_record",
        "patient_id": "P001",
        "encounter_date": "2026-06-10",
        "document_type": "SOAP",
    }


def test_nonstandard_patient_and_medical_knowledge_metadata(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    assert metadata_from_corpus_path(
        corpus / "patient_records" / "P002" / "random_note.txt", corpus
    ) == {"source_type": "patient_record", "patient_id": "P002"}
    assert metadata_from_corpus_path(
        corpus / "medical_knowledge" / "diabetes" / "targets.pdf", corpus
    ) == {"source_type": "medical_knowledge", "category": "diabetes"}


def test_scan_corpus_is_recursive_and_ignores_unsupported_hidden_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    supported = corpus / "patient_records" / "P001" / "note.TXT"
    supported.parent.mkdir(parents=True)
    supported.write_text("note", encoding="utf-8")
    (supported.parent / "image.png").write_bytes(b"x")
    hidden = corpus / ".cache" / "hidden.txt"
    hidden.parent.mkdir()
    hidden.write_text("hidden", encoding="utf-8")
    assert scan_corpus(corpus) == [supported]
    assert scan_corpus(tmp_path / "missing") == []


def test_reset_clears_only_runtime_and_preserves_corpus(tmp_path: Path) -> None:
    settings = Settings(
        raw_data_dir=Path("data/raw"),
        processed_data_dir=Path("data/processed"),
        corpus_data_dir=Path("data/corpus"),
        vector_db_dir=Path("vector_db"),
    )
    for relative in ("data/raw/a.txt", "data/processed/a.json", "vector_db/db.bin"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime", encoding="utf-8")
    corpus_file = tmp_path / "data/corpus/patient_records/P001/note.txt"
    corpus_file.parent.mkdir(parents=True)
    corpus_file.write_text("permanent", encoding="utf-8")
    outside = tmp_path / "README.keep"
    outside.write_text("keep", encoding="utf-8")

    reset_rag_data(settings, tmp_path)

    assert corpus_file.read_text(encoding="utf-8") == "permanent"
    assert outside.exists()
    assert all(not any((tmp_path / path).iterdir()) for path in ("data/raw", "data/processed", "vector_db"))


def test_batch_ingestion_preserves_metadata_and_is_idempotent(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus/patient_records/P001/2026-06-10_SOAP.txt"
    corpus_file.parent.mkdir(parents=True)
    corpus_file.write_text("Patient ID: P001\nEncounter Date: 2026-06-10\n\nS - Subjective\nStable symptoms.", encoding="utf-8")
    settings = Settings(
        corpus_data_dir=tmp_path / "corpus",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        vector_db_dir=tmp_path / "db",
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=1,
    )
    store = MemoryVectorStore("corpus_test")
    pipeline = IndexingPipeline(FakeEmbeddingBackend(), store)

    first = ingest_corpus(settings, pipeline)
    count = store.count()
    second = ingest_corpus(settings, pipeline)

    assert first.processed == second.processed == 1
    assert first.failed == second.failed == 0
    assert store.count() == count
    indexed_chunk = next(iter(store.records.values()))[0]
    metadata = indexed_chunk.metadata
    assert metadata["patient_id"] == "P001"
    assert metadata["source_type"] == "patient_record"
    assert metadata["encounter_date"] == "2026-06-10"
    assert metadata["document_type"] == "SOAP"
