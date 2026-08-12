"""Regression tests for deterministic section context across chunk boundaries."""

from app.ingestion.chunker import chunk_documents
from app.ingestion.models import LoadedDocument
from app.prompts.prompt_builder import build_rag_prompt
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import LightweightReranker
from app.retrieval.vector_retriever import VectorRetriever
from tests.fakes import FakeEmbeddingBackend
from tests.test_retrieval import MemorySearchStore


def document(text: str) -> LoadedDocument:
    return LoadedDocument(
        text=text,
        source="synthetic.txt",
        file_name="synthetic.txt",
        file_type="txt",
    )


def retrieval_result(chunk, score: float = 0.7) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        document_id="doc-synthetic",
        text=chunk.text,
        distance=1.0 - score,
        score=score,
        distance_metric="cosine",
        metadata=chunk.metadata,
    )


def test_parent_heading_propagates_to_later_chunk() -> None:
    text = (
        "=== 2026-06-10 Outpatient Visit ===\n"
        "History:\n" + "stable history details " * 12 + "\n"
        "Laboratory Results:\nHbA1c: 9.2%\n"
    )
    chunks = chunk_documents([document(text)], chunk_size=120, chunk_overlap=20, min_chunk_size=20)
    value_chunk = next(chunk for chunk in chunks if "9.2%" in chunk.text)
    assert value_chunk.metadata["section_title"] == "2026-06-10 Outpatient Visit"
    assert value_chunk.metadata["section_path"].startswith("2026-06-10 Outpatient Visit")


def test_overlap_containing_only_closing_delimiter_does_not_clear_parent() -> None:
    text = (
        "=== 2026-06-10 Outpatient Visit ===\n"
        + "Clinical history. " * 18
        + "\nLaboratory Results:\nHbA1c: 9.2%"
    )
    chunks = chunk_documents(
        [document(text)], chunk_size=80, chunk_overlap=40, min_chunk_size=10
    )
    value_chunk = next(chunk for chunk in chunks if "9.2%" in chunk.text)
    assert value_chunk.metadata["section_title"] == "2026-06-10 Outpatient Visit"


def test_new_parent_heading_replaces_previous_heading() -> None:
    text = (
        "=== 2026-06-10 Outpatient Visit ===\nJune information.\n"
        "=== 2026-09-12 Follow-up Visit ===\nCreatinine: 1.0 mg/dL\n"
    )
    chunks = chunk_documents([document(text)], chunk_size=70, chunk_overlap=10, min_chunk_size=10)
    september = next(chunk for chunk in chunks if "Creatinine" in chunk.text)
    assert september.metadata["section_title"] == "2026-09-12 Follow-up Visit"
    assert "2026-06-10" not in september.metadata["section_path"]


def test_docx_heading_style_preserves_arbitrary_parent_title() -> None:
    documents = [
        LoadedDocument(
            text="Renal Follow-up Encounter",
            source="record.docx",
            file_name="record.docx",
            file_type="docx",
            paragraph_number=1,
            metadata={
                "heading_title": "Renal Follow-up Encounter",
                "heading_level": 1,
            },
        ),
        LoadedDocument(
            text="Laboratory Results:\nCreatinine: 1.0 mg/dL",
            source="record.docx",
            file_name="record.docx",
            file_type="docx",
            paragraph_number=2,
        ),
    ]
    chunks = chunk_documents(
        documents, chunk_size=80, chunk_overlap=10, min_chunk_size=30
    )
    value_chunk = next(chunk for chunk in chunks if "Creatinine" in chunk.text)
    assert value_chunk.metadata["section_title"] == "Renal Follow-up Encounter"
    assert value_chunk.metadata["section_path"].startswith(
        "Renal Follow-up Encounter"
    )


def test_plain_text_without_heading_remains_unenriched() -> None:
    chunks = chunk_documents(
        [document("This is an ordinary sentence with no structural heading." )],
        chunk_size=30,
        chunk_overlap=5,
        min_chunk_size=5,
    )
    assert chunks
    assert all("section_title" not in chunk.metadata for chunk in chunks)


def test_prompt_enriches_section_context_and_supports_legacy_metadata() -> None:
    contextual = RetrievalResult(
        chunk_id="contextual",
        text="Laboratory Results:\nCreatinine: 1.0 mg/dL",
        distance=0.2,
        score=0.8,
        distance_metric="cosine",
        metadata={"section_path": "2026-09-12 Follow-up Visit > Laboratory Results"},
    )
    legacy = contextual.model_copy(
        update={"chunk_id": "legacy", "text": "Legacy evidence", "metadata": {}}
    )
    prompt = build_rag_prompt("What was the creatinine?", [contextual, legacy])
    assert "[Document section: 2026-09-12 Follow-up Visit > Laboratory Results]" in prompt.user
    assert "[Source 2]\nLegacy evidence" in prompt.user


def test_section_metadata_round_trips_through_vector_store_and_retrieval() -> None:
    chunks = chunk_documents(
        [document("=== 2026-09-12 Follow-up Visit ===\nCreatinine: 1.0 mg/dL")],
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=10,
    )
    backend = FakeEmbeddingBackend()
    store = MemorySearchStore()
    store.add_chunks(chunks, backend.embed_documents([chunk.text for chunk in chunks]), "doc-roundtrip")
    retriever = VectorRetriever(backend, store, default_top_k=1, max_top_k=5)
    retrieved = retriever.retrieve("creatinine", top_k=1)[0]
    assert retrieved.metadata["section_title"] == "2026-09-12 Follow-up Visit"
    assert retrieved.metadata["section_path"].startswith("2026-09-12 Follow-up Visit")


def test_reranked_cross_chunk_evidence_keeps_parent_context_in_prompt() -> None:
    text = (
        "=== 2026-06-10 Outpatient Visit ===\n"
        + "Clinical history and medication review. " * 14
        + "\nLaboratory Results:\nHbA1c: 9.2%\n"
    )
    chunks = chunk_documents([document(text)], chunk_size=140, chunk_overlap=20, min_chunk_size=20)
    results = [retrieval_result(chunk, 0.90 - index * 0.04) for index, chunk in enumerate(chunks)]
    ranked = LightweightReranker().rerank(
        "What was the HbA1c on 2026-06-10?", results
    )
    selected = next(item for item in ranked if "9.2%" in item.text)
    prompt = build_rag_prompt("What was the HbA1c on 2026-06-10?", [selected])
    assert "2026-06-10 Outpatient Visit" in prompt.user
    assert "HbA1c: 9.2%" in prompt.user
