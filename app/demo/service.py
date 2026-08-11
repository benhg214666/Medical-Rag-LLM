"""Safe Phase 8 orchestration over the existing ingestion/RAG/evaluation layers."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from app.core.config import Settings
from app.demo.data import (
    DEMO_CHUNK_OVERLAP,
    DEMO_CHUNK_SIZE,
    DEMO_MIN_CHUNK_SIZE,
    DemoCase,
    validate_demo_data,
)
from app.embeddings.base import EmbeddingBackend
from app.embeddings.dependencies import get_embedding_backend_for
from app.evaluation.evaluator import Evaluator, aggregate_results
from app.evaluation.models import EvaluationReport
from app.indexing.pipeline import IndexingPipeline
from app.ingestion.pipeline import ingest_document
from app.llm.dependencies import get_llm_provider
from app.rag.models import RAGAnswer
from app.rag.service import RAGService
from app.retrieval.vector_retriever import VectorRetriever
from app.vector_store.base import VectorStore
from app.vector_store.factory import create_vector_store

DEMO_COLLECTION_NAME = "medical_demo_v1"
# Deterministic lexical sanity gate for the fixed presentation cases. This is
# deliberately conservative and is not a factual, semantic, or clinical score.
DEMO_REFERENCE_F1_MIN = 0.30


class DemoError(RuntimeError):
    """A demo stage failed safely with an actionable message."""


@dataclass(frozen=True)
class SeedSummary:
    collection_name: str
    record_count: int
    chunk_count: int


@dataclass(frozen=True)
class DemoRunResult:
    case: DemoCase
    answer: RAGAnswer | None
    error: str | None = None


def demo_settings(base: Settings) -> Settings:
    """Clone normal settings while forcing the dedicated demo collection."""
    return base.model_copy(
        update={
            "chroma_collection_name": DEMO_COLLECTION_NAME,
            "chunk_size": DEMO_CHUNK_SIZE,
            "chunk_overlap": DEMO_CHUNK_OVERLAP,
            "min_chunk_size": DEMO_MIN_CHUNK_SIZE,
        }
    )


def assert_demo_store(store: VectorStore) -> None:
    if store.collection_name != DEMO_COLLECTION_NAME:
        raise DemoError(
            "refusing destructive operation: target is not the exact "
            f"demo collection {DEMO_COLLECTION_NAME!r}"
        )


def reset_demo_collection(store: VectorStore) -> None:
    assert_demo_store(store)
    store.delete_collection()


def validate_demo_index(
    store: VectorStore,
    cases: list[DemoCase],
    *,
    allow_empty: bool = False,
) -> int:
    """Prove the collection contains exactly the fixed demo evidence IDs."""
    assert_demo_store(store)
    actual_count = store.count()
    if actual_count == 0 and allow_empty:
        return 0

    expected_document_ids = {
        document_id
        for case in cases
        for document_id in case.expected_document_ids
    }
    expected_chunk_ids = {
        chunk_id for case in cases for chunk_id in case.expected_chunk_ids
    }
    if actual_count != len(expected_chunk_ids):
        raise DemoError(
            "demo collection does not match the fixed dataset: "
            f"expected {len(expected_chunk_ids)} chunks, found {actual_count}; "
            "run `python scripts/demo.py seed --reset`"
        )
    actual_chunk_ids = set().union(
        *(store.get_document_chunk_ids(item) for item in expected_document_ids)
    )
    if actual_chunk_ids != expected_chunk_ids:
        raise DemoError(
            "demo collection contains missing or stale evidence IDs; "
            "run `python scripts/demo.py seed --reset`"
        )
    return actual_count


def seed_demo_collection(
    settings: Settings,
    embedding_backend: EmbeddingBackend,
    store: VectorStore,
) -> SeedSummary:
    assert_demo_store(store)
    records, cases = validate_demo_data()
    pipeline = IndexingPipeline(
        embedding_backend=embedding_backend,
        vector_store=store,
        batch_size=settings.embedding_batch_size,
    )
    chunk_count = 0
    for record_path in records:
        ingestion = ingest_document(record_path, settings, write_output=False)
        result = pipeline.index_payload(ingestion.model_dump(mode="json"))
        chunk_count += result.indexed_chunks
    actual = validate_demo_index(store, cases)
    if actual != chunk_count:
        raise DemoError("demo seed verification produced an unexpected chunk count")
    return SeedSummary(DEMO_COLLECTION_NAME, len(records), actual)


def build_retriever(settings: Settings, backend: EmbeddingBackend, store: VectorStore) -> VectorRetriever:
    retriever = VectorRetriever(
        embedding_backend=backend,
        vector_store=store,
        default_top_k=settings.retrieval_top_k,
        max_top_k=settings.retrieval_max_top_k,
    )
    retriever.ensure_ready()
    return retriever


def create_demo_rag_service(settings: Settings) -> RAGService:
    store = create_vector_store(settings)
    _, cases = validate_demo_data()
    validate_demo_index(store, cases)
    backend = get_embedding_backend_for(settings)
    return RAGService(
        build_retriever(settings, backend, store),
        get_llm_provider(settings),
    )


def select_cases(cases: list[DemoCase], case_id: str | None) -> list[DemoCase]:
    if case_id is None:
        return cases
    selected = [case for case in cases if case.id == case_id]
    if not selected:
        raise DemoError(f"unknown demo case: {case_id}")
    return selected


def run_demo_cases(
    rag_service: RAGService,
    cases: list[DemoCase],
) -> list[DemoRunResult]:
    results: list[DemoRunResult] = []
    for case in cases:
        try:
            answer = rag_service.answer(case.query, case.top_k)
            returned_chunk_ids = {source.chunk_id for source in answer.sources}
            if returned_chunk_ids.isdisjoint(case.expected_chunk_ids):
                results.append(
                    DemoRunResult(
                        case,
                        answer,
                        "expected demo evidence was not retrieved; verify the demo seed "
                        "and embedding configuration",
                    )
                )
                continue
            results.append(DemoRunResult(case, answer))
        except Exception as exc:
            results.append(DemoRunResult(case, None, f"{type(exc).__name__}: {exc}"))
    return results


class _RecordingRAG:
    """Delegates to the real service while retaining its output for export."""

    def __init__(self, delegate: RAGService) -> None:
        self.delegate = delegate
        self.last_answer: RAGAnswer | None = None

    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
        self.last_answer = self.delegate.answer(query, top_k)
        return self.last_answer


def evaluate_demo_cases(
    rag_service: RAGService,
    cases: list[DemoCase],
    *,
    model_name: str,
    now: Callable[[], datetime] | None = None,
) -> dict:
    """Run the Phase 6 evaluator and enrich its metrics with evidence and status."""
    successful_results = []
    case_rows: list[dict] = []
    for case in cases:
        recorder = _RecordingRAG(rag_service)
        try:
            evaluated = Evaluator(rag_service=recorder).evaluate_dataset([case], "rag").cases[0]
            successful_results.append(evaluated)
            answer = recorder.last_answer
            assert answer is not None
            answer_metrics = evaluated.answer
            retrieval_pass = evaluated.retrieval.hit_at_k == 1.0
            citation_presence_pass = bool(
                answer_metrics is not None
                and answer_metrics.citation_presence is True
            )
            citation_validity_pass = bool(
                answer_metrics is not None
                and answer_metrics.citation_validity == 1.0
            )
            citation_relevance_pass = bool(
                answer_metrics is not None
                and answer_metrics.citation_relevance == 1.0
            )
            reference_f1_applicable = case.reference_answer is not None
            reference_f1 = (
                answer_metrics.reference_token_f1
                if answer_metrics is not None
                else None
            )
            reference_f1_pass: bool | None = (
                reference_f1 is not None
                and reference_f1 >= DEMO_REFERENCE_F1_MIN
                if reference_f1_applicable
                else None
            )
            passed = bool(
                retrieval_pass
                and citation_presence_pass
                and citation_validity_pass
                and citation_relevance_pass
                and (reference_f1_pass if reference_f1_applicable else True)
            )
            status = "pass" if passed else "fail"
            case_rows.append(
                {
                    "case_id": case.id,
                    "question": case.query,
                    "expected_document_ids": case.expected_document_ids,
                    "expected_chunk_ids": case.expected_chunk_ids,
                    "expected_facts": case.expected_facts,
                    "reference_answer": case.reference_answer,
                    "answer": answer.answer,
                    "sources": [source.model_dump(mode="json") for source in answer.sources],
                    "metrics": evaluated.model_dump(mode="json"),
                    "quality_gate": {
                        "retrieval_hit_pass": retrieval_pass,
                        "citation_presence_pass": citation_presence_pass,
                        "citation_validity_pass": citation_validity_pass,
                        "citation_relevance_pass": citation_relevance_pass,
                        "reference_token_f1_applicable": reference_f1_applicable,
                        "reference_token_f1": reference_f1,
                        "reference_token_f1_pass": reference_f1_pass,
                    },
                    "status": status,
                    "error": None,
                }
            )
        except Exception as exc:
            case_rows.append(
                {
                    "case_id": case.id,
                    "question": case.query,
                    "expected_document_ids": case.expected_document_ids,
                    "expected_chunk_ids": case.expected_chunk_ids,
                    "expected_facts": case.expected_facts,
                    "reference_answer": case.reference_answer,
                    "answer": None,
                    "sources": [],
                    "metrics": None,
                    "quality_gate": {
                        "evaluated": False,
                        "reference_token_f1_applicable": (
                            case.reference_answer is not None
                        ),
                    },
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report = EvaluationReport(
        mode="rag",
        aggregate=aggregate_results(successful_results),
        cases=successful_results,
    )
    timestamp = (now or (lambda: datetime.now(timezone.utc)))().astimezone(timezone.utc)
    return {
        "schema_version": 1,
        "generated_at": timestamp.isoformat(),
        "environment": {
            "model": model_name,
            "collection": DEMO_COLLECTION_NAME,
            "data_classification": "synthetic/de-identified; not for clinical use",
        },
        "totals": {
            "case_count": len(cases),
            "pass_count": sum(row["status"] == "pass" for row in case_rows),
            "fail_count": sum(row["status"] == "fail" for row in case_rows),
            "error_count": sum(row["status"] == "error" for row in case_rows),
        },
        "aggregate_metrics": report.aggregate.model_dump(mode="json"),
        "quality_gate": {
            "reference_token_f1_min": DEMO_REFERENCE_F1_MIN,
            "reference_token_f1_applies_when": "reference_answer is present",
            "purpose": "deterministic lexical demo sanity check only",
            "not_proof_of": [
                "semantic equivalence",
                "medical correctness",
                "clinical safety",
            ],
        },
        "cases": case_rows,
        "limitations": [
            "Generated wording can vary even with temperature zero.",
            "This synthetic demo is not for clinical decision-making.",
            "ROCm/GPU behavior must be verified on the target AMD host.",
        ],
    }
