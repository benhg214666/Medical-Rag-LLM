"""Evaluation runner that observes the existing Phase 4/5 services."""

from typing import Literal, Protocol

from app.evaluation.answer_metrics import (
    citation_relevance,
    citation_validity,
    extract_citations,
    is_abstention,
    reference_token_f1,
)
from app.evaluation.models import (
    AggregateMetrics,
    AnswerMetrics,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
)
from app.evaluation.retrieval_metrics import retrieval_metrics
from app.rag.models import RAGAnswer
from app.retrieval.models import RetrievalResult


class RetrieverLike(Protocol):
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]: ...


class RAGServiceLike(Protocol):
    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer: ...


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_results(results: list[EvaluationCaseResult]) -> AggregateMetrics:
    retrieval = [item.retrieval for item in results if item.retrieval.hit_at_k is not None]
    answers = [item.answer for item in results if item.answer is not None]
    presence = [float(item.citation_presence) for item in answers if item.citation_presence is not None]
    validity = [item.citation_validity for item in answers if item.citation_validity is not None]
    relevance = [item.citation_relevance for item in answers if item.citation_relevance is not None]
    abstention = [float(item.abstained) for item in answers if item.abstained is not None]
    reference_f1 = [item.reference_token_f1 for item in answers if item.reference_token_f1 is not None]
    return AggregateMetrics(
        case_count=len(results),
        answerable_count=sum(item.answerable for item in results),
        unanswerable_count=sum(not item.answerable for item in results),
        retrieval_applicable_count=len(retrieval),
        retrieval_hit_rate_at_k=_mean([item.hit_at_k for item in retrieval if item.hit_at_k is not None]),
        mean_recall_at_k=_mean([item.recall_at_k for item in retrieval if item.recall_at_k is not None]),
        mean_reciprocal_rank=_mean([item.reciprocal_rank for item in retrieval if item.reciprocal_rank is not None]),
        citation_presence_applicable_count=len(presence),
        citation_presence_rate=_mean(presence),
        citation_validity_applicable_count=len(validity),
        citation_validity_rate=_mean(validity),
        citation_relevance_applicable_count=len(relevance),
        citation_relevance_rate=_mean(relevance),
        abstention_applicable_count=len(abstention),
        abstention_accuracy=_mean(abstention),
        reference_f1_applicable_count=len(reference_f1),
        mean_reference_token_f1=_mean(reference_f1),
    )


class Evaluator:
    def __init__(
        self,
        *,
        retriever: RetrieverLike | None = None,
        rag_service: RAGServiceLike | None = None,
    ) -> None:
        self.retriever = retriever
        self.rag_service = rag_service

    def evaluate_dataset(
        self,
        cases: list[EvaluationCase],
        mode: Literal["retrieval", "rag"],
    ) -> EvaluationReport:
        if mode not in ("retrieval", "rag"):
            raise ValueError(f"unsupported evaluation mode: {mode}")
        if not cases:
            raise ValueError("evaluation dataset contains no cases")
        if mode == "retrieval" and self.retriever is None:
            raise ValueError("retrieval mode requires a retriever")
        if mode == "rag" and self.rag_service is None:
            raise ValueError("rag mode requires a RAG service")

        results = [self._evaluate_case(case, mode) for case in cases]
        return EvaluationReport(
            mode=mode,
            aggregate=aggregate_results(results),
            cases=results,
        )

    def _evaluate_case(
        self, case: EvaluationCase, mode: Literal["retrieval", "rag"]
    ) -> EvaluationCaseResult:
        if mode == "retrieval":
            assert self.retriever is not None
            sources = self.retriever.retrieve(case.query, case.top_k)
            return EvaluationCaseResult(
                case_id=case.id,
                query=case.query,
                answerable=case.answerable,
                top_k=case.top_k,
                retrieval=retrieval_metrics(case, sources),
            )

        assert self.rag_service is not None
        rag_answer = self.rag_service.answer(case.query, case.top_k)
        sources = rag_answer.sources
        citations = extract_citations(rag_answer.answer)
        source_numbers = {source.source_number for source in sources}
        invalid = [number for number in citations if number not in source_numbers]
        answer_metrics = AnswerMetrics(
            generated_answer=rag_answer.answer,
            citation_presence=bool(citations) if case.answerable else None,
            citation_validity=citation_validity(citations, sources) if case.answerable else None,
            citation_relevance=citation_relevance(case, citations, sources) if case.answerable else None,
            cited_source_numbers=citations,
            invalid_citation_numbers=invalid,
            reference_token_f1=(
                reference_token_f1(rag_answer.answer, case.reference_answer)
                if case.reference_answer is not None
                else None
            ),
            abstained=(
                is_abstention(rag_answer.answer, has_sources=bool(sources))
                if not case.answerable
                else None
            ),
        )
        return EvaluationCaseResult(
            case_id=case.id,
            query=case.query,
            answerable=case.answerable,
            top_k=case.top_k,
            retrieval=retrieval_metrics(case, sources),
            answer=answer_metrics,
        )
