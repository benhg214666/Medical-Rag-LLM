"""Pydantic contracts for datasets and evaluation reports."""

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictInt, field_validator


class EvaluationCase(BaseModel):
    """One manually curated gold evaluation case."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answerable: StrictBool
    top_k: StrictInt = Field(default=5, ge=1, le=50)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("expected_document_ids", "expected_chunk_ids", "tags")
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("list values must not be blank")
        return list(dict.fromkeys(normalized))

    @field_validator("reference_answer")
    @classmethod
    def normalize_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class RetrievalMetrics(BaseModel):
    hit_at_k: float | None = None
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None


class AnswerMetrics(BaseModel):
    generated_answer: str
    citation_presence: bool | None = None
    citation_validity: float | None = None
    citation_relevance: float | None = None
    cited_source_numbers: list[int] = Field(default_factory=list)
    invalid_citation_numbers: list[int] = Field(default_factory=list)
    reference_token_f1: float | None = None
    abstained: bool | None = None


class EvaluationCaseResult(BaseModel):
    case_id: str
    query: str
    answerable: bool
    top_k: int
    retrieval: RetrievalMetrics
    answer: AnswerMetrics | None = None


class AggregateMetrics(BaseModel):
    case_count: int
    answerable_count: int
    unanswerable_count: int
    retrieval_applicable_count: int
    retrieval_hit_rate_at_k: float | None = None
    mean_recall_at_k: float | None = None
    mean_reciprocal_rank: float | None = None
    citation_presence_applicable_count: int = 0
    citation_presence_rate: float | None = None
    citation_validity_applicable_count: int = 0
    citation_validity_rate: float | None = None
    citation_relevance_applicable_count: int = 0
    citation_relevance_rate: float | None = None
    abstention_applicable_count: int = 0
    abstention_accuracy: float | None = None
    reference_f1_applicable_count: int = 0
    mean_reference_token_f1: float | None = None


class EvaluationReport(BaseModel):
    mode: Literal["retrieval", "rag"]
    aggregate: AggregateMetrics
    cases: list[EvaluationCaseResult]
