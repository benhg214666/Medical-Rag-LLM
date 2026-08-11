"""Internal RAG result models."""

from typing import Any

from pydantic import BaseModel, Field


class RAGSource(BaseModel):
    source_number: int
    chunk_id: str
    document_id: str | None = None
    text: str
    distance: float
    score: float | None = None
    distance_metric: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGAnswer(BaseModel):
    answer: str
    model: str
    sources: list[RAGSource] = Field(default_factory=list)
