"""Deterministic relevance metrics for ranked retrieval results."""

from typing import Protocol

from app.evaluation.models import EvaluationCase, RetrievalMetrics


class RetrievedSource(Protocol):
    chunk_id: str
    document_id: str | None


def relevant_ranks(
    case: EvaluationCase, results: list[RetrievedSource]
) -> list[int] | None:
    """Return 1-based relevant ranks; chunk labels take precedence over documents."""
    if case.expected_chunk_ids:
        expected = set(case.expected_chunk_ids)
        return [
            rank for rank, item in enumerate(results[: case.top_k], start=1)
            if item.chunk_id in expected
        ]
    if case.expected_document_ids:
        expected = set(case.expected_document_ids)
        return [
            rank for rank, item in enumerate(results[: case.top_k], start=1)
            if item.document_id in expected
        ]
    return None


def retrieval_metrics(
    case: EvaluationCase, results: list[RetrievedSource]
) -> RetrievalMetrics:
    ranks = relevant_ranks(case, results)
    if ranks is None:
        return RetrievalMetrics()

    expected_count = (
        len(case.expected_chunk_ids)
        if case.expected_chunk_ids
        else len(case.expected_document_ids)
    )
    # Multiple chunks from one expected document count once for document recall.
    if case.expected_chunk_ids:
        retrieved_relevant = {
            item.chunk_id for item in results[: case.top_k]
            if item.chunk_id in set(case.expected_chunk_ids)
        }
    else:
        retrieved_relevant = {
            item.document_id for item in results[: case.top_k]
            if item.document_id in set(case.expected_document_ids)
        }
    return RetrievalMetrics(
        hit_at_k=1.0 if ranks else 0.0,
        recall_at_k=len(retrieved_relevant) / expected_count,
        reciprocal_rank=1.0 / ranks[0] if ranks else 0.0,
    )
