"""Focused regression tests for candidate expansion and lightweight reranking."""

from __future__ import annotations

import pytest

from app.retrieval.models import RetrievalResult
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import LightweightReranker, extract_dates
from app.retrieval.vector_retriever import VectorRetriever


def result(chunk_id: str, text: str, semantic_score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        distance=1.0 - semantic_score,
        score=semantic_score,
        distance_metric="cosine",
    )


class StubVectorRetriever:
    default_top_k = 5
    max_top_k = 50

    def __init__(self, candidates: list[RetrievalResult]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def resolve_top_k(self, top_k: int | None) -> int:
        resolved = self.default_top_k if top_k is None else top_k
        if resolved <= 0 or resolved > self.max_top_k:
            raise ValueError("invalid test top_k")
        return resolved

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        assert top_k is not None
        self.calls.append((query, top_k))
        return self.candidates[:top_k]


def test_exact_date_and_measurement_term_promote_lower_semantic_candidate() -> None:
    candidates = [
        result("wrong-date", "2026-09-12 follow-up: HbA1c 7.4%", 0.90),
        result("correct", "2026-06-10 laboratory results: HbA1c 9.2%", 0.64),
    ]
    ranked = LightweightReranker().rerank(
        "What was the patient's HbA1c on 2026-06-10?", candidates
    )
    assert ranked[0].chunk_id == "correct"


def test_wrong_date_competing_value_ranks_below_requested_date() -> None:
    candidates = [
        result("september", "2026-09-12 HbA1c: 7.4%", 0.82),
        result("june", "2026-06-10 HbA1c: 9.2%", 0.70),
    ]
    ranked = LightweightReranker().rerank(
        "HbA1c on 2026-06-10", candidates
    )
    assert [item.chunk_id for item in ranked] == ["june", "september"]


def test_query_without_date_applies_no_arbitrary_date_preference() -> None:
    candidates = [
        result("september", "2026-09-12 HbA1c: 7.4%", 0.80),
        result("june", "2026-06-10 HbA1c: 9.2%", 0.80),
    ]
    ranked = LightweightReranker().rerank("What was the HbA1c?", candidates)
    assert [item.chunk_id for item in ranked] == ["september", "june"]


def test_generic_term_matching_is_not_field_specific() -> None:
    candidates = [
        result("general", "Renal follow-up laboratory panel was reviewed.", 0.80),
        result("specific", "Creatinine was 1.8 mg/dL.", 0.72),
    ]
    ranked = LightweightReranker().rerank("What was the creatinine?", candidates)
    assert ranked[0].chunk_id == "specific"


def test_reranking_is_deterministic_and_stable_for_ties() -> None:
    candidates = [
        result("first", "unrelated alpha", 0.50),
        result("second", "unrelated beta", 0.50),
    ]
    reranker = LightweightReranker()
    expected = ["first", "second"]
    for _ in range(5):
        assert [item.chunk_id for item in reranker.rerank("query", candidates)] == expected


def test_candidate_expansion_returns_only_final_top_k() -> None:
    candidates = [result(f"chunk-{index}", f"text {index}", 1 - index / 100) for index in range(12)]
    vector = StubVectorRetriever(candidates)
    pipeline = RetrievalPipeline(
        vector,  # type: ignore[arg-type]
        LightweightReranker(),
        candidate_multiplier=2,
        min_candidate_k=10,
    )
    selected = pipeline.retrieve("query", top_k=5)
    assert vector.calls == [("query", 10)]
    assert len(selected) == 5


def test_rank_nine_regression_is_promoted_into_final_five() -> None:
    candidates = [
        result(f"distractor-{index}", f"General diabetes follow-up note {index}", 0.92 - index * 0.02)
        for index in range(8)
    ]
    candidates.append(
        result(
            "correct-evidence",
            "=== 2026-06-10 Outpatient Visit === Laboratory Results: HbA1c: 9.2%",
            0.55,
        )
    )
    vector = StubVectorRetriever(candidates)
    pipeline = RetrievalPipeline(vector, LightweightReranker())  # type: ignore[arg-type]
    selected = pipeline.retrieve(
        "What was the patient's HbA1c on 2026-06-10?", top_k=5
    )
    assert vector.calls[0][1] == 10
    assert "correct-evidence" in [item.chunk_id for item in selected]


def test_date_normalization_accepts_common_numeric_separators() -> None:
    assert extract_dates("2026/6/10 and 2026.06.10") == frozenset({"2026-06-10"})


def test_pipeline_rejects_invalid_candidate_configuration() -> None:
    vector = StubVectorRetriever([])
    with pytest.raises(ValueError, match="candidate_multiplier"):
        RetrievalPipeline(
            vector, LightweightReranker(), candidate_multiplier=0  # type: ignore[arg-type]
        )


def test_public_vector_top_k_resolver_preserves_validation() -> None:
    assert hasattr(VectorRetriever, "resolve_top_k")
