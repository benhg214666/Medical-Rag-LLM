"""Two-stage retrieval: vector candidates followed by lightweight reranking."""

from __future__ import annotations

import logging

from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import LightweightReranker
from app.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Preserve caller-facing top_k while expanding the internal candidate pool."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        reranker: LightweightReranker,
        *,
        reranking_enabled: bool = True,
        candidate_multiplier: int = 2,
        min_candidate_k: int = 10,
    ) -> None:
        if (
            isinstance(candidate_multiplier, bool)
            or not isinstance(candidate_multiplier, int)
            or candidate_multiplier < 1
        ):
            raise ValueError("candidate_multiplier must be an integer of at least 1")
        if (
            isinstance(min_candidate_k, bool)
            or not isinstance(min_candidate_k, int)
            or min_candidate_k < 1
        ):
            raise ValueError("min_candidate_k must be a positive integer")
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self.reranking_enabled = reranking_enabled
        self.candidate_multiplier = candidate_multiplier
        self.min_candidate_k = min_candidate_k
        self.default_top_k = vector_retriever.default_top_k
        self.max_top_k = vector_retriever.max_top_k

    def _candidate_top_k(self, final_top_k: int) -> int:
        expanded = max(final_top_k * self.candidate_multiplier, self.min_candidate_k)
        return min(expanded, self.max_top_k)

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievalResult]:
        final_top_k = self.vector_retriever.resolve_top_k(top_k)
        candidate_top_k = self._candidate_top_k(final_top_k)
        candidates = self.vector_retriever.retrieve(query, candidate_top_k)
        ranked = self.reranker.rerank(query, candidates) if self.reranking_enabled else candidates
        selected = ranked[:final_top_k]

        logger.info(
            "Retrieval pipeline complete: final_top_k=%d candidate_top_k=%d "
            "candidates=%d reranking_enabled=%s selected_chunk_ids=%s",
            final_top_k,
            candidate_top_k,
            len(candidates),
            self.reranking_enabled,
            [item.chunk_id for item in selected],
        )
        return selected
