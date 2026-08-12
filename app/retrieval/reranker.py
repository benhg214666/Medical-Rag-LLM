"""Deterministic lexical reranking for vector-retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import unicodedata

from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_DATE_RE = re.compile(
    r"(?<!\d)(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})(?!\d)"
)
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "be", "did", "do", "does", "for",
        "from", "had", "has", "have", "in", "is", "it", "of", "on", "or",
        "patient", "patients", "report", "reported", "the", "to", "was", "were",
        "what", "when", "which", "with",
    }
)


def normalize_text(text: str) -> str:
    """Apply the same Unicode-safe, case-insensitive normalization everywhere."""
    return unicodedata.normalize("NFKC", text).casefold()


def extract_dates(text: str) -> frozenset[str]:
    """Extract ISO-like dates and canonicalize separator and zero padding."""
    normalized = normalize_text(text)
    return frozenset(
        f"{match.group('year')}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"
        for match in _DATE_RE.finditer(normalized)
    )


def extract_meaningful_terms(text: str) -> frozenset[str]:
    """Return generic alphanumeric terms without requiring an NLP dependency."""
    without_dates = _DATE_RE.sub(" ", normalize_text(text))
    return frozenset(
        token
        for token in _TOKEN_RE.findall(without_dates)
        if token not in _STOP_WORDS and (len(token) > 1 or any(char.isdigit() for char in token))
    )


@dataclass(frozen=True)
class RerankingWeights:
    lexical: float = 0.15
    exact_date: float = 0.35
    exact_term: float = 0.10

    def __post_init__(self) -> None:
        if min(self.lexical, self.exact_date, self.exact_term) < 0:
            raise ValueError("reranking weights must be non-negative")


@dataclass(frozen=True)
class _ScoredCandidate:
    result: RetrievalResult
    original_rank: int
    semantic_score: float
    lexical_bonus: float
    date_bonus: float
    term_bonus: float

    @property
    def final_score(self) -> float:
        return self.semantic_score + self.lexical_bonus + self.date_bonus + self.term_bonus


class LightweightReranker:
    """Combine semantic relevance with bounded, query-derived exact-match signals."""

    def __init__(self, weights: RerankingWeights | None = None) -> None:
        self.weights = weights or RerankingWeights()

    @staticmethod
    def _semantic_score(result: RetrievalResult) -> float:
        # A native similarity score is preferred. Negative distance preserves the
        # vector store's ordering for metrics that cannot be converted safely.
        return result.score if result.score is not None else -result.distance

    def _score(
        self,
        result: RetrievalResult,
        original_rank: int,
        query_terms: frozenset[str],
        query_dates: frozenset[str],
    ) -> _ScoredCandidate:
        chunk_terms = extract_meaningful_terms(result.text)
        overlap = query_terms.intersection(chunk_terms)

        # Dice rewards shared vocabulary while penalizing broadly themed chunks.
        lexical_ratio = (
            (2.0 * len(overlap)) / (len(query_terms) + len(chunk_terms))
            if query_terms and chunk_terms
            else 0.0
        )
        # Coverage specifically rewards chunks that contain the user's meaningful
        # terms (e.g. creatinine or LDL), without any medical-term dictionary.
        term_coverage = len(overlap) / len(query_terms) if query_terms else 0.0
        date_match = bool(query_dates.intersection(extract_dates(result.text)))

        return _ScoredCandidate(
            result=result,
            original_rank=original_rank,
            semantic_score=self._semantic_score(result),
            lexical_bonus=self.weights.lexical * lexical_ratio,
            date_bonus=self.weights.exact_date if query_dates and date_match else 0.0,
            term_bonus=self.weights.exact_term * term_coverage,
        )

    def rerank(
        self, query: str, candidates: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        query_terms = extract_meaningful_terms(query)
        query_dates = extract_dates(query)
        scored = [
            self._score(candidate, rank, query_terms, query_dates)
            for rank, candidate in enumerate(candidates)
        ]
        # Python's stable sort plus original_rank makes tie behavior explicit and
        # deterministic, while preserving vector order when every bonus is zero.
        scored.sort(key=lambda item: (-item.final_score, item.original_rank))

        logger.debug(
            "Reranking scores: %s",
            [
                {
                    "chunk_id": item.result.chunk_id,
                    "semantic": round(item.semantic_score, 6),
                    "reranking": round(item.final_score, 6),
                }
                for item in scored
            ],
        )
        return [item.result for item in scored]
