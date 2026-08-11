"""Transparent lexical and citation checks for generated answers."""

import re
from collections import Counter

from app.evaluation.models import EvaluationCase
from app.evaluation.retrieval_metrics import RetrievedSource
from app.rag.constants import INSUFFICIENT_CONTEXT_ANSWER

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)


def extract_citations(answer: str) -> list[int]:
    """Extract unique positive source numbers in first-appearance order."""
    return list(dict.fromkeys(int(value) for value in _CITATION_PATTERN.findall(answer)))


def citation_validity(citations: list[int], sources: list[RetrievedSource]) -> float | None:
    if not citations:
        return None
    available = {
        int(getattr(source, "source_number", index))
        for index, source in enumerate(sources, start=1)
    }
    return sum(number in available for number in citations) / len(citations)


def citation_relevance(
    case: EvaluationCase,
    citations: list[int],
    sources: list[RetrievedSource],
) -> float | None:
    if not citations or not (case.expected_chunk_ids or case.expected_document_ids):
        return None
    source_by_number = {
        int(getattr(source, "source_number", index)): source
        for index, source in enumerate(sources, start=1)
    }
    relevant = 0
    for number in citations:
        source = source_by_number.get(number)
        if source is None:
            continue
        if case.expected_chunk_ids:
            relevant += source.chunk_id in set(case.expected_chunk_ids)
        else:
            relevant += source.document_id in set(case.expected_document_ids)
    return relevant / len(citations)


def is_abstention(answer: str, *, has_sources: bool) -> bool:
    """Check the explicit insufficient-context contract or zero-source path."""
    canonical = INSUFFICIENT_CONTEXT_ANSWER.strip().casefold()
    return not has_sources or answer.strip().casefold() == canonical


def _is_cjk_ideograph(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def normalize_tokens(text: str) -> list[str]:
    """Tokenize non-CJK alphanumerics as runs and CJK ideographs as characters."""
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for character in text.casefold():
        if _is_cjk_ideograph(character):
            flush()
            tokens.append(character)
        elif character.isalnum():
            current.append(character)
        else:
            flush()
    flush()
    return tokens


def reference_token_f1(generated: str, reference: str) -> float:
    """Lexical token F1 baseline; this is not a medical-correctness score."""
    generated_tokens = normalize_tokens(generated)
    reference_tokens = normalize_tokens(reference)
    if not generated_tokens and not reference_tokens:
        return 1.0
    if not generated_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(generated_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(generated_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
