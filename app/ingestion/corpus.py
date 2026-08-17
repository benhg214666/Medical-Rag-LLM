"""Corpus discovery, path metadata, and shared batch-ingestion orchestration."""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.core.config import Settings, ensure_data_directories
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import compute_document_id, ingest_document, sanitize_filename
from app.indexing.pipeline import IndexingPipeline

_PATIENT_FILENAME = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<type>[^.]+)$", re.IGNORECASE
)
_HIDDEN_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})


class CorpusBatchResult(BaseModel):
    discovered: int = 0
    processed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[str] = Field(default_factory=list)


def is_hidden_or_system(path: Path, corpus_dir: Path) -> bool:
    """Return whether any corpus-relative component is hidden/system-managed."""
    try:
        parts = path.relative_to(corpus_dir).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") or part.casefold() in _HIDDEN_NAMES for part in parts)


def scan_corpus(corpus_dir: Path) -> list[Path]:
    """Recursively and deterministically find supported corpus documents."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in corpus_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not is_hidden_or_system(path, corpus_dir)
        ),
        key=lambda path: path.relative_to(corpus_dir).as_posix().casefold(),
    )


def metadata_from_corpus_path(file_path: Path, corpus_dir: Path) -> dict[str, Any]:
    """Extract robust corpus metadata without requiring a filename convention."""
    try:
        relative = Path(file_path).relative_to(corpus_dir)
    except ValueError:
        return {}
    parts = relative.parts
    if len(parts) >= 3 and parts[0].casefold() == "patient_records":
        metadata: dict[str, Any] = {
            "source_type": "patient_record",
            "patient_id": parts[1],
        }
        match = _PATIENT_FILENAME.fullmatch(file_path.stem)
        if match:
            metadata["encounter_date"] = match.group("date")
            metadata["document_type"] = match.group("type").upper()
        return metadata
    if len(parts) >= 3 and parts[0].casefold() == "medical_knowledge":
        return {"source_type": "medical_knowledge", "category": parts[1]}
    return {}


def _raw_destination(file_path: Path, settings: Settings) -> Path:
    document_id = compute_document_id(file_path)
    safe = Path(sanitize_filename(file_path.name))
    return settings.raw_data_dir / f"{safe.stem}_{document_id[:8]}{safe.suffix.lower()}"


def ingest_corpus(
    settings: Settings,
    indexing_pipeline: IndexingPipeline,
    *,
    progress: Callable[[int, int, Path], None] | None = None,
) -> CorpusBatchResult:
    """Copy corpus sources to raw storage, then use normal ingestion and indexing."""
    files = scan_corpus(settings.corpus_data_dir)
    summary = CorpusBatchResult(discovered=len(files))
    ensure_data_directories(settings)
    for position, file_path in enumerate(files, 1):
        if progress:
            progress(position, len(files), file_path)
        try:
            raw_path = _raw_destination(file_path, settings)
            if not raw_path.exists() or compute_document_id(raw_path) != compute_document_id(file_path):
                shutil.copy2(file_path, raw_path)
            relative_source = file_path.relative_to(settings.corpus_data_dir).as_posix()
            result = ingest_document(
                raw_path,
                settings,
                metadata=metadata_from_corpus_path(file_path, settings.corpus_data_dir),
                source=relative_source,
            )
            indexed = indexing_pipeline.index_payload(result.model_dump(mode="json"))
            summary.processed += 1
            summary.chunks_created += result.statistics.chunk_count
            summary.chunks_indexed += indexed.indexed_chunks
        except Exception as exc:  # one bad document must not abort the corpus
            summary.failed += 1
            summary.failures.append(f"{file_path.name}: {type(exc).__name__}: {exc}")
    return summary


def corpus_breakdown(files: list[Path], corpus_dir: Path) -> tuple[Counter[str], Counter[str]]:
    patients: Counter[str] = Counter()
    knowledge: Counter[str] = Counter()
    for path in files:
        metadata = metadata_from_corpus_path(path, corpus_dir)
        if metadata.get("source_type") == "patient_record":
            patients[str(metadata["patient_id"])] += 1
        elif metadata.get("source_type") == "medical_knowledge":
            knowledge[str(metadata["category"])] += 1
    return patients, knowledge
