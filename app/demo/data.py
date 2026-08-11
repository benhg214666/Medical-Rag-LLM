"""Deterministic Phase 8 data loading and cross-reference validation."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from app.core.config import Settings
from app.evaluation.models import EvaluationCase
from app.ingestion.pipeline import compute_document_id, ingest_document

DEMO_ROOT = Path("data/demo")
DEMO_RECORDS_DIR = DEMO_ROOT / "records"
DEMO_CASES_PATH = DEMO_ROOT / "cases.jsonl"
SYNTHETIC_MARKER = "SYNTHETIC DE-IDENTIFIED DEMO RECORD"
CLINICAL_USE_MARKER = "NOT FOR CLINICAL USE"
DEMO_CHUNK_SIZE = 500
DEMO_CHUNK_OVERLAP = 100
DEMO_MIN_CHUNK_SIZE = 50


class DemoDataError(ValueError):
    """The versioned demo data is absent, unsafe, or internally inconsistent."""


class DemoCase(EvaluationCase):
    expected_document_ids: list[str] = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(min_length=1)
    expected_facts: list[str] = Field(min_length=1)
    synthetic: Literal[True]
    not_for_clinical_use: Literal[True]

    @field_validator("expected_facts")
    @classmethod
    def normalize_expected_facts(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("expected facts must not be blank")
        return list(dict.fromkeys(normalized))


def demo_record_paths(records_dir: Path = DEMO_RECORDS_DIR) -> list[Path]:
    if not records_dir.is_dir():
        raise DemoDataError(f"demo records directory does not exist: {records_dir}")
    paths = sorted(records_dir.glob("DEMO-P*.txt"), key=lambda item: item.name)
    if not paths:
        raise DemoDataError(f"demo records directory is empty: {records_dir}")
    return paths


def load_demo_cases(path: Path = DEMO_CASES_PATH) -> list[DemoCase]:
    if not path.is_file():
        raise DemoDataError(f"demo cases file does not exist: {path}")
    cases: list[DemoCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                case = DemoCase.model_validate(json.loads(raw_line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise DemoDataError(
                    f"invalid demo case at line {line_number}: {exc}"
                ) from exc
            if case.id in seen:
                raise DemoDataError(f"duplicate demo case id: {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise DemoDataError("demo cases file contains no cases")
    return cases


def validate_demo_data(
    records_dir: Path = DEMO_RECORDS_DIR,
    cases_path: Path = DEMO_CASES_PATH,
) -> tuple[list[Path], list[DemoCase]]:
    records = demo_record_paths(records_dir)
    document_ids: set[str] = set()
    chunk_ids: set[str] = set()
    record_text_by_document: dict[str, str] = {}
    chunk_ids_by_document: dict[str, set[str]] = {}
    chunk_settings = Settings(
        chunk_size=DEMO_CHUNK_SIZE,
        chunk_overlap=DEMO_CHUNK_OVERLAP,
        min_chunk_size=DEMO_MIN_CHUNK_SIZE,
    )
    for record in records:
        text = record.read_text(encoding="utf-8")
        if SYNTHETIC_MARKER not in text or CLINICAL_USE_MARKER not in text:
            raise DemoDataError(
                f"demo record lacks required safety markers: {record.name}"
            )
        document_id = compute_document_id(record)
        if document_id in document_ids:
            raise DemoDataError(f"duplicate demo document content: {record.name}")
        document_ids.add(document_id)
        record_text_by_document[document_id] = text
        ingestion = ingest_document(record, chunk_settings, write_output=False)
        chunk_ids_by_document[document_id] = set()
        for chunk in ingestion.chunks:
            if chunk.chunk_id in chunk_ids:
                raise DemoDataError(f"duplicate demo chunk ID: {chunk.chunk_id}")
            chunk_ids.add(chunk.chunk_id)
            chunk_ids_by_document[document_id].add(chunk.chunk_id)

    cases = load_demo_cases(cases_path)
    for case in cases:
        missing = set(case.expected_document_ids) - document_ids
        if missing:
            raise DemoDataError(
                f"case {case.id} references missing document IDs: {sorted(missing)}"
            )
        missing_chunks = set(case.expected_chunk_ids) - chunk_ids
        if missing_chunks:
            raise DemoDataError(
                f"case {case.id} references missing chunk IDs: {sorted(missing_chunks)}"
            )
        expected_document_chunks = set().union(
            *(chunk_ids_by_document[item] for item in case.expected_document_ids)
        )
        mismatched_chunks = set(case.expected_chunk_ids) - expected_document_chunks
        if mismatched_chunks:
            raise DemoDataError(
                f"case {case.id} chunk IDs do not belong to its expected documents: "
                f"{sorted(mismatched_chunks)}"
            )
        expected_text = "\n".join(
            record_text_by_document[item] for item in case.expected_document_ids
        ).casefold()
        missing_facts = [
            fact for fact in case.expected_facts if fact.casefold() not in expected_text
        ]
        if missing_facts:
            raise DemoDataError(
                f"case {case.id} expected facts are absent from its records: "
                f"{missing_facts}"
            )
    return records, cases
