"""Load and validate line-delimited evaluation cases."""

import json
from pathlib import Path

from pydantic import ValidationError

from app.evaluation.models import EvaluationCase


class EvaluationDatasetError(ValueError):
    """The evaluation dataset is missing, malformed, or internally inconsistent."""


def load_jsonl_dataset(path: Path) -> list[EvaluationCase]:
    if not path.is_file():
        raise EvaluationDatasetError(f"evaluation dataset does not exist: {path}")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                case = EvaluationCase.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise EvaluationDatasetError(
                    f"invalid evaluation case at line {line_number}: {exc}"
                ) from exc
            if case.id in seen_ids:
                raise EvaluationDatasetError(f"duplicate evaluation case id: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)

    if not cases:
        raise EvaluationDatasetError("evaluation dataset contains no cases")
    return cases
