"""Linux-friendly command line entry point for Phase 6 evaluation."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fastapi import HTTPException

from app.core.config import get_settings
from app.evaluation.dataset import EvaluationDatasetError, load_jsonl_dataset
from app.evaluation.evaluator import Evaluator
from app.evaluation.report import format_summary, write_json_report
from app.llm.dependencies import get_llm_provider
from app.rag.service import RAGService
from app.retrieval.dependencies import get_vector_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Medical-Rag-LLM quality")
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL gold dataset")
    parser.add_argument("--mode", required=True, choices=("retrieval", "rag"))
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cases = load_jsonl_dataset(args.dataset)
        settings = get_settings()
        retriever = get_vector_retriever(settings)
        if args.mode == "retrieval":
            evaluator = Evaluator(retriever=retriever)
        else:
            rag_service = RAGService(retriever, get_llm_provider(settings))
            evaluator = Evaluator(rag_service=rag_service)
        report = evaluator.evaluate_dataset(cases, args.mode)
        if args.output is not None:
            write_json_report(report, args.output)
        print(format_summary(report))
        return 0
    except (
        EvaluationDatasetError,
        HTTPException,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
