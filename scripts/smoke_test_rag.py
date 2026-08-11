"""End-to-end smoke test using production retrieval and RAG dependencies."""

import argparse
import logging
import sys
from pathlib import Path
from typing import TextIO

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.llm.base import LLMError
from app.llm.factory import create_llm_provider
from app.rag.service import RAGService
from app.retrieval.dependencies import get_vector_retriever
from app.retrieval.exceptions import RetrievalError
from app.vector_store.base import VectorStoreError
from app.vector_store.factory import create_vector_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the production RAG path")
    parser.add_argument("--query", required=True, help="question about indexed synthetic data")
    parser.add_argument("--top-k", type=int, default=None)
    return parser


def run_rag_smoke(
    *,
    query: str,
    top_k: int | None,
    service: RAGService,
    indexed_count: int,
    output: TextIO,
    error: TextIO,
) -> int:
    if indexed_count <= 0:
        print(
            "RAG smoke test failed: vector store has no indexed chunks; "
            "ingest and index a synthetic or de-identified document first.",
            file=error,
        )
        return 1
    try:
        result = service.answer(query, top_k)
    except (LLMError, RetrievalError, ValueError) as exc:
        print(f"RAG smoke test failed: {exc}", file=error)
        return 1
    if not result.sources:
        print(
            "RAG smoke test failed: retrieval returned no usable sources for the query.",
            file=error,
        )
        return 1
    print(f"Answer: {result.answer}", file=output)
    print(f"Model: {result.model}", file=output)
    for source in result.sources:
        print(
            f"Source {source.source_number}: chunk_id={source.chunk_id} "
            f"document_id={source.document_id or 'unknown'} "
            f"distance={source.distance:.6f}",
            file=output,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.query.strip():
        print("RAG smoke test failed: --query must not be blank", file=sys.stderr)
        return 2
    try:
        settings = Settings()
        indexed_count = create_vector_store(settings).count()
        if indexed_count <= 0:
            print(
                "RAG smoke test failed: vector store has no indexed chunks; "
                "ingest and index a synthetic or de-identified document first.",
                file=sys.stderr,
            )
            return 1
        # This CLI owns its error presentation. Prevent production logger.exception
        # calls from printing stack traces for expected runtime setup failures.
        previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            retriever = get_vector_retriever(settings)
            service = RAGService(retriever, create_llm_provider(settings))
            return run_rag_smoke(
                query=args.query,
                top_k=args.top_k,
                service=service,
                indexed_count=indexed_count,
                output=sys.stdout,
                error=sys.stderr,
            )
        finally:
            logging.disable(previous_logging_disable)
    except (
        HTTPException,
        LLMError,
        RetrievalError,
        VectorStoreError,
        ValueError,
    ) as exc:
        print(f"RAG smoke test setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
