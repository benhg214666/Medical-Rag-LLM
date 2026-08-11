"""Small, presentation-friendly Phase 8 command line interface."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from app.core.config import get_settings
from app.demo.data import validate_demo_data
from app.demo.report import write_evaluation_artifacts
from app.demo.service import (
    DEMO_COLLECTION_NAME,
    DemoError,
    build_retriever,
    create_demo_rag_service,
    demo_settings,
    evaluate_demo_cases,
    reset_demo_collection,
    run_demo_cases,
    seed_demo_collection,
    select_cases,
    validate_demo_index,
)
from app.embeddings.dependencies import get_embedding_backend_for
from app.embeddings.base import EmbeddingError
from app.llm.network import validate_local_llm_base_url
from app.retrieval.exceptions import RetrievalBackendError
from app.vector_store.base import VectorStoreError
from app.vector_store.factory import create_vector_store

DEFAULT_OUTPUT_DIR = Path("reports/demo")
_LOCAL_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible Medical-Rag-LLM demo (synthetic data only)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check data, index, and local services")
    preflight.add_argument("--api-url", default="http://127.0.0.1:8000")
    preflight.add_argument("--require-api", action="store_true")
    preflight.add_argument(
        "--allow-unseeded",
        action="store_true",
        help="permit an empty demo collection before the first seed",
    )

    reset = subparsers.add_parser("reset", help="delete only the dedicated demo collection")
    reset.add_argument("--yes", action="store_true", help="confirm the demo-only reset")

    seed = subparsers.add_parser("seed", help="idempotently ingest and index demo records")
    seed.add_argument("--reset", action="store_true", help="safely rebuild the demo collection first")

    run = subparsers.add_parser("run", help="run fixed cases through the real RAG pipeline")
    run.add_argument("--case", dest="case_id", help="run one case ID")
    run.add_argument("--verbose", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="export JSON and Markdown evaluation reports")
    evaluate.add_argument("--case", dest="case_id", help="evaluate one case ID")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    all_command = subparsers.add_parser("all", help="preflight, reset/seed, run, and evaluate")
    all_command.add_argument("--api-url", default="http://127.0.0.1:8000")
    all_command.add_argument("--require-api", action="store_true")
    all_command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    all_command.add_argument("--verbose", action="store_true")
    return parser


def _read_json_endpoint(
    url: str,
    label: str,
    *,
    required: bool = True,
) -> dict | None:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with _LOCAL_URL_OPENER.open(request, timeout=3) as response:
            if response.status >= 400:
                raise DemoError(f"{label} returned HTTP {response.status}: {url}")
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise DemoError(f"{label} returned a non-object JSON response: {url}")
        return payload
    except DemoError:
        if required:
            raise
        print(f"[warn] {label} returned an invalid response at {url}")
        return None
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        urllib.error.URLError,
        urllib.error.HTTPError,
    ) as exc:
        message = f"{label} is unavailable or returned invalid JSON at {url}: {exc}"
        if required:
            raise DemoError(message) from exc
        print(f"[warn] {message}")
        return None


def _check_llm_models(url: str, configured_model: str) -> None:
    if not configured_model.strip():
        raise DemoError("LLM_MODEL_NAME must not be blank")
    payload = _read_json_endpoint(url, "local vLLM/OpenAI-compatible service")
    assert payload is not None
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise DemoError(f"vLLM models response is missing the data list: {url}")
    served_models = {
        item.get("id")
        for item in raw_models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if configured_model not in served_models:
        raise DemoError(
            f"configured model {configured_model!r} is not served by {url}; "
            f"available models: {sorted(served_models)}"
        )
    print(f"[ok] local vLLM model: {configured_model} at {url}")


def _check_fastapi_health(url: str, *, required: bool) -> None:
    payload = _read_json_endpoint(url, "FastAPI service", required=required)
    if payload is None:
        return
    if payload.get("status") != "healthy":
        message = f"FastAPI health response is not healthy at {url}"
        if required:
            raise DemoError(message)
        print(f"[warn] {message}")
        return
    print(f"[ok] FastAPI service: {url}")


def preflight(*, api_url: str, require_api: bool, require_seed: bool = True) -> None:
    if sys.version_info < (3, 12):
        raise DemoError("Python 3.12 or newer is required for this repository")
    print(f"[ok] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    records, cases = validate_demo_data()
    print(f"[ok] synthetic demo data: {len(records)} records, {len(cases)} cases")
    settings = demo_settings(get_settings())
    if settings.vector_store_provider.lower() != "chroma":
        raise DemoError("Phase 8 demo requires the repository's existing Chroma provider")
    if settings.embedding_provider.lower() != "local":
        raise DemoError("Phase 8 demo requires the repository's existing local embedding provider")
    if not settings.embedding_model_name.strip():
        raise DemoError("EMBEDDING_MODEL_NAME must not be blank")
    if not settings.embedding_model_revision.strip():
        raise DemoError("EMBEDDING_MODEL_REVISION must not be blank")
    if settings.embedding_batch_size <= 0:
        raise DemoError("EMBEDDING_BATCH_SIZE must be greater than zero")
    if settings.llm_provider.lower() != "openai_compatible":
        raise DemoError("Phase 8 demo requires the existing openai_compatible LLM provider")
    store = create_vector_store(settings)
    count = validate_demo_index(store, cases, allow_empty=not require_seed)
    label = "seeded" if count else "empty (will be seeded by `all`)"
    print(f"[ok] demo collection {DEMO_COLLECTION_NAME}: {label}, chunks={count}")
    try:
        embedding_backend = get_embedding_backend_for(settings)
        dimension = embedding_backend.dimension
        if count:
            build_retriever(settings, embedding_backend, store)
    except (EmbeddingError, RetrievalBackendError, VectorStoreError, ValueError) as exc:
        raise DemoError(
            "embedding preflight failed; pre-cache the configured model and verify "
            "its revision matches the demo index"
        ) from exc
    print(
        f"[ok] embedding model: {settings.embedding_model_name} "
        f"revision={settings.embedding_model_revision} dimension={dimension}"
    )

    llm_base_url = validate_local_llm_base_url(
        settings.llm_base_url,
        allow_private_network=settings.llm_allow_private_network,
    )
    _check_llm_models(f"{llm_base_url}/models", settings.llm_model_name)
    local_api_url = validate_local_llm_base_url(api_url, allow_private_network=False)
    _check_fastapi_health(f"{local_api_url}/health", required=require_api)


def _seed(*, reset: bool) -> None:
    settings = demo_settings(get_settings())
    store = create_vector_store(settings)
    if reset:
        reset_demo_collection(store)
        print(f"[ok] reset demo collection: {DEMO_COLLECTION_NAME}")
        store = create_vector_store(settings)
    summary = seed_demo_collection(settings, get_embedding_backend_for(settings), store)
    print(
        f"[ok] seeded collection={summary.collection_name} "
        f"records={summary.record_count} chunks={summary.chunk_count}"
    )


def _display_run(case_id: str | None, *, verbose: bool) -> bool:
    _, validated_cases = validate_demo_data()
    cases = select_cases(validated_cases, case_id)
    results = run_demo_cases(create_demo_rag_service(demo_settings(get_settings())), cases)
    success = True
    for result in results:
        print(f"\n[{result.case.id}] {result.case.query}")
        if result.error is not None or result.answer is None:
            success = False
            print(f"STATUS: ERROR — {result.error}")
            continue
        print(f"ANSWER: {result.answer.answer}")
        print("SOURCES:")
        for source in result.answer.sources:
            location = source.metadata.get("source", source.metadata.get("file_name", "unknown"))
            print(
                f"  [{source.source_number}] {location} "
                f"document={source.document_id} chunk={source.chunk_id}"
            )
            if verbose:
                print(f"      {source.text}")
        print("STATUS: OK")
    return success


def _evaluate(case_id: str | None, output_dir: Path) -> bool:
    settings = demo_settings(get_settings())
    _, validated_cases = validate_demo_data()
    cases = select_cases(validated_cases, case_id)
    artifact = evaluate_demo_cases(
        create_demo_rag_service(settings), cases, model_name=settings.llm_model_name
    )
    json_path, markdown_path = write_evaluation_artifacts(artifact, output_dir)
    totals = artifact["totals"]
    print(
        f"[ok] evaluation cases={totals['case_count']} pass={totals['pass_count']} "
        f"fail={totals['fail_count']} error={totals['error_count']}"
    )
    print(f"JSON: {json_path.as_posix()}")
    print(f"Markdown: {markdown_path.as_posix()}")
    return totals["fail_count"] == 0 and totals["error_count"] == 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The demo never silently fetches model files. Operators must pre-cache the
    # pinned embedding model during an explicit setup step.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        if args.command == "preflight":
            preflight(
                api_url=args.api_url,
                require_api=args.require_api,
                require_seed=not args.allow_unseeded,
            )
        elif args.command == "reset":
            if not args.yes:
                raise DemoError("reset requires --yes and affects only medical_demo_v1")
            reset_demo_collection(create_vector_store(demo_settings(get_settings())))
            print(f"[ok] reset demo collection: {DEMO_COLLECTION_NAME}")
        elif args.command == "seed":
            _seed(reset=args.reset)
        elif args.command == "run":
            return 0 if _display_run(args.case_id, verbose=args.verbose) else 1
        elif args.command == "evaluate":
            return 0 if _evaluate(args.case_id, args.output_dir) else 1
        elif args.command == "all":
            preflight(
                api_url=args.api_url,
                require_api=args.require_api,
                require_seed=False,
            )
            _seed(reset=True)
            if not _display_run(None, verbose=args.verbose):
                return 1
            return 0 if _evaluate(None, args.output_dir) else 1
        return 0
    except (DemoError, OSError, RuntimeError, ValueError) as exc:
        print(f"demo {args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
