"""Connectivity smoke test for the configured local chat-completions server."""

import sys
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.llm.base import LLMError, LLMProvider
from app.llm.factory import create_llm_provider


def run_smoke(provider: LLMProvider, *, output: TextIO, error: TextIO) -> int:
    try:
        answer = provider.generate(
            system_prompt="This is a local infrastructure check.",
            user_prompt="Reply with exactly: LOCAL_LLM_OK",
        )
    except LLMError as exc:
        print(f"Local LLM smoke test failed: {exc}", file=error)
        return 1
    if not answer.strip():
        print("Local LLM smoke test failed: model returned empty text", file=error)
        return 1
    print(
        f"Local LLM smoke test passed: model={provider.model_name} "
        f"response_characters={len(answer)}",
        file=output,
    )
    return 0


def main() -> int:
    try:
        provider = create_llm_provider(Settings())
    except ValueError as exc:
        print(f"Local LLM configuration rejected: {exc}", file=sys.stderr)
        return 2
    return run_smoke(provider, output=sys.stdout, error=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
