"""Clear generated RAG data while preserving the permanent corpus."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.runtime_data import reset_rag_data


def main() -> int:
    settings = get_settings()
    print("Resetting RAG runtime data...\n")
    cleared = reset_rag_data(settings, PROJECT_ROOT)
    for directory in cleared:
        print(f"Cleared: {directory.relative_to(PROJECT_ROOT)}")
    corpus = settings.corpus_data_dir.resolve().relative_to(PROJECT_ROOT.resolve())
    print(f"\nPreserved: {corpus}")
    print("\nRAG runtime data reset complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
