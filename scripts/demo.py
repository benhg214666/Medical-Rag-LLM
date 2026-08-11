"""Repository-root entry point for the Phase 8 demo."""

import sys
from pathlib import Path

# Python starts a script with scripts/ (not the repository root) on sys.path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.demo.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
