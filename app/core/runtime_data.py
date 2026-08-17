"""Strictly scoped cleanup for generated RAG runtime data."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.core.config import Settings


def _resolve_from_root(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validated_runtime_path(path: Path, project_root: Path, corpus_dir: Path) -> Path:
    resolved = _resolve_from_root(path, project_root)
    root = project_root.resolve()
    corpus = _resolve_from_root(corpus_dir, project_root)
    if resolved in {root, corpus} or corpus == resolved or corpus.is_relative_to(resolved):
        raise ValueError(f"Refusing unsafe runtime path: {path}")
    if not resolved.is_relative_to(root):
        raise ValueError(f"Runtime path is outside project root: {path}")
    return resolved


def reset_rag_data(settings: Settings, project_root: Path) -> list[Path]:
    """Clear only configured raw, processed, and vector-store directories."""
    targets = (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.vector_db_dir,
    )
    validated = [
        _validated_runtime_path(path, project_root, settings.corpus_data_dir)
        for path in targets
    ]
    if len(set(validated)) != len(validated):
        raise ValueError("Runtime directories must be distinct")
    for directory in validated:
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if child.name == ".gitkeep" and child.is_file():
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    _resolve_from_root(settings.corpus_data_dir, project_root).mkdir(parents=True, exist_ok=True)
    return validated
