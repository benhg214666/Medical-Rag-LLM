"""Batch-import the configured permanent corpus."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.embeddings.dependencies import get_embedding_backend_for
from app.indexing.pipeline import IndexingPipeline
from app.ingestion.corpus import corpus_breakdown, ingest_corpus, scan_corpus
from app.vector_store.factory import create_vector_store


def main() -> int:
    settings = get_settings()
    files = scan_corpus(settings.corpus_data_dir)
    patients, knowledge = corpus_breakdown(files, settings.corpus_data_dir)
    print("Scanning corpus...\n")
    print(f"Found {len(files)} supported documents.\n")
    if patients:
        print("Patient records:")
        for patient_id, count in sorted(patients.items()):
            print(f"  {patient_id}: {count} documents")
    if knowledge:
        print("\nMedical knowledge:")
        for category, count in sorted(knowledge.items()):
            print(f"  {category}: {count} documents")
    if not files:
        print("\nCorpus ingestion complete (nothing to process).")
        return 0

    pipeline = IndexingPipeline(
        get_embedding_backend_for(settings),
        create_vector_store(settings),
        settings.embedding_batch_size,
    )
    print("\nProcessing...")
    result = ingest_corpus(
        settings,
        pipeline,
        progress=lambda current, total, path: print(f"[{current}/{total}] {path.name}"),
    )
    print(f"\nDocuments processed: {result.processed}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    print(f"Skipped: {result.skipped}")
    print(f"Failed: {result.failed}")
    for failure in result.failures:
        print(f"  - {failure}")
    print("\nCorpus ingestion complete.")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
