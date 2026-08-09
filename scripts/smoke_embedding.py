"""CPU real-model smoke test for the pinned Phase 3 embedding contract.

This script intentionally downloads the model on first use. Run it only after
the operator has approved network/model downloads.
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.embeddings.local_embedding import LocalEmbeddingBackend


def _assert_unit_vector(vector: list[float]) -> None:
    assert len(vector) == 384
    assert all(math.isfinite(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5)


def main() -> None:
    settings = Settings(embedding_device="cpu")
    backend = LocalEmbeddingBackend(
        model_name=settings.embedding_model_name,
        revision=settings.embedding_model_revision,
        device="cpu",
    )

    assert backend.model_name == settings.embedding_model_name
    assert backend.model_revision == settings.embedding_model_revision
    assert backend.dimension == 384

    model = backend._load_model()
    assert model.model_card_data.base_model == settings.embedding_model_name
    assert (
        model.model_card_data.base_model_revision
        == settings.embedding_model_revision
    )

    passage_text = "病患主訴胸痛。"
    query_text = "胸痛的可能原因"
    passage = backend.embed_documents([passage_text])[0]
    query = backend.embed_query(query_text)
    _assert_unit_vector(passage)
    _assert_unit_vector(query)

    # Compare with explicit E5 inputs to prove the backend added each prefix.
    direct = model.encode(
        [f"passage: {passage_text}", f"query: {query_text}"],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    for actual, expected in zip((passage, query), direct, strict=True):
        assert all(
            math.isclose(left, float(right), rel_tol=1e-6, abs_tol=1e-6)
            for left, right in zip(actual, expected, strict=True)
        )

    print(
        "PASS: "
        f"model={backend.model_name} "
        f"revision={backend.model_revision} "
        "dimension=384 normalization=L2 "
        "vectors=document:1,query:1 finite=true "
        "prefixes=passage:,query:"
    )


if __name__ == "__main__":
    main()
