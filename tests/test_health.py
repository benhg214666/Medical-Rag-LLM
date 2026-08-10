"""系統與各模組 status endpoint 的基本測試。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_200() -> None:
    response = client.get("/")
    assert response.status_code == 200


def test_root_status_is_running() -> None:
    response = client.get("/")
    body = response.json()
    assert body["status"] == "running"
    assert body["project"] == "Medical Local RAG"
    assert body["version"] == "0.1.0"


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_is_healthy() -> None:
    response = client.get("/health")
    assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize(
    ("module_name", "expected_status"),
    [
        ("query", "not_implemented"),
        ("documents", "available"),
        # Phase 4 起 retrieval 模組已實作。
        ("retrieval", "available"),
    ],
)
def test_module_status_endpoints(
    module_name: str,
    expected_status: str,
) -> None:
    response = client.get(f"/api/{module_name}/status")

    assert response.status_code == 200
    assert response.json() == {
        "module": module_name,
        "status": expected_status,
    }


def test_models_status_reports_embedding_configuration() -> None:
    response = client.get("/api/models/status")
    assert response.status_code == 200
    assert response.json() == {
        "module": "models",
        "status": "available",
        "embedding_provider": "local",
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_model_revision": (
            "614241f622f53c4eeff9890bdc4f31cfecc418b3"
        ),
        "embedding_device": "cpu",
    }
