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


@pytest.mark.parametrize("module", ["query", "documents", "models"])
def test_module_status_endpoints(module: str) -> None:
    response = client.get(f"/api/{module}/status")
    assert response.status_code == 200
    assert response.json() == {"module": module, "status": "not_implemented"}
