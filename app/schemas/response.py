"""API 回應（response）資料結構。"""

from pydantic import BaseModel


class RootResponse(BaseModel):
    """GET / 的回應內容。"""

    project: str
    version: str
    status: str


class HealthResponse(BaseModel):
    """GET /health 的回應內容。"""

    status: str


class ModuleStatusResponse(BaseModel):
    """各功能模組 status endpoint 的回應內容。"""

    module: str
    status: str
