"""Phase 7 runtime integration tests; all run without GPU, server, or network."""

import io
import logging
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.llm.base import LLMError
from app.llm.factory import create_llm_provider
from app.llm.local_backend import OllamaOpenAICompatibleLLM, OpenAICompatibleLLM
from app.llm.network import validate_local_llm_base_url
from app.rag.models import RAGAnswer, RAGSource
from scripts.check_runtime import (
    check_llm_configuration,
    collect_runtime_info,
    format_runtime_report,
)
from scripts.smoke_test_llm import run_smoke
from scripts.smoke_test_rag import run_rag_smoke
import scripts.smoke_test_rag as rag_smoke_module


class FakeCuda:
    def __init__(
        self, available: bool, count: int = 0, gfx_architecture: str | None = None
    ) -> None:
        self.available = available
        self.count = count
        self.gfx_architecture = gfx_architecture

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, index: int) -> str:
        return f"Synthetic AMD Device {index}"

    def get_device_properties(self, index: int):
        return SimpleNamespace(gcnArchName=self.gfx_architecture)


def fake_torch(
    *,
    available: bool,
    count: int,
    hip: str | None,
    gfx_architecture: str | None = None,
):
    return SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(hip=hip),
        cuda=FakeCuda(available, count, gfx_architecture),
    )


class FakeProvider:
    model_name = "fake-local-model"

    def __init__(self, answer: str = "LOCAL_LLM_OK", failure: Exception | None = None) -> None:
        self.answer = answer
        self.failure = failure

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.failure is not None:
            raise self.failure
        return self.answer


class FakeRAGService:
    def __init__(self, result: RAGAnswer) -> None:
        self.result = result
        self.calls: list[tuple[str, int | None]] = []

    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
        self.calls.append((query, top_k))
        return self.result


def synthetic_source() -> RAGSource:
    return RAGSource(
        source_number=1,
        chunk_id="synthetic-chunk",
        document_id="synthetic-document",
        text="Synthetic record text.",
        distance=0.1,
        score=0.9,
        distance_metric="cosine",
        metadata={},
    )


class TestRuntimeDiagnostics:
    def test_missing_torch_is_warning_not_traceback(self) -> None:
        def missing(name: str):
            raise ModuleNotFoundError(name)

        info = collect_runtime_info(missing, lambda name: None)
        assert info["torch"]["importable"] is False
        assert "ModuleNotFoundError" in info["torch"]["warning"]

    def test_no_accelerator_and_no_hip_is_not_rocm_available(self) -> None:
        no_gpu = collect_runtime_info(
            lambda name: fake_torch(available=False, count=0, hip=None),
            lambda name: None,
        )
        assert no_gpu["torch"]["accelerator_available"] is False
        assert no_gpu["torch"]["rocm_build"] is False
        assert no_gpu["torch"]["rocm_device_available"] is False
        assert no_gpu["torch"]["device_names"] == []

    def test_accelerator_without_hip_is_never_reported_as_rocm(self) -> None:
        cuda_only = collect_runtime_info(
            lambda name: fake_torch(available=True, count=1, hip=None),
            lambda name: None,
        )
        assert cuda_only["torch"]["accelerator_available"] is True
        assert cuda_only["torch"]["rocm_build"] is False
        assert cuda_only["torch"]["rocm_device_available"] is False
        report = format_runtime_report(cuda_only, (True, "accepted"))
        assert "ROCm device available to PyTorch: no" in report
        assert "not a HIP/ROCm PyTorch build" in report

    def test_accelerator_with_hip_is_rocm_and_reports_gfx_when_available(self) -> None:
        rocm = collect_runtime_info(
            lambda name: fake_torch(
                available=True,
                count=1,
                hip="6.test",
                gfx_architecture="gfx-test:sramecc+",
            ),
            lambda name: f"/safe/{name}",
        )
        assert rocm["torch"]["hip"] == "6.test"
        assert rocm["torch"]["rocm_build"] is True
        assert rocm["torch"]["rocm_device_available"] is True
        assert rocm["torch"]["device_names"] == ["Synthetic AMD Device 0"]
        assert rocm["torch"]["gfx_architectures"] == ["gfx-test:sramecc+"]

    def test_missing_gfx_property_is_ignored_safely(self) -> None:
        rocm = collect_runtime_info(
            lambda name: fake_torch(available=True, count=1, hip="6.test"),
            lambda name: None,
        )
        assert rocm["torch"]["rocm_device_available"] is True
        assert rocm["torch"]["gfx_architectures"] == []

    def test_report_does_not_print_configured_url_or_secret(self) -> None:
        info = collect_runtime_info(
            lambda name: fake_torch(available=False, count=0, hip=None),
            lambda name: None,
        )
        settings = Settings(llm_base_url="http://127.0.0.1:8123/v1")
        report = format_runtime_report(info, check_llm_configuration(settings))
        assert "127.0.0.1" not in report
        assert "8123" not in report
        assert "token" not in report.casefold()


class TestLocalNetworkPolicy:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8001/v1",
            "http://LOCALHOST.:8001/v1/",
            "http://127.0.0.1:8001/v1",
            "http://[::1]:8001/v1",
        ],
    )
    def test_loopback_is_accepted(self, url: str) -> None:
        assert validate_local_llm_base_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "http://localhost.example.com/v1",
            "http://8.8.8.8/v1",
            "http://127.0.0.1:not-a-port/v1",
            "not-a-url",
        ],
    )
    def test_public_or_misleading_endpoint_is_rejected(self, url: str) -> None:
        with pytest.raises(ValueError):
            validate_local_llm_base_url(url)

    def test_private_ip_requires_explicit_opt_in(self) -> None:
        url = "http://192.168.50.20:8001/v1"
        with pytest.raises(ValueError):
            validate_local_llm_base_url(url)
        assert validate_local_llm_base_url(url, allow_private_network=True) == url

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.1.2:8001/v1",
            "http://192.0.2.10:8001/v1",
            "http://[fe80::1]:8001/v1",
        ],
    )
    def test_opt_in_does_not_admit_link_local_or_reserved_ranges(self, url: str) -> None:
        with pytest.raises(ValueError):
            validate_local_llm_base_url(url, allow_private_network=True)

    def test_ipv6_unique_local_address_requires_opt_in(self) -> None:
        url = "http://[fd00::20]:8001/v1"
        with pytest.raises(ValueError):
            validate_local_llm_base_url(url)
        assert validate_local_llm_base_url(url, allow_private_network=True) == url

    def test_factory_enforces_policy_but_direct_client_injection_remains_available(self) -> None:
        with pytest.raises(ValueError):
            create_llm_provider(Settings(llm_base_url="https://example.com/v1"))
        client = httpx.Client(
            base_url="http://test.invalid",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "ok"}}]},
                )
            ),
        )
        provider = OpenAICompatibleLLM(
            base_url="https://public.invalid/v1",
            model_name="fake",
            temperature=0,
            max_tokens=1,
            timeout=1,
            client=client,
        )
        assert provider.generate(system_prompt="s", user_prompt="u") == "ok"
        client.close()

    def test_factory_selects_ollama_dialect_only_when_explicit(self) -> None:
        standard = create_llm_provider(
            Settings(_env_file=None, llm_compatibility_mode="standard")
        )
        ollama = create_llm_provider(
            Settings(_env_file=None, llm_compatibility_mode="ollama")
        )
        try:
            assert type(standard) is OpenAICompatibleLLM
            assert type(ollama) is OllamaOpenAICompatibleLLM
        finally:
            standard._client.close()  # type: ignore[attr-defined]
            ollama._client.close()  # type: ignore[attr-defined]


class TestLLMSmoke:
    def test_controlled_connection_failure(self) -> None:
        output, error = io.StringIO(), io.StringIO()
        code = run_smoke(
            FakeProvider(failure=LLMError("connection refused")),
            output=output,
            error=error,
        )
        assert code == 1
        assert "connection refused" in error.getvalue()
        assert "Traceback" not in error.getvalue()

    def test_success_requires_nonempty_output(self) -> None:
        output, error = io.StringIO(), io.StringIO()
        assert run_smoke(FakeProvider("   "), output=output, error=error) == 1
        assert "empty" in error.getvalue()
        error = io.StringIO()
        assert run_smoke(FakeProvider(), output=output, error=error) == 0
        assert "passed" in output.getvalue()


class TestRAGSmoke:
    def test_cli_restores_global_logging_state_after_setup_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SimpleNamespace(count=lambda: 1)
        monkeypatch.setattr(rag_smoke_module, "create_vector_store", lambda settings: store)

        def fail(settings):
            raise ValueError("synthetic setup failure")

        monkeypatch.setattr(rag_smoke_module, "get_vector_retriever", fail)
        previous = logging.root.manager.disable
        assert rag_smoke_module.main(["--query", "synthetic question"]) == 1
        assert logging.root.manager.disable == previous

    def test_empty_index_fails_without_calling_service(self) -> None:
        service = FakeRAGService(RAGAnswer(answer="unused", model="fake", sources=[]))
        error = io.StringIO()
        code = run_rag_smoke(
            query="synthetic question",
            top_k=2,
            service=service,  # type: ignore[arg-type]
            indexed_count=0,
            output=io.StringIO(),
            error=error,
        )
        assert code == 1
        assert service.calls == []
        assert "indexed" in error.getvalue()

    def test_wrapper_calls_existing_rag_service_and_prints_traceability(self) -> None:
        service = FakeRAGService(
            RAGAnswer(
                answer="Synthetic answer [1].",
                model="fake-local-model",
                sources=[synthetic_source()],
            )
        )
        output = io.StringIO()
        code = run_rag_smoke(
            query="synthetic question",
            top_k=3,
            service=service,  # type: ignore[arg-type]
            indexed_count=1,
            output=output,
            error=io.StringIO(),
        )
        assert code == 0
        assert service.calls == [("synthetic question", 3)]
        assert "synthetic-chunk" in output.getvalue()
        assert "synthetic-document" in output.getvalue()
