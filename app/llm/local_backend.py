"""HTTP client for a locally hosted OpenAI-compatible chat endpoint."""

from typing import Any

import httpx

from app.llm.base import LLMError


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            trust_env=False,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise LLMError("本地 LLM 回應逾時") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError("本地 LLM 服務回傳錯誤狀態") from exc
        except httpx.RequestError as exc:
            raise LLMError("無法連線至本地 LLM 服務") from exc
        except ValueError as exc:
            raise LLMError("本地 LLM 回傳無效 JSON") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("本地 LLM 回應格式不正確") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("本地 LLM 未回傳有效答案")
        return content.strip()
