import logging
import time

import httpx

from fin_agent.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Модель недоступна или вернула ошибку после всех попыток"""


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings().llm
        self._model = settings.model
        self._max_retries = settings.max_retries
        self._backoff = settings.retry_backoff_seconds
        self._timeout_seconds = settings.timeout_seconds
        self._client = httpx.Client(
            base_url=settings.url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=httpx.Timeout(settings.timeout_seconds, connect=5.0),
        )

    def close(self) -> None:
        self._client.close()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict = {"model": self._model, "messages": messages}
        if tools:
            payload["tools"] = tools

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = LLMError(f"HTTP {response.status_code}")
                    logger.warning(
                        "Модель вернула %s, попытка %d из %d",
                        response.status_code,
                        attempt + 1,
                        self._max_retries + 1,
                    )
                    self._sleep(attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError as exc:
                last_error = exc
                logger.warning("Нет соединения с моделью, попытка %d", attempt + 1)
                self._sleep(attempt)
            except httpx.TimeoutException as exc:
                raise LLMError(
                    f"Модель не ответила за {self._timeout_seconds} с"
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Модель отклонила запрос: HTTP {exc.response.status_code}"
                ) from exc

        raise LLMError(
            f"Модель недоступна после {self._max_retries + 1} попыток: {last_error}"
        )

    def _sleep(self, attempt: int) -> None:
        if attempt < self._max_retries:
            time.sleep(self._backoff * 2 ** attempt)

    @staticmethod
    def usage(response: dict) -> tuple[int, int]:
        usage = response.get("usage") or {}
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
