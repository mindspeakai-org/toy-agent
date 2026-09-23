"""Cloud answer generator integrating with Google Gemini API."""

import logging
from typing import Any, Dict, Optional

import httpx

from app.config.settings import get_settings
from app.generation.base import AnswerGenerator
from app.generation.exceptions import (
    CloudConfigurationError,
    CloudGenerationError,
    CloudTimeoutError,
    EmptyResponseError,
)
from app.memory.base import MemoryContext

logger = logging.getLogger("toy_agent.generation.cloud")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class CloudAnswerGenerator(AnswerGenerator):
    """Generates child-facing answers using external Google Gemini LLM API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.timeout_seconds = timeout_seconds or settings.gemini_timeout_seconds
        self._http_client = http_client

    @property
    def name(self) -> str:
        return f"CloudAnswerGenerator(Gemini/{self.model})"

    async def generate(
        self,
        query: str,
        memory_context: Optional[MemoryContext] = None,
    ) -> str:
        """Call Gemini API to generate response for the user query.

        Args:
            query: The user query text.
            memory_context: Optional memory context (typically None for CLOUD queries).

        Returns:
            str: Generated natural language answer text.

        Raises:
            CloudConfigurationError: If GEMINI_API_KEY is not set.
            CloudTimeoutError: If the request times out.
            CloudGenerationError: If the API returns an error status or malformed response.
            EmptyResponseError: If response text is empty.
        """
        if not self.api_key or not self.api_key.strip():
            logger.error("Cloud generation requested but GEMINI_API_KEY is missing")
            raise CloudConfigurationError("GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment.")

        endpoint = f"{GEMINI_BASE_URL}/{self.model}:generateContent"
        params = {"key": self.api_key}

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": query.strip()}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 256,
            },
        }

        # Do NOT log the API key or raw endpoint containing query params with key
        logger.info("[CLOUD GENERATION] Sending query to Gemini model '%s' (endpoint: %s)", self.model, endpoint)

        client = self._http_client
        should_close = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout_seconds)
            should_close = True

        try:
            response = await client.post(
                endpoint,
                params=params,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                error_msg = f"Gemini API returned status code {response.status_code}: {response.text}"
                logger.error("Gemini API error: HTTP %d", response.status_code)
                raise CloudGenerationError(error_msg)

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise EmptyResponseError("Gemini API returned no candidates")

            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            if not parts:
                raise EmptyResponseError("Gemini candidate contains no text parts")

            text_answer = parts[0].get("text", "").strip()
            if not text_answer:
                raise EmptyResponseError("Gemini candidate text is empty")

            return text_answer

        except httpx.TimeoutException as exc:
            logger.error("Gemini API request timed out after %.1fs", self.timeout_seconds)
            raise CloudTimeoutError(f"Gemini API timed out after {self.timeout_seconds}s") from exc
        except (CloudConfigurationError, CloudGenerationError, CloudTimeoutError, EmptyResponseError):
            raise
        except Exception as exc:
            logger.error("Unexpected error communicating with Gemini API: %s", exc)
            raise CloudGenerationError(f"Gemini API communication failed: {exc}") from exc
        finally:
            if should_close:
                await client.aclose()


class MockCloudAnswerGenerator(AnswerGenerator):
    """Deterministic mock for Cloud generation during automated testing."""

    def __init__(
        self,
        canned_response: str = "Mock cloud response for testing.",
    ) -> None:
        self._canned_response = canned_response
        self.call_count = 0
        self.last_query: Optional[str] = None

    @property
    def name(self) -> str:
        return "MockCloudAnswerGenerator"

    def set_response(self, response: str) -> None:
        self._canned_response = response

    async def generate(
        self,
        query: str,
        memory_context: Optional[MemoryContext] = None,
    ) -> str:
        self.call_count += 1
        self.last_query = query
        return self._canned_response
