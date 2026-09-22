"""Dedicated client for communicating with the external slm-router service."""

import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.config.settings import get_settings
from app.router.exceptions import (
    RouterConnectionError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.models import ProcessingType, RoutingDecision

logger = logging.getLogger(__name__)


class RouterClient:
    """HTTP client communicating with the external SLM Router service.

    Responsible only for sending text queries to the router and parsing the structured
    decision response conforming to the authoritative contract:
    {
      "processing": "LOCAL" | "CLOUD",
      "memory_required": bool,
      "memory_request": {"keys": [...]} | null
    }
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.router_base_url).rstrip("/")
        self.endpoint = endpoint or settings.router_route_endpoint
        if not self.endpoint.startswith("/"):
            self.endpoint = f"/{self.endpoint}"
        self.timeout = timeout or settings.router_timeout_seconds
        self._external_client = client is not None
        self._client = client

    @property
    def target_url(self) -> str:
        """Full URL targeted by this client."""
        return f"{self.base_url}{self.endpoint}"

    @property
    def health_url(self) -> str:
        """Health check endpoint URL."""
        return f"{self.base_url}/health"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or initialize the underlying httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def check_health(self) -> Dict[str, Any]:
        """Perform a single health check request against the external SLM Router.

        Returns:
            Dict[str, Any]: Health status data returned by router.

        Raises:
            RouterConnectionError: Router service is unreachable.
            RouterTimeoutError: Health check timed out.
            RouterResponseError: Unexpected status code or non-JSON response.
        """
        client = await self._get_client()
        try:
            response = await client.get(self.health_url, timeout=min(self.timeout, 5.0))
        except httpx.TimeoutException as exc:
            raise RouterTimeoutError(f"Health check timed out: {exc}") from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise RouterConnectionError(f"Router service unreachable at {self.health_url}: {exc}") from exc
        except Exception as exc:
            raise RouterConnectionError(f"Health check failed: {exc}") from exc

        if response.status_code != 200:
            raise RouterResponseError(f"Router returned health status {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            if not isinstance(data, dict):
                raise RouterResponseError("Health check payload must be a JSON dictionary")
            return data
        except Exception as exc:
            raise RouterResponseError(f"Invalid JSON from health check: {exc}") from exc

    async def is_healthy(self) -> bool:
        """Convenience method checking if the external router is reachable and healthy."""
        try:
            data = await self.check_health()
            status = data.get("status", "").lower()
            return status in {"ok", "healthy"}
        except Exception:
            return False

    async def route(self, text: str) -> RoutingDecision:
        """Send text to the external router and return a strictly validated RoutingDecision.

        Args:
            text: User text to classify.

        Returns:
            RoutingDecision: Strongly typed routing decision.

        Raises:
            RouterConnectionError: Router cannot be reached.
            RouterTimeoutError: Router call timed out.
            RouterResponseError: Invalid HTTP response, malformed JSON, or schema violation.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return RoutingDecision(
                processing=ProcessingType.LOCAL,
                memory_required=False,
                memory_request=None,
                query=text,
            )

        client = await self._get_client()
        payload: Dict[str, Any] = {
            "query": cleaned_text,
        }

        logger.debug("Dispatching request to router at %s: %s", self.target_url, cleaned_text)
        t_start = time.perf_counter()

        try:
            response = await client.post(
                self.target_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            logger.warning("Router request timed out after %.2fs: %s", self.timeout, exc)
            raise RouterTimeoutError(f"Router request timed out after {self.timeout} seconds") from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.warning("Failed to connect to router at %s: %s", self.target_url, exc)
            raise RouterConnectionError(f"Could not connect to router at {self.target_url}") from exc
        except Exception as exc:
            logger.warning("Unexpected network error communicating with router: %s", exc)
            raise RouterConnectionError(f"Router communication failed: {exc}") from exc

        http_latency = round(time.perf_counter() - t_start, 3)

        if response.status_code != 200:
            logger.warning("Router responded with status code %d: %s", response.status_code, response.text)
            raise RouterResponseError(
                f"Router returned error status {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.warning("Router response was not valid JSON: %s", response.text[:200])
            raise RouterResponseError(f"Malformed JSON response from router: {exc}") from exc

        if not isinstance(data, dict):
            logger.warning("Router response JSON is not an object: %s", type(data))
            raise RouterResponseError("Router payload must be a JSON dictionary")

        try:
            decision = RoutingDecision.from_payload(data, query=cleaned_text, http_latency=http_latency)
        except Exception as exc:
            logger.warning("Invalid decision schema received from router: %s", exc)
            raise RouterResponseError(f"Invalid decision schema received from router: {exc}") from exc

        logger.debug(
            "Received routing decision: processing=%s, memory_required=%s",
            decision.processing.value,
            decision.memory_required,
        )
        return decision

    async def aclose(self) -> None:
        """Close the underlying HTTP client if managed internally."""
        if self._client and not self._client.is_closed and not self._external_client:
            await self._client.aclose()

    async def __aenter__(self) -> "RouterClient":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
