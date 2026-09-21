"""Cloud AI client abstraction and Phase 1 stub."""

import logging
from typing import Optional

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class CloudClient:
    """Client abstraction for external cloud-hosted LLMs (e.g., Gemini, OpenAI, Claude).

    In Phase 1, this acts as a safe, deterministic stub to avoid unnecessary external
    cloud dependencies and API keys while establishing the architectural interface.
    """

    def __init__(self, stub_mode: Optional[bool] = None) -> None:
        settings = get_settings()
        self.stub_mode = stub_mode if stub_mode is not None else settings.cloud_provider_stub_mode

    async def generate(self, text: str) -> str:
        """Generate response from cloud AI model.

        Args:
            text: Query to dispatch to cloud model.

        Returns:
            str: Generated text response.
        """
        logger.info("CloudClient received query for cloud processing: '%s'", text)

        if self.stub_mode:
            # Deterministic, child-friendly stub for Phase 1
            return (
                "That sounds like a wonderful big question! "
                "This question would be sent to the cloud AI."
            )

        # In Phase 2, real cloud provider integration (e.g. Gemini) will plug in here.
        raise NotImplementedError("Live cloud providers are not configured for Phase 1.")
