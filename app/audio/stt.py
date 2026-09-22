"""Speech-to-Text (STT) interface and development provider.

Designed for future embedded / ESP32 companion architectures without platform-specific lock-in.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the STT provider."""
        pass

    @abstractmethod
    async def transcribe(self, audio_data: bytes, **kwargs) -> str:
        """Transcribe raw audio bytes into text.

        Args:
            audio_data: Raw audio byte stream (e.g. PCM 16-bit, WAV).

        Returns:
            str: Transcribed text query.
        """
        pass


class DevelopmentSTTProvider(STTProvider):
    """Platform-independent development STT provider.

    Accepts test audio byte payloads (e.g. UTF-8 encoded test text or mock audio)
    allowing end-to-end voice pipeline verification during development without
    depending on macOS-specific speech engines or un-quantized desktop models.
    """

    def __init__(self, fallback_text: str = "Hello") -> None:
        self.fallback_text = fallback_text

    @property
    def name(self) -> str:
        return "DevelopmentSTTProvider"

    async def transcribe(self, audio_data: bytes, **kwargs) -> str:
        """Transcribe development audio payload."""
        if not audio_data:
            return ""

        # Allow passing UTF-8 string directly in byte buffer for unit & integration testing
        try:
            decoded = audio_data.decode("utf-8").strip()
            if decoded:
                logger.debug("DevelopmentSTTProvider decoded direct text payload: '%s'", decoded)
                return decoded
        except UnicodeDecodeError:
            pass

        logger.debug("DevelopmentSTTProvider returning fallback text: '%s'", self.fallback_text)
        return self.fallback_text
