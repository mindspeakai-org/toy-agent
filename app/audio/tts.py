"""Text-to-Speech (TTS) interface and development provider.

Designed for future embedded / ESP32 companion architectures without platform-specific lock-in.
"""

from abc import ABC, abstractmethod
import io
import logging
import math
import struct
import wave

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Abstract interface for Text-to-Speech synthesis providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the TTS provider."""
        pass

    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Synthesize text into audio bytes.

        Args:
            text: Text to synthesize.

        Returns:
            bytes: Synthesized audio bytes (standard 16-bit PCM WAV).
        """
        pass


class DevelopmentTTSProvider(TTSProvider):
    """Platform-independent development TTS provider.

    Synthesizes valid standard 16-bit mono 16kHz PCM WAV audio streams using
    only the standard Python library (`wave`, `struct`). Requires zero macOS
    or platform-specific dependencies, ensuring cross-platform portability.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    @property
    def name(self) -> str:
        return "DevelopmentTTSProvider"

    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Synthesize a lightweight, valid WAV audio byte stream representing text."""
        cleaned = text.strip()
        if not cleaned:
            return b""

        logger.debug("DevelopmentTTSProvider synthesizing %d chars: '%s'", len(cleaned), cleaned[:40])

        # Generate a brief 100ms tone packet packaged in valid WAV format
        duration_s = 0.1
        total_samples = int(self.sample_rate * duration_s)
        frequency = 440.0  # A4

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            samples = []
            for i in range(total_samples):
                val = int(32767.0 * 0.1 * math.sin(2.0 * math.pi * frequency * (i / self.sample_rate)))
                samples.append(struct.pack("<h", val))

            wav_file.writeframes(b"".join(samples))

        wav_bytes = buffer.getvalue()
        logger.debug("DevelopmentTTSProvider generated %d WAV audio bytes", len(wav_bytes))
        return wav_bytes
