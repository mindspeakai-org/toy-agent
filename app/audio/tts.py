"""Text-to-Speech (TTS) interface and development provider.

Designed for future embedded / ESP32 companion architectures without platform-specific lock-in.
"""

from abc import ABC, abstractmethod
import io
import logging
import math
import struct
from typing import Any, Optional
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


class PiperTTSProvider(TTSProvider):
    """Offline neural Text-to-Speech provider using Piper ONNX voice models.

    Completely offline and cross-platform (CPU/ARM/x86).
    Produces 16-bit mono 22050Hz PCM WAV audio.
    Downloads/caches the model weights once and keeps the loaded voice model warm
    in memory across syntheses.
    """

    DEFAULT_VOICE_REPO = "rhasspy/piper-voices"
    DEFAULT_MODEL_FILE = "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    DEFAULT_CONFIG_FILE = "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        voice_repo: str = DEFAULT_VOICE_REPO,
        model_file: str = DEFAULT_MODEL_FILE,
        config_file: str = DEFAULT_CONFIG_FILE,
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.voice_repo = voice_repo
        self.model_file = model_file
        self.config_file = config_file
        self._voice: Any = None

    @property
    def name(self) -> str:
        return "PiperTTSProvider(en_US-lessac-medium)"

    def _load_voice(self) -> Any:
        """Load PiperVoice lazily and cache for subsequent calls."""
        if self._voice is not None:
            return self._voice

        from app.audio.exceptions import SynthesisError

        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise SynthesisError(
                "Missing dependency 'piper-tts'. Install via 'uv add piper-tts'."
            ) from exc

        try:
            # Resolve model and config paths
            resolved_model = self.model_path
            resolved_config = self.config_path

            if not resolved_model:
                from huggingface_hub import hf_hub_download

                logger.info("Resolving Piper voice model from cache: %s", self.model_file)
                resolved_model = hf_hub_download(
                    repo_id=self.voice_repo,
                    filename=self.model_file,
                )
            if not resolved_config:
                from huggingface_hub import hf_hub_download

                resolved_config = hf_hub_download(
                    repo_id=self.voice_repo,
                    filename=self.config_file,
                )

            logger.info("Loading Piper ONNX voice model into memory: %s", resolved_model)
            self._voice = PiperVoice.load(resolved_model, config_path=resolved_config)
            return self._voice
        except Exception as exc:
            logger.error("Failed to load Piper voice model: %s", exc)
            raise SynthesisError(f"Piper TTS model initialization failed: {exc}") from exc

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis worker executed in background thread."""
        voice = self._load_voice()
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, set_wav_format=True)

        return buffer.getvalue()

    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Synthesize text into 16-bit mono 22050Hz PCM WAV bytes."""
        from app.audio.exceptions import SynthesisError

        cleaned = text.strip()
        if not cleaned:
            return b""

        import asyncio

        try:
            return await asyncio.to_thread(self._synthesize_sync, cleaned)
        except SynthesisError:
            raise
        except Exception as exc:
            logger.error("Piper TTS synthesis error: %s", exc)
            raise SynthesisError(f"Piper TTS synthesis failed: {exc}") from exc


class MockTTSProvider(TTSProvider):
    """Deterministic mock TTS provider for automated testing."""

    def __init__(self, canned_audio: Optional[bytes] = None, name: str = "MockTTSProvider") -> None:
        self._name = name
        self.call_count = 0
        self.call_history: list[str] = []
        self._injected_error: Optional[Exception] = None

        if canned_audio is not None:
            self._canned_audio = canned_audio
        else:
            # Produce a valid 100ms 16-bit mono 16kHz WAV header + frames
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 160)
            self._canned_audio = buffer.getvalue()

    @property
    def name(self) -> str:
        return self._name

    def inject_error(self, exc: Exception) -> None:
        """Inject an exception to be raised on synthesize."""
        self._injected_error = exc

    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Record synthesis call and return canned audio or raise injected error."""
        self.call_count += 1
        self.call_history.append(text)

        if self._injected_error is not None:
            raise self._injected_error

        cleaned = text.strip()
        if not cleaned:
            return b""

        return self._canned_audio

