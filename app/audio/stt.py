"""Speech-to-Text (STT) interface and development provider.

Designed for future embedded / companion architectures without platform-specific lock-in.

NOTE ON DEVELOPMENT IMPLEMENTATION:
`DevelopmentSTTProvider` is a deterministic test double and development fixture.
It does NOT perform actual deep learning speech recognition (e.g. Whisper / Kaldi).
It allows reproducible local testing of the voice pipeline across platforms
without microphone or heavy engine requirements.
"""

from abc import ABC, abstractmethod
import io
import logging
from typing import Dict, Optional
import wave

from app.audio.exceptions import AudioInputError, TranscriptionError

logger = logging.getLogger(__name__)


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text providers.

    All implementations (development test doubles, cloud STT services,
    or on-device embedded speech engines) must conform to this contract.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the STT provider."""
        pass

    @abstractmethod
    async def transcribe(self, audio_data: bytes, **kwargs) -> str:
        """Transcribe raw audio bytes into text.

        Args:
            audio_data: Raw audio byte stream (e.g. PCM 16-bit, WAV container, or encoded byte payload).

        Returns:
            str: Transcribed text query.

        Raises:
            AudioInputError: If audio_data is None, empty, or invalid type.
            TranscriptionError: If audio payload cannot be decoded, is corrupted, or unsupported.
        """
        pass


class DevelopmentSTTProvider(STTProvider):
    """Platform-independent development STT provider.

    Acts as a deterministic test double for the future real STT engine.
    Supports:
    1. Direct UTF-8 text encoded in audio byte buffers (deterministic testing).
    2. Registered byte-pattern mappings (e.g. specific audio frames -> specific text).
    3. Configurable fallback text when binary data is unrecognized.
    4. Validation and error raising for None, empty, or corrupted payloads.
    """

    def __init__(
        self,
        fallback_text: Optional[str] = "Hello",
        strict: bool = False,
    ) -> None:
        self.fallback_text = fallback_text
        self.strict = strict
        self._registered_mappings: Dict[bytes, str] = {}

    @property
    def name(self) -> str:
        return "DevelopmentSTTProvider"

    def register_mapping(self, audio_bytes: bytes, transcribed_text: str) -> None:
        """Register an exact deterministic mapping from audio bytes to transcribed text."""
        self._registered_mappings[bytes(audio_bytes)] = transcribed_text

    async def transcribe(self, audio_data: bytes, **kwargs) -> str:
        """Transcribe development audio payload into text.

        Args:
            audio_data: Byte payload to transcribe.

        Returns:
            str: Transcribed text.

        Raises:
            AudioInputError: If audio_data is None, not bytes, or empty.
            TranscriptionError: If audio payload is explicitly malformed or unparseable in strict mode.
        """
        if audio_data is None:
            raise AudioInputError("Audio data cannot be None")

        if not isinstance(audio_data, (bytes, bytearray)):
            raise AudioInputError(f"Expected bytes audio data, got {type(audio_data).__name__}")

        if len(audio_data) == 0:
            raise AudioInputError("Audio data cannot be empty")

        raw_bytes = bytes(audio_data)

        # Explicit corrupted payload sentinel check
        if raw_bytes.startswith(b"CORRUPT_AUDIO_PAYLOAD"):
            raise TranscriptionError("Malformed audio payload: corrupted audio data encountered")

        # Check for malformed WAV header
        if raw_bytes.startswith(b"RIFF") and len(raw_bytes) < 44:
            raise TranscriptionError("Malformed audio payload: truncated WAV header")

        if raw_bytes.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
                    _ = wav_file.getnframes()
            except Exception as exc:
                raise TranscriptionError(f"Malformed audio payload: invalid WAV data ({exc})") from exc

        # Check registered exact mappings first
        if raw_bytes in self._registered_mappings:
            mapped = self._registered_mappings[raw_bytes]
            logger.debug("DevelopmentSTTProvider matched registered mapping -> '%s'", mapped)
            return mapped

        # Check for UTF-8 test string payload
        try:
            decoded = raw_bytes.decode("utf-8").strip()
            if decoded:
                logger.debug("DevelopmentSTTProvider decoded direct text payload: '%s'", decoded)
                return decoded
        except UnicodeDecodeError:
            pass

        # Check strict mode
        is_strict = kwargs.get("strict", self.strict)
        if is_strict:
            raise TranscriptionError(
                "Strict transcription failed: audio payload cannot be decoded and no matching mapping was found"
            )

        # Return fallback text if defined
        if self.fallback_text is not None:
            logger.debug("DevelopmentSTTProvider returning fallback text: '%s'", self.fallback_text)
            return self.fallback_text

        raise TranscriptionError("Unrecognized audio payload and no fallback text configured")
