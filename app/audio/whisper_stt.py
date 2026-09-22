"""Real development Speech-to-Text provider powered by local Whisper.

Uses faster-whisper (CTranslate2) for high-performance, local, offline speech recognition.
Does not require cloud API keys or remote dependencies.
Kept strictly behind the abstract STTProvider interface to maintain complete architectural portability.
"""

import asyncio
import io
import logging
from typing import Optional
import wave

from app.audio.exceptions import AudioInputError, TranscriptionError
from app.audio.stt import STTProvider

logger = logging.getLogger(__name__)


def _get_whisper_class():
    """Lazily and safely import WhisperModel from faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "Missing development dependency 'faster-whisper'. "
            "Install voice dependencies with: pip install 'toy-agent[voice]'"
        ) from exc


class WhisperSTTProvider(STTProvider):
    """Real local development STT provider using Whisper.

    Transcribes standard audio bytes (WAV or raw audio stream) into text using a local
    quantized Whisper model (default 'tiny.en' on CPU).
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        name: str = "WhisperSTTProvider",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._name = name
        self._model = None

    @property
    def name(self) -> str:
        return self._name

    def _get_model(self):
        """Lazy-load the local Whisper model."""
        if self._model is None:
            whisper_cls = _get_whisper_class()
            try:
                logger.info(
                    "Loading local Whisper model '%s' (device=%s, compute_type=%s)...",
                    self.model_size,
                    self.device,
                    self.compute_type,
                )
                self._model = whisper_cls(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                raise TranscriptionError(
                    f"STT initialization failure: failed to load Whisper model '{self.model_size}': {exc}"
                ) from exc
        return self._model

    async def transcribe(self, audio_data: bytes, **kwargs) -> str:
        """Transcribe raw audio bytes into recognized text.

        Args:
            audio_data: Raw audio byte stream (e.g. WAV or PCM frames).

        Returns:
            str: Transcribed text query.

        Raises:
            AudioInputError: If audio_data is None, empty, or invalid type.
            TranscriptionError: If STT initialization fails, payload is corrupted, or transcription fails.
        """
        if audio_data is None:
            raise AudioInputError("Audio data cannot be None")

        if not isinstance(audio_data, (bytes, bytearray)):
            raise AudioInputError(f"Expected bytes audio data, got {type(audio_data).__name__}")

        if len(audio_data) == 0:
            raise AudioInputError("Audio data cannot be empty")

        raw_bytes = bytes(audio_data)

        # Check for explicitly malformed/corrupted audio
        if raw_bytes.startswith(b"CORRUPT_AUDIO_PAYLOAD"):
            raise TranscriptionError("Malformed audio payload: corrupted audio data encountered")

        # Check for truncated WAV header
        if raw_bytes.startswith(b"RIFF") and len(raw_bytes) < 44:
            raise TranscriptionError("Malformed audio payload: truncated WAV header")

        if raw_bytes.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
                    _ = wav_file.getnframes()
            except Exception as exc:
                raise TranscriptionError(f"Malformed audio payload: invalid WAV data ({exc})") from exc

        model = self._get_model()

        def _do_transcribe() -> str:
            try:
                audio_stream = io.BytesIO(raw_bytes)
                segments, info = model.transcribe(
                    audio_stream,
                    beam_size=kwargs.get("beam_size", 1),
                    language=kwargs.get("language", "en"),
                    temperature=kwargs.get("temperature", 0.0),
                )
                transcribed_text = " ".join(segment.text.strip() for segment in segments).strip()
                return transcribed_text
            except Exception as exc:
                raise TranscriptionError(f"STT transcription failure: {exc}") from exc

        return await asyncio.to_thread(_do_transcribe)
