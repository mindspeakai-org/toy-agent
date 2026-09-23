"""Audio output abstraction and implementations (Speaker and Buffer).

Designed for modular development:
- SpeakerAudioOutput: Plays standard PCM WAV audio via system default output device using sounddevice.
- BufferAudioOutput: Captures audio bytes in memory for deterministic unit testing.
"""

from abc import ABC, abstractmethod
import asyncio
import io
import logging
from typing import List, Optional
import wave

from app.audio.exceptions import AudioOutputError

logger = logging.getLogger(__name__)


class AudioOutput(ABC):
    """Abstract interface for audio output playback sinks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for the audio output sink."""
        pass

    @abstractmethod
    async def play(self, audio_data: bytes) -> None:
        """Play audio bytes through the output sink.

        Args:
            audio_data: Raw audio byte payload (standard PCM WAV format).

        Raises:
            AudioOutputError: If playback fails or audio data is invalid.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop current audio playback if supported."""
        pass


class BufferAudioOutput(AudioOutput):
    """In-memory audio output sink for test fixtures and diagnostics.

    Captures all played audio payloads in memory without requiring a physical speaker.
    """

    def __init__(self, name: str = "BufferAudioOutput") -> None:
        self._name = name
        self.played_payloads: List[bytes] = []

    @property
    def name(self) -> str:
        return self._name

    async def play(self, audio_data: bytes) -> None:
        """Store audio payload in memory."""
        if not audio_data:
            raise AudioOutputError("Audio payload is empty")
        self.played_payloads.append(audio_data)

    async def stop(self) -> None:
        pass

    def clear(self) -> None:
        """Clear recorded payloads."""
        self.played_payloads.clear()


class SpeakerAudioOutput(AudioOutput):
    """Plays audio through the system default output device using sounddevice.

    Cross-platform (macOS, Linux, Windows), portable, and hardware-agnostic.
    Uses PortAudio system default device without hardcoding Mac-specific hardware.
    """

    def __init__(self) -> None:
        self._is_playing = False

    @property
    def name(self) -> str:
        return "SpeakerAudioOutput(SystemDefault)"

    def _play_sync(self, audio_data: bytes) -> None:
        """Synchronous playback worker executed in background thread."""
        if not audio_data:
            raise AudioOutputError("Cannot play empty audio payload")

        try:
            import numpy as np
            import sounddevice as sd

            buffer = io.BytesIO(audio_data)
            with wave.open(buffer, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())

            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                dtype = np.uint8

            audio_array = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)

            self._is_playing = True
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
            self._is_playing = False

        except Exception as exc:
            self._is_playing = False
            logger.error("Speaker playback failed: %s", exc)
            raise AudioOutputError(f"Speaker playback failed: {exc}") from exc

    async def play(self, audio_data: bytes) -> None:
        """Play audio asynchronously through the system default speaker."""
        await asyncio.to_thread(self._play_sync, audio_data)

    async def stop(self) -> None:
        """Halt playback immediately."""
        try:
            import sounddevice as sd
            sd.stop()
            self._is_playing = False
        except Exception:
            pass
