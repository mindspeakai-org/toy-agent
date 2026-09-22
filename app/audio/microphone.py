"""Real development microphone input implementation.

Captures real-time audio from the local development machine's physical microphone.
This adapter is isolated to the development audio layer and wraps cross-platform
audio capture (sounddevice/PortAudio) so that the core agent architecture remains
completely hardware- and platform-agnostic (portable to ESP32 I2S or companion streams).
"""

import asyncio
import io
import logging
from typing import Optional, Union
import wave

from app.audio.exceptions import AudioInputError
from app.audio.input import AudioInput

logger = logging.getLogger(__name__)


def _get_sounddevice():
    """Lazily and safely import sounddevice."""
    try:
        import sounddevice as sd
        return sd
    except ImportError as exc:
        raise AudioInputError(
            "Missing development dependency 'sounddevice'. "
            "Install voice dependencies with: pip install 'toy-agent[voice]'"
        ) from exc


def _get_numpy():
    """Lazily and safely import numpy."""
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise AudioInputError(
            "Missing development dependency 'numpy'. "
            "Install voice dependencies with: pip install 'toy-agent[voice]'"
        ) from exc


class MicrophoneAudioInput(AudioInput):
    """Real development audio input capturing physical microphone audio.

    Records audio frames from the default (or specified) local microphone,
    verifies non-empty capture, and packages the recording into standard 16kHz mono 16-bit WAV bytes.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        default_duration: float = 4.0,
        device: Optional[Union[int, str]] = None,
        name: str = "MicrophoneAudioInput",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.default_duration = default_duration
        self.device = device
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def read(self, duration: Optional[float] = None) -> bytes:
        """Capture live audio from the physical microphone.

        Args:
            duration: Recording duration in seconds. Defaults to self.default_duration.

        Returns:
            bytes: Valid 16-bit 16kHz WAV audio bytes.

        Raises:
            AudioInputError: If dependencies are missing, microphone is unavailable,
                             or captured audio is empty.
        """
        sd = _get_sounddevice()
        np = _get_numpy()

        # Check device availability
        try:
            devices = sd.query_devices()
            if not devices:
                raise AudioInputError("Microphone unavailable: no audio devices detected")
        except Exception as exc:
            if isinstance(exc, AudioInputError):
                raise
            raise AudioInputError(f"Microphone unavailable: {exc}") from exc

        record_duration = duration if duration is not None else self.default_duration
        if record_duration <= 0:
            raise AudioInputError("Invalid recording duration: duration must be positive")

        num_frames = int(record_duration * self.sample_rate)

        def _record_sync() -> bytes:
            try:
                recording = sd.rec(
                    num_frames,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    device=self.device,
                )
                sd.wait()
            except Exception as exc:
                raise AudioInputError(f"Microphone capture failed: {exc}") from exc

            if recording is None or len(recording) == 0:
                raise AudioInputError("Captured audio is empty: no audio frames acquired")

            raw_frames = recording.tobytes()
            if len(raw_frames) == 0:
                raise AudioInputError("Captured audio is empty: zero-length frame buffer")

            # Check if audio is completely flatline silence of length 0 or no signal
            # Package as valid standard WAV
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(raw_frames)

            wav_bytes = wav_buffer.getvalue()
            if len(wav_bytes) <= 44:  # WAV header alone is 44 bytes
                raise AudioInputError("Captured audio is empty: WAV payload contains no frames")

            return wav_bytes

        return await asyncio.to_thread(_record_sync)
