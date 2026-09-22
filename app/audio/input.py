"""AudioInput abstraction for capturing or supplying audio data.

Provides a hardware- and platform-agnostic interface for audio sources.
The source may be an embedded microphone, companion stream, test fixture, or buffer.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional, Union

from app.audio.exceptions import AudioInputError

logger = logging.getLogger(__name__)


class AudioInput(ABC):
    """Abstract interface for audio input sources.

    Decouples speech processing from physical hardware (ESP32 I2S microphone,
    Linux ALSA/PulseAudio, network audio stream, test fixture, or mobile companion).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name or identifier of the audio input source."""
        pass

    @abstractmethod
    async def read(self) -> bytes:
        """Acquire or read an audio payload as raw bytes.

        Returns:
            bytes: Audio byte payload (e.g. PCM frames, WAV container).

        Raises:
            AudioInputError: If audio cannot be read, is None, or is empty.
        """
        pass


class BufferAudioInput(AudioInput):
    """Supplies audio data from an in-memory byte buffer.

    Useful for test fixtures, network buffers, or pre-recorded audio packets.
    """

    def __init__(
        self,
        data: Optional[Union[bytes, bytearray]] = None,
        name: str = "BufferAudioInput",
    ) -> None:
        self._data = bytes(data) if data is not None else None
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def set_data(self, data: Optional[Union[bytes, bytearray]]) -> None:
        """Update buffer contents."""
        self._data = bytes(data) if data is not None else None

    async def read(self) -> bytes:
        """Read buffer contents.

        Raises:
            AudioInputError: If data is None or empty.
        """
        if self._data is None:
            raise AudioInputError("Audio input buffer is None")
        if len(self._data) == 0:
            raise AudioInputError("Audio input buffer is empty")
        return self._data


class DevelopmentAudioInput(AudioInput):
    """Deterministic development audio source.

    Wraps deterministic payloads (such as test phrases) encoded into byte buffers
    for reproducible development and testing without requiring hardware microphones.
    """

    def __init__(
        self,
        payload: Optional[Union[str, bytes]] = "Hello",
        name: str = "DevelopmentAudioInput",
    ) -> None:
        self._name = name
        self.set_payload(payload)

    @property
    def name(self) -> str:
        return self._name

    def set_payload(self, payload: Optional[Union[str, bytes]]) -> None:
        """Configure the payload from text or bytes."""
        if payload is None:
            self._data: Optional[bytes] = None
        elif isinstance(payload, str):
            self._data = payload.encode("utf-8")
        else:
            self._data = bytes(payload)

    async def read(self) -> bytes:
        """Return the configured audio payload.

        Raises:
            AudioInputError: If payload is None or empty.
        """
        if self._data is None:
            raise AudioInputError("Development audio input payload is None")
        if len(self._data) == 0:
            raise AudioInputError("Development audio input payload is empty")
        return self._data
