"""Audio exceptions for input acquisition and speech-to-text processing.

Platform-independent and embedded-friendly error definitions.
"""


class AudioError(Exception):
    """Base exception for all audio subsystem errors."""

    pass


class AudioInputError(AudioError):
    """Raised when audio input cannot be acquired, is None, or is empty."""

    pass


class TranscriptionError(AudioError):
    """Raised when speech-to-text transcription fails or audio data is malformed."""

    pass
