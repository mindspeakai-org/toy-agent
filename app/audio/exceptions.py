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


class AudioOutputError(AudioError):
    """Raised when audio playback or output stream encounters a failure."""

    pass


class TTSError(AudioError):
    """Base exception for text-to-speech synthesis failures."""

    pass


class SynthesisError(TTSError):
    """Raised when speech synthesis fails or audio model encounters an error."""

    pass

