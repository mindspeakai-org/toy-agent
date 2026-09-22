"""Audio subsystem for speech input, speech-to-text, and speech synthesis."""

from app.audio.exceptions import AudioError, AudioInputError, TranscriptionError
from app.audio.input import AudioInput, BufferAudioInput, DevelopmentAudioInput
from app.audio.stt import DevelopmentSTTProvider, STTProvider
from app.audio.tts import DevelopmentTTSProvider, TTSProvider

__all__ = [
    "AudioError",
    "AudioInputError",
    "TranscriptionError",
    "AudioInput",
    "BufferAudioInput",
    "DevelopmentAudioInput",
    "STTProvider",
    "DevelopmentSTTProvider",
    "TTSProvider",
    "DevelopmentTTSProvider",
]
