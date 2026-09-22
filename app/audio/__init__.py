"""Audio subsystem for speech input, speech-to-text, and speech synthesis."""

from app.audio.exceptions import AudioError, AudioInputError, TranscriptionError
from app.audio.input import AudioInput, BufferAudioInput, DevelopmentAudioInput
from app.audio.microphone import MicrophoneAudioInput
from app.audio.stt import DevelopmentSTTProvider, STTProvider
from app.audio.tts import DevelopmentTTSProvider, TTSProvider
from app.audio.whisper_stt import WhisperSTTProvider

__all__ = [
    "AudioError",
    "AudioInputError",
    "TranscriptionError",
    "AudioInput",
    "BufferAudioInput",
    "DevelopmentAudioInput",
    "MicrophoneAudioInput",
    "STTProvider",
    "DevelopmentSTTProvider",
    "WhisperSTTProvider",
    "TTSProvider",
    "DevelopmentTTSProvider",
]

