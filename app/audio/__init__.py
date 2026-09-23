"""Audio subsystem for speech input, speech-to-text, and speech synthesis."""

from app.audio.exceptions import (
    AudioError,
    AudioInputError,
    AudioOutputError,
    SynthesisError,
    TranscriptionError,
    TTSError,
)
from app.audio.input import AudioInput, BufferAudioInput, DevelopmentAudioInput
from app.audio.microphone import MicrophoneAudioInput
from app.audio.output import AudioOutput, BufferAudioOutput, SpeakerAudioOutput
from app.audio.stt import DevelopmentSTTProvider, STTProvider
from app.audio.tts import DevelopmentTTSProvider, MockTTSProvider, PiperTTSProvider, TTSProvider
from app.audio.whisper_stt import WhisperSTTProvider

__all__ = [
    "AudioError",
    "AudioInputError",
    "AudioOutputError",
    "TTSError",
    "SynthesisError",
    "TranscriptionError",
    "AudioInput",
    "BufferAudioInput",
    "DevelopmentAudioInput",
    "MicrophoneAudioInput",
    "AudioOutput",
    "BufferAudioOutput",
    "SpeakerAudioOutput",
    "STTProvider",
    "DevelopmentSTTProvider",
    "WhisperSTTProvider",
    "TTSProvider",
    "DevelopmentTTSProvider",
    "PiperTTSProvider",
    "MockTTSProvider",
]


