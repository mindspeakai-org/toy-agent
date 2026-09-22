"""Unit tests for the portable AudioInput, STTProvider, and TTSProvider interfaces."""

import io
import wave
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.audio.exceptions import AudioError, AudioInputError, TranscriptionError
from app.audio.input import BufferAudioInput, DevelopmentAudioInput
from app.audio.stt import DevelopmentSTTProvider, STTProvider
from app.audio.tts import DevelopmentTTSProvider
from app.memory.store import MemoryStore
from app.router.mock import MockRouterClient
from app.router.models import ProcessingType, RouteType, RoutingDecision


# --- AudioInput Tests ---

@pytest.mark.asyncio
async def test_buffer_audio_input_valid() -> None:
    """Test BufferAudioInput delivers valid audio payload."""
    audio_source = BufferAudioInput(b"sample_audio_stream")
    assert audio_source.name == "BufferAudioInput"
    data = await audio_source.read()
    assert data == b"sample_audio_stream"


@pytest.mark.asyncio
async def test_buffer_audio_input_empty_and_none() -> None:
    """Test BufferAudioInput rejects None and empty byte buffers with AudioInputError."""
    # None buffer
    empty_source = BufferAudioInput(None)
    with pytest.raises(AudioInputError):
        await empty_source.read()

    # Empty byte buffer
    empty_bytes_source = BufferAudioInput(b"")
    with pytest.raises(AudioInputError):
        await empty_bytes_source.read()


@pytest.mark.asyncio
async def test_development_audio_input_deterministic() -> None:
    """Test DevelopmentAudioInput provides deterministic payloads from string or bytes."""
    dev_input = DevelopmentAudioInput("what is my favorite animal")
    assert dev_input.name == "DevelopmentAudioInput"
    data = await dev_input.read()
    assert data == b"what is my favorite animal"

    # Dynamic update
    dev_input.set_payload("turn on the lights")
    assert await dev_input.read() == b"turn on the lights"

    # Rejection of empty/None payload
    dev_input.set_payload("")
    with pytest.raises(AudioInputError):
        await dev_input.read()

    dev_input.set_payload(None)
    with pytest.raises(AudioInputError):
        await dev_input.read()


# --- STTProvider & DevelopmentSTTProvider Tests ---

@pytest.mark.asyncio
async def test_development_stt_deterministic_transcription() -> None:
    """Test DevelopmentSTTProvider deterministic decoding and custom mappings."""
    stt = DevelopmentSTTProvider()

    # 1. Direct text in byte stream
    audio_payload = b"what is my favorite animal"
    text = await stt.transcribe(audio_payload)
    assert text == "what is my favorite animal"

    # 2. Registered exact byte mapping
    custom_bytes = b"\x01\x02\x03\x04\x05"
    stt.register_mapping(custom_bytes, "registered query mapping")
    text_mapped = await stt.transcribe(custom_bytes)
    assert text_mapped == "registered query mapping"

    # 3. Fallback for unrecognized non-utf8 binary
    unrecognized_binary = b"\xff\xfe\x00\x00\x80\x00"
    fallback_text = await stt.transcribe(unrecognized_binary)
    assert fallback_text == "Hello"


@pytest.mark.asyncio
async def test_development_stt_none_and_empty_rejected() -> None:
    """Test STT validation rejects None, empty, or invalid type payloads."""
    stt = DevelopmentSTTProvider()

    # None payload
    with pytest.raises(AudioInputError) as exc_none:
        await stt.transcribe(None)  # type: ignore[arg-type]
    assert "cannot be None" in str(exc_none.value)

    # Empty bytes
    with pytest.raises(AudioInputError) as exc_empty:
        await stt.transcribe(b"")
    assert "cannot be empty" in str(exc_empty.value)

    # Invalid type
    with pytest.raises(AudioInputError) as exc_type:
        await stt.transcribe(12345)  # type: ignore[arg-type]
    assert "Expected bytes" in str(exc_type.value)


@pytest.mark.asyncio
async def test_development_stt_malformed_corrupt_payload_rejected() -> None:
    """Test malformed or corrupted audio payloads raise TranscriptionError."""
    stt = DevelopmentSTTProvider()

    # Explicit corrupted payload sentinel
    corrupt_audio = b"CORRUPT_AUDIO_PAYLOAD\x00\x01\x02"
    with pytest.raises(TranscriptionError) as exc_corrupt:
        await stt.transcribe(corrupt_audio)
    assert "corrupted audio" in str(exc_corrupt.value).lower()

    # Truncated WAV header
    truncated_wav = b"RIFF\x00\x00"
    with pytest.raises(TranscriptionError) as exc_wav:
        await stt.transcribe(truncated_wav)
    assert "truncated wav" in str(exc_wav.value).lower()

    # Strict mode failure when binary data has no mapping and no fallback
    strict_stt = DevelopmentSTTProvider(fallback_text=None, strict=True)
    with pytest.raises(TranscriptionError):
        await strict_stt.transcribe(b"\xfe\xff\x80\x81")


def test_audio_subsystem_no_mac_native_imports() -> None:
    """Verify that app/audio does NOT import any Mac-native or platform-specific libraries."""
    import inspect
    import app.audio
    import app.audio.exceptions
    import app.audio.input
    import app.audio.stt
    import app.audio.tts

    mac_prohibited = [
        "appkit",
        "avfoundation",
        "coreaudio",
        "pyobjc",
        "speech",
        "speech_recognition",
        "pyaudio",
        "sounddevice",
    ]

    modules = [app.audio, app.audio.exceptions, app.audio.input, app.audio.stt, app.audio.tts]
    for mod in modules:
        source = inspect.getsource(mod).lower()
        for prohibited in mac_prohibited:
            assert f"import {prohibited}" not in source, f"Found prohibited import '{prohibited}' in {mod.__name__}"
            assert f"from {prohibited}" not in source, f"Found prohibited import '{prohibited}' in {mod.__name__}"


# --- Orchestrator STT Integration Tests ---

@pytest.mark.asyncio
async def test_orchestrator_transcribe_audio_method() -> None:
    """Test AgentOrchestrator.transcribe_audio exposes the clean STT step."""
    stt = DevelopmentSTTProvider()
    orchestrator = AgentOrchestrator(stt_provider=stt)
    transcribed = await orchestrator.transcribe_audio(b"what is my favorite animal")
    assert transcribed == "what is my favorite animal"


# --- Existing Voice Pipeline & TTS Tests ---

@pytest.mark.asyncio
async def test_development_tts_synthesis() -> None:
    """Test DevelopmentTTSProvider generates valid standard WAV audio bytes."""
    tts = DevelopmentTTSProvider(sample_rate=16000)

    # Empty text
    assert await tts.synthesize("") == b""
    assert await tts.synthesize("   ") == b""

    # Non-empty text
    wav_bytes = await tts.synthesize("Hello there, friend!")
    assert len(wav_bytes) > 44  # WAV header is at least 44 bytes

    # Verify standard WAV header using Python standard library wave module
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() > 0


@pytest.mark.asyncio
async def test_orchestrator_process_voice(tmp_path) -> None:
    """Test end-to-end voice pipeline: Audio In -> STT -> Router -> Handler -> TTS -> Audio Out."""
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "turn on the lights",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=False,
            memory_request=None,
        ),
    )

    stt = DevelopmentSTTProvider()
    tts = DevelopmentTTSProvider()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        stt_provider=stt,
        tts_provider=tts,
    )

    audio_in = b"turn on the lights"
    response, audio_out = await orchestrator.process_voice(audio_in)

    assert response.success is True
    assert response.processing == ProcessingType.LOCAL
    assert response.decision is not None
    assert len(audio_out) > 44

    # Verify voice telemetry
    assert "voice" in response.metadata
    voice_meta = response.metadata["voice"]
    assert voice_meta["stt_provider"] == "DevelopmentSTTProvider"
    assert voice_meta["tts_provider"] == "DevelopmentTTSProvider"
    assert voice_meta["stt_latency_s"] >= 0.0
    assert voice_meta["tts_latency_s"] >= 0.0
    assert voice_meta["audio_output_bytes"] == len(audio_out)
