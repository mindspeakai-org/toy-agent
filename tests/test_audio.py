"""Unit tests for the portable STT and TTS interfaces and development providers."""

import io
import wave
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.audio.stt import DevelopmentSTTProvider
from app.audio.tts import DevelopmentTTSProvider
from app.memory.store import MemoryStore
from app.router.mock import MockRouterClient
from app.router.models import RouteType, RoutingDecision


@pytest.mark.asyncio
async def test_development_stt_transcription() -> None:
    """Test DevelopmentSTTProvider with text and fallback bytes."""
    stt = DevelopmentSTTProvider()

    # Empty payload
    assert await stt.transcribe(b"") == ""

    # Text embedded in byte stream
    audio_payload = b"Turn on the lights."
    text = await stt.transcribe(audio_payload)
    assert text == "Turn on the lights."

    # Binary non-utf8 payload falls back gracefully
    binary_audio = b"\xff\xfe\x00\x00\x80\x00"
    fallback_text = await stt.transcribe(binary_audio)
    assert fallback_text == "Hello"


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
            route=RouteType.COMMAND,
            intent="DEVICE_ACTION",
            raw_response={"response": "The lights have been turned on."},
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
    assert response.route == RouteType.COMMAND
    assert "lights" in response.text.lower()
    assert len(audio_out) > 44

    # Verify voice telemetry
    assert "voice" in response.metadata
    voice_meta = response.metadata["voice"]
    assert voice_meta["stt_provider"] == "DevelopmentSTTProvider"
    assert voice_meta["tts_provider"] == "DevelopmentTTSProvider"
    assert voice_meta["stt_latency_s"] >= 0.0
    assert voice_meta["tts_latency_s"] >= 0.0
    assert voice_meta["audio_output_bytes"] == len(audio_out)
