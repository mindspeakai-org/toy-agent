"""Unit tests for Phase 5: Text-to-Speech (Piper) and AudioOutput subsystem."""

import io
from pathlib import Path
import wave
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.audio.exceptions import AudioOutputError, SynthesisError, TTSError
from app.audio.output import BufferAudioOutput, SpeakerAudioOutput
from app.audio.tts import MockTTSProvider, PiperTTSProvider
from app.generation.base import AnswerGenerator
from app.generation.exceptions import GenerationError
from app.memory.base import MemoryContext
from app.memory.store import MemoryStore
from app.models.responses import AgentResponse
from app.router.mock import MockRouterClient
from app.router.models import ProcessingType, RoutingDecision


class FixedLocalGenerator(AnswerGenerator):
    """Deterministic local answer generator for testing."""

    def __init__(self, answer: str = "This is a deterministic local answer.") -> None:
        self.answer = answer
        self.call_count = 0

    @property
    def name(self) -> str:
        return "FixedLocalGenerator"

    async def generate(self, query: str, memory_context: MemoryContext | None = None) -> str:
        self.call_count += 1
        return self.answer


class FixedCloudGenerator(AnswerGenerator):
    """Deterministic cloud answer generator for testing."""

    def __init__(self, answer: str = "This is a deterministic cloud answer.") -> None:
        self.answer = answer
        self.call_count = 0

    @property
    def name(self) -> str:
        return "FixedCloudGenerator"

    async def generate(self, query: str, memory_context: MemoryContext | None = None) -> str:
        self.call_count += 1
        return self.answer


class FailingGenerator(AnswerGenerator):
    """Generator that always raises GenerationError."""

    @property
    def name(self) -> str:
        return "FailingGenerator"

    async def generate(self, query: str, memory_context: MemoryContext | None = None) -> str:
        raise GenerationError("Simulated LLM inference failure")


# ============================================================================
# 1. PiperTTSProvider & MockTTSProvider Unit Tests
# ============================================================================

def test_piper_tts_provider_initialization() -> None:
    """Test PiperTTSProvider metadata and lazy initial state."""
    provider = PiperTTSProvider()
    assert "PiperTTSProvider" in provider.name
    assert provider._voice is None
    assert provider.model_file == "en/en_US/lessac/medium/en_US-lessac-medium.onnx"


@pytest.mark.asyncio
async def test_piper_tts_provider_empty_input() -> None:
    """Test PiperTTSProvider returns empty bytes for empty or whitespace text."""
    provider = PiperTTSProvider()
    assert await provider.synthesize("") == b""
    assert await provider.synthesize("   \n\t  ") == b""


def test_piper_tts_provider_lazy_loading_and_warm_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PiperTTSProvider loads the voice model lazily on first access and reuses it."""
    load_calls = []

    class FakePiperVoice:
        pass

    def mock_load(model_path: str, config_path: str | None = None) -> FakePiperVoice:
        load_calls.append((model_path, config_path))
        return FakePiperVoice()

    monkeypatch.setattr("piper.voice.PiperVoice.load", mock_load)

    provider = PiperTTSProvider(model_path="/fake/model.onnx", config_path="/fake/config.json")
    assert provider._voice is None

    # First load
    voice1 = provider._load_voice()
    assert isinstance(voice1, FakePiperVoice)
    assert len(load_calls) == 1

    # Second load reuses existing warm instance without calling load again
    voice2 = provider._load_voice()
    assert voice2 is voice1
    assert len(load_calls) == 1


@pytest.mark.asyncio
async def test_piper_tts_provider_synthesizes_valid_wav_with_mocked_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test PiperTTSProvider _synthesize_sync produces valid 16-bit mono WAV audio."""

    class FakePiperVoice:
        def synthesize_wav(self, text: str, wav_file: wave.Wave_write, set_wav_format: bool = True) -> None:
            if set_wav_format:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
            # Write 220 samples (10ms of 22050Hz)
            wav_file.writeframes(b"\x00\x00" * 220)

    provider = PiperTTSProvider()
    provider._voice = FakePiperVoice()

    audio_bytes = await provider.synthesize("Hello child!")
    assert len(audio_bytes) > 44

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 220


@pytest.mark.asyncio
async def test_mock_tts_provider_records_calls_and_canned_audio() -> None:
    """Test MockTTSProvider records call history and returns valid canned WAV audio."""
    mock_tts = MockTTSProvider()
    assert mock_tts.call_count == 0

    audio = await mock_tts.synthesize("Test speech text")
    assert mock_tts.call_count == 1
    assert mock_tts.call_history == ["Test speech text"]
    assert len(audio) > 44

    # Empty text returns empty bytes
    empty_audio = await mock_tts.synthesize("")
    assert empty_audio == b""


# ============================================================================
# 2. AudioOutput & SpeakerAudioOutput Unit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_buffer_audio_output_captures_bytes() -> None:
    """Test BufferAudioOutput captures audio payloads deterministically in memory."""
    sink = BufferAudioOutput()
    assert sink.name == "BufferAudioOutput"
    assert sink.played_payloads == []

    await sink.play(b"sample_pcm_audio_bytes_1")
    await sink.play(b"sample_pcm_audio_bytes_2")

    assert len(sink.played_payloads) == 2
    assert sink.played_payloads[0] == b"sample_pcm_audio_bytes_1"
    assert sink.played_payloads[1] == b"sample_pcm_audio_bytes_2"

    sink.clear()
    assert sink.played_payloads == []


@pytest.mark.asyncio
async def test_buffer_audio_output_rejects_empty_payload() -> None:
    """Test BufferAudioOutput raises AudioOutputError on empty bytes."""
    sink = BufferAudioOutput()
    with pytest.raises(AudioOutputError):
        await sink.play(b"")


@pytest.mark.asyncio
async def test_speaker_audio_output_rejects_empty_payload() -> None:
    """Test SpeakerAudioOutput raises AudioOutputError on empty bytes."""
    speaker = SpeakerAudioOutput()
    with pytest.raises(AudioOutputError):
        await speaker.play(b"")


# ============================================================================
# 3. Phase 5 Orchestrator Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_phase5_local_answer_synthesizes_and_plays(tmp_path: Path) -> None:
    """Test Phase 5: answer_text -> TTS -> WAV bytes -> AudioOutput for LOCAL route."""
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "tell me a joke",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=False,
            memory_request=None,
        ),
    )

    mock_tts = MockTTSProvider()
    buffer_out = BufferAudioOutput()
    local_gen = FixedLocalGenerator("Why did the chicken cross the road?")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        local_generator=local_gen,
        tts_provider=mock_tts,
        audio_output=buffer_out,
    )

    response = await orchestrator.process("tell me a joke")

    assert response.success is True
    assert response.answer_text == "Why did the chicken cross the road?"
    # TTS invoked exactly once
    assert mock_tts.call_count == 1
    # TTS received strictly the final answer_text
    assert mock_tts.call_history[0] == "Why did the chicken cross the road?"
    # Response contains synthesized audio
    assert response.audio is not None
    assert len(response.audio) > 44
    # BufferAudioOutput captured the synthesized audio
    assert len(buffer_out.played_payloads) == 1
    assert buffer_out.played_payloads[0] == response.audio


@pytest.mark.asyncio
async def test_orchestrator_phase5_cloud_answer_synthesizes_and_plays(tmp_path: Path) -> None:
    """Test Phase 5: answer_text -> TTS -> AudioOutput for CLOUD route."""
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "what is the weather today?",
        RoutingDecision(
            processing=ProcessingType.CLOUD,
            memory_required=False,
            memory_request=None,
        ),
    )

    mock_tts = MockTTSProvider()
    buffer_out = BufferAudioOutput()
    cloud_gen = FixedCloudGenerator("The weather is sunny and 75 degrees.")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        cloud_generator=cloud_gen,
        tts_provider=mock_tts,
        audio_output=buffer_out,
    )

    response = await orchestrator.process("what is the weather today?")

    assert response.success is True
    assert response.answer_text == "The weather is sunny and 75 degrees."
    assert mock_tts.call_count == 1
    assert mock_tts.call_history[0] == "The weather is sunny and 75 degrees."
    assert response.audio is not None
    assert len(buffer_out.played_payloads) == 1
    assert buffer_out.played_payloads[0] == response.audio


@pytest.mark.asyncio
async def test_orchestrator_phase5_tts_receives_only_final_answer_text(tmp_path: Path) -> None:
    """Constraint 7: TTS must receive ONLY final answer_text.

    It must NOT receive the query, RoutingDecision, MemoryContext, or MemoryStore.
    """
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "what is my pet name?",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=True,
            memory_request={"keys": ["pet_name"]},
        ),
    )

    mem_store = MemoryStore(tmp_path / "mem.json")
    mem_store.set("pet_name", "Bubbles")

    mock_tts = MockTTSProvider()
    buffer_out = BufferAudioOutput()
    local_gen = FixedLocalGenerator("Your pet's name is Bubbles!")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=mem_store,
        local_generator=local_gen,
        tts_provider=mock_tts,
        audio_output=buffer_out,
    )

    response = await orchestrator.process("what is my pet name?")

    assert response.success is True
    assert response.answer_text == "Your pet's name is Bubbles!"
    # Verify strict parameter isolation
    assert mock_tts.call_count == 1
    received_tts_text = mock_tts.call_history[0]
    assert received_tts_text == "Your pet's name is Bubbles!"
    # Crucial: Must not be or contain raw query or JSON structures
    assert "what is my pet name?" not in received_tts_text
    assert "ProcessingType" not in received_tts_text
    assert "MemoryContext" not in received_tts_text


@pytest.mark.asyncio
async def test_orchestrator_phase5_generation_failure_skips_tts(tmp_path: Path) -> None:
    """When answer generation fails, TTS and AudioOutput must NOT be invoked."""
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "tell me something",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=False,
            memory_request=None,
        ),
    )

    mock_tts = MockTTSProvider()
    buffer_out = BufferAudioOutput()
    failing_gen = FailingGenerator()

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        local_generator=failing_gen,
        tts_provider=mock_tts,
        audio_output=buffer_out,
    )

    response = await orchestrator.process("tell me something")

    assert response.success is False
    assert response.intent == "GENERATION_FAILURE"
    assert response.audio is None
    # TTS was never called
    assert mock_tts.call_count == 0
    # No audio played
    assert len(buffer_out.played_payloads) == 0


@pytest.mark.asyncio
async def test_orchestrator_phase5_tts_failure_is_observable(tmp_path: Path) -> None:
    """When TTS synthesis fails, the error must be observable with success=False."""
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "hello",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=False,
            memory_request=None,
        ),
    )

    mock_tts = MockTTSProvider()
    mock_tts.inject_error(SynthesisError("Piper ONNX synthesis internal error"))
    buffer_out = BufferAudioOutput()
    local_gen = FixedLocalGenerator("Hello there!")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        local_generator=local_gen,
        tts_provider=mock_tts,
        audio_output=buffer_out,
    )

    response = await orchestrator.process("hello")

    assert response.success is False
    assert response.intent == "TTS_FAILURE"
    assert "[TTS FAILURE (SynthesisError)]" in response.text
    assert response.answer_text == "Hello there!"
    assert response.audio is None
    assert len(buffer_out.played_payloads) == 0


@pytest.mark.asyncio
async def test_orchestrator_phase5_audio_output_failure_is_observable(tmp_path: Path) -> None:
    """When AudioOutput fails, the error must be observable with success=False."""

    class FailingAudioOutput(BufferAudioOutput):
        async def play(self, audio_data: bytes) -> None:
            raise AudioOutputError("Default sound device is busy or disconnected")

    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "hello",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=False,
            memory_request=None,
        ),
    )

    mock_tts = MockTTSProvider()
    failing_out = FailingAudioOutput()
    local_gen = FixedLocalGenerator("Hello there!")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        local_generator=local_gen,
        tts_provider=mock_tts,
        audio_output=failing_out,
    )

    response = await orchestrator.process("hello")

    assert response.success is False
    assert response.intent == "AUDIO_OUTPUT_FAILURE"
    assert "[AUDIO OUTPUT FAILURE (AudioOutputError)]" in response.text
    assert response.answer_text == "Hello there!"
    assert response.audio is not None  # TTS succeeded, playback failed


@pytest.mark.asyncio
async def test_orchestrator_phase5_empty_input_skips_tts(tmp_path: Path) -> None:
    """When input is empty or whitespace, TTS must not be invoked."""
    mock_tts = MockTTSProvider()
    buffer_out = BufferAudioOutput()

    orchestrator = AgentOrchestrator(
        tts_provider=mock_tts,
        audio_output=buffer_out,
    )

    response = await orchestrator.process("   \t  ")

    assert response.success is True
    assert response.intent == "EMPTY_INPUT"
    assert mock_tts.call_count == 0
    assert len(buffer_out.played_payloads) == 0
