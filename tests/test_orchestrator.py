"""Unit and integration tests for AgentOrchestrator under the Phase 2 routing boundary."""

from pathlib import Path
import httpx
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.generation.base import MockAnswerGenerator
from app.generation.cloud import MockCloudAnswerGenerator
from app.generation.exceptions import CloudConfigurationError, LocalGenerationError
from app.memory.store import MemoryStore
from app.router.client import RouterClient
from app.router.exceptions import (
    RouterConnectionError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.mock import MockRouterClient
from app.router.models import MemoryRequest, ProcessingType, RoutingDecision


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_local_no_memory(tmp_path: Path) -> None:
    """Test text reaches router and LOCAL decision without memory is handled locally."""
    mock_router = MockRouterClient()
    mock_local = MockAnswerGenerator(canned_response="Why did the teddy bear say no to dessert? It was stuffed!")
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        local_generator=mock_local,
    )

    res = await orchestrator.process("Tell me a joke")
    assert res.success is True
    assert res.processing == ProcessingType.LOCAL
    assert res.decision is not None
    assert res.decision.processing == ProcessingType.LOCAL
    assert res.decision.memory_required is False
    assert res.decision.memory_request is None
    assert res.handler == "LocalAnswerGenerator"
    assert res.answer_text == "Why did the teddy bear say no to dessert? It was stuffed!"
    assert res.memory_context is None


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_local_with_memory(tmp_path: Path) -> None:
    """Test LOCAL decision with required memory retrieves memory and produces local answer."""
    mock_router = MockRouterClient()
    mock_local = MockAnswerGenerator(canned_response="Your favorite animal is a tiger!")
    store = MemoryStore(tmp_path / "mem.json")
    store.set("favorite_animal", "tiger")
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
        local_generator=mock_local,
    )

    res = await orchestrator.process("What is my favorite animal?")
    assert res.success is True
    assert res.processing == ProcessingType.LOCAL
    assert res.decision is not None
    assert res.decision.memory_required is True
    assert res.decision.memory_request is not None
    assert res.decision.memory_request.keys == ["favorite_animal"]
    assert res.memory_context is not None
    assert res.memory_context.hits == {"favorite_animal": "tiger"}
    assert res.answer_text == "Your favorite animal is a tiger!"


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_multiple_memories(tmp_path: Path) -> None:
    """Test LOCAL decision with multiple required memory keys retrieves both and generates answer."""
    mock_router = MockRouterClient()
    mock_local = MockAnswerGenerator(canned_response="Your name is Alex and you love tigers!")
    store = MemoryStore(tmp_path / "mem.json")
    store.set("child_name", "Alex")
    store.set("favorite_animal", "tiger")
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
        local_generator=mock_local,
    )

    res = await orchestrator.process("What is my name and favorite animal?")
    assert res.success is True
    assert res.decision is not None
    assert res.decision.memory_required is True
    assert res.decision.memory_request is not None
    assert res.decision.memory_request.keys == ["child_name", "favorite_animal"]
    assert res.memory_context is not None
    assert res.memory_context.hits == {"child_name": "Alex", "favorite_animal": "tiger"}
    assert res.answer_text == "Your name is Alex and you love tigers!"


@pytest.mark.asyncio
async def test_orchestrator_preserves_routing_decision_cloud(tmp_path: Path) -> None:
    """Test CLOUD decision is processed via CloudAnswerGenerator without memory access."""
    mock_router = MockRouterClient()
    mock_cloud = MockCloudAnswerGenerator(canned_response="The weather is 72 degrees and sunny.")
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        cloud_generator=mock_cloud,
    )

    res = await orchestrator.process("What is the weather today?")
    assert res.success is True
    assert res.processing == ProcessingType.CLOUD
    assert res.decision is not None
    assert res.decision.processing == ProcessingType.CLOUD
    assert res.decision.memory_required is False
    assert res.memory_context is None
    assert res.handler == "CloudAnswerGenerator"
    assert res.answer_text == "The weather is 72 degrees and sunny."
    assert mock_cloud.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_empty_input(tmp_path: Path) -> None:
    """Test empty string handling short-circuits gracefully."""
    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("   ")
    assert "didn't hear anything" in res.text.lower()
    assert res.success is True


@pytest.mark.asyncio
async def test_orchestrator_resilience_router_timeout(tmp_path: Path) -> None:
    """Test graceful fallback on router timeout."""
    mock_router = MockRouterClient()
    mock_router.inject_error(RouterTimeoutError("Gateway timed out"))
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Any query")
    assert res.success is False
    assert res.intent == "ROUTER_TIMEOUT"
    assert "taking a little too long" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_resilience_router_connection_error(tmp_path: Path) -> None:
    """Test graceful fallback on router connection refusal."""
    mock_router = MockRouterClient()
    mock_router.inject_error(RouterConnectionError("Connection refused"))
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Any query")
    assert res.success is False
    assert res.intent == "ROUTER_UNAVAILABLE"
    assert "trouble connecting" in res.text.lower()


@pytest.mark.asyncio
async def test_orchestrator_resilience_router_response_error(tmp_path: Path) -> None:
    """Test graceful fallback on malformed response."""
    mock_router = MockRouterClient()
    mock_router.inject_error(RouterResponseError("Malformed JSON"))
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
    )

    res = await orchestrator.process("Any query")
    assert res.success is False
    assert res.intent == "ROUTER_MALFORMED_RESPONSE"
    assert "trouble understanding" in res.text.lower()


# --- Live Integration Test ---

@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_slm_router_integration() -> None:
    """Live integration test connecting to active slm-router on port 8008.

    Validates that real router output adheres to the Phase 2 contract:
    {
      "processing": "LOCAL" | "CLOUD",
      "memory_required": bool,
      "memory_request": {"keys": [...]} | null
    }
    """
    from app.config.settings import get_settings
    settings = get_settings()
    health_url = f"{settings.router_base_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(health_url)
            if resp.status_code != 200:
                pytest.skip(f"Service at {health_url} is not healthy (HTTP {resp.status_code})")
    except Exception:
        pytest.skip(f"External slm-router service is not reachable at {health_url}")

    real_router = RouterClient(
        base_url=settings.router_base_url,
        endpoint=settings.router_route_endpoint,
        timeout=30.0,
    )
    orchestrator = AgentOrchestrator(router_client=real_router)

    try:
        # Test 1: Turn on the lights -> LOCAL, memory_required=False
        res_cmd = await orchestrator.process("Turn on the lights")
        assert res_cmd.success is True
        assert res_cmd.decision is not None
        assert res_cmd.decision.processing == ProcessingType.LOCAL
        assert res_cmd.decision.memory_required is False
        assert res_cmd.decision.memory_request is None

        # Test 2: What is my favorite animal? -> LOCAL, memory_required=True
        res_mem = await orchestrator.process("What is my favorite animal?")
        assert res_mem.success is True
        assert res_mem.decision is not None
        assert res_mem.decision.processing == ProcessingType.LOCAL
        assert res_mem.decision.memory_required is True
        assert res_mem.decision.memory_request is not None
        assert "favorite_animal" in res_mem.decision.memory_request.keys

    finally:
        await orchestrator.aclose()


# --- Voice -> STT -> Router Integration Unit Tests ---

@pytest.mark.asyncio
async def test_orchestrator_route_audio_local_with_memory() -> None:
    """Test deterministic route_audio transcribes bytes and routes to LOCAL with memory."""
    from app.audio.stt import DevelopmentSTTProvider

    mock_router = MockRouterClient()
    stt = DevelopmentSTTProvider()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        stt_provider=stt,
    )

    audio_bytes = b"What is my favorite animal?"
    text, decision = await orchestrator.route_audio(audio_bytes)

    assert text == "What is my favorite animal?"
    assert decision.processing == ProcessingType.LOCAL
    assert decision.memory_required is True
    assert decision.memory_request is not None
    assert decision.memory_request.keys == ["favorite_animal"]


@pytest.mark.asyncio
async def test_orchestrator_route_audio_cloud() -> None:
    """Test deterministic route_audio routes weather queries to CLOUD."""
    from app.audio.stt import DevelopmentSTTProvider

    mock_router = MockRouterClient()
    stt = DevelopmentSTTProvider()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        stt_provider=stt,
    )

    audio_bytes = b"What is the weather today?"
    text, decision = await orchestrator.route_audio(audio_bytes)

    assert text == "What is the weather today?"
    assert decision.processing == ProcessingType.CLOUD
    assert decision.memory_required is False
    assert decision.memory_request is None


@pytest.mark.asyncio
async def test_orchestrator_route_voice_configured_source() -> None:
    """Test route_voice captures from configured AudioInput and returns RoutingDecision."""
    from app.audio.input import BufferAudioInput
    from app.audio.stt import DevelopmentSTTProvider

    mock_router = MockRouterClient()
    stt = DevelopmentSTTProvider()
    audio_input = BufferAudioInput(b"Tell me a joke.")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        stt_provider=stt,
        audio_input=audio_input,
    )

    text, decision = await orchestrator.route_voice()
    assert text == "Tell me a joke."
    assert decision.processing == ProcessingType.LOCAL
    assert decision.memory_required is False


@pytest.mark.asyncio
async def test_orchestrator_route_voice_missing_audio_input() -> None:
    """Test route_voice raises AudioInputError when no AudioInput is provided or configured."""
    from app.audio.exceptions import AudioInputError
    from app.audio.stt import DevelopmentSTTProvider

    mock_router = MockRouterClient()
    stt = DevelopmentSTTProvider()

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        stt_provider=stt,
        audio_input=None,
    )

    with pytest.raises(AudioInputError) as exc:
        await orchestrator.route_voice()
    assert "No audio input source provided" in str(exc.value)


@pytest.mark.asyncio
async def test_orchestrator_route_voice_stt_transcription_error() -> None:
    """Test route_voice propagates TranscriptionError when audio data is corrupted."""
    from app.audio.exceptions import TranscriptionError
    from app.audio.input import BufferAudioInput
    from app.audio.stt import DevelopmentSTTProvider

    mock_router = MockRouterClient()
    stt = DevelopmentSTTProvider()
    audio_input = BufferAudioInput(b"CORRUPT_AUDIO_PAYLOAD\x00\x01")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        stt_provider=stt,
        audio_input=audio_input,
    )

    with pytest.raises(TranscriptionError):
        await orchestrator.route_voice()


# --- Phase 4 Runtime Branch Tests ---

@pytest.mark.asyncio
async def test_phase4_branch_a_local_with_memory(tmp_path: Path) -> None:
    """Branch A: LOCAL + MEMORY -> MemoryStore -> LocalAnswerGenerator -> answer_text."""
    mock_router = MockRouterClient()
    mock_local = MockAnswerGenerator(canned_response="You love tigers!")
    mock_cloud = MockCloudAnswerGenerator(canned_response="Should not be called")

    store = MemoryStore(tmp_path / "mem.json")
    store.set("favorite_animal", "tiger")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
        local_generator=mock_local,
        cloud_generator=mock_cloud,
    )

    res = await orchestrator.process("What is my favorite animal?")

    assert res.success is True
    assert res.processing == ProcessingType.LOCAL
    assert res.decision.memory_required is True

    # 1. MemoryStore was accessed and resolved
    assert res.memory_context is not None
    assert res.memory_context.hits == {"favorite_animal": "tiger"}
    assert res.memory_context.is_complete is True

    # 2. LocalAnswerGenerator received query and MemoryContext
    assert len(mock_local.call_history) == 1
    assert mock_local.call_history[0]["query"] == "What is my favorite animal?"
    assert mock_local.call_history[0]["memory_context"] == res.memory_context

    # 3. CloudAnswerGenerator was NOT called
    assert mock_cloud.call_count == 0

    # 4. answer_text is returned
    assert res.answer_text == "You love tigers!"
    assert res.text == "You love tigers!"
    assert res.handler == "LocalAnswerGenerator"


@pytest.mark.asyncio
async def test_phase4_branch_b_local_no_memory(tmp_path: Path) -> None:
    """Branch B: LOCAL + NO MEMORY -> LocalAnswerGenerator -> answer_text."""
    mock_router = MockRouterClient()
    mock_local = MockAnswerGenerator(canned_response="Why did the teddy bear say no? It was stuffed!")
    mock_cloud = MockCloudAnswerGenerator(canned_response="Should not be called")

    store = MemoryStore(tmp_path / "mem.json")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
        local_generator=mock_local,
        cloud_generator=mock_cloud,
    )

    res = await orchestrator.process("Tell me a joke")

    assert res.success is True
    assert res.processing == ProcessingType.LOCAL
    assert res.decision.memory_required is False

    # 1. MemoryStore was NOT accessed
    assert res.memory_context is None

    # 2. LocalAnswerGenerator was called with memory_context=None
    assert len(mock_local.call_history) == 1
    assert mock_local.call_history[0]["query"] == "Tell me a joke"
    assert mock_local.call_history[0]["memory_context"] is None

    # 3. CloudAnswerGenerator was NOT called
    assert mock_cloud.call_count == 0

    # 4. answer_text is returned
    assert res.answer_text == "Why did the teddy bear say no? It was stuffed!"
    assert res.handler == "LocalAnswerGenerator"


@pytest.mark.asyncio
async def test_phase4_branch_c_cloud(tmp_path: Path) -> None:
    """Branch C: CLOUD -> CloudAnswerGenerator -> answer_text (zero real Gemini calls)."""
    mock_router = MockRouterClient()
    mock_local = MockAnswerGenerator(canned_response="Should not be called")
    mock_cloud = MockCloudAnswerGenerator(canned_response="Mock cloud response: It's sunny outside!")

    store = MemoryStore(tmp_path / "mem.json")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
        local_generator=mock_local,
        cloud_generator=mock_cloud,
    )

    res = await orchestrator.process("What is the weather today?")

    assert res.success is True
    assert res.processing == ProcessingType.CLOUD
    assert res.decision.memory_required is False

    # 1. CloudAnswerGenerator was called
    assert mock_cloud.call_count == 1
    assert mock_cloud.last_query == "What is the weather today?"

    # 2. LocalAnswerGenerator was NOT called
    assert len(mock_local.call_history) == 0

    # 3. MemoryStore was NOT accessed
    assert res.memory_context is None

    # 4. answer_text is returned
    assert res.answer_text == "Mock cloud response: It's sunny outside!"
    assert res.handler == "CloudAnswerGenerator"


@pytest.mark.asyncio
async def test_phase4_local_generation_failure_observability(tmp_path: Path) -> None:
    """Verify local generation failure is observable and returns success=False without fake child text."""
    class FailingLocalGenerator(MockAnswerGenerator):
        async def generate(self, query: str, memory_context=None) -> str:
            raise LocalGenerationError("Simulated on-device model OOM error")

    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        local_generator=FailingLocalGenerator(),
    )

    res = await orchestrator.process("Tell me a joke")

    assert res.success is False
    assert res.intent == "GENERATION_FAILURE"
    assert res.answer_text is None
    assert "[LOCAL GENERATION FAILURE (LocalGenerationError)]" in res.text
    assert "Simulated on-device model OOM error" in res.text
    assert res.metadata["error_type"] == "LocalGenerationError"


@pytest.mark.asyncio
async def test_phase4_cloud_generation_failure_observability(tmp_path: Path) -> None:
    """Verify cloud generation failure is observable and returns success=False without fake child text."""
    class FailingCloudGenerator(MockAnswerGenerator):
        async def generate(self, query: str, memory_context=None) -> str:
            raise CloudConfigurationError("GEMINI_API_KEY is not configured")

    mock_router = MockRouterClient()
    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=MemoryStore(tmp_path / "mem.json"),
        cloud_generator=FailingCloudGenerator(),
    )

    res = await orchestrator.process("What is the weather today?")

    assert res.success is False
    assert res.intent == "GENERATION_FAILURE"
    assert res.answer_text is None
    assert "[CLOUD GENERATION FAILURE (CloudConfigurationError)]" in res.text
    assert "GEMINI_API_KEY is not configured" in res.text
    assert res.metadata["error_type"] == "CloudConfigurationError"


@pytest.mark.asyncio
async def test_phase4_branch_a_multiple_memories(tmp_path: Path) -> None:
    """Verify multiple requested memory keys are resolved and supplied to local generator."""
    mock_router = MockRouterClient()
    mock_local = MockAnswerGenerator(canned_response="Your name is Alex and you love tigers!")
    mock_cloud = MockCloudAnswerGenerator(canned_response="Should not be called")

    store = MemoryStore(tmp_path / "mem.json")
    store.set("child_name", "Alex")
    store.set("favorite_animal", "tiger")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
        local_generator=mock_local,
        cloud_generator=mock_cloud,
    )

    res = await orchestrator.process("What is my name and favorite animal?")
    assert res.success is True
    assert res.memory_context is not None
    assert res.memory_context.hits == {"child_name": "Alex", "favorite_animal": "tiger"}
    assert res.memory_context.is_complete is True
    assert mock_local.call_history[0]["memory_context"] == res.memory_context
    assert res.answer_text == "Your name is Alex and you love tigers!"


@pytest.mark.asyncio
async def test_phase4_branch_a_partial_missing_memory(tmp_path: Path) -> None:
    """Verify partial memory hit supplies hits and misses to local generator."""
    mock_router = MockRouterClient()
    mock_router.set_mock_response(
        "What is my favorite animal and song?",
        RoutingDecision(
            processing=ProcessingType.LOCAL,
            memory_required=True,
            memory_request=MemoryRequest(keys=["favorite_animal", "favorite_song"]),
        ),
    )
    mock_local = MockAnswerGenerator(
        canned_response="Your favorite animal is tiger, but I don't know your favorite song yet."
    )
    store = MemoryStore(tmp_path / "mem.json")
    store.set("favorite_animal", "tiger")

    orchestrator = AgentOrchestrator(
        router_client=mock_router,
        memory_store=store,
        local_generator=mock_local,
    )

    res = await orchestrator.process("What is my favorite animal and song?")
    assert res.success is True
    assert res.memory_context is not None
    assert res.memory_context.hits == {"favorite_animal": "tiger"}
    assert "favorite_song" in res.memory_context.misses
    assert res.memory_context.is_complete is False
    assert mock_local.call_history[0]["memory_context"] == res.memory_context
    assert "tiger" in res.answer_text

