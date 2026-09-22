"""Central agent orchestrator coordinating routing, memory, and handlers."""

import logging
import time
from typing import Optional, Union

from app.audio.exceptions import AudioInputError
from app.audio.input import AudioInput
from app.audio.stt import DevelopmentSTTProvider, STTProvider
from app.audio.tts import DevelopmentTTSProvider, TTSProvider
from app.cloud.client import CloudClient
from app.handlers.cloud import CloudHandler
from app.handlers.command import CommandHandler
from app.handlers.local import LocalHandler
from app.handlers.memory import MemoryHandler
from app.memory.store import MemoryStore
from app.models.responses import AgentResponse
from app.router.client import RouterClient
from app.router.exceptions import (
    RouterConnectionError,
    RouterError,
    RouterResponseError,
    RouterTimeoutError,
)
from app.router.mock import MockRouterClient
from app.router.models import ProcessingType, RouteType, RoutingDecision

logger = logging.getLogger("toy_agent.orchestrator")


class AgentOrchestrator:
    """Central orchestrator for the AI Toy Agent.

    In Phase 2, coordinates:
    Audio Input (optional) -> STT -> Text -> RouterClient -> RoutingDecision.
    Downstream generation (memory retrieval, local/cloud model execution) is
    deferred to subsequent phases.
    """

    def __init__(
        self,
        router_client: Optional[Union[RouterClient, MockRouterClient]] = None,
        memory_store: Optional[MemoryStore] = None,
        local_handler: Optional[LocalHandler] = None,
        memory_handler: Optional[MemoryHandler] = None,
        command_handler: Optional[CommandHandler] = None,
        cloud_handler: Optional[CloudHandler] = None,
        stt_provider: Optional[STTProvider] = None,
        tts_provider: Optional[TTSProvider] = None,
        audio_input: Optional[AudioInput] = None,
    ) -> None:
        self.router = router_client or RouterClient()
        self.memory = memory_store or MemoryStore()

        self.local_handler = local_handler or LocalHandler()
        self.memory_handler = memory_handler or MemoryHandler(self.memory)
        self.command_handler = command_handler or CommandHandler()
        self.cloud_handler = cloud_handler or CloudHandler(CloudClient())

        self.stt = stt_provider or DevelopmentSTTProvider()
        self.tts = tts_provider or DevelopmentTTSProvider()
        self.audio_input = audio_input

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Convert input audio data to text query using the configured STTProvider.

        Args:
            audio_data: Raw input audio bytes.

        Returns:
            str: Transcribed text query.
        """
        return await self.stt.transcribe(audio_data)

    async def route_audio(self, audio_data: bytes) -> tuple[str, RoutingDecision]:
        """Convert audio bytes to text via STTProvider and route to SLM-Router.

        Terminates strictly at the RoutingDecision without memory retrieval,
        response generation, or TTS execution.

        Args:
            audio_data: Raw input audio bytes.

        Returns:
            tuple[str, RoutingDecision]: Transcribed query text and structured routing decision.
        """
        transcribed_text = await self.transcribe_audio(audio_data)
        logger.info("[VOICE->ROUTER] Recognized query: '%s'", transcribed_text)
        decision = await self.router.route(transcribed_text)
        logger.info("[VOICE->ROUTER] Decision: %s (memory_required=%s)", decision.processing.value, decision.memory_required)
        return transcribed_text, decision

    async def route_voice(
        self,
        audio_input: Optional[AudioInput] = None,
        duration: Optional[float] = None,
    ) -> tuple[str, RoutingDecision]:
        """Capture audio from an AudioInput source, transcribe via STT, and route to SLM-Router.

        Args:
            audio_input: Optional AudioInput instance (defaults to self.audio_input if configured).
            duration: Optional duration in seconds for recording.

        Returns:
            tuple[str, RoutingDecision]: Transcribed query text and structured routing decision.

        Raises:
            AudioInputError: If no audio input source is provided or reading fails.
        """
        source = audio_input or self.audio_input
        if source is None:
            raise AudioInputError("No audio input source provided or configured in orchestrator")

        try:
            audio_bytes = await source.read(duration=duration)
        except TypeError:
            audio_bytes = await source.read()

        return await self.route_audio(audio_bytes)

    async def process_voice(self, audio_data: bytes) -> tuple[AgentResponse, bytes]:
        """Process incoming voice audio through the Phase 1 audio pipeline and routing.

        Workflow:
            Audio Input -> STT -> AgentOrchestrator -> RouterClient -> RoutingDecision -> TTS -> Audio Output

        Args:
            audio_data: Raw input audio bytes.

        Returns:
            tuple[AgentResponse, bytes]: The text response object and synthesized output audio bytes.
        """
        t_voice_start = time.perf_counter()

        # Step 1: Speech-to-Text
        transcribed_text = await self.transcribe_audio(audio_data)
        stt_duration = round(time.perf_counter() - t_voice_start, 4)
        logger.info("[VOICE STT] Transcribed %d audio bytes -> '%s' (%.3fs)", len(audio_data), transcribed_text, stt_duration)

        # Step 2: Text Routing Process
        response = await self.process(transcribed_text)

        # Step 3: Text-to-Speech synthesis
        t_tts_start = time.perf_counter()
        audio_output = await self.tts.synthesize(response.text)
        tts_duration = round(time.perf_counter() - t_tts_start, 4)
        logger.info("[VOICE TTS] Synthesized %d chars -> %d audio bytes (%.3fs)", len(response.text), len(audio_output), tts_duration)

        response.metadata["voice"] = {
            "stt_provider": self.stt.name,
            "tts_provider": self.tts.name,
            "stt_latency_s": stt_duration,
            "tts_latency_s": tts_duration,
            "audio_output_bytes": len(audio_output),
        }

        return response, audio_output

    async def process(self, text: str) -> AgentResponse:
        """Process user text through RouterClient to produce a structured RoutingDecision.

        Guarantees that a child-safe AgentResponse is returned even in the event
        of network timeouts, router connection drops, or malformed data.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return AgentResponse(
                text="I didn't hear anything! What's on your mind?",
                processing=ProcessingType.LOCAL,
                route=RouteType.UNKNOWN,
                handler="Orchestrator",
                intent="EMPTY_INPUT",
                metadata={},
                success=True,
            )

        t_e2e_start = time.perf_counter()

        logger.info("================ PIPELINE START ================")
        logger.info("[TOY_AGENT INPUT]  : %s", cleaned_text)
        target_url = getattr(self.router, "target_url", "internal_mock")
        logger.info("[ROUTER REQUEST]   : POST %s", target_url)

        # Step 1: Query Router
        try:
            decision = await self.router.route(cleaned_text)
        except RouterTimeoutError as exc:
            logger.error("ROUTER_ERROR: Timeout occurred - %s", exc)
            return self._build_fallback(
                "I'm taking a little too long to think right now. Could you ask me again?",
                reason="ROUTER_TIMEOUT",
                error=str(exc),
            )
        except RouterConnectionError as exc:
            logger.error("ROUTER_ERROR: Connection failed - %s", exc)
            return self._build_fallback(
                "I'm having trouble connecting right now. Let's try again in a moment!",
                reason="ROUTER_UNAVAILABLE",
                error=str(exc),
            )
        except RouterResponseError as exc:
            logger.error("ROUTER_ERROR: Malformed or error response - %s", exc)
            return self._build_fallback(
                "I had trouble understanding that. Let's try something else!",
                reason="ROUTER_MALFORMED_RESPONSE",
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("ROUTER_ERROR: Unexpected error - %s", exc)
            return self._build_fallback(
                "I'm having trouble understanding that right now.",
                reason="UNEXPECTED_ROUTER_ERROR",
                error=str(exc),
            )

        # Step 2: Log Routing Decision & Metadata
        logger.info("[ROUTING DECISION] : Processing=%s | MemoryRequired=%s", decision.processing.value, decision.memory_required)
        if decision.memory_request:
            logger.info("[MEMORY KEYS]      : %s", decision.memory_request.keys)

        e2e_total_latency = round(time.perf_counter() - t_e2e_start, 4)

        # Telemetry metrics collection
        timing_metrics = {
            "http_latency_s": decision.http_latency,
            "router_total_s": decision.timings.get("total"),
            "e2e_total_s": e2e_total_latency,
        }

        # In Phase 2, the pipeline terminates at RoutingDecision (no downstream generation).
        # We wrap the decision in AgentResponse for consumer inspection.
        response = AgentResponse(
            text=f"Routing decision: {decision.processing.value}",
            processing=decision.processing,
            decision=decision,
            route=RouteType(decision.processing.value),
            handler="SLMRouter",
            metadata={
                "timings": timing_metrics,
                "raw_response": decision.raw_response,
                "memory_required": decision.memory_required,
                "memory_keys": decision.memory_request.keys if decision.memory_request else [],
            },
            success=True,
        )

        logger.info(
            "[LATENCY METRICS]  : HTTP=%.3fs | E2E_Tot=%.3fs",
            decision.http_latency or 0.0,
            e2e_total_latency,
        )
        logger.info("[DECISION SUMMARY] : %s (memory_required=%s)", decision.processing.value, decision.memory_required)
        logger.info("================= PIPELINE END =================")
        return response

    def _build_fallback(self, message: str, reason: str, error: str) -> AgentResponse:
        """Construct child-safe fallback response while preserving debug details in metadata."""
        logger.info("FALLBACK_RESPONSE: %s (Reason: %s)", message, reason)
        logger.info("================= PIPELINE END =================")
        return AgentResponse(
            text=message,
            processing=ProcessingType.LOCAL,
            decision=None,
            route=RouteType.UNKNOWN,
            handler="FallbackHandler",
            intent=reason,
            metadata={"reason": reason, "technical_error": error},
            success=False,
        )

    async def aclose(self) -> None:
        """Cleanly close network and resource handles."""
        if hasattr(self.router, "aclose"):
            await self.router.aclose()
