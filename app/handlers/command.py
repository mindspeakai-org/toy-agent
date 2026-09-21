"""Command handler for safe device and toy action acknowledgements."""

import logging
from typing import Any, Callable, Dict, Optional

from app.handlers.base import BaseHandler
from app.models.responses import AgentResponse
from app.router.models import RouteType, RoutingDecision

logger = logging.getLogger(__name__)


class CommandHandler(BaseHandler):
    """Processes device actions and commands.

    In Phase 1, executes simulated acknowledgements without physical hardware.
    Designed with execution hooks so hardware drivers (ESP32/BLE/GPIO) can be
    injected seamlessly in subsequent phases.
    """

    def __init__(self, action_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None) -> None:
        self.action_executor = action_executor

    @property
    def name(self) -> str:
        return "CommandHandler"

    async def handle(self, query: str, decision: RoutingDecision) -> AgentResponse:
        """Acknowledge or execute device action."""
        cleaned = query.strip()
        lower = cleaned.lower()

        # If an external hardware executor is provided (Phase 2 readiness)
        if self.action_executor is not None:
            try:
                self.action_executor(query, decision.raw_response)
            except Exception as exc:
                logger.error("Action executor failed on '%s': %s", query, exc)

        # Check if external router already provided dynamic confirmation text
        raw_router_response: Optional[str] = decision.raw_response.get("response")
        if raw_router_response and isinstance(raw_router_response, str) and raw_router_response.strip():
            return AgentResponse(
                text=raw_router_response.strip(),
                route=RouteType.COMMAND,
                handler=self.name,
                intent=decision.intent or "COMMAND_EXECUTED",
                metadata={"executed": True, "source": "router_payload"},
                success=True,
            )

        # Deterministic Phase 1 acknowledgements
        if ("volume" in lower and "up" in lower) or "louder" in lower:
            text = "Okay, I turned the volume up for you!"
            action = "VOLUME_UP"
        elif ("volume" in lower and "down" in lower) or "softer" in lower or "quiet" in lower:
            text = "Okay, I turned the volume down."
            action = "VOLUME_DOWN"
        elif "stop" in lower or "pause" in lower:
            text = "Okay, I've stopped."
            action = "STOP"
        elif "play music" in lower or "start music" in lower:
            text = "Okay! Pretending to play your favorite tune!"
            action = "PLAY_MUSIC"
        elif "sleep" in lower or "bedtime" in lower or "shut down" in lower:
            text = "Okay, going to sleep now. Goodnight!"
            action = "SLEEP"
        elif "light" in lower or "glow" in lower:
            text = "Okay, I adjusted my lights for you!"
            action = "LIGHT_TOGGLE"
        else:
            text = f"Okay, I'll do that for you right away!"
            action = "GENERIC_ACTION"

        logger.info("Command acknowledged: '%s' -> action='%s'", cleaned, action)

        return AgentResponse(
            text=text,
            route=RouteType.COMMAND,
            handler=self.name,
            intent="DEVICE_ACTION",
            metadata={"action": action, "hardware_simulated": True},
            success=True,
        )
