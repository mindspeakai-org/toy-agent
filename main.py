"""Interactive CLI REPL for manual testing of the toy-agent pipeline."""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from app.agent.orchestrator import AgentOrchestrator
from app.config.settings import get_settings
from app.memory.store import MemoryStore
from app.router.client import RouterClient
from app.router.mock import MockRouterClient


def configure_logging(verbose: bool) -> None:
    """Set up console logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def interactive_repl(orchestrator: AgentOrchestrator, mode_desc: str, verbose: bool) -> None:
    """Run interactive text loop."""
    print("=" * 60)
    print("🤖 TOY AGENT — PHASE 2 INTERFACE")
    print(f"Mode: {mode_desc}")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 60)

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", ":q"}:
                print("Toy: Bye for now! See you next time!")
                break

            response = await orchestrator.process(user_input)

            if verbose:
                if response.decision:
                    keys = response.decision.memory_request.keys if response.decision.memory_request else []
                    print(f"   [Processing: {response.decision.processing.value} | MemoryRequired: {response.decision.memory_required} | Keys: {keys}]")
                else:
                    print(f"   [Response: success={response.success} | handler={response.handler}]")

            print(f"Toy: {response.text}")

    finally:
        await orchestrator.aclose()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Toy Agent Interactive CLI")
    parser.add_argument(
        "--mock-router",
        action="store_true",
        help="Use built-in MockRouterClient instead of connecting to an external HTTP router service.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for the external SLM router (e.g. http://localhost:8008).",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run health check diagnostic against external SLM router and exit.",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default=None,
        help="Custom JSON file path for memory persistence.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Run real voice capture (Microphone -> STT) and display recognized text.",
    )
    parser.add_argument(
        "--voice-router",
        action="store_true",
        help="Run real voice pipeline (Microphone -> STT -> Router) and display routing decision.",
    )
    parser.add_argument(
        "--voice-duration",
        type=float,
        default=4.0,
        help="Duration in seconds to record from microphone (default: 4.0s).",
    )
    parser.add_argument(
        "--voice-loop",
        action="store_true",
        help="Run voice pipeline continuously in an interactive loop.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging to observe pipeline transitions.",
    )
    return parser.parse_args()


async def run_voice_mode(duration: float, loop: bool, verbose: bool) -> None:
    """Execute real microphone recording and Whisper speech-to-text pipeline."""
    from app.audio.exceptions import AudioError
    from app.audio.microphone import MicrophoneAudioInput
    from app.audio.whisper_stt import WhisperSTTProvider

    print("=" * 60)
    print("🎤 TOY AGENT — REAL VOICE PIPELINE (PHASE 1)")
    print("Audio Input:  MicrophoneAudioInput (16kHz mono)")
    print("STT Engine:   WhisperSTTProvider (faster-whisper 'tiny.en')")
    print(f"Duration:     {duration:.1f}s")
    print("=" * 60)

    try:
        audio_input = MicrophoneAudioInput(default_duration=duration)
        stt_provider = WhisperSTTProvider(model_size="tiny.en")
    except Exception as exc:
        print(f"❌ Failed to initialize voice components: {exc}")
        return

    # Pre-warm local Whisper model
    if verbose:
        print("Pre-warming local Whisper model...")
    try:
        _ = stt_provider._get_model()
    except Exception as exc:
        print(f"❌ STT initialization failure: {exc}")
        return

    while True:
        try:
            if loop and sys.stdin.isatty():
                prompt = input("\nPress Enter to record (or type 'exit' to quit): ").strip()
                if prompt.lower() in {"exit", "quit", ":q"}:
                    print("Exiting voice mode.")
                    break

            print("\n🎤 Speak now...")
            audio_bytes = await audio_input.read(duration=duration)
            print("[capture audio]")
            print("[run real STT]")

            recognized_text = await stt_provider.transcribe(audio_bytes)
            print("\nRecognized text:")
            print(f'"{recognized_text}"')

            if not loop:
                break

        except AudioError as exc:
            print(f"\n❌ Voice error: {exc}")
            if not loop:
                break
        except KeyboardInterrupt:
            print("\nRecording cancelled.")
            break


async def run_voice_router_mode(
    duration: float,
    loop: bool,
    verbose: bool,
    base_url: Optional[str] = None,
    mock_router: bool = False,
) -> None:
    """Execute real microphone recording -> Whisper STT -> SLM Router classification."""
    import json
    from app.audio.exceptions import AudioError
    from app.audio.microphone import MicrophoneAudioInput
    from app.audio.whisper_stt import WhisperSTTProvider

    print("=" * 60)
    print("TOY AGENT — REAL VOICE → STT → ROUTER")
    print("=" * 60)

    try:
        audio_input = MicrophoneAudioInput(default_duration=duration)
        stt_provider = WhisperSTTProvider(model_size="tiny.en")
    except Exception as exc:
        print(f"❌ Failed to initialize audio components: {exc}")
        return

    # Initialize RouterClient
    if mock_router:
        router_client = MockRouterClient()
    else:
        router_client = RouterClient(base_url=base_url)
        try:
            health_info = await router_client.check_health()
            if verbose:
                print(f"[SLM-Router connected: {health_info.get('model', 'SLM')}]")
        except Exception as exc:
            print(f"⚠️ Warning: SLM-Router not reachable at {router_client.health_url}: {exc}")

    orchestrator = AgentOrchestrator(
        router_client=router_client,
        stt_provider=stt_provider,
        audio_input=audio_input,
    )

    # Pre-warm local Whisper model
    if verbose:
        print("Pre-warming local Whisper model...")
    try:
        _ = stt_provider._get_model()
    except Exception as exc:
        print(f"❌ STT initialization failure: {exc}")
        return

    while True:
        try:
            if loop and sys.stdin.isatty():
                prompt = input("\nPress Enter to record (or type 'exit' to quit): ").strip()
                if prompt.lower() in {"exit", "quit", ":q"}:
                    print("Exiting voice-router mode.")
                    break

            print("\n🎤 Speak now...")
            audio_bytes = await audio_input.read(duration=duration)
            print("\n[capture audio]")

            recognized_text = await stt_provider.transcribe(audio_bytes)
            print("\nRecognized text:")
            print(f'"{recognized_text}"')

            if not recognized_text:
                print("\n[No speech recognized. Skipping router call.]")
                if not loop:
                    break
                continue

            print("\nSending to SLM Router...")
            decision = await router_client.route(recognized_text)

            decision_dict = {
                "processing": decision.processing.value,
                "memory_required": decision.memory_required,
                "memory_request": (
                    {"keys": decision.memory_request.keys}
                    if decision.memory_request
                    else None
                ),
            }

            print("\nRouting decision:")
            print(json.dumps(decision_dict, indent=2))

            # Step 4: Device-side memory retrieval (Phase 3)
            if decision.memory_required:
                memory_ctx = orchestrator.memory_handler.retrieve(decision)
                print("\nDevice Memory Context:")
                if memory_ctx:
                    print(json.dumps(memory_ctx.model_dump(), indent=2))
                else:
                    print("null")
            else:
                print("\nDevice Memory Context:")
                print("[Bypassed: memory_required=false]")

            print("\nPipeline:")
            print("✓ physical microphone")
            print("✓ real speech-to-text")
            print("✓ router classification")
            print("✓ device-side memory retrieval")
            print("\nSTOP HERE.")
            print("\nDo not generate a response.")
            print("Do not execute anything.")

            if not loop:
                break

        except AudioError as exc:
            print(f"\n❌ Audio error: {exc}")
            if not loop:
                break
        except Exception as exc:
            print(f"\n❌ Router error: {exc}")
            if not loop:
                break
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        finally:
            await orchestrator.aclose()


async def main_async() -> None:
    """Async main entrypoint."""
    args = parse_args()
    configure_logging(args.verbose)

    # Real voice pipeline mode (Phase 1: Mic -> STT)
    if args.voice:
        await run_voice_mode(
            duration=args.voice_duration,
            loop=args.voice_loop,
            verbose=args.verbose,
        )
        return

    # Real voice-to-router pipeline mode (Phase 1 -> Phase 2: Mic -> STT -> Router)
    if args.voice_router:
        await run_voice_router_mode(
            duration=args.voice_duration,
            loop=args.voice_loop,
            verbose=args.verbose,
            base_url=args.base_url,
            mock_router=args.mock_router,
        )
        return

    settings = get_settings()

    # Diagnostic health check mode
    if args.health:
        target_url = args.base_url or settings.router_base_url
        client = RouterClient(base_url=target_url)
        print(f"Checking health of SLM Router at {client.health_url}...")
        try:
            health_data = await client.check_health()
            print(f"✅ SLM Router is HEALTHY:")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Model:  {health_data.get('model')}")
        except Exception as exc:
            print(f"❌ SLM Router health check FAILED: {exc}")
            sys.exit(1)
        finally:
            await client.aclose()
        return

    memory_store = MemoryStore(args.memory_path)

    if args.mock_router:
        mode_desc = "Internal Mock Router (Standalone Test Mode)"
        router_client = MockRouterClient()
    else:
        target_url = args.base_url or settings.router_url
        mode_desc = f"External SLM Router API ({target_url})"
        router_client = RouterClient(base_url=args.base_url)

        # Startup health verification
        try:
            health_info = await router_client.check_health()
            logging.info("Connected to SLM Router [%s] - Status: %s", health_info.get("model", "SLM"), health_info.get("status"))
        except Exception as exc:
            logging.warning("SLM Router not reachable at startup (%s): %s", router_client.health_url, exc)
            logging.warning("Continuing in resilient mode (queries will use safe fallbacks if router remains offline).")

    orchestrator = AgentOrchestrator(
        router_client=router_client,
        memory_store=memory_store,
    )

    await interactive_repl(orchestrator, mode_desc, args.verbose)


def main() -> None:
    """CLI script entrypoint."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nSession ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()
