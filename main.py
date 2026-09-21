"""Interactive CLI REPL for manual testing of the toy-agent pipeline."""

import argparse
import asyncio
import logging
import sys

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
    print("🤖 TOY AGENT — PHASE 1 TEXT INTERFACE")
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
                print(f"   [Route: {response.route.value} | Handler: {response.handler} | Intent: {response.intent}]")

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
        help="Custom base URL for the external SLM router (e.g. http://localhost:8000).",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default=None,
        help="Custom JSON file path for memory persistence.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging to observe pipeline transitions.",
    )
    return parser.parse_args()


async def main_async() -> None:
    """Async main entrypoint."""
    args = parse_args()
    configure_logging(args.verbose)

    settings = get_settings()
    memory_store = MemoryStore(args.memory_path)

    if args.mock_router:
        mode_desc = "Internal Mock Router (Standalone Test Mode)"
        router_client = MockRouterClient()
    else:
        target_url = args.base_url or settings.router_url
        mode_desc = f"External SLM Router API ({target_url})"
        router_client = RouterClient(base_url=args.base_url)

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
