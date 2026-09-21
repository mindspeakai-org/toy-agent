# Toy Agent 🧸⚡

The software brain and conversational orchestrator for an AI-powered smart toy for children.

---

## 1. Architectural Boundary & Repository Responsibilities

This system is built with a clean separation of concerns across two distinct projects:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              slm-router                                │
│  - Standalone service / API                                            │
│  - Responsible ONLY for query classification using an SLM              │
│  - Decides destination: LOCAL, MEMORY, COMMAND, CLOUD                  │
│  - Does not manage toy hardware, long-term memory, or agent flows      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP (POST /route)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                              toy-agent                                 │
│  - THIS REPOSITORY                                                     │
│  - Conversational Agent Brain & Orchestrator                           │
│  - Consumes the slm-router API via RouterClient                        │
│  - Dispatches to specialized handlers:                                 │
│      • MemoryHandler  (structured key-value storage)                   │
│      • LocalHandler   (deterministic chit-chat & facts)                │
│      • CommandHandler (safe device/action confirmations)               │
│      • CloudHandler   (client stub for heavy queries)                  │
│  - Child-safe error handling and failure resilience                    │
│  - Foundation for future STT, TTS, BLE, and ESP32 hardware             │
└────────────────────────────────────────────────────────────────────────┘
```

> **Strict Architectural Rule:**
> `toy-agent` communicates with `slm-router` solely across its HTTP API boundary. No models, classification heuristics, or internal modules from `slm-router` are copied, vendored, or imported.

---

## 2. Phase 1 Workflow

In Phase 1, the pipeline operates end-to-end on clean text before any audio or hardware layers are introduced:

```
                  ┌───────────────┐
                  │   User Text   │
                  └───────┬───────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   AgentOrchestrator   │
              └───────────┬───────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   RouterClient   │
                └─────────┬────────┘
                          │  HTTP POST /route
                          ▼
               ┌─────────────────────┐
               │   slm-router API    │
               └──────────┬──────────┘
                          │  JSON Routing Decision
                          ▼
              ┌───────────────────────┐
              │    RoutingDecision    │
              └───────────┬───────────┘
                          │
         ┌────────────────┼────────────────┬───────────────┐
         ▼                ▼                ▼               ▼
   [ Route: LOCAL ] [ Route: MEMORY ] [ Route: COMMAND ] [ Route: CLOUD ]
         │                │                │               │
         ▼                ▼                ▼               ▼
   LocalHandler     MemoryHandler    CommandHandler  CloudHandler
   (Deterministic   (Atomic JSON     (Simulated      (Safe Stub)
    Chit-chat)       Key-Value)       Actions)
         │                │                │               │
         └────────────────┴────────────────┴───────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Child-Safe Text Response│
                     └─────────────────────────┘
```

---

## 3. Project Structure

```text
toy-agent/
├── app/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   └── orchestrator.py        # Central agent workflow & error fallbacks
│   ├── cloud/
│   │   ├── __init__.py
│   │   └── client.py              # Cloud AI client interface & safe stub
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Pydantic Settings loaded from .env
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract BaseHandler interface
│   │   ├── cloud.py               # Cloud route handler
│   │   ├── command.py             # Device action handler
│   │   ├── local.py               # Deterministic conversational handler
│   │   └── memory.py              # Memory query and update handler
│   ├── memory/
│   │   ├── __init__.py
│   │   └── store.py               # Atomic JSON-backed key-value store
│   ├── models/
│   │   ├── __init__.py
│   │   └── responses.py           # Standardized AgentResponse model
│   └── router/
│       ├── __init__.py
│       ├── client.py              # Async HTTP client for slm-router API
│       ├── exceptions.py          # Custom domain exception hierarchy
│       ├── mock.py                # Deterministic MockRouterClient for testing
│       └── models.py              # RoutingDecision & RouteType models
├── data/                          # Default directory for local memory storage
├── tests/
│   ├── __init__.py
│   ├── test_handlers.py           # Unit tests for all handlers
│   ├── test_memory.py             # Unit tests for memory persistence
│   ├── test_orchestrator.py       # Unit & integration tests for orchestrator
│   └── test_router_client.py      # Unit tests for router HTTP client
├── .env.example                   # Environment variable template
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── main.py                        # Interactive CLI REPL for manual testing
```

---

## 4. Setup Instructions

### Prerequisites
- Python 3.11+
- `uv` (recommended) or standard `python3 -m venv`

### Installation

1. Clone the repository and enter the directory:
   ```bash
   cd toy-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   # Using uv:
   uv venv .venv
   source .venv/bin/activate

   # Or using standard venv:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create your local environment configuration:
   ```bash
   cp .env.example .env
   ```

---

## 5. Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROUTER_BASE_URL` | `http://localhost:8000` | Base URL of the external `slm-router` HTTP service |
| `ROUTER_ROUTE_ENDPOINT` | `/route` | Route classification endpoint path |
| `ROUTER_TIMEOUT_SECONDS` | `5.0` | Maximum wait time for router responses |
| `MEMORY_STORAGE_PATH` | `data/memory.json` | File path for local key-value persistence |
| `ENVIRONMENT` | `development` | Deployment environment (`development`, `test`, `production`) |
| `LOG_LEVEL` | `INFO` | Console logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CLOUD_PROVIDER_STUB_MODE` | `true` | When true, returns deterministic stub without cloud API keys |

---

## 6. How to Run

### Option A: Standalone Mode with Mock Router
Test the complete conversational pipeline without spinning up the external `slm-router` service:

```bash
python main.py --mock-router --verbose
```

### Option B: Connected to Live `slm-router` Service
1. In the `slm-router` project directory, start the API service (e.g. on port 8000):
   ```bash
   # (In slm-router directory)
   uv run uvicorn api:app --port 8000
   ```
2. In `toy-agent`, launch the agent:
   ```bash
   python main.py --verbose
   ```

---

## 7. Example Interaction

```text
🤖 TOY AGENT — PHASE 1 TEXT INTERFACE
Mode: Internal Mock Router (Standalone Test Mode)
Type 'exit', 'quit', or press Ctrl+C to stop.
============================================================

You: Hello!
Toy: Hello there! I'm your toy friend. What would you like to talk about today?

You: My favorite animal is a tiger.
Toy: Got it! I'll remember that your favorite animal is tiger.

You: What is my favorite animal?
Toy: Your favorite animal is tiger.

You: Turn up the volume.
Toy: Okay, I turned the volume up for you!

You: Why is the sky blue?
Toy: That sounds like a wonderful big question! This question would be sent to the cloud AI.

You: exit
Toy: Bye for now! See you next time!
```

---

## 8. Failure Modes & Resilience

The toy is designed so that network anomalies or external service downtime never crash the application or expose stack traces to the child:

- **Router Unavailable / Connection Refused**: Returns `"I'm having trouble connecting right now. Let's try again in a moment!"`
- **Router Timeout**: Returns `"I'm taking a little too long to think right now. Could you ask me again?"`
- **Router Malformed JSON**: Returns `"I had trouble understanding that. Let's try something else!"`
- **Memory Key Miss**: Friendly prompt: `"I don't know your favorite color yet! What is it?"`
- **Handler Execution Failure**: Returns `"Oops, something went a little wobbly on my end. Can you say that again?"`

---

## 9. Running Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

To run only unit tests:
```bash
pytest tests/ -v -m "not integration"
```

To run the live integration test (when `slm-router` is active on port 8000):
```bash
pytest tests/test_orchestrator.py -k test_live_slm_router_integration -v
```

---

## 10. Phase 1 Limitations & Phase 2 Roadmap

### Current Phase 1 Boundaries
- Text-only input and text-only response (no audio processing).
- Device actions are simulated with acknowledgements (no physical actuators or GPIO).
- Key-value memory store without vector embeddings or semantic search.
- Cloud handler operates as a safe stub without production cloud API dependencies.

### Future Phase 2 Roadmap
In Phase 2, the pipeline expands naturally into voice and hardware without altering the core orchestrator:

```
Microphone → Audio Frontend (Noise Suppression) → STT (Sherpa-ONNX / Whisper)
    ↓
Agent Orchestrator
    ↓
RouterClient (slm-router)
    ↓
Specialized Handler (Memory / Hardware Actuator / Cloud LLM)
    ↓
Response Text
    ↓
TTS Engine → Speaker
```
