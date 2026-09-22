# Toy Agent 🧸⚡

The software brain and conversational orchestrator for an AI-powered smart toy for children.

---

## 1. Architectural Boundary & Repository Responsibilities

This system maintains a strict separation of concerns across two independent repositories:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              slm-router                                │
│  - Independent service exposing a FastAPI endpoint                     │
│  - Hosts the real Qwen2.5-1.5B-Instruct Small Language Model           │
│  - Classifies natural language requests on-device (LOCAL/COMMAND/CLOUD)│
│  - Exposes POST /route with execution latency telemetry                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP (POST /route)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                              toy-agent                                 │
│  - THIS REPOSITORY                                                     │
│  - Conversational Agent Brain & Orchestrator                           │
│  - Consumes the real slm-router API via RouterClient                   │
│  - Dispatches to specialized handlers:                                 │
│      • LocalHandler   (answers on-device queries)                      │
│      • CommandHandler (safe device/action confirmations)               │
│      • MemoryHandler  (structured key-value storage)                   │
│      • CloudHandler   (client stub / bridge for complex queries)       │
│  - Voice interfaces (STTProvider, TTSProvider) for speech I/O          │
│  - Child-safe error handling and failure resilience                    │
│  - Foundation for future ESP32-S3 hardware & voice streaming           │
└────────────────────────────────────────────────────────────────────────┘
```

> **Strict Architectural Rule:**
> `toy-agent` communicates with `slm-router` solely across its HTTP API boundary. No models, classification heuristics, or internal modules from `slm-router` are copied, vendored, or imported into `toy-agent`.

---

## 2. Phase 2 Voice-Ready Architecture

```
                    ┌─────────────────────────┐
                    │    Raw Audio / Mic      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │      STTProvider      │  (Abstract STT Interface / Dev STT)
                     └───────────┬───────────┘
                                 │ Text Transcription
                                 ▼
                     ┌───────────────────────┐
                     │   AgentOrchestrator   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                       ┌───────────────────┐
                       │   RouterClient    │
                       └─────────┬─────────┘
                                 │  HTTP POST /route {"query": "..."}
                                 ▼
                     ┌───────────────────────┐
                     │    slm-router API     │  (FastAPI on :8008)
                     └───────────┬───────────┘
                                 │  Calls Router.route()
                                 ▼
                     ┌───────────────────────┐
                     │  Qwen2.5-1.5B-Instruct│  (On-device SLM)
                     └───────────┬───────────┘
                                 │  Returns real routing decision & text
                                 ▼
                     ┌───────────────────────┐
                     │    RoutingDecision    │  (Normalized in toy-agent)
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   [ Route: LOCAL ]        [ Route: COMMAND ]      [ Route: CLOUD ]
         │                       │                       │
         ▼                       ▼                       ▼
   LocalHandler            CommandHandler          CloudHandler
   (Direct SLM Answer)     (Dynamic SLM Action)    (Cloud Bridge / Stub)
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                     ┌─────────────────────────┐
                     │ Child-Safe Text Response│
                     └───────────┬─────────────┘
                                 │ Text Response
                                 ▼
                     ┌─────────────────────────┐
                     │       TTSProvider       │  (Abstract TTS Interface / Dev PCM WAV)
                     └───────────┬─────────────┘
                                 │ Synthesized Audio (WAV PCM)
                                 ▼
                     ┌─────────────────────────┐
                     │    Audio Output / DAC   │
                     └─────────────────────────┘
```

---

## 3. Real Integration & End-to-End Workflow

To run the real, unmocked conversational toy pipeline:

### Step 1: Start the Real `slm-router` Service
In the `slm-router` repository:

```bash
# Terminal 1: In slm-router directory:
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8008
```

Verify it is active:
```bash
curl http://127.0.0.1:8008/health
# {"status":"healthy","model":"Qwen/Qwen2.5-1.5B-Instruct"}
```

### Step 2: Diagnostic Health Check via `toy-agent`
In the `toy-agent` directory:

```bash
.venv/bin/python main.py --health
# ✅ SLM Router is healthy at http://localhost:8008: {'status': 'healthy', 'model': 'Qwen/Qwen2.5-1.5B-Instruct'}
```

### Step 3: Start `toy-agent` CLI
```bash
# Terminal 2: In toy-agent directory:
.venv/bin/python main.py --verbose
```

### Step 4: Converse
Interact with the real Qwen2.5-1.5B model in real-time:
```text
You: Hello
Toy: Hello! How can I assist you today?

You: Turn on the lights
Toy: I have turned on the lights for you. Enjoy your evening!

You: Tell me a joke
Toy: Why don't scientists trust atoms? Because they make up everything!

You: What is the capital of France?
Toy: The capital of France is Paris.
```

---

## 4. Voice Interfaces & Audio Layer (`app.audio`)

Phase 2 establishes clean, portable interfaces for speech input and speech synthesis without any coupling to macOS-specific APIs (no `say`, CoreAudio, or Apple frameworks):

- **`STTProvider` (`app.audio.stt`)**: Abstract base class defining `async def transcribe(audio_data: bytes, format: str = "wav") -> str`. Includes `DevelopmentSTTProvider` for development and testing.
- **`TTSProvider` (`app.audio.tts`)**: Abstract base class defining `async def synthesize(text: str) -> bytes`. Includes `DevelopmentTTSProvider` which produces valid standard 16kHz mono PCM RIFF WAV audio bytes using Python's built-in `wave` and `struct` libraries without third-party audio dependencies.
- **Orchestrator Voice Method**: `AgentOrchestrator.process_voice(audio_data: bytes, format: str = "wav") -> tuple[AgentResponse, bytes]` pipes audio through STT -> Orchestrator Router/Handlers -> TTS.

---

## 5. Configuration & Environment Variables

Configuration is loaded from environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `ROUTER_BASE_URL` | `http://localhost:8008` | URL of the running `slm-router` API |
| `ROUTER_ROUTE_ENDPOINT` | `/route` | Router classification endpoint path |
| `ROUTER_HEALTH_ENDPOINT` | `/health` | Router health check endpoint path |
| `ROUTER_TIMEOUT_SECONDS` | `15.0` | Timeout allowing full SLM token generation |
| `MEMORY_STORAGE_PATH` | `data/memory.json` | File path for local key-value persistence |
| `ENVIRONMENT` | `development` | Deployment environment |
| `LOG_LEVEL` | `INFO` | Console log verbosity (`DEBUG`, `INFO`) |
| `CLOUD_PROVIDER_STUB_MODE` | `true` | When true, returns safe stub for CLOUD route |

---

## 6. Measured Performance & Latency Telemetry

When connected to the real `Qwen2.5-1.5B-Instruct` model on Apple Silicon (MPS), real queries exhibit the following baseline latencies:

| Query | Classified Route | SLM Cls Latency | SLM Handler Latency | HTTP Latency | End-to-End Total |
|---|---|---|---|---|---|
| `"Hello"` | `LOCAL` | 1.59s | 0.47s | 2.10s | **2.10s** |
| `"Turn on the lights"` | `COMMAND` | 1.03s | 0.79s | 1.83s | **1.83s** |
| `"Tell me a joke"` | `LOCAL` | 1.04s | 0.78s | 1.83s | **1.83s** |
| `"What is the capital of France?"` | `LOCAL` | 1.10s | 0.42s | 1.54s | **1.54s** |

---

## 7. Testing

### Run All Unit & Integration Tests
```bash
.venv/bin/pytest tests/ -v
```

All 43 tests pass cleanly:
- **RouterClient tests**: health check success/failure, timeout handling, connection refused, malformed response, HTTP errors.
- **Audio tests**: `STTProvider`, `TTSProvider` standard WAV generation, orchestrator voice pipeline.
- **Orchestrator tests**: routing decision dispatch, failure recovery, telemetry capture.
- **Memory & Handler tests**: structured KV storage, local handler, command handler, cloud stub.
- **Live Integration test**: verifies end-to-end HTTP communication when `slm-router` is active.

---

## 8. Troubleshooting

- **Router connection refused (`http://localhost:8008/route`)**:
  - Ensure the `slm-router` uvicorn server is running: `curl http://127.0.0.1:8008/health`.
  - Check that port 8008 is not blocked by a firewall.
- **Port 8000 conflict**:
  - Port 8000 is frequently reserved on development machines. `slm-router` and `toy-agent` default to port `8008` to prevent port collisions.
- **Timeout on long queries**:
  - On CPU/MPS devices, generating multi-paragraph responses can take ~10 seconds. Keep `ROUTER_TIMEOUT_SECONDS=15.0`.

---

## 9. Architectural Roadmap

```
[Phase 1]
Text Input ──► toy-agent (Mock Router) ──► Handlers ──► Text Output

[Phase 1.5]
Text Input ──► toy-agent ──► RouterClient ──► real slm-router API (Qwen 1.5B) ──► Handlers ──► Text Output

[Phase 2 - Current]
Audio In ──► STTProvider ──► AgentOrchestrator ──► RouterClient (POST /route) ──► Handlers ──► TTSProvider ──► Audio Out
(Portable abstract interfaces, zero macOS lock-in, health diagnostics, full telemetry)

[Phase 3 - Future Target]
Hardware Integration: ESP32-S3 I2S Mic (INMP441) ──► Agent ──► I2S Amp (MAX98357A) Speaker
Natural Female Voice TTS Benchmark & Selection
```
